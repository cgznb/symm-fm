"""Scratch VQ-GAN training with generator-counted updates and FP32 EMA VQ."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import registered_roi32_data as data_module
from . import registered_roi32_runtime as runtime
from . import perceptual as perceptual_module
from . import vqgan as vq_module
from .perceptual import UncheckedLPIPSLoss
from .registered_roi32_data import CropDataset, file_identity, read_json, write_json
from .vqgan import MRILevelVQGAN, PatchDiscriminator, VQGANConfig, VectorQuantizerEMA, _hinge_discriminator, _orthogonal_slices, _random_slice_indices


class FP32Quantizer(VectorQuantizerEMA):
    def forward(self, latent):
        with torch.autocast(device_type=latent.device.type, enabled=False):
            return super().forward(latent.float())


def build_autoencoder(config):
    architecture = VQGANConfig(**config["vq"]["model"], commitment_weight=config["vq"]["commitment_weight"])
    model = MRILevelVQGAN(architecture)
    model.quantizer = FP32Quantizer(architecture)
    return model


def adversarial_factor(step, start, ramp):
    return min(1.0, max(0.0, (step - start) / max(1, ramp)))


class ROI32VQ(torch.nn.Module):
    def __init__(self, config, perceptual=None):
        super().__init__()
        self.autoencoder = build_autoencoder(config)
        self.image_discriminator = PatchDiscriminator(2, 64, 3)
        self.volume_discriminator = PatchDiscriminator(3, 64, 3)
        self.perceptual = perceptual if perceptual is not None else UncheckedLPIPSLoss.vgg()
        self.perceptual.requires_grad_(False).eval()
        self.training_config = config["vq"]

    def train(self, mode=True):
        super().train(mode)
        self.perceptual.eval()
        return self

    def discriminators(self, *, train, requires_grad):
        for network in (self.image_discriminator, self.volume_discriminator):
            network.train(train).requires_grad_(requires_grad)

    @staticmethod
    def perceptual_view(value):
        return value.float().repeat(1, 3, 1, 1).clamp(-1, 1)

    def reconstruction_losses(self, image, indices=None):
        reconstruction, quantizer = self.autoencoder(image)
        indices = indices if indices is not None else _random_slice_indices(image)
        real = _orthogonal_slices(image, indices)
        fake = _orthogonal_slices(reconstruction, indices)
        # Keep the frozen perceptual network in FP32 as well.
        with torch.autocast(device_type=image.device.type, enabled=False):
            lpips = torch.stack([self.perceptual(self.perceptual_view(f), self.perceptual_view(r)).mean()
                                 for f, r in zip(fake, real, strict=True)]).sum()
        losses = {"reconstruction": F.l1_loss(reconstruction.float(), image.float()),
                  "slice_lpips": lpips, "commitment": quantizer["commitment_loss"], "perplexity": quantizer["perplexity"]}
        losses["composite"] = (self.training_config["reconstruction_weight"] * losses["reconstruction"]
                                + self.training_config["perceptual_weight"] * lpips + losses["commitment"])
        return losses, reconstruction, real, fake, quantizer

    def generator_loss(self, image, step):
        self.discriminators(train=False, requires_grad=False)
        losses, reconstruction, real, fake, quantizer = self.reconstruction_losses(image)
        config = self.training_config
        factor = adversarial_factor(step, config["discriminator_start"], config["discriminator_ramp"])
        total = losses["composite"]
        if factor:
            fake2, features2 = self.image_discriminator(fake[0])
            fake3, features3 = self.volume_discriminator(reconstruction)
            with torch.no_grad():
                _, reference2 = self.image_discriminator(real[0])
                _, reference3 = self.volume_discriminator(image)
            matching = sum(F.l1_loss(f.float(), r.float()) for f, r in zip(features2[:-1] + features3[:-1], reference2[:-1] + reference3[:-1], strict=True))
            total = total + factor * (-config["image_gan_weight"] * fake2.float().mean()
                                      - config["volume_gan_weight"] * fake3.float().mean()
                                      + config["feature_matching_weight"] * matching)
            losses["feature_matching"] = matching
        losses["generator_total"] = total
        return total, losses

    def discriminator_loss(self, image, step):
        self.discriminators(train=True, requires_grad=True)
        mode = self.autoencoder.training
        self.autoencoder.eval()
        with torch.no_grad():
            reconstruction, _ = self.autoencoder(image)
        self.autoencoder.train(mode)
        indices = _random_slice_indices(image)
        real = _orthogonal_slices(image, indices)[0]
        fake = _orthogonal_slices(reconstruction, indices)[0]
        real2, _ = self.image_discriminator(real)
        fake2, _ = self.image_discriminator(fake.detach())
        real3, _ = self.volume_discriminator(image)
        fake3, _ = self.volume_discriminator(reconstruction.detach())
        config = self.training_config
        factor = adversarial_factor(step, config["discriminator_start"], config["discriminator_ramp"])
        return factor * (config["image_gan_weight"] * _hinge_discriminator(real2.float(), fake2.float())
                         + config["volume_gan_weight"] * _hinge_discriminator(real3.float(), fake3.float()))


def optimizers(system, config):
    generator = torch.optim.Adam(system.autoencoder.parameters(), lr=config["vq"]["learning_rate"], betas=(0.5, 0.9))
    parameters = list(system.image_discriminator.parameters()) + list(system.volume_discriminator.parameters())
    discriminator = torch.optim.Adam(parameters, lr=config["vq"]["discriminator_learning_rate"], betas=(0.5, 0.9))
    return generator, discriminator


def update(system, batches, generator, discriminator, *, step, device, config):
    system.train()
    decay = step >= config["vq"]["lr_decay_update"]
    generator.param_groups[0]["lr"] = config["vq"]["learning_rate"] * (0.5 if decay else 1)
    discriminator.param_groups[0]["lr"] = config["vq"]["discriminator_learning_rate"] * (0.1 if decay else 1)
    generator.zero_grad(set_to_none=True)
    discriminator.zero_grad(set_to_none=True)
    total_examples = sum(len(batch["image"]) for batch in batches)
    metrics = {}
    for batch in batches:
        image = batch["image"].to(device, non_blocking=True)
        weight = len(image) / total_examples
        with runtime.autocast(device):
            loss, values = system.generator_loss(image, step)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite VQ generator loss")
        (loss * weight).backward()
        for key, value in values.items():
            metrics[key] = metrics.get(key, 0.0) + float(value.detach()) * weight
    metrics["generator_gradient_norm"] = runtime.finite_gradients(system.autoencoder.parameters(), config["runtime"]["gradient_clip"])
    generator.step()
    factor = adversarial_factor(step, config["vq"]["discriminator_start"], config["vq"]["discriminator_ramp"])
    metrics["adversarial_factor"] = factor
    metrics["discriminator_total"] = 0.0
    if factor:
        for batch in batches:
            image = batch["image"].to(device, non_blocking=True)
            weight = len(image) / total_examples
            with runtime.autocast(device):
                loss = system.discriminator_loss(image, step)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite VQ discriminator loss")
            (loss * weight).backward()
            metrics["discriminator_total"] += float(loss.detach()) * weight
        parameters = list(system.image_discriminator.parameters()) + list(system.volume_discriminator.parameters())
        metrics["discriminator_gradient_norm"] = runtime.finite_gradients(parameters, config["runtime"]["gradient_clip"])
        discriminator.step()
    generator.zero_grad(set_to_none=True)
    discriminator.zero_grad(set_to_none=True)
    return metrics


def fixed_slices(batch, device):
    # Stable per-visit slice choices also survive physical-batch fallback.
    import zlib
    axes = [[], [], []]
    shape = batch["image"].shape[2:]
    for visit in batch["visit_id"]:
        generator = np.random.default_rng(2026 + zlib.crc32(visit.encode()))
        for axis, size in enumerate(shape):
            axes[axis].append(int(generator.integers(size)))
    return tuple(torch.tensor(indices, device=device) for indices in axes)


@torch.no_grad()
def validate(system, dataset, *, batch_size, device, config):
    was_training = system.training
    system.eval()
    totals, examples = {}, 0
    counts = torch.zeros(config["vq"]["model"]["n_codes"], dtype=torch.int64, device=device)
    before = {k: v.clone() for k, v in system.autoencoder.quantizer.state_dict().items()}
    with runtime.fixed_rng(2026):
        loader = runtime.evaluation_loader(dataset, batch_size, config["runtime"]["loader_workers"])
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            with runtime.autocast(device):
                losses, reconstruction, _, _, quantizer = system.reconstruction_losses(image, fixed_slices(batch, device))
            mask = batch["mask"].to(device)
            roi_count = mask.flatten(1).sum(1)
            roi = ((reconstruction.float() - image).abs() * mask).flatten(1).sum(1) / roi_count.clamp_min(1)
            losses["t0_roi_l1"] = roi.mean()
            for key, value in losses.items():
                if not torch.isfinite(value):
                    raise FloatingPointError("Non-finite VQ validation metric")
                totals[key] = totals.get(key, 0.0) + float(value) * len(image)
            counts += torch.bincount(quantizer["indices"].reshape(-1), minlength=len(counts))
            examples += len(image)
    for key, value in before.items():
        if not torch.equal(value, system.autoencoder.quantizer.state_dict()[key]):
            raise RuntimeError("Validation modified the frozen codebook")
    system.train(was_training)
    probabilities = counts.double() / counts.sum()
    metrics = {key: value / examples for key, value in totals.items()}
    metrics.update(visits=examples, active_codes=int((counts > 0).sum()),
                   codebook_perplexity=float(torch.exp(-(probabilities * probabilities.clamp_min(1e-30).log()).sum())))
    return metrics


def contract_for(config):
    return runtime.stage_contract(config, "vq", [__file__, data_module.__file__, runtime.__file__, vq_module.__file__, perceptual_module.__file__],
                                  initialization="scratch_all_model_and_codebook_weights", latent_shape=[8, 8, 32, 32])


def train(config, device, batch_size):
    root = Path(config["output_dir"]) / "vq"
    contract = contract_for(config)
    if runtime.stage_complete(config, "vq", contract):
        return True
    runtime.seed_all(config["seed"])
    training, validation = CropDataset(config, "train"), CropDataset(config, "val")
    system = ROI32VQ(config).to(device)
    generator, discriminator = optimizers(system, config)
    stream = runtime.TrainingBatches(training, batch_size=batch_size, effective_batch=config["vq"]["batch_size"],
                                      seed=config["seed"], workers=config["runtime"]["loader_workers"])
    state = {"step": 0, "best": None, "bad_validations": 0, "last_validation": None}
    last, best = root / "last.pt", root / "best.pt"
    if last.exists():
        saved = runtime.load_checkpoint(last, contract, device)
        system.load_state_dict(saved["model"])
        generator.load_state_dict(saved["generator"])
        discriminator.load_state_dict(saved["discriminator"])
        stream.load_state_dict(saved["stream"])
        state = saved["training_state"]
        runtime.restore_rng(saved["rng"])
        del saved
    clock = runtime.UpdateClock(state["step"])

    def checkpoint(path):
        runtime.save_checkpoint(path, {"contract": contract, "model": system.state_dict(),
                                      "generator": generator.state_dict(), "discriminator": discriminator.state_dict(),
                                      "stream": stream.state_dict(), "training_state": dict(state), "rng": runtime.rng_state()})

    try:
        while state["step"] < config["vq"]["max_updates"]:
            batches = [next(stream) for _ in range(config["vq"]["batch_size"] // batch_size)]
            metrics = update(system, batches, generator, discriminator, step=state["step"] + 1, device=device, config=config)
            state["step"] += 1
            step = state["step"]
            if step == 1 or step % config["runtime"]["log_interval"] == 0:
                runtime.log_event(config, "vq", "training", step=step, epoch=stream.epoch, batch_size=batch_size,
                                  effective_batch=config["vq"]["batch_size"], **metrics, **clock.metrics(step, config["vq"]["max_updates"]))
                runtime.periodic_guard(config, contract)
            if step % config["vq"]["validation_interval"] == 0:
                metrics = validate(system, validation, batch_size=batch_size, device=device, config=config)
                state["last_validation"] = metrics
                if step >= config["vq"]["discriminator_start"] + config["vq"]["discriminator_ramp"]:
                    improved = state["best"] is None or metrics["composite"] < state["best"]
                    if improved:
                        state["best"] = metrics["composite"]
                        state["bad_validations"] = 0
                        checkpoint(best)
                    elif step >= config["vq"]["early_stopping_start"]:
                        state["bad_validations"] += 1
                runtime.log_event(config, "vq", "validation", step=step, **metrics)
            if step % config["runtime"]["checkpoint_interval"] == 0 or runtime.STOP_REQUESTED:
                checkpoint(last)
            if runtime.STOP_REQUESTED:
                runtime.log_event(config, "vq", "paused", step=step)
                return False
            if state["bad_validations"] >= config["vq"]["early_stopping_patience"]:
                break
        checkpoint(last)
        if not best.exists():
            raise RuntimeError("No post-warmup VQ checkpoint was selected")
        chosen = runtime.load_checkpoint(best, contract, device)
        chosen_metrics = chosen["training_state"]["last_validation"]
        if chosen_metrics["active_codes"] <= 1:
            raise RuntimeError("Selected VQ codebook collapsed; FM is blocked")
        runtime.stage_finished(config, "vq", contract, [best, last], best_validation=chosen_metrics,
                               best_step=chosen["training_state"]["step"], last_step=state["step"])
        runtime.log_event(config, "vq", "complete", step=state["step"], best_step=chosen["training_state"]["step"])
        return True
    finally:
        stream.close()


def load_frozen(config, device):
    path = Path(config["output_dir"]) / "vq" / "best.pt"
    contract = contract_for(config)
    if not runtime.stage_complete(config, "vq", contract):
        raise ValueError("VQ stage has not passed its gate")
    payload = runtime.load_checkpoint(path, contract, "cpu")
    model = build_autoencoder(config)
    prefix = "autoencoder."
    model.load_state_dict({k[len(prefix):]: v for k, v in payload["model"].items() if k.startswith(prefix)}, strict=True)
    if any(not torch.isfinite(v).all() for v in model.state_dict().values() if v.is_floating_point()):
        raise FloatingPointError("VQ checkpoint contains non-finite state")
    return model.to(device).eval().requires_grad_(False), file_identity(path)
