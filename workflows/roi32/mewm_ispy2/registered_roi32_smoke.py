"""Real-data optimizer, frozen-codebook and recovery gates before long runs."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Subset

from . import registered_roi32_fm as fm
from . import registered_roi32_runtime as runtime
from . import registered_roi32_vq as vq
from .registered_roi32_data import CropDataset, read_json, write_json
from .registered_roi32_latents import LatentPairs, fit_channel_statistics


def snapshot(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items() if not key.startswith("perceptual.")}


def assert_replay(expected, actual):
    if set(expected) != set(actual):
        raise AssertionError("Recovery changed state keys")
    maximum = 0.0
    for key, value in expected.items():
        observed = actual[key]
        if value.is_floating_point():
            maximum = max(maximum, float((value - observed).abs().max()))
            if not torch.allclose(value, observed, atol=1e-6, rtol=1e-5):
                raise AssertionError(f"Recovery replay differs: {key}, max error {maximum}")
        elif not torch.equal(value, observed):
            raise AssertionError(f"Recovery changed integer state: {key}")
    return maximum


def real_crop_subset(dataset, size):
    selected = []
    for visit in ("T0", "T1", "T2", "T3"):
        candidates = [i for i, row in enumerate(dataset.records) if row["visit"] == visit]
        if candidates:
            selected.append(candidates[0])
    selected += [i for i in range(len(dataset)) if i not in selected]
    return Subset(dataset, selected[:size])


def vq_smoke(config, device, batch_size):
    contract = vq.contract_for(config)
    output = Path(config["output_dir"]) / "smoke"
    runtime.seed_all(config["seed"] + 41)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    dataset = CropDataset(config, "train")
    subset = real_crop_subset(dataset, config["vq"]["batch_size"])
    batches = list(runtime.evaluation_loader(subset, batch_size))
    system = vq.ROI32VQ(config).to(device)
    generator, discriminator = vq.optimizers(system, config)
    before_g = system.autoencoder.encoder.input.weight.detach().clone()
    before_d = snapshot(system.image_discriminator)
    before_d3 = snapshot(system.volume_discriminator)
    warmup = vq.update(system, batches, generator, discriminator, step=1, device=device, config=config)
    if torch.equal(before_g, system.autoencoder.encoder.input.weight):
        raise AssertionError("Generator did not update")
    assert_replay(before_d, snapshot(system.image_discriminator))
    assert_replay(before_d3, snapshot(system.volume_discriminator))
    if discriminator.state or warmup["adversarial_factor"] != 0:
        raise AssertionError("Discriminator updated during GAN warmup")
    full_step = config["vq"]["discriminator_start"] + config["vq"]["discriminator_ramp"]
    before_d = system.image_discriminator.blocks[0][0].weight.detach().clone()
    before_d3 = system.volume_discriminator.blocks[0][0].weight.detach().clone()
    full = vq.update(system, batches, generator, discriminator, step=full_step, device=device, config=config)
    if full["adversarial_factor"] != 1 or torch.equal(before_d, system.image_discriminator.blocks[0][0].weight):
        raise AssertionError("GAN discriminator did not update after warmup")
    if torch.equal(before_d3, system.volume_discriminator.blocks[0][0].weight):
        raise AssertionError("3D GAN discriminator did not update after warmup")
    recovery = output / "vq-recovery.pt"
    runtime.save_checkpoint(recovery, {"contract": contract, "model": system.state_dict(), "generator": generator.state_dict(),
                                      "discriminator": discriminator.state_dict(), "rng": runtime.rng_state()})
    vq.update(system, batches, generator, discriminator, step=full_step + 1, device=device, config=config)
    expected = snapshot(system)
    saved = runtime.load_checkpoint(recovery, contract, device)
    system.load_state_dict(saved["model"])
    generator.load_state_dict(saved["generator"])
    discriminator.load_state_dict(saved["discriminator"])
    runtime.restore_rng(saved["rng"])
    del saved
    vq.update(system, batches, generator, discriminator, step=full_step + 1, device=device, config=config)
    replay_error = assert_replay(expected, snapshot(system))
    first = vq.validate(system, subset, batch_size=batch_size, device=device, config=config)
    second = vq.validate(system, subset, batch_size=batch_size, device=device, config=config)
    if first != second:
        raise AssertionError("Fixed-slice validation is not reproducible")
    with torch.no_grad(), runtime.autocast(device):
        system.autoencoder.eval()
        image = batches[0]["image"].to(device)
        latent = system.autoencoder.encode_continuous(image)
        reconstruction, _ = system.autoencoder(image)
    if tuple(latent.shape[1:]) != (8, 8, 32, 32) or reconstruction.shape != image.shape:
        raise AssertionError("Actual VQ shape test failed")
    result = {"status": "passed", "contract": contract, "device": str(device), "batch_size": batch_size,
              "effective_batch": config["vq"]["batch_size"], "generator_updated": True, "gan_warmup_checked": True,
              "both_discriminators_updated": True,
              "recovery_max_error": replay_error, "deterministic_validation": True,
              "validation_codebook_frozen": True, "image_shape": list(image.shape), "latent_shape": list(latent.shape),
              "validation": first, "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2 if device.type == "cuda" else None}
    write_json(output / ("vq.json" if device.type == "cuda" else "vq-cpu.json"), result)
    recovery.unlink()
    return result


def fm_smoke(config, device, batch_size):
    module = fm.bridge(config)
    output = Path(config["output_dir"]) / "smoke"
    contract = fm.contract_for(config)
    runtime.seed_all(config["seed"] + 42)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    dataset = LatentPairs(config, "train")
    subset = Subset(dataset, list(range(config["fm"]["batch_size"])))
    batches = list(runtime.evaluation_loader(subset, batch_size, collate_fn=module.collate_pairs))
    model, optimizer, scheduler, ema, trainer = fm.build(config, dataset.inventory["pairs"], device)
    trainer.gradient_accumulation = config["fm"]["batch_size"] // batch_size

    def one_update():
        for batch in batches:
            values = trainer.train_batch(batch["later_latent"], batch["earlier_latent"], batch["conditions"])
        if not values["optimizer_updated"]:
            raise AssertionError("FM accumulation does not end at an optimizer update")
        return values

    first = one_update()
    recovery = output / "fm-recovery.pt"
    runtime.save_checkpoint(recovery, {"contract": contract, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                      "scheduler": scheduler.state_dict(), "ema": ema.state_dict(), "micro_step": trainer.micro_step,
                                      "scaler": trainer.scaler.state_dict(), "rng": runtime.rng_state()})
    one_update()
    expected = snapshot(model)
    expected_ema = {key: value.detach().cpu().clone() for key, value in ema.shadow.items()}
    saved = runtime.load_checkpoint(recovery, contract, device)
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    ema.load_state_dict(saved["ema"])
    trainer.micro_step = saved["micro_step"]
    trainer.scaler.load_state_dict(saved["scaler"])
    runtime.restore_rng(saved["rng"])
    del saved
    one_update()
    error = assert_replay(expected, snapshot(model))
    ema_error = assert_replay(expected_ema, {key: value.detach().cpu() for key, value in ema.shadow.items()})
    model.eval()
    batch = batches[0]
    source = batch["earlier_latent"].to(device)
    with torch.no_grad(), runtime.autocast(device), ema.average_parameters(model.velocity_model):
        generated = model.sample(source, batch["conditions"], torch.zeros_like(source),
                                 steps=config["fm"]["sampling_steps"], solver=config["fm"]["solver"])
    if generated.shape != source.shape or not torch.isfinite(generated).all():
        raise AssertionError("FM Heun endpoint is invalid")
    result = {"status": "passed", "contract": contract, "device": str(device), "batch_size": batch_size,
              "effective_batch": config["fm"]["batch_size"], "recovery_max_error": error, "ema_recovery_max_error": ema_error,
              "joint_shape": [len(source), 16, 8, 32, 32], "first_update": first, "heun_endpoint_finite": True,
              "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2 if device.type == "cuda" else None}
    write_json(output / ("fm.json" if device.type == "cuda" else "fm-cpu.json"), result)
    recovery.unlink()
    return result


def fm_cpu_architecture(config):
    """Exercise the real FM network before the production VQ has been trained."""
    import numpy as np
    device = torch.device("cpu")
    torch.set_num_threads(config["runtime"]["cpu_threads"])
    runtime.seed_all(config["seed"] + 43)
    root = Path(config["output_dir"])
    inventory = read_json(root / "data" / "inventory.json")
    records = []
    for earlier, later in (("T0", "T1"), ("T0", "T3"), ("T1", "T2"), ("T2", "T3")):
        records.append(next(r for r in inventory["pairs"] if r["split"] == "train" and r["earlier_stage"] == earlier and r["later_stage"] == later))
    visit_ids = sorted({r[key] for r in records for key in ("earlier_visit_id", "later_visit_id")})
    chosen = [r for r in inventory["visits"] if r["visit_id"] in visit_ids]
    crops = CropDataset(config, records=chosen)
    encoder = vq.build_autoencoder(config).eval().requires_grad_(False)
    values = {}
    with torch.no_grad():
        for i in range(len(crops)):
            item = crops[i]
            values[item["visit_id"]] = encoder.encode_continuous(item["image"][None])[0].numpy()
    del encoder
    moments = fit_channel_statistics(values.items())
    mean = np.array(moments["mean"], dtype=np.float32).reshape(8, 1, 1, 1)
    std = np.array(moments["std"], dtype=np.float32).reshape(8, 1, 1, 1)
    module = fm.bridge(config)
    items = [{"record": row, "earlier_latent": torch.from_numpy((values[row["earlier_visit_id"]] - mean) / std),
              "later_latent": torch.from_numpy((values[row["later_visit_id"]] - mean) / std)} for row in records]
    batch = module.collate_pairs(items)
    model, optimizer, scheduler, ema, trainer = fm.build(config, inventory["pairs"], device)
    contract = runtime.stage_contract(config, "fm_cpu_architecture", [__file__, fm.__file__, module.__file__],
                                      codec="untrained_scratch_encoder_diagnostic_only")
    before = snapshot(model)
    first = trainer.train_batch(batch["later_latent"], batch["earlier_latent"], batch["conditions"])
    after = snapshot(model)
    if not any(not torch.equal(value, after[key]) for key, value in before.items()):
        raise AssertionError("Real FM architecture did not update")
    del before, after
    recovery = root / "smoke" / "fm-cpu-architecture-recovery.pt"
    runtime.save_checkpoint(recovery, {"contract": contract, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                      "scheduler": scheduler.state_dict(), "ema": ema.state_dict(), "rng": runtime.rng_state(),
                                      "micro_step": trainer.micro_step})
    trainer.train_batch(batch["later_latent"], batch["earlier_latent"], batch["conditions"])
    expected = snapshot(model)
    expected_ema = {key: value.clone() for key, value in ema.shadow.items()}
    saved = runtime.load_checkpoint(recovery, contract, device)
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    ema.load_state_dict(saved["ema"])
    trainer.micro_step = saved["micro_step"]
    runtime.restore_rng(saved["rng"])
    del saved
    trainer.train_batch(batch["later_latent"], batch["earlier_latent"], batch["conditions"])
    error = assert_replay(expected, snapshot(model))
    ema_error = assert_replay(expected_ema, ema.shadow)
    del expected, expected_ema
    model.eval()
    with torch.no_grad(), ema.average_parameters(model.velocity_model):
        endpoint = model.sample(batch["earlier_latent"], batch["conditions"], torch.zeros_like(batch["earlier_latent"]),
                                steps=config["fm"]["sampling_steps"], solver=config["fm"]["solver"])
    if not torch.isfinite(endpoint).all():
        raise FloatingPointError("Real FM Heun endpoint is non-finite")
    result = {"status": "passed", "contract": contract, "batch_size": len(records), "real_training_visits": len(visit_ids),
              "latent_shape": list(endpoint.shape), "recovery_max_error": error, "ema_recovery_max_error": ema_error,
              "first_update": first, "heun_steps": config["fm"]["sampling_steps"],
              "limitation": "CPU FP32 architecture/recovery check using real MRI and a fresh untrained encoder; production VQ latents and GPU BF16 smoke remain separate gates."}
    write_json(root / "smoke" / "fm-cpu-architecture.json", result)
    recovery.unlink()
    return result
