from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from .source_bridge import write_json


def finite_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: finite_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_values(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def patient_macro(rows):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (row["method"], row["modality"], row["region"])
        grouped[key][row["patient_id"]].append(row)
    result = []
    for (method, modality, region), patients in sorted(grouped.items()):
        metrics = sorted(
            {
                key
                for values in patients.values()
                for row in values
                for key, value in row.items()
                if isinstance(value, float)
            }
        )
        for metric in metrics:
            means = []
            for values in patients.values():
                observed = [
                    row[metric]
                    for row in values
                    if metric in row
                    and isinstance(row[metric], float)
                    and math.isfinite(row[metric])
                ]
                if observed:
                    means.append(sum(observed) / len(observed))
            result.append(
                {
                    "method": method,
                    "modality": modality,
                    "region": region,
                    "metric": metric,
                    "patient_count": len(means),
                    "mean": sum(means) / len(means) if means else None,
                }
            )
    return result


def change_metrics(predicted, source, target, region):
    region = torch.as_tensor(region, dtype=torch.bool)
    if not region.any():
        return {
            "change_cosine": None,
            "predicted_change_mae": None,
            "true_change_mae": None,
        }
    estimate = (predicted - source)[region].double()
    actual = (target - source)[region].double()
    denominator = estimate.norm() * actual.norm()
    return {
        "change_cosine": (estimate.dot(actual) / denominator).item()
        if denominator > 0
        else None,
        "predicted_change_mae": estimate.abs().mean().item(),
        "true_change_mae": actual.abs().mean().item(),
    }


def render_comparison(
    path, source, target, reconstruction, mean, std, mask, modalities
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mask = mask.reshape(-1, *source.shape[-3:]).any(0)
    slice_index = (
        int(mask.sum((1, 2)).argmax()) if mask.any() else source.shape[-3] // 2
    )
    figure, axes = plt.subplots(
        len(modalities),
        5,
        figsize=(15, 3 * len(modalities)),
        squeeze=False,
        layout="constrained",
    )
    for row, modality in enumerate(modalities):
        foreground = source[row][source[row] != 0]
        limits = (
            torch.quantile(foreground, torch.tensor([0.01, 0.99]))
            if foreground.numel()
            else torch.tensor([0.0, 1.0])
        )
        for column, (name, volume) in enumerate(
            zip(
                ("Source", "Target", "Target VQ", "Prediction mean", "Sample SD"),
                (source, target, reconstruction, mean, std),
                strict=True,
            )
        ):
            vmax = max(float(std[row].max()), 1e-6) if column == 4 else float(limits[1])
            axes[row, column].imshow(
                volume[row, slice_index].numpy(),
                origin="lower",
                cmap="magma" if column == 4 else "gray",
                vmin=0 if column == 4 else float(limits[0]),
                vmax=vmax,
            )
            axes[row, column].set_title(f"{modality}: {name}", fontsize=10)
            axes[row, column].axis("off")
    figure.savefig(path, dpi=120)
    plt.close(figure)


def decode_and_evaluate(project, root: Path, representatives: dict[str, int]):
    family = project.experiment["family"]
    if family == "ispy2":
        from .backend import load_transition_records
        from .ispy2_biflow_cohort_evaluation import (
            comparison_metrics,
            region_masks,
            valid_ssim_centers,
        )
        from .ispy2_biflow_latent_contract import decode_ispy2_biflow_continuous
        from .vqgan import REGISTERED_VQGAN_NUMERIC_CONTRACT
        from .workflows import load_mri_vqgan

        data = project.base.base.data
        codec = (
            load_mri_vqgan(
                data.vqgan_checkpoint,
                expected_numeric_contract=REGISTERED_VQGAN_NUMERIC_CONTRACT,
            )
            .cuda()
            .eval()
        )
        visits = load_transition_records(
            data.bundle_json, data.phase_manifest_csv, backend="registered_t0"
        ).visits

        def decode(latent):
            value = decode_ispy2_biflow_continuous(
                codec, project.latents.denormalize(latent.cuda())
            )
            return value[0, :, 0].float().cpu()

        modalities = ("dce0",)
    else:
        from .mu_glioma_biflow_cohort_evaluation import (
            _decoded_modalities,
            _load_formal_visit,
        )
        from .mu_glioma_biflow_evaluation import compute_mu_glioma_biflow_volume_metrics
        from .mu_glioma_biflow_workflow import (
            _load_vqgan,
            decode_mu_glioma_biflow_prediction,
        )

        codec = _load_vqgan(project.base, device=torch.device("cuda")).eval()

        def decode(latent):
            value = decode_mu_glioma_biflow_prediction(
                codec,
                latent.cuda(),
                codebook_min=project.latents.codebook_min,
                codebook_max=project.latents.codebook_max,
            )
            return _decoded_modalities(value, "bridge candidate")

        modalities = tuple(project.base.data.modalities)
    codec.requires_grad_(False)
    rows = []
    for index, pair in enumerate(project.val_pairs):
        from .source_bridge_workflow import pair_key

        shard = torch.load(
            root / "candidates" / f"{index:05d}.pt",
            weights_only=True,
            map_location="cpu",
            mmap=True,
        )
        if shard["pair_id"] != pair_key(pair):
            raise ValueError("Generated latent cohort order changed")
        if family == "ispy2":
            source_record = project.conditions.load(pair.source_visit)
            target_record = project.conditions.load(visits[pair.target_visit_id])
            source, target = source_record.image.float(), target_record.image.float()
            source_mask = source_record.mask.bool()
            regions = region_masks(
                source_record.valid_foreground[0].numpy(),
                target_record.valid_foreground[0].numpy(),
                source_mask[0].numpy(),
                target_record.mask[0].numpy(),
            )
            common = torch.from_numpy(regions["common_foreground"]).unsqueeze(0)
            target_mask = target_record.mask
            target_latent = project.val_dataset.dataset._target_latent(
                pair.target_visit_id
            ).unsqueeze(0)
        else:
            source, source_fg, _ = _load_formal_visit(
                pair.source_samples, include_target_mask=False
            )
            target, target_fg, target_mask = _load_formal_visit(
                pair.target_samples, include_target_mask=True
            )
            _, source_mask = project.conditions.load_visit(pair.source_samples)
            common = source_fg & target_fg
            target_latent = project.val_dataset.dataset._latent(
                pair.target_samples
            ).unsqueeze(0)
        mean = second = None
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for sample_index, candidate in enumerate(shard["samples"]):
                decoded = decode(candidate)
                if mean is None:
                    mean, second = decoded, torch.zeros_like(decoded)
                else:
                    delta = decoded - mean
                    mean = mean + delta / (sample_index + 1)
                    second = second + delta * (decoded - mean)
            reconstruction = decode(target_latent)
        std = (second / len(shard["samples"])).clamp_min(0).sqrt()
        for method, prediction in (
            ("generation", mean),
            ("source_copy", source),
            ("target_vq_floor", reconstruction),
        ):
            for modality_index, modality in enumerate(modalities):
                if family == "ispy2":
                    metrics = comparison_metrics(
                        prediction[0],
                        target[0],
                        regions,
                        valid_ssim_centers(regions["common_foreground"]),
                    )
                else:
                    tumor = None if target_mask is None else target_mask != 0
                    metrics = compute_mu_glioma_biflow_volume_metrics(
                        prediction[modality_index],
                        target[modality_index],
                        common_foreground=common[modality_index],
                        target_tumor=tumor,
                    )
                for region, values in metrics.items():
                    rows.append(
                        finite_values(
                            {
                                "pair_id": pair_key(pair),
                                "patient_id": pair.patient_id,
                                "method": method,
                                "modality": modality,
                                "region": region,
                                **values,
                            }
                        )
                    )
                if method == "generation":
                    for region, mask in (
                        ("change_common_foreground", common[modality_index]),
                        (
                            "change_tumor_union",
                            (source_mask != 0).reshape(-1, *source.shape[-3:]).any(0)
                            if target_mask is None
                            else ((source_mask != 0) | (target_mask != 0))
                            .reshape(-1, *source.shape[-3:])
                            .any(0),
                        ),
                    ):
                        rows.append(
                            {
                                "pair_id": pair_key(pair),
                                "patient_id": pair.patient_id,
                                "method": method,
                                "modality": modality,
                                "region": region,
                                **change_metrics(
                                    prediction[modality_index],
                                    source[modality_index],
                                    target[modality_index],
                                    mask,
                                ),
                            }
                        )
        if pair_key(pair) in representatives:
            slot = representatives[pair_key(pair)]
            render_comparison(
                root / f"representative_{slot}.png",
                source,
                target,
                reconstruction,
                mean,
                std,
                source_mask,
                modalities,
            )
        if (index + 1) % 10 == 0:
            print(
                f"decoded/compared {index + 1}/{len(project.val_pairs)} validation pairs",
                flush=True,
            )
    write_json(root / "image_pair_metrics.json", rows)
    result = {
        "pairs": len(project.val_pairs),
        "patient_macro": patient_macro(rows),
        "prediction": "voxelwise_mean_of_individually_decoded_candidates",
        "sample_sd": "uncalibrated_decoded_candidate_population_standard_deviation",
        "representative_selection": "minimum_median_maximum_interval_before_inference",
        "slice_selection": "source_mask_maximum_area_axial",
    }
    write_json(root / "image_summary.json", result)
    # These temporary candidates can be regenerated from the retained checkpoint and seeds.
    for path in (root / "candidates").glob("*.pt"):
        path.unlink()
    return result
