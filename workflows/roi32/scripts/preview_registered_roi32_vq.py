"""Preview an immutable best-VQ snapshot on CPU while formal training continues."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from mewm_ispy2 import registered_roi32_runtime as runtime
from mewm_ispy2 import registered_roi32_vq as vq
from mewm_ispy2.perceptual import UncheckedLPIPSLoss
from mewm_ispy2.registered_roi32_data import CropDataset, SPACING, file_identity, read_config, timestamp, write_json
from mewm_ispy2.registered_roi32_evaluation import crop_reports, gallery, image_metrics, select_examples, write_csv


def crop_status(report):
    if report["largest_clipped"]:
        return "Largest T0 region clipped"
    if report["all_clipped"]:
        return "Largest T0 region contained; other regions clipped"
    return "T0 mask contained"


def render_case(path, title, original, reconstruction, mask, window):
    anchors = [int(mask.sum(axis=tuple(i for i in range(3) if i != axis)).argmax()) for axis in range(3)]
    error = np.abs(reconstruction - original)
    figure, axes = plt.subplots(3, 3, figsize=(11.4, 9.4), constrained_layout=True)
    figure.suptitle(title + f"\nGreen: T0 predicted mask | Shared window [{window[0]:.2f}, {window[1]:.2f}]", fontsize=12)
    for axis, plane in enumerate(("Axial", "Coronal", "Sagittal")):
        spacing = [SPACING[i] for i in range(3) if i != axis]
        contour = np.take(mask, anchors[axis], axis=axis)
        for column, (name, volume) in enumerate((("Original DCE0", original), ("VQ reconstruction", reconstruction), ("Absolute error", error))):
            panel = axes[axis, column]
            values = np.take(volume, anchors[axis], axis=axis)
            artist = panel.imshow(values, origin="lower", interpolation="nearest", aspect=spacing[0] / spacing[1],
                                  cmap="hot" if column == 2 else "gray", vmin=0 if column == 2 else window[0],
                                  vmax=0.5 if column == 2 else window[1])
            if contour.any() and not contour.all():
                panel.contour(contour, levels=[0.5], colors=["#38c896"], linewidths=0.65, origin="lower")
            panel.set_title(f"{name} | {plane} {anchors[axis]}", fontsize=10)
            panel.set_axis_off()
        colorbar_artist = artist
    figure.colorbar(colorbar_artist, ax=axes[:, 2], shrink=0.65, extend="max", label="Absolute error (normalized intensity)")
    figure.savefig(path, dpi=140, facecolor="white")
    plt.close(figure)
    return anchors


def render_overview(path, cases, step):
    figure, axes = plt.subplots(4, 3, figsize=(12, 13.6), constrained_layout=True)
    figure.suptitle(f"VQ-GAN step {step} | Validation examples\nOriginal (left) / reconstruction (right) | Green: T0 predicted mask", fontsize=13)
    for panel, case in zip(axes.flat, cases, strict=True):
        axis = case["anchors"][0]
        original, reconstruction, mask = (case[key][axis] for key in ("original", "reconstruction", "mask"))
        window = case["display_window"]
        gap = np.full((original.shape[0], 3), window[0], dtype=np.float32)
        joined = np.concatenate([original, gap, reconstruction], axis=1)
        joined_mask = np.concatenate([mask, np.zeros_like(gap, dtype=bool), mask], axis=1)
        panel.imshow(joined, cmap="gray", origin="lower", interpolation="nearest", vmin=window[0], vmax=window[1])
        if joined_mask.any():
            panel.contour(joined_mask, levels=[0.5], colors=["#38c896"], linewidths=0.5, origin="lower")
        status = "largest clipped" if case["largest_clipped"] else "largest contained"
        panel.set_title(f"{case['visit_id']} | {status}\nL1 {case['metrics']['l1']:.4f}", fontsize=9)
        panel.set_axis_off()
    figure.savefig(path, dpi=140, facecolor="white")
    plt.close(figure)


@torch.inference_mode()
def preview(config, threads):
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    checkpoint = Path(config["output_dir"]) / "vq" / "best.pt"
    # The trainer replaces checkpoints atomically; the open descriptor pins this version.
    with checkpoint.open("rb") as handle:
        stat = os.fstat(handle.fileno())
        source_identity = {"path": str(checkpoint.resolve()), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        payload = torch.load(handle, map_location="cpu", weights_only=False)
    contract = payload["contract"]
    if contract != vq.contract_for(config):
        raise ValueError("Best checkpoint does not match the current experiment contract")
    step = payload["training_state"]["step"]
    validation = payload["training_state"]["last_validation"]
    model = vq.build_autoencoder(config).eval().requires_grad_(False)
    prefix = "autoencoder."
    model.load_state_dict({key[len(prefix):]: value for key, value in payload["model"].items() if key.startswith(prefix)}, strict=True)
    if any(not torch.isfinite(value).all() for value in model.state_dict().values() if value.is_floating_point()):
        raise FloatingPointError("Non-finite snapshot model state")
    del payload
    output = Path(config["output_dir"]) / "vq_previews" / f"step_{step:06d}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "codec_snapshot.pt"
    runtime.save_checkpoint(snapshot, {"model": model.state_dict(), "model_configuration": config["vq"]["model"],
                                       "commitment_weight": config["vq"]["commitment_weight"], "selected_update": step,
                                       "source_checkpoint": source_identity, "source_contract": contract,
                                       "formal_best_validation": validation})
    codebook_before = {key: value.clone() for key, value in model.quantizer.state_dict().items()}
    dataset = CropDataset(config, "val")
    reports = crop_reports(config)
    selected = select_examples(dataset.records, reports, count=12)
    crops = CropDataset(config, records=selected)
    perceptual = UncheckedLPIPSLoss.vgg().eval().requires_grad_(False)
    rows, entries, fixed_entries, cases = [], [], [], []
    for index in range(len(crops)):
        item = crops[index]
        original = item["image"][None]
        reconstruction, _ = model(original)
        if reconstruction.shape != original.shape or not torch.isfinite(reconstruction).all():
            raise FloatingPointError("Invalid preview reconstruction")
        metrics = image_metrics(reconstruction, original, item["mask"][None], item["valid"][None], perceptual, item["visit_id"])
        report = reports[item["patient_id"]]
        name = item["visit_id"].replace(":", "_") + ".png"
        title = f"{item['visit_id']} | Step {step} | {crop_status(report)}"
        original_array = original[0, 0].numpy()
        reconstruction_array = reconstruction[0, 0].numpy()
        mask = item["mask"][0].numpy()
        window = np.percentile(original_array[item["valid"][0].numpy()], [0.5, 99.5]).tolist()
        if window[1] - window[0] < 1e-6:
            raise ValueError("Reference image has no usable display range")
        anchors = render_case(output / name, title, original_array, reconstruction_array, mask, window)
        fixed_name = name.replace(".png", "_fixed.png")
        render_case(output / fixed_name, title, original_array, reconstruction_array, mask, [-1, 3])
        rows.append({"visit_id": item["visit_id"], "visit": item["visit"], "t0_largest_clipped": report["largest_clipped"],
                     "t0_any_component_clipped": report["all_clipped"], "display_min": window[0], "display_max": window[1], **metrics})
        entries.append((name, f"{title} | L1 {metrics['l1']:.4f} | SSIM {metrics['ssim_clamped_neg1_pos3']:.4f}"))
        fixed_entries.append((fixed_name, entries[-1][1]))
        cases.append({"visit_id": item["visit_id"], "anchors": anchors, "original": original_array,
                      "reconstruction": reconstruction_array, "mask": mask, "display_window": window,
                      "largest_clipped": report["largest_clipped"], "metrics": metrics})
        print(json.dumps({"completed": index + 1, "total": len(crops), "checkpoint_step": step, "updated_at": timestamp()}), flush=True)
    for key, value in codebook_before.items():
        if not torch.equal(value, model.quantizer.state_dict()[key]):
            raise RuntimeError("Preview changed the frozen codebook")
    render_overview(output / "overview.png", cases, step)
    gallery(output / "index.html", f"VQ-GAN reconstructions | Step {step} | 12 validation visits", entries)
    gallery(output / "fixed_window.html", f"VQ-GAN reconstructions | Step {step} | Fixed window [-1, 3]", fixed_entries)
    write_csv(output / "metrics.csv", rows)
    images = [output / name for name, _ in entries + fixed_entries] + [output / "overview.png"]
    for path in images:
        with Image.open(path) as rendered:
            pixels = np.asarray(rendered.convert("RGB"))
            if min(rendered.size) < 500 or pixels.std() < 10:
                raise ValueError(f"Blank or undersized visualization: {path}")
    write_json(output / "report.json", {"status": "passed", "updated_at": timestamp(), "checkpoint_step": step,
                                         "source_checkpoint": source_identity, "codec_snapshot": file_identity(snapshot),
                                         "preview_script": file_identity(__file__), "formal_best_validation": validation,
                                         "prepared_data": contract["prepared_data"], "library_versions": contract["library_versions"],
                                         "normalization": {key: crops.normalization[key] for key in ("mean", "std", "background_and_padding")},
                                         "selection": "same fixed 12 validation visits as crop preview and final VQ audit",
                                         "visit_ids": [r["visit_id"] for r in selected], "device": "cpu", "precision": "fp32",
                                         "is_final_vq_acceptance": False, "shape_zyx": [32, 128, 128], "spacing_zyx_mm": SPACING.tolist(),
                                         "intensity_display_window": "reference foreground percentiles 0.5 and 99.5, shared by original and reconstruction",
                                         "fixed_intensity_display_window": [-1, 3], "error_display_window": [0, 0.5],
                                         "crop_geometry_sources": [file_identity(Path(config["output_dir"]) / "data/patients" / (pid + ".json"))
                                                                   for pid in sorted({r["patient_id"] for r in selected})],
                                         "contour": "T0 first-post model prediction, fixed across visits; not a reconstruction-derived or follow-up segmentation",
                                         "limitation": "Interim CPU FP32 reconstruction preview; differs numerically from formal GPU BF16 validation and is not an FM forecast.",
                                         "examples": rows})
    print(json.dumps({"status": "passed", "checkpoint_step": step, "output": str(output), "gallery": str(output / "index.html")}), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO / "configs/registered_dce0_roi32_firstpostmask_v1.yaml"))
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    preview(read_config(args.config), args.threads)


if __name__ == "__main__":
    main()
