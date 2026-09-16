"""Registered crop, VQ reconstruction and forecast reports on fixed examples."""

from __future__ import annotations

import csv
import html
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import structural_similarity

from . import registered_roi32_fm as fm
from . import registered_roi32_runtime as runtime
from . import registered_roi32_vq as vq
from .perceptual import UncheckedLPIPSLoss
from .registered_roi32_data import CropDataset, SPACING, check_disk, file_identity, read_json, save_npz, write_json
from .registered_roi32_latents import LatentPairs, decode_normalized


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def crop_reports(config):
    root = Path(config["output_dir"]) / "data"
    return {p["patient_id"]: read_json(root / "patients" / (p["patient_id"] + ".json"))
            for p in read_json(root / "inventory.json")["patients"]}


def select_examples(records, reports, count=12):
    chosen = []
    for visit in ("T0", "T1", "T2", "T3"):
        for clipped in (False, True):
            candidates = [r for r in records if r["visit"] == visit and reports[r["patient_id"]]["largest_clipped"] == clipped]
            if candidates:
                chosen.append(min(candidates, key=lambda r: (reports[r["patient_id"]]["largest_source_center_retention"], r["visit_id"])))
    seen = {r["visit_id"] for r in chosen}
    chosen += [r for r in sorted(records, key=lambda r: r["visit_id"]) if r["visit_id"] not in seen][:count - len(chosen)]
    return chosen[:count]


def plot_case(path, title, volumes, mask):
    mask = np.asarray(mask).squeeze()
    anchors = [int(np.argmax(mask.sum(axis=tuple(i for i in range(3) if i != axis)))) for axis in range(3)]
    columns = len(volumes)
    figure, axes = plt.subplots(3, columns, figsize=(4 * columns, 9), squeeze=False, constrained_layout=True)
    figure.suptitle(textwrap.fill(title, width=45 * columns), fontsize=12)
    for axis, plane in enumerate(("Axial", "Coronal", "Sagittal")):
        physical = [SPACING[i] for i in range(3) if i != axis]
        contour = np.take(mask, anchors[axis], axis=axis)
        for column, (name, volume) in enumerate(volumes.items()):
            image = np.take(np.asarray(volume).squeeze(), anchors[axis], axis=axis)
            panel = axes[axis, column]
            panel.imshow(image, cmap="gray", origin="lower", vmin=-1, vmax=3, aspect=physical[0] / physical[1])
            if contour.any() and not contour.all():
                panel.contour(contour, levels=[0.5], colors=["#38c896"], linewidths=0.6, origin="lower")
            panel.set_title(f"{name} | {plane}", fontsize=10)
            panel.set_axis_off()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120, facecolor="white")
    plt.close(figure)


def gallery(path, title, examples):
    entries = "\n".join(f'<figure><a href="{html.escape(name)}"><img src="{html.escape(name)}" loading="lazy" alt="{html.escape(label)}"></a><figcaption>{html.escape(label)}</figcaption></figure>' for name, label in examples)
    text = ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{html.escape(title)}</title><style>body{{font:15px system-ui;margin:24px;background:#f4f6f5;color:#18231f}}'
            'h1{font-size:24px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,520px),1fr));gap:20px}'
            'figure{margin:0;border:1px solid #cbd3cf;border-radius:4px;background:white;overflow:hidden}img{display:block;width:100%;height:auto}'
            'figcaption{padding:10px;overflow-wrap:anywhere}@media(max-width:500px){body{margin:12px}}</style>'
            f'<h1>{html.escape(title)}</h1><main>{entries}</main></html>')
    Path(path).write_text(text)


def preview_crops(config):
    root = Path(config["output_dir"]) / "crop_preview"
    dataset = CropDataset(config, "val")
    reports = crop_reports(config)
    chosen = select_examples(dataset.records, reports)
    selected = CropDataset(config, records=chosen)
    entries = []
    for i in range(len(selected)):
        batch = selected[i]
        pid = batch["patient_id"]
        status = "T0 largest region clipped" if reports[pid]["largest_clipped"] else "T0 largest region contained"
        title = f"{batch['visit_id']} | 32 x 128 x 128 | {status}"
        name = batch["visit_id"].replace(":", "_") + ".png"
        plot_case(root / name, title, {"Registered DCE0": batch["image"].numpy()}, batch["mask"].numpy())
        entries.append((name, title))
    gallery(root / "index.html", "Registered DCE0 | 32 x 128 x 128", entries)
    write_json(root / "selection.json", {"visit_ids": [r["visit_id"] for r in chosen], "split": "val",
                                          "contour": "retained T0 model mask, fixed across visits", "spacing_zyx_mm": SPACING.tolist()})
    return root / "index.html"


@torch.no_grad()
def image_metrics(predicted, target, mask, valid, perceptual, visit_id):
    error = (predicted.float() - target.float()).abs()
    if not torch.isfinite(predicted).all():
        raise FloatingPointError("Non-finite decoded image")
    result = {"l1": float(error.mean()), "valid_l1": float(error[valid].mean()),
              "t0_roi_l1": float(error[mask].mean()) if mask.any() else None}
    prediction_array = predicted.detach().float().cpu().numpy().squeeze().clip(-1, 3)
    target_array = target.detach().float().cpu().numpy().squeeze().clip(-1, 3)
    result["ssim_clamped_neg1_pos3"] = float(structural_similarity(target_array, prediction_array, data_range=4))
    batch = {"image": target, "visit_id": [visit_id]}
    indices = vq.fixed_slices(batch, predicted.device)
    real = vq._orthogonal_slices(target, indices)
    fake = vq._orthogonal_slices(predicted, indices)
    with torch.autocast(device_type=predicted.device.type, enabled=False):
        result["lpips_three_slices_sum"] = float(sum(perceptual(vq.ROI32VQ.perceptual_view(f), vq.ROI32VQ.perceptual_view(r)).mean() for f, r in zip(fake, real, strict=True)))
    return result


def aggregate(rows, groups):
    metrics = ("l1", "valid_l1", "t0_roi_l1", "ssim_clamped_neg1_pos3", "lpips_three_slices_sum")
    output = []
    for fields in groups:
        buckets = defaultdict(list)
        for row in rows:
            buckets[tuple(row[key] for key in fields)].append(row)
        for key, bucket in sorted(buckets.items(), key=lambda item: str(item[0])):
            result = {"group": dict(zip(fields, key, strict=True)), "count": len(bucket)}
            for metric in metrics:
                values = [row[metric] for row in bucket if row[metric] is not None]
                result[metric] = float(np.mean(values)) if values else None
            output.append(result)
    return output


@torch.no_grad()
def evaluate_vq(config, device):
    root = Path(config["output_dir"]) / "vq_evaluation"
    model, identity = vq.load_frozen(config, device)
    contract = runtime.stage_contract(config, "vq_evaluation", [__file__, vq.__file__], codec=identity)
    if runtime.stage_complete(config, "vq_evaluation", contract):
        return True
    dataset = CropDataset(config, "val")
    reports = crop_reports(config)
    chosen = {r["visit_id"] for r in select_examples(dataset.records, reports)}
    perceptual = UncheckedLPIPSLoss.vgg().to(device).eval()
    rows, examples = [], []
    counts = torch.zeros(config["vq"]["model"]["n_codes"], device=device, dtype=torch.int64)
    for index in range(len(dataset)):
        batch = dataset[index]
        image = batch["image"][None].to(device)
        with runtime.autocast(device):
            reconstruction, quantizer = model(image)
        counts += torch.bincount(quantizer["indices"].flatten(), minlength=len(counts))
        metrics = image_metrics(reconstruction, image, batch["mask"][None].to(device), batch["valid"][None].to(device), perceptual, batch["visit_id"])
        rows.append({"visit_id": batch["visit_id"], "patient_id": batch["patient_id"], "visit": batch["visit"],
                     "t0_largest_clipped": reports[batch["patient_id"]]["largest_clipped"], **metrics})
        if batch["visit_id"] in chosen:
            name = batch["visit_id"].replace(":", "_") + ".png"
            plot_case(root / name, batch["visit_id"], {"Real": image.cpu().numpy(), "VQ reconstruction": reconstruction.float().cpu().numpy()}, batch["mask"].numpy())
            examples.append((name, batch["visit_id"]))
        if (index + 1) % 25 == 0:
            runtime.log_event(config, "vq_evaluation", "evaluating", completed=index + 1, total=len(dataset))
        if runtime.STOP_REQUESTED:
            return False
    active = int((counts > 0).sum())
    if active <= 1:
        raise RuntimeError("VQ codebook collapsed in reconstruction audit")
    write_csv(root / "visits.csv", rows)
    write_json(root / "summary.json", {"split": "val", "active_codes": active,
                                         "groups": aggregate(rows, [[], ["visit"], ["t0_largest_clipped"]])})
    gallery(root / "index.html", "Registered ROI32 | VQ reconstructions", examples)
    runtime.stage_finished(config, "vq_evaluation", contract, [root / "visits.csv", root / "summary.json", root / "index.html"], visits=len(rows))
    return True


@torch.no_grad()
def evaluate_fm(config, device):
    module = fm.bridge(config)
    root = Path(config["output_dir"]) / "fm_evaluation"
    codec, codec_identity = vq.load_frozen(config, device)
    contract = runtime.stage_contract(config, "fm_evaluation", [__file__, fm.__file__], codec=codec_identity,
                                      flow=file_identity(Path(config["output_dir"]) / "fm" / "best-endpoint.pt"))
    if runtime.stage_complete(config, "fm_evaluation", contract):
        return True
    model = fm.load_selected(config, device)
    dataset = LatentPairs(config, "val")
    images = CropDataset(config, "val")
    image_indices = {row["visit_id"]: i for i, row in enumerate(images.records)}
    reports = crop_reports(config)
    perceptual = UncheckedLPIPSLoss.vgg().to(device).eval()
    examples, rows = [], []
    for index, record in enumerate(dataset.records):
        check_disk(config, 24 * 1024**2)
        case_name = record["pair_id"].replace(":", "_").replace("->", "_")
        case_report = root / "cases" / (case_name + ".json")
        if case_report.exists():
            saved = read_json(case_report)
            if saved["contract"] != contract:
                raise ValueError("Evaluation cache belongs to another selected model")
            runtime.verify_identity(saved["samples_identity"])
            rows.extend(saved["rows"])
            if saved.get("example"):
                examples.append(tuple(saved["example"]))
            continue
        batch = module.collate_pairs([dataset[index]])
        source_item = images[image_indices[record["earlier_visit_id"]]]
        target_item = images[image_indices[record["later_visit_id"]]]
        source = batch["earlier_latent"].to(device)
        target_image = target_item["image"][None].to(device)
        source_image = source_item["image"][None].to(device)
        mask = target_item["mask"][None].to(device)
        valid = target_item["valid"][None].to(device)
        with runtime.autocast(device):
            reconstruction, _ = codec(target_image)
        predictions = []
        for sample in range(config["fm"]["final_samples"]):
            generator = torch.Generator(device=device).manual_seed(config["fm"]["validation_seed"] + index * config["fm"]["final_samples"] + sample)
            noise = torch.randn(source.shape, device=device, generator=generator)
            with runtime.autocast(device):
                latent = model.sample(source, batch["conditions"], noise, steps=config["fm"]["sampling_steps"], solver=config["fm"]["solver"])
            predictions.append(decode_normalized(codec, latent, dataset.statistics))
        predictive_mean = torch.stack(predictions).mean(0)
        comparisons = [("sample", i, value) for i, value in enumerate(predictions)]
        comparisons += [("sample_mean", -1, predictive_mean), ("source_copy", -1, source_image), ("vq_reconstruction", -1, reconstruction)]
        case_rows = []
        for method, sample, predicted in comparisons:
            metrics = image_metrics(predicted, target_image, mask, valid, perceptual, record["later_visit_id"])
            case_rows.append({"pair_id": record["pair_id"], "patient_id": record["patient_id"],
                              "time_pair": record["earlier_stage"] + "->" + record["later_stage"],
                              "t0_largest_clipped": reports[record["patient_id"]]["largest_clipped"],
                              "method": method, "sample_index": sample, **metrics})
        samples_path = root / "samples" / (case_name + ".npz")
        stored = torch.stack(predictions).float().cpu().numpy().astype(np.float16)
        if not np.isfinite(stored).all():
            raise FloatingPointError("Generated images overflow FP16 storage")
        save_npz(samples_path, samples=stored[:, 0], predictive_mean=predictive_mean[0].float().cpu().numpy(),
                 affine_lps=np.asarray(reports[record["patient_id"]]["crop_affine_lps"]),
                 earlier_visit_id=record["earlier_visit_id"], later_visit_id=record["later_visit_id"])
        example = None
        if index < 12:
            name = case_name + ".png"
            plot_case(root / name, record["pair_id"], {"Source": source_image.cpu().numpy(), "Target": target_image.cpu().numpy(),
                                                      "VQ target": reconstruction.float().cpu().numpy(),
                                                      "Sample 1": predictions[0].float().cpu().numpy(), "Sample mean": predictive_mean.float().cpu().numpy()}, mask.cpu().numpy())
            example = (name, record["pair_id"])
            examples.append(example)
        write_json(case_report, {"contract": contract, "rows": case_rows, "samples_identity": file_identity(samples_path), "example": example})
        rows.extend(case_rows)
        runtime.log_event(config, "fm_evaluation", "evaluating", completed=index + 1, total=len(dataset))
        if runtime.STOP_REQUESTED:
            return False
    write_csv(root / "metrics.csv", rows)
    write_json(root / "summary.json", {"split": "val", "pairs": len(dataset), "samples_per_pair": config["fm"]["final_samples"],
                                         "groups": aggregate(rows, [["method"], ["method", "time_pair"], ["method", "t0_largest_clipped"],
                                                                     ["method", "time_pair", "t0_largest_clipped"]]), "test_split": None})
    gallery(root / "index.html", "Registered ROI32 | SymmFlow forecasts", examples)
    runtime.stage_finished(config, "fm_evaluation", contract, [root / "metrics.csv", root / "summary.json", root / "index.html"], pairs=len(dataset))
    return True
