from __future__ import annotations

import numpy as np
import torch


def truncate_temporal_splits(splits, max_tp):
    """Return copies with timepoints at or after ``max_tp`` masked and zeroed."""
    depth = int(max_tp)
    if depth not in (1, 2, 3, 4):
        raise ValueError("max_tp must be one of 1, 2, 3, or 4")
    truncated = {}
    for split, values in splits.items():
        embeddings = np.asarray(values["embs"])
        masks = np.asarray(values["masks"])
        days = np.asarray(values["days"])
        if embeddings.ndim != 3 or masks.shape != embeddings.shape[:2]:
            raise ValueError(f"invalid embedding/mask shape for {split}")
        if days.shape != masks.shape or embeddings.shape[1] != 4:
            raise ValueError(f"invalid elapsed-day shape for {split}")
        embeddings = embeddings.copy()
        masks = masks.copy()
        days = days.copy()
        embeddings[:, depth:] = 0
        masks[:, depth:] = 0
        days[:, depth:] = 0
        truncated[split] = {
            **values,
            "embs": embeddings,
            "masks": masks,
            "days": days,
        }
    return truncated


def mask_torch_temporal_prefix(embeddings, masks, days, depths):
    """Return tensors with each patient's calendar-time prefix applied.

    ``depths`` contains values from one through the tensor's time dimension. A
    missing visit inside a retained prefix stays missing; later visits are not
    shifted forward to fill the gap.
    """
    if embeddings.ndim != 3 or masks.shape != embeddings.shape[:2]:
        raise ValueError("invalid embedding/mask shape")
    if days.shape != masks.shape:
        raise ValueError("invalid elapsed-day shape")
    depth_tensor = torch.as_tensor(depths, device=masks.device)
    if depth_tensor.ndim == 0:
        depth_tensor = depth_tensor.expand(embeddings.shape[0])
    if depth_tensor.shape != (embeddings.shape[0],):
        raise ValueError("depths must be scalar or have one value per patient")
    if depth_tensor.dtype == torch.bool or torch.is_floating_point(depth_tensor):
        raise ValueError("depths must contain integers")
    timepoints = embeddings.shape[1]
    if bool(((depth_tensor < 1) | (depth_tensor > timepoints)).any()):
        raise ValueError(f"depths must be between one and {timepoints}")
    keep = torch.arange(timepoints, device=masks.device).unsqueeze(0) < depth_tensor.unsqueeze(1)
    return (
        embeddings * keep.unsqueeze(-1).to(embeddings.dtype),
        masks * keep.to(masks.dtype),
        days * keep.to(days.dtype),
    )


def expand_all_torch_temporal_prefixes(embeddings, masks, days, *patient_tensors):
    """Stack all calendar prefixes for a batch, grouped by prefix depth."""
    if embeddings.ndim != 3:
        raise ValueError("embeddings must have shape [batch,time,features]")
    batch, timepoints, _ = embeddings.shape
    if masks.shape != (batch, timepoints) or days.shape != (batch, timepoints):
        raise ValueError("mask/day shape does not match embeddings")
    if any(value.shape[0] != batch for value in patient_tensors):
        raise ValueError("patient tensors must share the embedding batch dimension")
    expanded_embeddings = embeddings.repeat((timepoints, 1, 1))
    expanded_masks = masks.repeat((timepoints, 1))
    expanded_days = days.repeat((timepoints, 1))
    depths = torch.arange(1, timepoints + 1, device=masks.device).repeat_interleave(batch)
    expanded_embeddings, expanded_masks, expanded_days = mask_torch_temporal_prefix(
        expanded_embeddings, expanded_masks, expanded_days, depths
    )
    repeated = tuple(
        value.repeat((timepoints,) + (1,) * (value.ndim - 1))
        for value in patient_tensors
    )
    return expanded_embeddings, expanded_masks, expanded_days, repeated, depths


def contiguous_prefix_lengths(masks):
    """Number of consecutive present visits starting at T0 for each patient."""
    values = np.asarray(masks)
    if values.ndim != 2:
        raise ValueError("masks must have shape [patients,timepoints]")
    present = values > 0
    return np.cumprod(present.astype(np.int64), axis=1).sum(axis=1)


def canonicalize_temporal_prefix(embeddings, masks, days, max_tp):
    """Keep only the contiguous T0-starting prefix, capped at ``max_tp``."""
    embeddings = np.asarray(embeddings)
    masks = np.asarray(masks)
    days = np.asarray(days)
    if embeddings.ndim != 3 or masks.shape != embeddings.shape[:2]:
        raise ValueError("invalid embedding/mask shape")
    if days.shape != masks.shape:
        raise ValueError("invalid elapsed-day shape")
    depth = int(max_tp)
    if depth < 1 or depth > embeddings.shape[1]:
        raise ValueError("max_tp is outside the available time dimension")
    present = masks > 0
    contiguous = np.cumprod(present.astype(np.int64), axis=1).astype(bool)
    before_limit = np.arange(embeddings.shape[1])[None, :] < depth
    keep = contiguous & before_limit
    return (
        embeddings * keep[..., None].astype(embeddings.dtype),
        masks * keep.astype(masks.dtype),
        days * keep.astype(days.dtype),
    )


def canonicalize_torch_temporal_prefix(embeddings, masks, days, max_tp):
    """Torch equivalent of :func:`canonicalize_temporal_prefix`."""
    if embeddings.ndim != 3 or masks.shape != embeddings.shape[:2]:
        raise ValueError("invalid embedding/mask shape")
    if days.shape != masks.shape:
        raise ValueError("invalid elapsed-day shape")
    depth = int(max_tp)
    if depth < 1 or depth > embeddings.shape[1]:
        raise ValueError("max_tp is outside the available time dimension")
    contiguous = torch.cumprod((masks > 0).to(torch.int64), dim=1).bool()
    before_limit = (
        torch.arange(embeddings.shape[1], device=masks.device).unsqueeze(0) < depth
    )
    keep = contiguous & before_limit
    return (
        embeddings * keep.unsqueeze(-1).to(embeddings.dtype),
        masks * keep.to(masks.dtype),
        days * keep.to(days.dtype),
    )


def expand_contiguous_torch_temporal_prefixes(
    embeddings, masks, days, *patient_tensors
):
    """Stack each unique valid T0-starting prefix exactly once.

    Patients with no T0 do not produce a temporal view. The returned counts are
    ordered by prefix depth and let callers average per-depth losses rather than
    weighting depths by their different patient counts.
    """
    if embeddings.ndim != 3 or masks.shape != embeddings.shape[:2]:
        raise ValueError("invalid embedding/mask shape")
    if days.shape != masks.shape:
        raise ValueError("invalid elapsed-day shape")
    if any(value.shape[0] != embeddings.shape[0] for value in patient_tensors):
        raise ValueError("patient tensors must share the embedding batch dimension")

    lengths = torch.cumprod((masks > 0).to(torch.int64), dim=1).sum(dim=1)
    embedding_views = []
    mask_views = []
    day_views = []
    repeated = [[] for _ in patient_tensors]
    depths = []
    counts = []
    for depth in range(1, embeddings.shape[1] + 1):
        eligible = lengths >= depth
        count = int(eligible.sum().item())
        counts.append(count)
        if count == 0:
            continue
        prefix_embeddings, prefix_masks, prefix_days = (
            canonicalize_torch_temporal_prefix(
                embeddings[eligible], masks[eligible], days[eligible], depth
            )
        )
        embedding_views.append(prefix_embeddings)
        mask_views.append(prefix_masks)
        day_views.append(prefix_days)
        depths.append(
            torch.full((count,), depth, dtype=torch.long, device=masks.device)
        )
        for index, value in enumerate(patient_tensors):
            repeated[index].append(value[eligible])
    if not embedding_views:
        raise ValueError("batch has no patient with a valid T0-starting prefix")
    return (
        torch.cat(embedding_views, dim=0),
        torch.cat(mask_views, dim=0),
        torch.cat(day_views, dim=0),
        tuple(torch.cat(values, dim=0) for values in repeated),
        torch.cat(depths, dim=0),
        tuple(counts),
    )
