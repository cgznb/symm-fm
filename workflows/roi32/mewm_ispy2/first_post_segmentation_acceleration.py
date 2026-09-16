from __future__ import annotations

import itertools

import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.utilities.helpers import empty_cache


class BatchedNNUNetPredictor(nnUNetPredictor):
    """Batch the official sliding windows and mirror views without dropping either."""

    def __init__(self, *, tile_batch_size=1, mirror_batch_size=1, **kwargs):
        super().__init__(**kwargs)
        self.tile_batch_size = tile_batch_size
        self.mirror_batch_size = mirror_batch_size
        self.oom_reductions = 0
        self.largest_forward_batch = 0

    def _internal_maybe_mirror_and_predict(self, x):
        if self.mirror_batch_size == 1:
            self.largest_forward_batch = max(self.largest_forward_batch, len(x))
            return super()._internal_maybe_mirror_and_predict(x)
        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
        combinations = [()]
        if mirror_axes:
            if max(mirror_axes) > x.ndim - 3:
                raise ValueError("Mirror axes do not match input dimensions")
            axes = [axis + 2 for axis in mirror_axes]
            combinations.extend(
                combination
                for length in range(1, len(axes) + 1)
                for combination in itertools.combinations(axes, length)
            )
        prediction = None
        for offset in range(0, len(combinations), self.mirror_batch_size):
            group = combinations[offset : offset + self.mirror_batch_size]
            work = torch.cat([torch.flip(x, axes) if axes else x for axes in group])
            self.largest_forward_batch = max(self.largest_forward_batch, len(work))
            outputs = self.network(work)
            for index, axes in enumerate(group):
                output = outputs[index * len(x) : (index + 1) * len(x)]
                output = torch.flip(output, axes) if axes else output
                if prediction is None:
                    prediction = output.clone()
                else:
                    prediction += output
            del outputs, work
        return prediction / len(combinations)

    def _internal_predict_sliding_window_return_logits(
        self, data, slicers, do_on_device=True
    ):
        results_device = self.device if do_on_device else torch.device("cpu")
        data = data.to(results_device)
        predicted_logits = torch.zeros(
            (self.label_manager.num_segmentation_heads, *data.shape[1:]),
            dtype=torch.half,
            device=results_device,
        )
        n_predictions = torch.zeros(
            data.shape[1:], dtype=torch.half, device=results_device
        )
        gaussian = (
            compute_gaussian(
                tuple(self.configuration_manager.patch_size),
                sigma_scale=1.0 / 8,
                value_scaling_factor=10,
                device=results_device,
            )
            if self.use_gaussian
            else 1
        )
        offset = 0
        while offset < len(slicers):
            group = slicers[offset : offset + self.tile_batch_size]
            work = prediction = None
            try:
                work = torch.stack([data[sl] for sl in group]).to(self.device)
                prediction = self._internal_maybe_mirror_and_predict(work).to(
                    results_device
                )
            except torch.cuda.OutOfMemoryError:
                work = prediction = None
                empty_cache(self.device)
                if self.tile_batch_size > 1:
                    self.tile_batch_size = max(1, self.tile_batch_size // 2)
                elif self.mirror_batch_size > 1:
                    self.mirror_batch_size //= 2
                else:
                    raise
                self.oom_reductions += 1
                continue
            # Preserve the original tile and mirror accumulation order in FP16.
            for index, sl in enumerate(group):
                predicted_logits[sl] += prediction[index] * gaussian
                n_predictions[sl[1:]] += gaussian
            offset += len(group)
            del prediction, work
        predicted_logits /= n_predictions
        if not torch.isfinite(predicted_logits).all():
            raise RuntimeError("Non-finite batched sliding-window logits")
        return predicted_logits
