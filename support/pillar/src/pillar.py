import glob
import gzip
import os
import re
import zipfile

import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch

TARGET_HW, TARGET_D = 384, 192

# NIfTI member/path ending: <pid>/dce/T<t>/<...>_dce_aqc_<idx>.nii.gz  (any prefix allowed)
_AQC = re.compile(r"([^/]+)/dce/T(\d)/[^/]*_dce_aqc_(\d+)\.nii\.gz$")


# --------------------------------------------------------------------- frozen backbone

def _ensure_transformers_remote_code_compatibility(pretrained_model_class):
    """Supply metadata expected by Transformers 5 for older remote model code."""
    if not hasattr(pretrained_model_class, "all_tied_weights_keys"):
        # Pillar's remote wrapper predates the post_init() contract and declares
        # no tied weights. Normal Transformers models replace this class default
        # with their instance-specific map during post_init().
        pretrained_model_class.all_tied_weights_keys = {}


def load_pillar(device="cuda", revision="main"):
    """Load the frozen Pillar-0 backbone in eval mode (gated weights; requires HF access)."""
    from transformers import AutoModel, PreTrainedModel

    _ensure_transformers_remote_code_compatibility(PreTrainedModel)
    model = AutoModel.from_pretrained(
        "YalaLab/Pillar0-BreastMRI",
        revision=revision,
        trust_remote_code=True,
        low_cpu_mem_usage=False,
    )
    return model.eval().to(device)


@torch.no_grad()
def global_embedding(model, volume):
    """Return the 1152-D pooled global embedding for one [1, 3, 384, 384, 192] input volume."""
    return model.extract_vision_feats({"breast_mr": volume}).squeeze(0).float().cpu()


# --------------------------------------------------------------------- volume construction

def _img_to_vol(img):
    """nibabel image -> ((D, H, W) float32, [dz, dy, dx] voxel spacing)."""
    vol = np.transpose(img.get_fdata().astype(np.float32), (2, 1, 0))  # (X,Y,Z) -> (D,H,W)
    dx, dy, dz = img.header.get_zooms()[:3]
    return vol, [float(dz), float(dy), float(dx)]


def resample_volume(volume, spacing, target_spacing=(1.0, 1.0, 1.0), interp="linear"):
    """Resample a (D, H, W) volume to `target_spacing` (1 mm isotropic) with linear interpolation."""
    v = sitk.GetImageFromArray(volume)
    v.SetSpacing([spacing[2], spacing[1], spacing[0]])
    new_size = [int(round(volume.shape[2] * spacing[2] / target_spacing[2])),
                int(round(volume.shape[1] * spacing[1] / target_spacing[1])),
                int(round(volume.shape[0] * spacing[0] / target_spacing[0]))]
    r = sitk.ResampleImageFilter()
    r.SetOutputSpacing([target_spacing[2], target_spacing[1], target_spacing[0]])
    r.SetSize(new_size)
    r.SetInterpolator(sitk.sitkNearestNeighbor if interp == "nearest" else sitk.sitkLinear)
    r.SetOutputOrigin(v.GetOrigin())
    r.SetOutputDirection(v.GetDirection())
    return sitk.GetArrayFromImage(r.Execute(v))


def normalize_volume(volume):
    """Clip to the 1st–99th intensity percentiles and min-max scale to [0, 1]."""
    vmin, vmax = np.percentile(volume, 1), np.percentile(volume, 99)
    vol = np.clip(volume, vmin, vmax)
    return (vol - vmin) / (vmax - vmin + 1e-8)


def pad_or_crop(t, target_d=TARGET_D, target_hw=TARGET_HW):
    """Center pad or crop a (D, H, W) tensor to (target_d, target_hw, target_hw)."""
    D, H, W = t.shape
    if D < target_d:
        pf = (target_d - D) // 2
        t = torch.nn.functional.pad(t, (0, 0, 0, 0, pf, target_d - D - pf))
    elif D > target_d:
        s = (D - target_d) // 2
        t = t[s:s + target_d]
    D, H, W = t.shape
    if H < target_hw:
        ph = (target_hw - H) // 2
        t = torch.nn.functional.pad(t, (0, 0, ph, target_hw - H - ph))
    elif H > target_hw:
        s = (H - target_hw) // 2
        t = t[:, s:s + target_hw, :]
    D, H, W = t.shape
    if W < target_hw:
        pw = (target_hw - W) // 2
        t = torch.nn.functional.pad(t, (pw, target_hw - W - pw))
    elif W > target_hw:
        s = (W - target_hw) // 2
        t = t[:, :, s:s + target_hw]
    return t


def _to_model_axes(t):
    """Reorder a (D, H, W) tensor to the model's (H, W, D) axis order."""
    return t.permute(1, 2, 0).contiguous()


class NiftiSource:
    """Reads BreastDCEDL DCE acquisitions from a directory or a `.zip` (in memory, no extraction)."""

    def __init__(self, path):
        self.aqc = {}                                    # (pid, t) -> {acq_index: name}
        if os.path.isdir(path):
            self.mode = "dir"
            for dp, _, files in os.walk(path):
                for fn in files:
                    full = os.path.join(dp, fn).replace(os.sep, "/")
                    m = _AQC.search(full)
                    if m:
                        pid, t, idx = m.group(1), int(m.group(2)), int(m.group(3))
                        self.aqc.setdefault((pid, t), {})[idx] = full
        else:
            self.mode = "zip"
            self.zf = zipfile.ZipFile(path)
            for n in self.zf.namelist():
                if n.startswith("__MACOSX") or n.endswith("/"):
                    continue
                m = _AQC.search(n)
                if m:
                    pid, t, idx = m.group(1), int(m.group(2)), int(m.group(3))
                    self.aqc.setdefault((pid, t), {})[idx] = n
        if not self.aqc:
            raise FileNotFoundError(f"No `*_dce_aqc_*.nii.gz` acquisitions found under {path}")

    def phase_members(self, pid, t, row):
        """The three pre/early-post/late-post acquisitions, chosen from the metadata phase columns.

        Uses `pre_T{t}`, `post_early_T{t}`, `post_late_T{t}` from `row`; falls back to the first three
        sorted acquisitions if any is missing. Returns None when the timepoint is absent.
        """
        avail = self.aqc.get((str(pid), t))
        if not avail:
            return None

        def byidx(idx):
            return None if pd.isna(idx) else avail.get(int(idx))

        members = [byidx(row.get(f"pre_T{t}")), byidx(row.get(f"post_early_T{t}")),
                   byidx(row.get(f"post_late_T{t}"))]
        if any(x is None for x in members):
            keys = sorted(avail)
            if len(keys) < 3:
                return None
            members = [avail[k] for k in keys[:3]]
        return members

    def load(self, name):
        """Read one acquisition (filesystem path or zip member) -> ((D, H, W), [dz, dy, dx])."""
        if self.mode == "dir":
            return _img_to_vol(nib.load(name))
        raw = self.zf.read(name)
        if name.endswith(".gz"):
            raw = gzip.decompress(raw)
        return _img_to_vol(nib.Nifti1Image.from_bytes(raw))


def build_volume(source, pid, t, row):
    """Assemble the [1, 3, 384, 384, 192] input volume for (patient, timepoint), or None if missing."""
    members = source.phase_members(pid, t, row)
    if members is None:
        return None
    channels = []
    for name in members:
        vol, spacing = source.load(name)
        vol = normalize_volume(resample_volume(vol, spacing, interp="linear"))
        channels.append(_to_model_axes(pad_or_crop(torch.from_numpy(vol).float())))
    return torch.stack(channels, dim=0).unsqueeze(0)
