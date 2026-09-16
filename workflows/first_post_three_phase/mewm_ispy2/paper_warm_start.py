from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, BinaryIO, Sequence

import torch

from .conditioning import MEDGEMMA_MODEL_ID, MEDGEMMA_REVISION
from .contracts import (
    ANCESTRAL_DDPM_SAMPLER,
    CONTINUOUS_CODEBOOK_MINMAX_CONTRACT,
    CT_DENOISER_ARCHITECTURE,
    DIFFUSION_CHECKPOINT_SCHEMA_VERSION,
    EPSILON_PREDICTION_TYPE,
)
from .paper_checkpoint import _paper_state_references
from .paper_contracts import PaperRuntimeContract
from .paper_diffusion import (
    PaperDiffusionConfig,
    PaperFaithfulLatentDiffusion,
)


COMPATIBLE_ROOTS = (
    "input.bias",
    "time_mlp.",
    "relative_position_bias.",
    "initial_temporal_attention.",
    "downs.",
    "middle_block1.",
    "middle_spatial_attention.",
    "middle_temporal_attention.",
    "middle_block2.",
    "ups.",
    "final.",
)

_HEX_SHA256 = re.compile(r"[0-9a-f]{64}", re.ASCII)
_SOURCE_DENOISER_PREFIXES = (
    "model.denoiser.",
    "model.ema_denoiser.",
)
_V4_TOP_LEVEL_KEYS = {
    "callbacks",
    "epoch",
    "global_step",
    "identity",
    "loops",
    "lr_schedulers",
    "optimizer_states",
    "pytorch-lightning_version",
    "schema_version",
    "state_dict",
}
_V4_IDENTITY_KEYS = {
    "medgemma_model_id",
    "medgemma_revision",
    "vqgan_sha256",
    "data_contract_sha256",
    "data_backend",
    "denoiser_architecture",
    "denoiser_input_channels",
    "semantic_channels",
    "prediction_type",
    "timesteps",
    "sampler",
    "latent_contract",
    "ema_decay",
}
_SOURCE_EPOCH = 24
_SOURCE_GLOBAL_STEP = 10600
_PARTIAL_CHANNELS = 17


@dataclass(frozen=True)
class PaperWarmStartReport:
    checkpoint_path: str
    checkpoint_sha256: str
    source_schema_version: str
    source_epoch: int
    source_global_step: int
    exact_online_keys: Sequence[str]
    exact_ema_keys: Sequence[str]
    partial_slices: Sequence[str]
    new_destination_keys: Sequence[str]
    ignored_source_keys: Sequence[str]
    rejected_keys: Sequence[str]
    unclassified_keys: Sequence[str]

    def __post_init__(self) -> None:
        for field_name in (
            "checkpoint_path",
            "checkpoint_sha256",
            "source_schema_version",
        ):
            if type(getattr(self, field_name)) is not str:
                raise TypeError(f"{field_name} must be an exact string")
        if not self.checkpoint_path:
            raise ValueError("checkpoint_path must be nonempty")
        if _HEX_SHA256.fullmatch(self.checkpoint_sha256) is None:
            raise ValueError(
                "checkpoint_sha256 must contain 64 lowercase hex characters"
            )
        if not self.source_schema_version:
            raise ValueError("source_schema_version must be nonempty")
        for field_name in ("source_epoch", "source_global_step"):
            if type(getattr(self, field_name)) is not int:
                raise TypeError(f"{field_name} must be an exact integer")
        for field_name in (
            "exact_online_keys",
            "exact_ema_keys",
            "partial_slices",
            "new_destination_keys",
            "ignored_source_keys",
            "rejected_keys",
            "unclassified_keys",
        ):
            value = getattr(self, field_name)
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                raise TypeError(f"{field_name} must be a sequence of strings")
            normalized = tuple(value)
            if any(type(item) is not str for item in normalized):
                raise TypeError(f"{field_name} must contain exact strings")
            object.__setattr__(self, field_name, normalized)


@dataclass(frozen=True)
class _ReportFileSnapshot:
    content: bytes
    mode: int
    atime_ns: int
    mtime_ns: int


def _validate_path(value: str | Path, *, name: str) -> Path:
    if type(value) is str:
        if not value:
            raise ValueError(f"{name} must be nonempty")
        return Path(value)
    if isinstance(value, Path):
        if not str(value):
            raise ValueError(f"{name} must be nonempty")
        return value
    raise TypeError(f"{name} must be an exact string or Path")


def _validate_sha256(value: str, *, name: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    if _HEX_SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must contain 64 lowercase hex characters")


def _sha256_open_file(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _validate_runtime_binding(
    model: PaperFaithfulLatentDiffusion,
    source_path: Path,
    *,
    expected_sha256: str,
    expected_vqgan_sha256: str,
    expected_data_contract_sha256: str,
) -> None:
    if type(model) is not PaperFaithfulLatentDiffusion:
        raise TypeError("model must be an exact PaperFaithfulLatentDiffusion")
    config = getattr(model, "config", None)
    if type(config) is not PaperDiffusionConfig:
        raise TypeError("model config must be an exact PaperDiffusionConfig")
    runtime = config.runtime
    if type(runtime) is not PaperRuntimeContract:
        raise TypeError("model runtime must be an exact PaperRuntimeContract")
    if not isinstance(runtime.warm_start_path, Path):
        raise TypeError("runtime warm_start_path must be a Path")
    for name, value in (
        ("runtime warm_start_sha256", runtime.warm_start_sha256),
        ("runtime vqgan_sha256", runtime.vqgan_sha256),
        ("runtime data_contract_sha256", runtime.data_contract_sha256),
    ):
        _validate_sha256(value, name=name)
    try:
        actual_path = source_path.resolve()
        runtime_path = runtime.warm_start_path.resolve()
    except OSError as error:
        raise ValueError("runtime warm_start_path could not be resolved") from error
    if actual_path != runtime_path:
        raise ValueError("runtime warm_start_path does not match checkpoint_path")
    bound_values = (
        ("warm_start_sha256", expected_sha256, runtime.warm_start_sha256),
        ("vqgan_sha256", expected_vqgan_sha256, runtime.vqgan_sha256),
        (
            "data_contract_sha256",
            expected_data_contract_sha256,
            runtime.data_contract_sha256,
        ),
    )
    for name, supplied, bound in bound_values:
        if supplied != bound:
            raise ValueError(f"runtime {name} does not match caller argument")


def _validate_cpu_destination(destination: dict[str, torch.Tensor]) -> None:
    non_cpu = sorted(
        name for name, value in destination.items() if value.device.type != "cpu"
    )
    if non_cpu:
        raise ValueError(
            "warm-start destination tensors must all be on CPU before source load: "
            f"{non_cpu[:5]}"
        )


def _validate_cpu_model_before_state_export(
    model: PaperFaithfulLatentDiffusion,
) -> None:
    non_cpu: list[str] = []
    for module_name in ("denoiser", "ema_denoiser"):
        module = getattr(model, module_name, None)
        if not isinstance(module, torch.nn.Module):
            raise TypeError(f"paper model is missing {module_name}")
        for name, value in (
            *module.named_parameters(recurse=True),
            *module.named_buffers(recurse=True),
        ):
            if value.device.type != "cpu":
                non_cpu.append(f"model.{module_name}.{name}")
    conditioner = getattr(model, "conditioner", None)
    if not isinstance(conditioner, torch.nn.Module):
        raise TypeError("paper model is missing conditioner")
    for name, parameter in conditioner.named_parameters(recurse=True):
        if parameter.requires_grad and parameter.device.type != "cpu":
            non_cpu.append(f"model.conditioner.{name}")
    if non_cpu:
        raise ValueError(
            "warm-start destination tensors must all be on CPU before state export: "
            f"{sorted(non_cpu)[:5]}"
        )


def _validate_exact_string(value: Any, *, name: str, expected: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    if value != expected:
        raise ValueError(f"{name} mismatch")


def _validate_exact_integer(value: Any, *, name: str, expected: int) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    if value != expected:
        raise ValueError(f"{name} mismatch")


def _validate_v4_payload(
    payload: Any,
    *,
    expected_vqgan_sha256: str,
    expected_data_contract_sha256: str,
) -> tuple[dict[str, torch.Tensor], str, int, int]:
    if type(payload) is not dict:
        raise TypeError("v4 checkpoint payload must be an exact dictionary")
    if set(payload) != _V4_TOP_LEVEL_KEYS:
        raise ValueError("v4 checkpoint top-level fields do not match the locked schema")
    _validate_exact_string(
        payload["schema_version"],
        name="source schema_version",
        expected=DIFFUSION_CHECKPOINT_SCHEMA_VERSION,
    )
    _validate_exact_integer(payload["epoch"], name="source epoch", expected=_SOURCE_EPOCH)
    _validate_exact_integer(
        payload["global_step"],
        name="source global_step",
        expected=_SOURCE_GLOBAL_STEP,
    )
    if type(payload["pytorch-lightning_version"]) is not str or not payload[
        "pytorch-lightning_version"
    ]:
        raise TypeError("v4 checkpoint pytorch-lightning_version must be nonempty")
    if type(payload["callbacks"]) is not dict:
        raise TypeError("v4 checkpoint callbacks must be an exact dictionary")
    if type(payload["loops"]) is not dict:
        raise TypeError("v4 checkpoint loops must be an exact dictionary")
    if type(payload["optimizer_states"]) is not list:
        raise TypeError("v4 checkpoint optimizer_states must be an exact list")
    if type(payload["lr_schedulers"]) is not list:
        raise TypeError("v4 checkpoint lr_schedulers must be an exact list")

    identity = payload["identity"]
    if type(identity) is not dict:
        raise TypeError("v4 checkpoint identity must be an exact dictionary")
    if set(identity) != _V4_IDENTITY_KEYS:
        raise ValueError("v4 checkpoint identity fields do not match the locked schema")
    locked_strings = {
        "medgemma_model_id": MEDGEMMA_MODEL_ID,
        "medgemma_revision": MEDGEMMA_REVISION,
        "vqgan_sha256": expected_vqgan_sha256,
        "data_contract_sha256": expected_data_contract_sha256,
        "data_backend": "current",
        "denoiser_architecture": CT_DENOISER_ARCHITECTURE,
        "prediction_type": EPSILON_PREDICTION_TYPE,
        "sampler": ANCESTRAL_DDPM_SAMPLER,
        "latent_contract": CONTINUOUS_CODEBOOK_MINMAX_CONTRACT,
    }
    for name, expected in locked_strings.items():
        _validate_exact_string(identity[name], name=f"source identity {name}", expected=expected)
    for name, expected in (
        ("denoiser_input_channels", 49),
        ("semantic_channels", 32),
        ("timesteps", 200),
    ):
        _validate_exact_integer(identity[name], name=f"source identity {name}", expected=expected)
    if type(identity["ema_decay"]) is not float:
        raise TypeError("source identity ema_decay must be an exact float")
    if not math.isfinite(identity["ema_decay"]) or identity["ema_decay"] != 0.995:
        raise ValueError("source identity ema_decay mismatch")

    state = payload["state_dict"]
    if type(state) is not dict:
        raise TypeError("v4 checkpoint state_dict must be an exact dictionary")
    if any(type(key) is not str for key in state):
        raise TypeError("v4 checkpoint state_dict keys must be exact strings")
    return (
        state,
        payload["schema_version"],
        payload["epoch"],
        payload["global_step"],
    )


def _compatible_suffix(suffix: str) -> bool:
    return suffix == "input.bias" or any(
        suffix.startswith(root) for root in COMPATIBLE_ROOTS if root.endswith(".")
    )


def _new_denoiser_suffix(suffix: str) -> bool:
    return suffix.startswith(
        (
            "semantic_projection.",
            "down_cross_attentions.",
            "middle_cross_attention.",
            "up_cross_attentions.",
        )
    )


def _tensor_is_finite(value: torch.Tensor) -> bool:
    return not (value.is_floating_point() or value.is_complex()) or bool(
        torch.isfinite(value).all()
    )


def _validate_source_tensor(value: Any, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"v4 checkpoint state is not a tensor: {name}")
    if value.device.type == "meta":
        raise ValueError(f"v4 checkpoint tensor device is invalid: {name}")
    if not _tensor_is_finite(value):
        raise FloatingPointError(f"v4 checkpoint tensor is nonfinite: {name}")
    return value


def _validate_tensor_match(
    source: torch.Tensor, destination: torch.Tensor, *, name: str
) -> None:
    if source.shape != destination.shape:
        raise ValueError(f"warm-start tensor shape mismatch: {name}")
    if source.dtype != destination.dtype:
        raise TypeError(f"warm-start tensor dtype mismatch: {name}")
    if destination.device.type == "meta":
        raise ValueError(f"paper destination tensor device is invalid: {name}")
    if not _tensor_is_finite(destination):
        raise FloatingPointError(f"paper destination tensor is nonfinite: {name}")


def _classify_and_prepare(
    destination: dict[str, torch.Tensor],
    source: dict[str, torch.Tensor],
) -> tuple[
    dict[str, torch.Tensor],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    exact_online: list[str] = []
    exact_ema: list[str] = []
    partial_keys: list[str] = []
    new_destination: list[str] = []
    unclassified_destination: list[str] = []

    for name, destination_value in destination.items():
        if not isinstance(destination_value, torch.Tensor):
            raise TypeError(f"paper destination state is not a tensor: {name}")
        if destination_value.device.type == "meta":
            raise ValueError(f"paper destination tensor device is invalid: {name}")
        if not _tensor_is_finite(destination_value):
            raise FloatingPointError(
                f"paper destination tensor is nonfinite: {name}"
            )
        if name.startswith("model.conditioner."):
            new_destination.append(name)
            continue
        matched_prefix = next(
            (prefix for prefix in _SOURCE_DENOISER_PREFIXES if name.startswith(prefix)),
            None,
        )
        if matched_prefix is None:
            unclassified_destination.append(name)
            continue
        suffix = name.removeprefix(matched_prefix)
        if suffix == "input.weight":
            partial_keys.append(name)
        elif _compatible_suffix(suffix):
            if matched_prefix == "model.denoiser.":
                exact_online.append(name)
            else:
                exact_ema.append(name)
        elif _new_denoiser_suffix(suffix):
            new_destination.append(name)
        else:
            unclassified_destination.append(name)

    rejected_source: list[str] = []
    ignored_source: list[str] = []
    classified_source: set[str] = set()
    for name, raw_value in source.items():
        _validate_source_tensor(raw_value, name=name)
        if name.startswith("model.conditioner."):
            ignored_source.append(name)
            classified_source.add(name)
            continue
        matched_prefix = next(
            (prefix for prefix in _SOURCE_DENOISER_PREFIXES if name.startswith(prefix)),
            None,
        )
        if matched_prefix is None:
            rejected_source.append(name)
            continue
        suffix = name.removeprefix(matched_prefix)
        if suffix.startswith("semantic_projection."):
            if name not in destination:
                rejected_source.append(name)
            else:
                ignored_source.append(name)
                classified_source.add(name)
        elif suffix == "input.weight" or _compatible_suffix(suffix):
            if name not in destination:
                rejected_source.append(name)
            else:
                classified_source.add(name)
        else:
            rejected_source.append(name)

    if rejected_source or unclassified_destination:
        raise ValueError(
            "warm-start classification failed: "
            f"rejected={sorted(rejected_source)[:5]}, "
            f"unclassified={sorted(unclassified_destination)[:5]}"
        )

    required_source = set(exact_online) | set(exact_ema) | set(partial_keys)
    required_source.update(
        name
        for name in new_destination
        if name.startswith(_SOURCE_DENOISER_PREFIXES)
        and name.removeprefix(
            next(prefix for prefix in _SOURCE_DENOISER_PREFIXES if name.startswith(prefix))
        ).startswith("semantic_projection.")
    )
    missing_source = sorted(required_source - classified_source)
    if missing_source:
        raise ValueError(f"warm-start source is missing classified tensors: {missing_source[:5]}")

    prepared = {
        name: value.detach().clone(memory_format=torch.preserve_format)
        for name, value in destination.items()
    }
    for name in (*exact_online, *exact_ema):
        source_value = source[name]
        destination_value = destination[name]
        _validate_tensor_match(source_value, destination_value, name=name)
        prepared[name] = source_value.detach().to(device=destination_value.device).clone()
    for name in partial_keys:
        source_value = source[name]
        destination_value = destination[name]
        _validate_tensor_match(source_value, destination_value, name=name)
        if destination_value.ndim < 2 or destination_value.shape[1] != 49:
            raise ValueError(f"paper input weight channel contract mismatch: {name}")
        replacement = prepared[name]
        replacement[:, :_PARTIAL_CHANNELS].copy_(
            source_value[:, :_PARTIAL_CHANNELS].to(device=replacement.device)
        )

    return (
        prepared,
        tuple(sorted(exact_online)),
        tuple(sorted(exact_ema)),
        tuple(sorted(new_destination)),
        tuple(sorted(ignored_source)),
    )


@torch.no_grad()
def _commit_prepared_state(
    destination: dict[str, torch.Tensor], prepared: dict[str, torch.Tensor]
) -> None:
    for name, value in destination.items():
        value.copy_(prepared[name])


@torch.no_grad()
def _restore_destination_state(
    destination: dict[str, torch.Tensor], snapshot: dict[str, torch.Tensor]
) -> None:
    failures: list[str] = []
    for name, value in destination.items():
        try:
            value.copy_(snapshot[name])
        except BaseException as error:
            failures.append(f"{name}: {type(error).__name__}: {error}")
    if failures:
        raise RuntimeError(
            "warm-start model rollback failed: " + "; ".join(failures[:5])
        )


def _report_json(report: PaperWarmStartReport) -> str:
    payload = {field.name: getattr(report, field.name) for field in fields(report)}
    return (
        json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _markdown_literal(value: str) -> str:
    encoded = json.dumps(value, ensure_ascii=True, allow_nan=False)
    return "`" + encoded.replace("`", r"\u0060") + "`"


def _report_markdown(report: PaperWarmStartReport) -> str:
    def section(title: str, values: Sequence[str]) -> list[str]:
        lines = [f"## {title}", ""]
        lines.extend(f"- {_markdown_literal(value)}" for value in values)
        if not values:
            lines.append("- None")
        lines.append("")
        return lines

    lines = [
        "# Paper Diffusion Warm-Start Report",
        "",
        "This operation is a warm-start, not resume.",
        "",
        f"- checkpoint path: {_markdown_literal(report.checkpoint_path)}",
        f"- checkpoint SHA256: {_markdown_literal(report.checkpoint_sha256)}",
        f"- source schema: {_markdown_literal(report.source_schema_version)}",
        f"- source epoch: {report.source_epoch}",
        f"- source global step: {report.source_global_step}",
        "- new epoch: 0",
        "- new global step: 0",
        "",
    ]
    lines.extend(section("Exact Online Tensors", report.exact_online_keys))
    lines.extend(section("Exact EMA Tensors", report.exact_ema_keys))
    lines.extend(section("Partial Tensor Slices", report.partial_slices))
    lines.extend(section("New Destination Tensors", report.new_destination_keys))
    lines.extend(section("Ignored Source Tensors", report.ignored_source_keys))
    lines.extend(section("Rejected Tensors", report.rejected_keys))
    lines.extend(section("Unclassified Tensors", report.unclassified_keys))
    return "\n".join(lines).rstrip() + "\n"


def _cleanup_temporary_path(path: Path) -> str | None:
    try:
        path.unlink(missing_ok=True)
        return None
    except BaseException as primary_error:
        try:
            os.unlink(path)
            return None
        except FileNotFoundError:
            return None
        except BaseException as fallback_error:
            try:
                remains = path.exists()
            except BaseException:
                remains = True
            if not remains:
                return None
            return (
                f"remaining path {path}: Path.unlink "
                f"{type(primary_error).__name__}: {primary_error}; os.unlink "
                f"{type(fallback_error).__name__}: {fallback_error}"
            )


def _stage_report(path: Path, content: bytes) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException as original_error:
        cleanup_failure = _cleanup_temporary_path(temporary_path)
        if cleanup_failure is not None:
            raise RuntimeError(
                "warm-start report staging failed and temporary cleanup also "
                f"failed: {cleanup_failure}"
            ) from original_error
        raise
    return temporary_path


def _backup_report(path: Path) -> Path:
    descriptor, backup_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".backup.tmp", dir=path.parent
    )
    os.close(descriptor)
    backup_path = Path(backup_name)
    try:
        shutil.copy2(path, backup_path)
    except BaseException as original_error:
        cleanup_failure = _cleanup_temporary_path(backup_path)
        if cleanup_failure is not None:
            raise RuntimeError(
                "warm-start report backup failed and temporary cleanup also "
                f"failed: {cleanup_failure}"
            ) from original_error
        raise
    return backup_path


def _snapshot_report(path: Path) -> _ReportFileSnapshot:
    metadata = path.stat()
    return _ReportFileSnapshot(
        content=path.read_bytes(),
        mode=stat.S_IMODE(metadata.st_mode),
        atime_ns=metadata.st_atime_ns,
        mtime_ns=metadata.st_mtime_ns,
    )


def _restore_report_snapshot(
    destination: Path, snapshot: _ReportFileSnapshot
) -> None:
    temporary_path = _stage_report(destination, snapshot.content)
    try:
        temporary_path.chmod(snapshot.mode)
        os.utime(
            temporary_path,
            ns=(snapshot.atime_ns, snapshot.mtime_ns),
        )
        os.replace(temporary_path, destination)
    except BaseException as original_error:
        cleanup_failure = _cleanup_temporary_path(temporary_path)
        if cleanup_failure is not None:
            raise RuntimeError(
                "warm-start report snapshot restore failed and temporary cleanup "
                f"also failed: {cleanup_failure}"
            ) from original_error
        raise


def _rollback_report_pair(
    destinations: Sequence[Path],
    existed_before: set[Path],
    backups: dict[Path, Path],
    snapshots: dict[Path, _ReportFileSnapshot],
) -> list[str]:
    failures: list[str] = []
    for destination in destinations:
        try:
            backup = backups.get(destination)
            if backup is not None and backup.exists():
                snapshot = snapshots[destination]
                backup.chmod(snapshot.mode)
                os.utime(
                    backup,
                    ns=(snapshot.atime_ns, snapshot.mtime_ns),
                )
                os.replace(backup, destination)
            elif destination in snapshots:
                _restore_report_snapshot(destination, snapshots[destination])
            elif destination not in existed_before:
                destination.unlink(missing_ok=True)
        except BaseException as rollback_error:
            failures.append(
                f"{destination}: {type(rollback_error).__name__}: "
                f"{rollback_error}"
            )
    return failures


def _cleanup_report_artifacts(paths: Sequence[Path]) -> list[str]:
    failures: list[str] = []
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        cleanup_failure = _cleanup_temporary_path(path)
        if cleanup_failure is not None:
            failures.append(cleanup_failure)
    return failures


def _write_reports(output_directory: Path, report: PaperWarmStartReport) -> None:
    if not output_directory.is_dir():
        raise ValueError(f"output_directory is not a directory: {output_directory}")
    destinations = (
        output_directory / "report.json",
        output_directory / "report.md",
    )
    existed_before: set[Path] = set()
    snapshots: dict[Path, _ReportFileSnapshot] = {}
    staged: dict[Path, Path] = {}
    backups: dict[Path, Path] = {}
    try:
        for destination in destinations:
            if destination.exists():
                if not destination.is_file():
                    raise ValueError(
                        f"warm-start report destination is not a file: {destination}"
                    )
                existed_before.add(destination)
                snapshots[destination] = _snapshot_report(destination)
                backups[destination] = _backup_report(destination)
        staged[destinations[0]] = _stage_report(
            destinations[0], _report_json(report).encode("ascii")
        )
        staged[destinations[1]] = _stage_report(
            destinations[1], _report_markdown(report).encode("ascii")
        )
        # Each fixed report file is atomic; handled failures roll the pair back.
        for destination in destinations:
            os.replace(staged[destination], destination)
    except BaseException as original_error:
        rollback_failures = _rollback_report_pair(
            destinations,
            existed_before,
            backups,
            snapshots,
        )
        rollback_failures.extend(
            _cleanup_report_artifacts((*staged.values(), *backups.values()))
        )
        if rollback_failures:
            raise RuntimeError(
                "warm-start report installation failed and report rollback also "
                "failed: "
                + "; ".join(rollback_failures[:5])
            ) from original_error
        raise
    cleanup_failures = _cleanup_report_artifacts(tuple(backups.values()))
    if cleanup_failures:
        cleanup_error = RuntimeError(
            "warm-start reports were installed but backup cleanup failed: "
            + "; ".join(cleanup_failures)
        )
        rollback_failures = _rollback_report_pair(
            destinations,
            existed_before,
            backups,
            snapshots,
        )
        rollback_failures.extend(
            _cleanup_report_artifacts((*staged.values(), *backups.values()))
        )
        if rollback_failures:
            raise RuntimeError(
                "warm-start backup cleanup failed and report rollback also failed: "
                + "; ".join(rollback_failures[:5])
            ) from cleanup_error
        raise cleanup_error


def _create_output_directories(
    output_directory: Path, created_directories: list[Path]
) -> None:
    missing: list[Path] = []
    current = output_directory
    while not current.exists():
        missing.append(current)
        if current.parent == current:
            break
        current = current.parent
    if not current.is_dir():
        raise ValueError(f"output_directory ancestor is not a directory: {current}")
    for directory in reversed(missing):
        directory.mkdir()
        created_directories.append(directory)
    if not output_directory.is_dir():
        raise ValueError(f"output_directory is not a directory: {output_directory}")


def _remove_created_output_directories(
    created_directories: Sequence[Path],
) -> None:
    failures: list[str] = []
    for directory in reversed(created_directories):
        if not directory.exists():
            continue
        try:
            directory.rmdir()
        except OSError as error:
            failures.append(f"{directory}: {type(error).__name__}: {error}")
    if failures:
        raise RuntimeError(
            "warm-start output rollback could not remove importer-created "
            "directories: "
            + "; ".join(failures[:5])
        )


def _raise_compounded_transaction_error(
    original_error: BaseException, rollback_failures: Sequence[BaseException]
) -> None:
    details = "; ".join(
        f"{type(error).__name__}: {error}" for error in rollback_failures
    )
    raise RuntimeError(
        "warm-start transaction failed and rollback also failed: " + details
    ) from original_error


def import_v4_paper_warm_start(
    model: PaperFaithfulLatentDiffusion,
    checkpoint_path: str | Path,
    *,
    expected_sha256: str,
    expected_vqgan_sha256: str,
    expected_data_contract_sha256: str,
    output_directory: str | Path,
) -> PaperWarmStartReport:
    source_path = _validate_path(checkpoint_path, name="checkpoint_path")
    report_directory = _validate_path(output_directory, name="output_directory")
    for name, value in (
        ("expected_sha256", expected_sha256),
        ("expected_vqgan_sha256", expected_vqgan_sha256),
        ("expected_data_contract_sha256", expected_data_contract_sha256),
    ):
        _validate_sha256(value, name=name)
    _validate_runtime_binding(
        model,
        source_path,
        expected_sha256=expected_sha256,
        expected_vqgan_sha256=expected_vqgan_sha256,
        expected_data_contract_sha256=expected_data_contract_sha256,
    )
    _validate_cpu_model_before_state_export(model)
    destination = _paper_state_references(model, prefix="model.")
    _validate_cpu_destination(destination)
    destination_snapshot = {
        name: value.detach().clone(memory_format=torch.preserve_format)
        for name, value in destination.items()
    }
    try:
        checkpoint_handle = source_path.open("rb")
    except OSError as error:
        raise ValueError(
            f"checkpoint_path is not a readable file: {source_path}"
        ) from error
    with checkpoint_handle:
        if checkpoint_handle.closed or not checkpoint_handle.readable():
            raise ValueError("checkpoint handle must be open and readable")
        if not checkpoint_handle.seekable():
            raise ValueError("checkpoint handle must be seekable")
        actual_sha256 = _sha256_open_file(checkpoint_handle)
        if actual_sha256 != expected_sha256:
            raise ValueError("checkpoint SHA256 mismatch")
        if checkpoint_handle.seek(0) != 0:
            raise OSError("checkpoint handle could not rewind to byte zero")
        payload = torch.load(
            checkpoint_handle,
            map_location="cpu",
            weights_only=True,
        )
    source, schema, epoch, global_step = _validate_v4_payload(
        payload,
        expected_vqgan_sha256=expected_vqgan_sha256,
        expected_data_contract_sha256=expected_data_contract_sha256,
    )
    del payload
    prepared, exact_online, exact_ema, new_destination, ignored_source = (
        _classify_and_prepare(destination, source)
    )

    partial_slices = (
        "model.denoiser.input.weight[:, :17]",
        "model.ema_denoiser.input.weight[:, :17]",
    )
    report = PaperWarmStartReport(
        checkpoint_path=str(source_path),
        checkpoint_sha256=actual_sha256,
        source_schema_version=schema,
        source_epoch=epoch,
        source_global_step=global_step,
        exact_online_keys=exact_online,
        exact_ema_keys=exact_ema,
        partial_slices=partial_slices,
        new_destination_keys=new_destination,
        ignored_source_keys=ignored_source,
        rejected_keys=(),
        unclassified_keys=(),
    )
    created_directories: list[Path] = []
    try:
        _commit_prepared_state(destination, prepared)
        _create_output_directories(report_directory, created_directories)
        _write_reports(report_directory, report)
    except BaseException as original_error:
        rollback_failures: list[BaseException] = []
        try:
            _restore_destination_state(destination, destination_snapshot)
        except BaseException as rollback_error:
            rollback_failures.append(rollback_error)
        try:
            _remove_created_output_directories(created_directories)
        except BaseException as rollback_error:
            rollback_failures.append(rollback_error)
        if rollback_failures:
            _raise_compounded_transaction_error(
                original_error,
                rollback_failures,
            )
        raise
    return report


__all__ = [
    "COMPATIBLE_ROOTS",
    "PaperWarmStartReport",
    "import_v4_paper_warm_start",
]
