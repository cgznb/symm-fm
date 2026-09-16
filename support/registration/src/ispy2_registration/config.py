from __future__ import annotations

from research_release import path as _release_path, load_yaml as _release_yaml

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path


DEFAULT_INPUT_ROOT = Path(
    _release_path('@data/manifest-1781750940386/ISPY2/processed_ispy2_world_phase3_scan')
)
DEFAULT_OUTPUT_ROOT = Path(
    _release_path('@data/manifest-1781750940386/ISPY2/processed_ispy2_world_phase3_scan_phase_aligned_multires_registered_to_T0')
)
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "ispy2_registration.toml"
)


class ConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class PathConfig:
    input_root: Path = DEFAULT_INPUT_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    manifest: Path | None = None
    manifest_name: str = "manifest/ispy2_breastdcedl_full_manifest.csv"

    @property
    def manifest_path(self) -> Path:
        return self.manifest if self.manifest is not None else self.input_root / self.manifest_name


@dataclass(frozen=True)
class RegistrationConfig:
    fixed_visit: str = "T0"
    moving_visits: tuple[str, ...] = ("T1", "T2", "T3")
    workers: int = 4
    elastix_threads: int = 8
    seed: int = 20260719
    tumor_weight: float = 0.5
    rigid_samples: int = 5000
    rigid_iterations: int = 500
    rigid_learning_rate: float = 2.0
    rigid_min_step: float = 1e-4
    rigid_relaxation_factor: float = 0.5
    rigid_shrink_factors: tuple[int, ...] = (4, 2, 1)
    rigid_smoothing_sigmas: tuple[float, ...] = (2.0, 1.0, 0.0)
    bspline_resolutions: int = 4
    bspline_iterations: tuple[int, ...] = (300, 300, 300, 1200)
    bspline_spatial_samples: int = 5000
    max_ftv_volume_change: float = 0.10
    similarity_tolerance: float = 1e-6
    protected_margin_mm: float = 10.0
    max_outside_fold_fraction: float = 0.02

    def __post_init__(self) -> None:
        if self.workers < 1:
            raise ValueError("workers must be at least 1")
        if self.elastix_threads < 1:
            raise ValueError("elastix_threads must be at least 1")
        if self.fixed_visit in self.moving_visits:
            raise ValueError("fixed_visit cannot also be a moving visit")
        if self.rigid_samples < 1 or self.rigid_iterations < 1:
            raise ValueError("rigid sample and iteration counts must be positive")
        if self.rigid_learning_rate <= 0 or self.rigid_min_step <= 0:
            raise ValueError("rigid learning rate and minimum step must be positive")
        if not 0.0 < self.rigid_relaxation_factor < 1.0:
            raise ValueError("rigid relaxation factor must be between zero and one")
        if (
            not self.rigid_shrink_factors
            or len(self.rigid_shrink_factors) != len(self.rigid_smoothing_sigmas)
            or any(value < 1 for value in self.rigid_shrink_factors)
            or any(value < 0 for value in self.rigid_smoothing_sigmas)
        ):
            raise ValueError(
                "rigid shrink factors and smoothing sigmas must be valid equal-length schedules"
            )
        if self.bspline_resolutions < 1:
            raise ValueError("bspline_resolutions must be positive")
        if len(self.bspline_iterations) != self.bspline_resolutions or any(
            value < 0 for value in self.bspline_iterations
        ):
            raise ValueError("bspline_iterations must match the resolution count")
        if self.bspline_spatial_samples < 1:
            raise ValueError("bspline_spatial_samples must be positive")
        if self.max_ftv_volume_change < 0 or self.similarity_tolerance < 0:
            raise ValueError("QC thresholds must be nonnegative")
        if self.protected_margin_mm < 0:
            raise ValueError("protected_margin_mm must be nonnegative")
        if not 0.0 <= self.max_outside_fold_fraction <= 1.0:
            raise ValueError("max_outside_fold_fraction must be between zero and one")


def _table(payload: dict[str, object], name: str) -> dict[str, object]:
    value = payload.get(name, {})
    if not isinstance(value, dict):
        raise ConfigurationError(f"TOML [{name}] must be a table")
    return dict(value)


def load_toml_config(
    path: Path | str = DEFAULT_CONFIG_PATH,
) -> tuple[PathConfig, RegistrationConfig]:
    config_path = Path(path)
    try:
        with config_path.open("rb") as handle:
            payload = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigurationError(f"cannot load configuration {config_path}: {exc}") from exc

    path_values = _table(payload, "paths")
    registration_values = _table(payload, "registration")
    registration_values.update(_table(payload, "qc"))

    allowed_paths = {"input_root", "output_root", "manifest"}
    unknown_paths = sorted(set(path_values) - allowed_paths)
    allowed_registration = {field.name for field in fields(RegistrationConfig)}
    unknown_registration = sorted(set(registration_values) - allowed_registration)
    if unknown_paths or unknown_registration:
        unknown = [*(f"paths.{name}" for name in unknown_paths), *unknown_registration]
        raise ConfigurationError(f"unknown configuration fields: {', '.join(unknown)}")

    for name in (
        "moving_visits",
        "rigid_shrink_factors",
        "rigid_smoothing_sigmas",
        "bspline_iterations",
    ):
        if name in registration_values:
            value = registration_values[name]
            if not isinstance(value, list):
                raise ConfigurationError(f"{name} must be a TOML array")
            registration_values[name] = tuple(value)

    try:
        paths = PathConfig(
            input_root=Path(path_values.get("input_root", DEFAULT_INPUT_ROOT)),
            output_root=Path(path_values.get("output_root", DEFAULT_OUTPUT_ROOT)),
            manifest=(
                Path(path_values["manifest"])
                if path_values.get("manifest") is not None
                else None
            ),
        )
        registration = RegistrationConfig(**registration_values)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"invalid configuration {config_path}: {exc}") from exc
    return paths, registration
