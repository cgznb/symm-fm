"""Command-line entry points for every stage of the research pipeline."""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
from typing import Any, Mapping

import click

from ispy2_symmflow.config import load_config


def _configuration(path: str, overrides: tuple[str, ...]) -> dict[str, Any]:
    return load_config(path, overrides)


def _json_ready(value: Any) -> Any:
    try:
        import torch
    except ImportError:
        torch = None  # type: ignore[assignment]
    if torch is not None and isinstance(value, torch.Tensor):
        values = value.detach().cpu()
        return _json_ready(values.item() if values.numel() == 1 else values.tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: str | Path, value: Any) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            _json_ready(value), indent=2, sort_keys=True, default=str, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def _require_checkpoint_time_pair(
    conditions: Mapping[str, Any], training_data: Mapping[str, Any]
) -> list[str]:
    from ispy2_symmflow.config import ConfigError, resolve_time_pairs

    try:
        allowed = resolve_time_pairs(training_data)
    except ConfigError as exc:
        raise click.ClickException(f"checkpoint time-pair contract is invalid: {exc}") from exc
    observed = [
        str(conditions.get("stage_i", "")),
        str(conditions.get("stage_j", "")),
    ]
    if tuple(observed) not in set(allowed):
        if len(allowed) == 1:
            raise click.ClickException(
                f"conditions stage_i/stage_j must match checkpoint time_pair "
                f"{list(allowed[0])}, got {observed}"
            )
        raise click.ClickException(
            "conditions stage_i/stage_j must match one checkpoint time pair "
            f"{[list(pair) for pair in allowed]}, got {observed}"
        )
    return observed


def _validate_sampling_conditions(
    conditions: Mapping[str, Any], checkpoint_header: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the checkpoint's train-observed condition availability contract."""

    from ispy2_symmflow.training.schema import (
        validate_sampling_condition_availability,
    )

    extra = checkpoint_header.get("extra")
    provenance = extra.get("schema_provenance") if isinstance(extra, Mapping) else None
    if not isinstance(provenance, Mapping):
        raise click.ClickException(
            "sampling checkpoint has no condition schema provenance"
        )
    try:
        validate_sampling_condition_availability(conditions, provenance)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    return dict(provenance)


def _checkpoint_manifest_provenance(
    header: Mapping[str, Any], *, ordered_key: str, label: str
) -> tuple[str, dict[str, Any]]:
    split_hash = str(header.get("split_hash", "")).strip()
    extra = header.get("extra")
    signature = extra.get("training_signature") if isinstance(extra, Mapping) else None
    if not split_hash:
        raise click.ClickException(f"{label} checkpoint has no training split_hash")
    if not isinstance(signature, Mapping):
        raise click.ClickException(f"{label} checkpoint has no training_signature")
    if not str(signature.get(ordered_key, "")).strip():
        raise click.ClickException(
            f"{label} checkpoint training_signature has no {ordered_key}"
        )
    try:
        manifest_record_count = int(signature.get("manifest_record_count", -1))
    except (TypeError, ValueError) as exc:
        raise click.ClickException(
            f"{label} checkpoint training_signature has an invalid manifest record count"
        ) from exc
    if manifest_record_count < 1:
        raise click.ClickException(
            f"{label} checkpoint training_signature has no valid manifest record count"
        )
    from ispy2_symmflow.training.provenance import (
        CACHED_PAIR_MANIFEST_FINGERPRINT,
        CACHED_PAIR_MANIFEST_RECORD_COUNT,
        require_latent_statistics_fingerprint,
    )

    statistics = header.get("latent_statistics")
    if not isinstance(statistics, Mapping):
        raise click.ClickException(f"{label} checkpoint has no latent statistics")
    try:
        require_latent_statistics_fingerprint(statistics)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    pair_fingerprint = str(
        statistics.get(CACHED_PAIR_MANIFEST_FINGERPRINT, "")
    ).strip()
    try:
        pair_count = int(statistics[CACHED_PAIR_MANIFEST_RECORD_COUNT])
    except (KeyError, TypeError, ValueError) as exc:
        raise click.ClickException(
            f"{label} checkpoint latent statistics have no cached-pair record count"
        ) from exc
    if not pair_fingerprint or pair_count < 1:
        raise click.ClickException(
            f"{label} checkpoint has no valid cached-pair manifest binding"
        )
    if str(signature.get(CACHED_PAIR_MANIFEST_FINGERPRINT, "")) != pair_fingerprint:
        raise click.ClickException(
            f"{label} checkpoint training signature cached-pair fingerprint differs "
            "from latent statistics"
        )
    try:
        signature_pair_count = int(
            signature.get(CACHED_PAIR_MANIFEST_RECORD_COUNT, -1)
        )
    except (TypeError, ValueError) as exc:
        raise click.ClickException(
            f"{label} checkpoint training signature has an invalid cached-pair count"
        ) from exc
    if signature_pair_count != pair_count:
        raise click.ClickException(
            f"{label} checkpoint training signature cached-pair count differs from "
            "latent statistics"
        )
    if manifest_record_count != pair_count:
        raise click.ClickException(
            f"{label} checkpoint training manifest count differs from its cached-pair binding"
        )
    return split_hash, dict(signature)


def _validate_sampling_source(
    image: Any,
    metadata: Mapping[str, Any],
    checkpoint_header: Mapping[str, Any],
    *,
    expected_stage: str,
) -> tuple[dict[str, Any], list[str], dict[str, str]]:
    """Bind a formal source archive to the checkpoint preprocessing contract."""

    from ispy2_symmflow.training.datasets import validate_prepared_image
    from ispy2_symmflow.utils.hashing import stable_hash

    identity: dict[str, str] = {}
    for key in ("patient_id", "visit_id", "study_uid", "visit_stage", "split"):
        value = str(metadata.get(key, "")).strip()
        if not value:
            raise click.ClickException(f"formal source archive is missing {key!r}")
        identity[key] = value
    if identity["split"] not in {"train", "val", "test"}:
        raise click.ClickException(
            "formal source archive split must be train, val, or test"
        )
    if identity["visit_stage"] != expected_stage:
        raise click.ClickException(
            f"sampling expects a {expected_stage} source, got {identity['visit_stage']}"
        )

    try:
        observed_signature = validate_prepared_image(
            image, metadata, label="formal sampling source"
        )
    except (TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    training_config = checkpoint_header.get("config")
    statistics = checkpoint_header.get("latent_statistics")
    if not isinstance(training_config, Mapping) or not isinstance(statistics, Mapping):
        raise click.ClickException(
            "sampling checkpoint lacks training configuration or latent statistics"
        )
    training_data = training_config.get("data")
    if not isinstance(training_data, Mapping):
        raise click.ClickException("sampling checkpoint has no data configuration")
    raw_roles = training_data.get("phase_channels")
    if not isinstance(raw_roles, (list, tuple)) or not raw_roles:
        raise click.ClickException("sampling checkpoint has no phase-channel contract")
    expected_roles = [str(value) for value in raw_roles]
    observed_roles = [str(value) for value in observed_signature["phase_roles"]]
    if observed_roles != expected_roles:
        raise click.ClickException(
            f"source phase roles {observed_roles} do not match checkpoint roles {expected_roles}"
        )

    expected_signature = statistics.get("source_preprocessing_signature")
    if not isinstance(expected_signature, Mapping):
        raise click.ClickException(
            "sampling checkpoint latent statistics have no source preprocessing signature"
        )
    if stable_hash(observed_signature) != stable_hash(expected_signature):
        raise click.ClickException(
            "source preprocessing contract differs from the checkpoint training data"
        )
    return observed_signature, observed_roles, identity


def _endpoint_provenance(
    *, sigma_min: float, direction: str, residual_consent: bool
) -> dict[str, Any]:
    if sigma_min == 0:
        return {
            "endpoint_semantics": "paper_clean_endpoints",
            "residual_endpoint_consent": False,
            "source_endpoint_approximation": False,
            "decoded_endpoint_contains_sigma_residual": False,
        }
    return {
        "endpoint_semantics": "upstream_sigma_compatibility_experiment",
        "residual_endpoint_consent": bool(residual_consent),
        "source_endpoint_approximation": direction == "backward",
        "decoded_endpoint_contains_sigma_residual": True,
        "endpoint_note": (
            "Backward compatibility sampling substitutes the clean later latent for "
            "x_1=x+sigma_min*epsilon_x; decoded endpoints are not clean-endpoint samples."
            if direction == "backward"
            else "The decoded later endpoint retains the configured sigma_min noise residual."
        ),
    }


def _config_options(function):
    function = click.option(
        "--set",
        "overrides",
        multiple=True,
        help="Override a YAML value with dotted.path=value; repeat as needed.",
    )(function)
    return click.option(
        "--config",
        "config_path",
        type=click.Path(exists=True, dir_okay=False, path_type=str),
        default="configs/base.yaml",
        show_default=True,
    )(function)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name="ispy2-symmflow3d")
def main() -> None:
    """Conditional 3D SymmFlow for longitudinal I-SPY2 breast MRI."""


@main.command("audit-data")
@_config_options
@click.option("--output-dir", type=click.Path(file_okay=False, path_type=str), required=True)
@click.option("--header-sample-limit", type=click.IntRange(min=0), default=64, show_default=True)
@click.option(
    "--build-manifests/--summary-only",
    default=True,
    help="Scan temporal headers and create visit/pair manifests after the fast summary.",
)
def audit_data(
    config_path: str,
    overrides: tuple[str, ...],
    output_dir: str,
    header_sample_limit: int,
    build_manifests: bool,
) -> None:
    """Audit DICOM metadata, conditions, masks, visits, and available pairs."""

    from ispy2_symmflow.data import audit_dataset, build_pair_manifest, build_visit_manifest
    from ispy2_symmflow.data.manifest import write_jsonl
    from ispy2_symmflow.data.split import (
        assign_patient_splits,
        write_split_manifest,
    )

    config = _configuration(config_path, overrides)
    data = config["data"]
    root = data["root"]
    clinical = data.get("clinical_path")
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    audit = audit_dataset(
        root,
        clinical,
        metadata_csv=data.get("metadata_csv"),
        header_sample_limit=header_sample_limit,
    )
    audit_path = destination / "data_audit.json"
    audit.write_json(audit_path)
    outputs: dict[str, Any] = {"audit": str(audit_path), **audit.to_dict()}
    if build_manifests:
        visits = build_visit_manifest(
            root,
            clinical_path=clinical,
            relative_dates_verified=bool(data.get("relative_dates_verified", False)),
            derived_phase_convention_verified=bool(
                data.get("derived_phase_convention_verified", False)
            ),
            scan_phase_headers=True,
        )
        fractions = [float(value) for value in data.get("split_fractions", (0.8, 0.1, 0.1))]
        ratios = dict(zip(("train", "val", "test"), fractions, strict=True))
        assignments = assign_patient_splits(
            sorted({visit.patient_id for visit in visits}),
            ratios=ratios,
            seed=int(data.get("split_seed", config["project"]["seed"])),
        )
        split_visits = [replace(visit, split=assignments[visit.patient_id]) for visit in visits]
        pair = data.get("time_pair", ["T0", "T1"])
        require_three = set(("pre", "early", "late")).issubset(data["phase_channels"])
        pairs = build_pair_manifest(
            split_visits,
            assignments,
            earlier_stage=str(pair[0]),
            later_stage=str(pair[1]),
            require_three_phase=require_three,
        )
        visits_path = destination / "visits.jsonl"
        pairs_path = destination / "pairs.jsonl"
        splits_path = destination / "splits.json"
        write_jsonl(split_visits, visits_path)
        write_jsonl(pairs, pairs_path)
        write_split_manifest(assignments, splits_path)
        outputs.update(
            visits=str(visits_path),
            pairs=str(pairs_path),
            splits=str(splits_path),
            manifest_visit_count=len(split_visits),
            manifest_pair_count=len(pairs),
            qc_passing_pair_count=sum(
                not any(event.severity == "error" for event in item.qc) for item in pairs
            ),
        )
    click.echo(json.dumps(_json_ready(outputs), indent=2, sort_keys=True))


@main.command("prepare-data")
@_config_options
@click.option(
    "--visit-manifest",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    required=True,
)
@click.option("--output-dir", type=click.Path(file_okay=False, path_type=str), required=True)
def prepare_data(
    config_path: str,
    overrides: tuple[str, ...],
    visit_manifest: str,
    output_dir: str,
) -> None:
    """Decode, resample, normalize, and source-crop QC-passing 3D visits."""

    from ispy2_symmflow.preprocessing import PreprocessConfig, prepare_dataset

    config = _configuration(config_path, overrides)
    data = config["data"]
    roles = tuple(str(value) for value in data["phase_channels"])
    settings = PreprocessConfig(
        phase_roles=roles,
        target_spacing_dhw=tuple(float(value) for value in data["target_spacing_mm"]),
        output_shape_dhw=tuple(int(value) for value in data["spatial_size"]),
        target_axis_codes=str(data.get("orientation", "SAR")),
        crop_mode=str(data.get("crop_mode", "fixed_center")),
        lower_percentile=float(data.get("intensity_lower_percentile", 0.5)),
        upper_percentile=float(data.get("intensity_upper_percentile", 99.5)),
    )
    fractions = [float(value) for value in data.get("split_fractions", (0.8, 0.1, 0.1))]
    pair = data.get("time_pair", ["T0", "T1"])
    result = prepare_dataset(
        visit_manifest,
        output_dir,
        split_ratios=dict(zip(("train", "val", "test"), fractions, strict=True)),
        split_seed=int(data.get("split_seed", config["project"]["seed"])),
        config=settings,
        earlier_stage=str(pair[0]),
        later_stage=str(pair[1]),
    )
    click.echo(
        json.dumps(
            {
                "visit_manifest": result.visit_manifest_path,
                "pair_manifest": result.pair_manifest_path,
                "intensity_stats": result.intensity_stats_path,
                "prepared_visits": len(result.visits),
                "prepared_pairs": len(result.pairs),
                "invalid_pairs": len(result.invalid_pair_ids),
                "skipped_visits": len(result.skipped_visit_ids),
            },
            indent=2,
            sort_keys=True,
        )
    )


@main.command("train-autoencoder")
@_config_options
@click.option("--manifest", type=click.Path(exists=True, dir_okay=False, path_type=str), required=True)
@click.option("--output-dir", type=click.Path(file_okay=False, path_type=str), default=None)
@click.option("--resume", type=click.Path(exists=True, dir_okay=False, path_type=str), default=None)
@click.option("--max-steps", type=click.IntRange(min=1), default=None)
def train_autoencoder_command(
    config_path: str,
    overrides: tuple[str, ...],
    manifest: str,
    output_dir: str | None,
    resume: str | None,
    max_steps: int | None,
) -> None:
    """Train the one shared 3D AutoencoderKL across training visits."""

    from ispy2_symmflow.training.run import train_autoencoder

    config = _configuration(config_path, overrides)
    destination = output_dir or str(Path(config["project"]["output_dir"]) / "autoencoder")
    result = train_autoencoder(
        config, manifest, output_dir=destination, resume=resume, max_steps=max_steps
    )
    click.echo(json.dumps(_json_ready(result), indent=2, sort_keys=True))


@main.command("cache-latents")
@_config_options
@click.option("--manifest", type=click.Path(exists=True, dir_okay=False, path_type=str), required=True)
@click.option("--pair-manifest", type=click.Path(exists=True, dir_okay=False, path_type=str), required=True)
@click.option(
    "--autoencoder-checkpoint",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    required=True,
)
@click.option("--output-dir", type=click.Path(file_okay=False, path_type=str), default=None)
def cache_latents_command(
    config_path: str,
    overrides: tuple[str, ...],
    manifest: str,
    pair_manifest: str,
    autoencoder_checkpoint: str,
    output_dir: str | None,
) -> None:
    """Fit train-only latent statistics and cache deterministic posterior means."""

    import torch

    from ispy2_symmflow.models import build_autoencoder_from_config
    from ispy2_symmflow.training.cache import cache_latents
    from ispy2_symmflow.training.checkpoint import load_checkpoint
    from ispy2_symmflow.utils.hashing import sha256_file

    config = _configuration(config_path, overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_header = torch.load(
        autoencoder_checkpoint, map_location="cpu", weights_only=False
    )
    checkpoint_config = (
        checkpoint_header.get("config")
        if isinstance(checkpoint_header, Mapping)
        else None
    )
    if not isinstance(checkpoint_config, Mapping):
        raise click.ClickException(
            "autoencoder checkpoint has no training configuration"
        )
    autoencoder = build_autoencoder_from_config(checkpoint_config).to(device)
    load_checkpoint(autoencoder_checkpoint, model=autoencoder, map_location=device)
    autoencoder.freeze()
    identifier = sha256_file(autoencoder_checkpoint)
    destination = output_dir or str(Path(config["project"]["output_dir"]) / "latents")
    result = cache_latents(
        autoencoder,
        manifest,
        destination,
        device=device,
        autoencoder_id=identifier,
        autoencoder_checkpoint_header=checkpoint_header,
        pair_manifest=pair_manifest,
    )
    click.echo(json.dumps(_json_ready(result), indent=2, sort_keys=True))


@main.command("import-mewm-latents")
@_config_options
@click.option(
    "--cache-identity",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    required=True,
)
@click.option(
    "--bundle-dir",
    type=click.Path(exists=True, file_okay=False, path_type=str),
    required=True,
)
@click.option(
    "--source-latent-dir",
    type=click.Path(exists=True, file_okay=False, path_type=str),
    required=True,
)
@click.option(
    "--upstream-metadata-dir",
    type=click.Path(exists=True, file_okay=False, path_type=str),
    required=True,
    help="Signed upstream registration metadata used to recover physical LPS grids.",
)
@click.option(
    "--vqgan-checkpoint",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    required=True,
)
@click.option(
    "--pair-id",
    "pair_ids",
    multiple=True,
    help="Import only this transition ID; repeat for a smoke subset.",
)
@click.option("--output-dir", type=click.Path(file_okay=False, path_type=str), required=True)
def import_mewm_latents_command(
    config_path: str,
    overrides: tuple[str, ...],
    cache_identity: str,
    bundle_dir: str,
    source_latent_dir: str,
    upstream_metadata_dir: str,
    vqgan_checkpoint: str,
    pair_ids: tuple[str, ...],
    output_dir: str,
) -> None:
    """Validate and import paired MeWM pre-quantization latent payloads."""

    from ispy2_symmflow.training.mewm_import import (
        import_mewm_continuous_latents,
    )

    from ispy2_symmflow.config import resolve_time_pairs

    config = _configuration(config_path, overrides)
    time_pairs = resolve_time_pairs(config["data"])
    result = import_mewm_continuous_latents(
        cache_identity,
        bundle_dir,
        source_latent_dir,
        vqgan_checkpoint,
        output_dir,
        upstream_metadata_dir=upstream_metadata_dir,
        time_pairs=time_pairs,
        selected_pair_ids=pair_ids or None,
    )
    click.echo(
        json.dumps(
            _json_ready(
                {
                    "pair_manifest": result.pair_manifest_path,
                    "visit_manifest": result.visit_manifest_path,
                    "latent_statistics": result.statistics_path,
                    "audit": result.audit_path,
                    "pair_count": result.pair_count,
                    "visit_count": result.visit_count,
                    "split_pair_counts": result.split_pair_counts,
                    "autoencoder_id": result.autoencoder_id,
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )


@main.command("import-mu-glioma-latents")
@_config_options
@click.option(
    "--cache-identity",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    required=True,
)
@click.option(
    "--source-latent-dir",
    type=click.Path(exists=True, file_okay=False, path_type=str),
    required=True,
)
@click.option(
    "--clinical-timeline",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    required=True,
)
@click.option(
    "--vqgan-checkpoint",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    required=True,
)
@click.option("--output-dir", type=click.Path(file_okay=False, path_type=str), required=True)
def import_mu_glioma_latents_command(
    config_path: str,
    overrides: tuple[str, ...],
    cache_identity: str,
    source_latent_dir: str,
    clinical_timeline: str,
    vqgan_checkpoint: str,
    output_dir: str,
) -> None:
    """Import the audited four-modality MU-Glioma longitudinal cache."""

    from ispy2_symmflow.training.mu_glioma_import import (
        MU_NORMALIZATION,
        import_mu_glioma_continuous_latents,
    )

    config = _configuration(config_path, overrides)
    result = import_mu_glioma_continuous_latents(
        cache_identity,
        source_latent_dir,
        clinical_timeline,
        vqgan_checkpoint,
        output_dir,
        flow_normalization=str(
            config.get("data", {}).get("latent_normalization", MU_NORMALIZATION)
        ),
    )
    click.echo(
        json.dumps(
            _json_ready(
                {
                    "pair_manifest": result.pair_manifest_path,
                    "visit_manifest": result.visit_manifest_path,
                    "latent_statistics": result.statistics_path,
                    "audit": result.audit_path,
                    "pair_count": result.pair_count,
                    "visit_count": result.visit_count,
                    "split_pair_counts": result.split_pair_counts,
                    "autoencoder_id": result.autoencoder_id,
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )


@main.command("train-symmflow")
@_config_options
@click.option(
    "--pair-manifest",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    required=True,
)
@click.option(
    "--autoencoder-checkpoint",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    required=True,
)
@click.option("--output-dir", type=click.Path(file_okay=False, path_type=str), default=None)
@click.option("--resume", type=click.Path(exists=True, dir_okay=False, path_type=str), default=None)
@click.option("--max-steps", type=click.IntRange(min=1), default=None)
def train_symmflow_command(
    config_path: str,
    overrides: tuple[str, ...],
    pair_manifest: str,
    autoencoder_checkpoint: str,
    output_dir: str | None,
    resume: str | None,
    max_steps: int | None,
) -> None:
    """Freeze cached latents and train one two-branch velocity network."""

    from ispy2_symmflow.training.run import train_symmflow
    from ispy2_symmflow.utils.hashing import sha256_file

    config = _configuration(config_path, overrides)
    destination = output_dir or str(Path(config["project"]["output_dir"]) / "symmflow")
    result = train_symmflow(
        config,
        pair_manifest,
        output_dir=destination,
        expected_autoencoder_id=sha256_file(autoencoder_checkpoint),
        resume=resume,
        max_steps=max_steps,
    )
    click.echo(json.dumps(_json_ready(result), indent=2, sort_keys=True))


@main.command("train-baseline")
@_config_options
@click.option(
    "--kind",
    type=click.Choice(["deterministic", "unidirectional_cfm"]),
    required=True,
)
@click.option(
    "--pair-manifest",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    required=True,
)
@click.option(
    "--autoencoder-checkpoint",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    required=True,
)
@click.option("--output-dir", type=click.Path(file_okay=False, path_type=str), default=None)
@click.option("--resume", type=click.Path(exists=True, dir_okay=False, path_type=str), default=None)
@click.option("--max-steps", type=click.IntRange(min=1), default=None)
def train_baseline_command(
    config_path: str,
    overrides: tuple[str, ...],
    kind: str,
    pair_manifest: str,
    autoencoder_checkpoint: str,
    output_dir: str | None,
    resume: str | None,
    max_steps: int | None,
) -> None:
    """Train one forward-only comparison on the cached latent pairs."""

    from ispy2_symmflow.training.baselines import train_baseline
    from ispy2_symmflow.utils.hashing import sha256_file

    config = _configuration(config_path, overrides)
    destination = output_dir or str(
        Path(config["project"]["output_dir"]) / "baselines" / kind
    )
    result = train_baseline(
        config,
        pair_manifest,
        kind=kind,
        output_dir=destination,
        expected_autoencoder_id=sha256_file(autoencoder_checkpoint),
        resume=resume,
        max_steps=max_steps,
    )
    click.echo(json.dumps(_json_ready(result), indent=2, sort_keys=True))


def _load_sampling_models(
    config: Mapping[str, Any],
    checkpoint: str,
    autoencoder_checkpoint: str,
    device,
):
    import torch
    from torch import nn

    from ispy2_symmflow.models import (
        ConditionSchema,
        StructuredConditionEncoder,
        build_velocity_model_from_config,
    )
    from ispy2_symmflow.training.checkpoint import load_checkpoint
    from ispy2_symmflow.training.ema import ExponentialMovingAverage
    from ispy2_symmflow.training.provenance import (
        require_latent_statistics_fingerprint,
    )
    from ispy2_symmflow.utils.hashing import sha256_file

    header = torch.load(checkpoint, map_location="cpu", weights_only=False)
    training_config = header["config"]
    schema = ConditionSchema.from_dict(header["feature_schema"])
    actual_autoencoder_id = sha256_file(autoencoder_checkpoint)
    if actual_autoencoder_id != header["autoencoder_id"]:
        raise click.ClickException("autoencoder checkpoint hash differs from the flow checkpoint")
    statistics = header.get("latent_statistics")
    if not isinstance(statistics, Mapping):
        raise click.ClickException("flow checkpoint has no latent normalization statistics")
    try:
        require_latent_statistics_fingerprint(statistics)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    autoencoder = _load_sampling_codec(
        training_config,
        statistics,
        autoencoder_checkpoint,
        device,
    )
    condition_encoder = StructuredConditionEncoder(schema).to(device).eval()
    velocity = build_velocity_model_from_config(training_config).to(device).eval()
    system = nn.ModuleDict({"velocity": velocity, "conditions": condition_encoder})
    ema = ExponentialMovingAverage(velocity)
    load_checkpoint(
        checkpoint,
        model=system,
        ema=ema,
        expected_autoencoder_id=actual_autoencoder_id,
        expected_feature_schema=schema.to_dict(),
        expected_latent_statistics=statistics,
        map_location=device,
    )
    for name, parameter in velocity.named_parameters():
        parameter.data.copy_(ema.shadow[name].to(parameter))
    return autoencoder, velocity, condition_encoder, header


def _load_sampling_codec(
    training_config: Mapping[str, Any],
    latent_statistics: Mapping[str, Any],
    autoencoder_checkpoint: str,
    device,
):
    """Load the checkpoint-bound codec used to produce the training latents."""

    import torch

    from ispy2_symmflow.models import build_codec_from_config
    from ispy2_symmflow.training.checkpoint import load_checkpoint

    raw_codec = training_config.get("codec")
    is_mewm = (
        isinstance(raw_codec, Mapping)
        and str(raw_codec.get("backend", "")).strip().lower()
        in {"mewm_vqgan", "mewm_mu_glioma_vqgan"}
    )
    if is_mewm:
        codec_config = dict(training_config)
        codec_config["codec"] = {
            **dict(raw_codec),
            "checkpoint_path": str(Path(autoencoder_checkpoint).expanduser().resolve()),
        }
        codec = build_codec_from_config(
            codec_config,
            latent_statistics=latent_statistics,
        ).to(device)
    else:
        autoencoder_header = torch.load(
            autoencoder_checkpoint, map_location="cpu", weights_only=False
        )
        checkpoint_config = (
            autoencoder_header.get("config")
            if isinstance(autoencoder_header, Mapping)
            else None
        )
        if not isinstance(checkpoint_config, Mapping):
            raise click.ClickException(
                "autoencoder checkpoint has no training configuration"
            )
        codec = build_codec_from_config(
            checkpoint_config,
            latent_statistics=latent_statistics,
        ).to(device)
        load_checkpoint(autoencoder_checkpoint, model=codec, map_location=device)
        # Local checkpoints include the codec buffers; restore the flow-fitted
        # normalization after loading so inference uses the latent-cache contract.
        codec.set_latent_statistics(
            latent_statistics["mean"], latent_statistics["std"]
        )
    return codec.freeze()


@main.command("sample")
@_config_options
@click.option("--checkpoint", type=click.Path(exists=True, dir_okay=False, path_type=str), required=True)
@click.option(
    "--autoencoder-checkpoint",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    required=True,
)
@click.option("--source", type=click.Path(exists=True, dir_okay=False, path_type=str), required=True)
@click.option("--conditions", type=click.Path(exists=True, dir_okay=False, path_type=str), required=True)
@click.option("--direction", type=click.Choice(["forward", "backward"]), required=True)
@click.option("--output", type=click.Path(dir_okay=False, path_type=str), required=True)
@click.option("--num-samples", type=click.IntRange(min=1), default=None)
@click.option("--steps", type=click.IntRange(min=1), default=None)
@click.option("--seed", type=int, default=None)
@click.option("--solver", type=click.Choice(["euler", "heun"]), default=None)
@click.option("--return-joint-state/--no-return-joint-state", default=True)
@click.option(
    "--allow-residual-endpoint",
    is_flag=True,
    help="Explicitly allow sigma_min>0 compatibility sampling with non-clean endpoints.",
)
def sample_command(
    config_path: str,
    overrides: tuple[str, ...],
    checkpoint: str,
    autoencoder_checkpoint: str,
    source: str,
    conditions: str,
    direction: str,
    output: str,
    num_samples: int | None,
    steps: int | None,
    seed: int | None,
    solver: str | None,
    return_joint_state: bool,
    allow_residual_endpoint: bool,
) -> None:
    """Generate candidates from one source MRI and known conditions only."""

    import torch

    from ispy2_symmflow.inference.sampler import SymmFlowSampler, save_sample_batch
    from ispy2_symmflow.training.datasets import load_prepared_image
    from ispy2_symmflow.utils.hashing import sha256_file, stable_hash

    config = _configuration(config_path, overrides)
    sampling = config["sampling"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder, velocity, encoder, flow_header = _load_sampling_models(
        config, checkpoint, autoencoder_checkpoint, device
    )
    try:
        condition_values = json.loads(Path(conditions).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"conditions file is not valid JSON: {exc}") from exc
    if not isinstance(condition_values, dict):
        raise click.ClickException("conditions JSON must contain one object")
    training_data = flow_header["config"]["data"]
    time_pair = _require_checkpoint_time_pair(condition_values, training_data)
    schema_provenance = _validate_sampling_conditions(condition_values, flow_header)
    training_split_hash, training_signature = _checkpoint_manifest_provenance(
        flow_header,
        ordered_key="ordered_manifest_fingerprint",
        label="SymmFlow",
    )
    expected_source_stage = time_pair[0] if direction == "forward" else time_pair[1]
    source_sha256 = sha256_file(source)
    try:
        source_image, source_metadata = load_prepared_image(source)
    except (OSError, TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    if sha256_file(source) != source_sha256:
        raise click.ClickException("source archive changed while it was being loaded")
    source_signature, observed_roles, source_identity = _validate_sampling_source(
        source_image,
        source_metadata,
        flow_header,
        expected_stage=expected_source_stage,
    )
    source_batch = source_image[None].to(device)
    tokens = encoder(condition_values, batch_size=1)
    sampler = SymmFlowSampler(
        autoencoder,
        velocity,
        sigma_min=float(flow_header["config"]["flow"].get("sigma_min", 0.0)),
    )
    kwargs = {
        "num_samples": int(num_samples or sampling.get("num_samples", 8)),
        "seed": int(seed if seed is not None else sampling.get("seed", 0)),
        "steps": int(steps or sampling.get("steps", 25)),
        "solver": str(solver or sampling.get("solver", "heun")),
        "return_joint_state": return_joint_state,
        "allow_residual_endpoint": allow_residual_endpoint,
    }
    if direction == "forward":
        batch = sampler.sample_forward(source_batch, tokens, **kwargs)
    else:
        batch = sampler.sample_backward(source_batch, tokens, **kwargs)
    array_path, sidecar = save_sample_batch(
        batch,
        output,
        metadata={
            "source_path": str(Path(source).resolve()),
            "source_sha256": source_sha256,
            "source_metadata": source_metadata,
            "conditions": condition_values,
            "solver": kwargs["solver"],
            "steps": kwargs["steps"],
            "sigma_min": sampler.sigma_min,
            **_endpoint_provenance(
                sigma_min=sampler.sigma_min,
                direction=direction,
                residual_consent=allow_residual_endpoint,
            ),
            "branch_order": ["later", "earlier"],
            "flow_checkpoint": str(Path(checkpoint).resolve()),
            "flow_checkpoint_sha256": sha256_file(checkpoint),
            "autoencoder_checkpoint": str(Path(autoencoder_checkpoint).resolve()),
            "autoencoder_checkpoint_sha256": sha256_file(autoencoder_checkpoint),
            "training_config_fingerprint": stable_hash(flow_header["config"]),
            "training_split_hash": training_split_hash,
            "training_signature": training_signature,
            "condition_schema": flow_header["feature_schema"],
            "condition_schema_provenance": schema_provenance,
            "source_phase_roles": observed_roles,
            "source_visit_stage": source_identity["visit_stage"],
            "source_split": source_identity["split"],
            "source_preprocessing_signature": source_signature,
            "source_preprocessing_provenance": source_metadata.get("provenance_json"),
        },
    )
    click.echo(json.dumps({"samples": str(array_path), "metadata": str(sidecar)}, indent=2))


def _load_baseline_sampling_models(
    checkpoint: str,
    autoencoder_checkpoint: str,
    device,
):
    import torch
    from torch import nn

    from ispy2_symmflow.models import (
        ConditionSchema,
        StructuredConditionEncoder,
        build_deterministic_baseline_from_config,
        build_unidirectional_cfm_from_config,
    )
    from ispy2_symmflow.training.baselines import BASELINE_KINDS
    from ispy2_symmflow.training.checkpoint import load_checkpoint
    from ispy2_symmflow.training.ema import ExponentialMovingAverage
    from ispy2_symmflow.training.provenance import (
        require_latent_statistics_fingerprint,
    )
    from ispy2_symmflow.utils.hashing import sha256_file

    header = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(header, Mapping):
        raise click.ClickException("baseline checkpoint must contain a mapping")
    training_config = header.get("config")
    feature_schema = header.get("feature_schema")
    statistics = header.get("latent_statistics")
    extra = header.get("extra")
    if not isinstance(training_config, Mapping):
        raise click.ClickException("baseline checkpoint has no training config")
    if not isinstance(feature_schema, Mapping) or not feature_schema:
        raise click.ClickException("baseline checkpoint has no condition schema")
    if not isinstance(statistics, Mapping) or any(
        key not in statistics for key in ("mean", "std")
    ):
        raise click.ClickException("baseline checkpoint has no latent normalization statistics")
    try:
        require_latent_statistics_fingerprint(statistics)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    if not isinstance(extra, Mapping):
        raise click.ClickException("baseline checkpoint has no baseline provenance")
    kind = str(extra.get("baseline_kind", ""))
    if kind not in BASELINE_KINDS:
        raise click.ClickException(f"unsupported baseline checkpoint kind {kind!r}")
    if extra.get("direction") != "forward":
        raise click.ClickException("baseline checkpoint is not marked as forward-only")

    schema = ConditionSchema.from_dict(feature_schema)
    if not {"stage_i", "stage_j"}.issubset(schema.field_names):
        raise click.ClickException(
            "baseline checkpoint condition schema must include stage_i and stage_j"
        )
    actual_autoencoder_id = sha256_file(autoencoder_checkpoint)
    if actual_autoencoder_id != str(header.get("autoencoder_id", "")):
        raise click.ClickException(
            "autoencoder checkpoint hash differs from the baseline checkpoint"
        )
    if str(statistics.get("autoencoder_id", "")) != actual_autoencoder_id:
        raise click.ClickException(
            "baseline latent statistics identify a different autoencoder checkpoint"
        )
    autoencoder = _load_sampling_codec(
        training_config,
        statistics,
        autoencoder_checkpoint,
        device,
    )

    condition_encoder = StructuredConditionEncoder(schema).to(device).eval()
    model = (
        build_deterministic_baseline_from_config(training_config)
        if kind == "deterministic"
        else build_unidirectional_cfm_from_config(training_config)
    ).to(device).eval()
    system = nn.ModuleDict({"model": model, "conditions": condition_encoder})
    ema = ExponentialMovingAverage(model)
    load_checkpoint(
        checkpoint,
        model=system,
        ema=ema,
        expected_autoencoder_id=actual_autoencoder_id,
        expected_feature_schema=schema.to_dict(),
        expected_latent_statistics=statistics,
        required_extra_keys=("baseline_kind", "direction"),
        map_location=device,
    )
    with torch.no_grad():
        parameters = dict(model.named_parameters())
        for name, average in ema.shadow.items():
            parameters[name].copy_(average.to(parameters[name]))
    return autoencoder, model.eval(), condition_encoder.eval(), header, kind


@main.command("sample-baseline")
@_config_options
@click.option("--checkpoint", type=click.Path(exists=True, dir_okay=False, path_type=str), required=True)
@click.option(
    "--autoencoder-checkpoint",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    required=True,
)
@click.option("--source", type=click.Path(exists=True, dir_okay=False, path_type=str), required=True)
@click.option("--conditions", type=click.Path(exists=True, dir_okay=False, path_type=str), required=True)
@click.option("--output", type=click.Path(dir_okay=False, path_type=str), required=True)
@click.option("--num-samples", "--k", type=click.IntRange(min=1), default=None)
@click.option("--steps", type=click.IntRange(min=1), default=None)
@click.option("--seed", type=int, default=None)
@click.option("--solver", type=click.Choice(["euler", "heun"]), default=None)
@click.option(
    "--allow-residual-endpoint",
    is_flag=True,
    help="Explicitly allow sigma_min>0 CFM compatibility sampling.",
)
def sample_baseline_command(
    config_path: str,
    overrides: tuple[str, ...],
    checkpoint: str,
    autoencoder_checkpoint: str,
    source: str,
    conditions: str,
    output: str,
    num_samples: int | None,
    steps: int | None,
    seed: int | None,
    solver: str | None,
    allow_residual_endpoint: bool,
) -> None:
    """Generate a forward prediction with a validation-selected baseline."""

    import torch

    from ispy2_symmflow.inference.baselines import (
        UnidirectionalCFMSampler,
        sample_deterministic_baseline,
    )
    from ispy2_symmflow.inference.sampler import save_sample_batch
    from ispy2_symmflow.training.datasets import load_prepared_image
    from ispy2_symmflow.utils.hashing import sha256_file, stable_hash

    config = _configuration(config_path, overrides)
    sampling = config["sampling"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder, model, encoder, baseline_header, kind = _load_baseline_sampling_models(
        checkpoint, autoencoder_checkpoint, device
    )
    if kind == "deterministic" and num_samples not in (None, 1):
        raise click.ClickException(
            "deterministic baseline produces exactly one candidate; use --num-samples 1"
        )

    try:
        condition_values = json.loads(Path(conditions).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"conditions file is not valid JSON: {exc}") from exc
    if not isinstance(condition_values, dict):
        raise click.ClickException("conditions JSON must contain one object")
    training_data = baseline_header["config"]["data"]
    time_pair = _require_checkpoint_time_pair(condition_values, training_data)
    schema_provenance = _validate_sampling_conditions(condition_values, baseline_header)
    training_split_hash, training_signature = _checkpoint_manifest_provenance(
        baseline_header,
        ordered_key="ordered_manifest_hash",
        label="baseline",
    )
    source_sha256 = sha256_file(source)
    try:
        source_image, source_metadata = load_prepared_image(source)
    except (OSError, TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    if sha256_file(source) != source_sha256:
        raise click.ClickException("source archive changed while it was being loaded")
    source_signature, observed_roles, source_identity = _validate_sampling_source(
        source_image,
        source_metadata,
        baseline_header,
        expected_stage=time_pair[0],
    )
    source_batch = source_image[None].to(device)

    effective_seed = int(seed if seed is not None else sampling.get("seed", 0))
    sigma_min = float(baseline_header["config"]["flow"].get("sigma_min", 0.0))
    baseline_settings = baseline_header["config"].get("baseline_training", {})
    cfm_base_distribution = str(
        baseline_settings.get("cfm_base_distribution", "standard_normal")
    )
    cfm_noise_scale = float(baseline_settings.get("cfm_noise_scale", 1.0))
    try:
        if kind == "deterministic":
            batch = sample_deterministic_baseline(
                autoencoder, encoder, model, source_batch, condition_values
            )
            effective_solver = "none"
            effective_steps = 0
        else:
            effective_solver = str(solver or sampling.get("solver", "heun"))
            effective_steps = int(steps or sampling.get("steps", 25))
            sampler = UnidirectionalCFMSampler(
                autoencoder,
                model,
                encoder,
                sigma_min=sigma_min,
                base_distribution=cfm_base_distribution,
                noise_scale=cfm_noise_scale,
            )
            batch = sampler.sample_forward(
                source_batch,
                condition_values,
                num_samples=int(
                    num_samples if num_samples is not None else sampling.get("num_samples", 8)
                ),
                seed=effective_seed,
                steps=effective_steps,
                solver=effective_solver,
                allow_residual_endpoint=allow_residual_endpoint,
            )
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    checkpoint_hash = sha256_file(checkpoint)
    autoencoder_hash = sha256_file(autoencoder_checkpoint)
    array_path, sidecar = save_sample_batch(
        batch,
        output,
        metadata={
            "model_family": "baseline",
            "baseline_kind": kind,
            "direction": "forward",
            "source_path": str(Path(source).resolve()),
            "source_sha256": source_sha256,
            "source_metadata": source_metadata,
            "conditions": condition_values,
            "time_pair": time_pair,
            "solver": effective_solver,
            "steps": effective_steps,
            "cfm_base_distribution": (
                cfm_base_distribution if kind == "unidirectional_cfm" else None
            ),
            "cfm_noise_scale": (
                cfm_noise_scale if kind == "unidirectional_cfm" else None
            ),
            "requested_seed": effective_seed,
            "sigma_min": sigma_min if kind == "unidirectional_cfm" else 0.0,
            **_endpoint_provenance(
                sigma_min=(sigma_min if kind == "unidirectional_cfm" else 0.0),
                direction="forward",
                residual_consent=allow_residual_endpoint,
            ),
            "branch_order": ["later"],
            "baseline_checkpoint": str(Path(checkpoint).resolve()),
            "baseline_checkpoint_sha256": checkpoint_hash,
            "autoencoder_checkpoint": str(Path(autoencoder_checkpoint).resolve()),
            "autoencoder_checkpoint_sha256": autoencoder_hash,
            "training_config_fingerprint": stable_hash(baseline_header["config"]),
            "training_split_hash": training_split_hash,
            "training_signature": training_signature,
            "condition_schema": baseline_header["feature_schema"],
            "condition_schema_provenance": schema_provenance,
            "latent_statistics": baseline_header["latent_statistics"],
            "ema_applied": True,
            "source_phase_roles": observed_roles,
            "source_visit_stage": source_identity["visit_stage"],
            "source_split": source_identity["split"],
            "source_preprocessing_signature": source_signature,
            "source_preprocessing_provenance": source_metadata.get("provenance_json"),
        },
    )
    click.echo(json.dumps({"samples": str(array_path), "metadata": str(sidecar)}, indent=2))


@main.command("evaluate")
@click.option("--prediction", type=click.Path(exists=True, dir_okay=False, path_type=str), required=True)
@click.option("--target", type=click.Path(exists=True, dir_okay=False, path_type=str), required=True)
@click.option("--source", type=click.Path(exists=True, dir_okay=False, path_type=str), default=None)
@click.option("--output", type=click.Path(dir_okay=False, path_type=str), required=True)
@click.option("--data-range", type=click.FloatRange(min=0, min_open=True), required=True)
@click.option("--foreground-threshold", type=float, default=-0.95, show_default=True)
@click.option(
    "--allow-unverified-geometry",
    is_flag=True,
    help="Allow exploratory scoring without a verifiable sampling sidecar/grid match.",
)
def evaluate_command(
    prediction: str,
    target: str,
    source: str | None,
    output: str,
    data_range: float,
    foreground_threshold: float,
    allow_unverified_geometry: bool,
) -> None:
    """Evaluate candidates separately from their mean and optional no-change baseline."""

    import numpy as np
    import torch

    from ispy2_symmflow.evaluation.aggregate import patient_bootstrap_mean_ci
    from ispy2_symmflow.evaluation.metrics import evaluate_prediction, evaluate_sample_set
    from ispy2_symmflow.inference.sampler import (
        validate_sample_array_binding,
        validate_source_archive_binding,
    )
    from ispy2_symmflow.training.datasets import load_prepared_image

    sidecar_path = Path(prediction).with_suffix(".json")
    sampling_record: dict[str, Any] | None = None
    sidecar_error: str | None = None
    if sidecar_path.is_file():
        try:
            value = json.loads(sidecar_path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("sampling sidecar must contain one object")
            validate_sample_array_binding(prediction, value)
            sampling_record = value
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            sidecar_error = str(exc)
    else:
        sidecar_error = "sampling sidecar is missing"
    if sampling_record is None and not allow_unverified_geometry:
        raise click.ClickException(
            f"prediction sampling sidecar/integrity verification failed: {sidecar_error}; "
            "use --allow-unverified-geometry only for explicitly exploratory scoring"
        )

    source_binding_error: str | None = None
    source_integrity_verified: bool | None = None
    source_integrity_status = "not_checked: --source was not provided"
    if source is not None:
        source_integrity_verified = False
        source_integrity_status = (
            f"unverified: {sidecar_error}"
            if sampling_record is None
            else "pending post-load verification"
        )
    if source is not None and sampling_record is not None:
        sample_metadata = sampling_record.get("metadata")
        if not isinstance(sample_metadata, Mapping):
            source_binding_error = "sampling sidecar metadata is not an object"
        else:
            recorded_source = str(sample_metadata.get("source_path", ""))
            if recorded_source and str(Path(source).resolve()) != recorded_source:
                raise click.ClickException(
                    "--source differs from the source recorded by sampling"
                )
            try:
                validate_source_archive_binding(source, sample_metadata)
            except (OSError, ValueError) as exc:
                source_binding_error = str(exc)
        if source_binding_error is not None:
            if not allow_unverified_geometry:
                raise click.ClickException(
                    "source archive integrity verification failed: "
                    f"{source_binding_error}; use --allow-unverified-geometry only for "
                    "explicitly exploratory scoring"
                )
            source_integrity_status = f"unverified: {source_binding_error}"

    with np.load(prediction, allow_pickle=False) as archive:
        if "samples" not in archive:
            raise click.ClickException("prediction NPZ must contain 'samples'")
        samples = torch.from_numpy(np.asarray(archive["samples"], dtype=np.float32))
    target_image, target_metadata = load_prepared_image(target)
    target_batch = target_image[None]
    foreground = target_batch > float(foreground_threshold)
    candidate_set = evaluate_sample_set(
        samples, target_batch, data_range=data_range, foreground_mask=foreground
    )
    patient_id = str(target_metadata.get("patient_id", "unknown"))
    prediction_integrity_verified = (
        sampling_record is not None and source_binding_error is None
    )
    prediction_integrity_status = (
        "matched_sampling_sidecar_sha256"
        if prediction_integrity_verified
        else f"unverified: {source_binding_error or sidecar_error}"
    )
    geometry_verified = False
    geometry_status = (
        "unverified" if sidecar_error is None else f"unverified: {sidecar_error}"
    )
    if sampling_record is None:
        pass
    else:
        sample_metadata = sampling_record.get("metadata", {})
        source_metadata = sample_metadata.get("source_metadata", {})
        source_patient = str(source_metadata.get("patient_id", "unknown"))
        if patient_id != "unknown" and source_patient != patient_id:
            raise click.ClickException(
                f"prediction source patient {source_patient} differs from target patient {patient_id}"
            )
        direction = sampling_record.get("direction")
        condition_values = sample_metadata.get("conditions", {})
        expected_target_key = "stage_j" if direction == "forward" else "stage_i"
        expected_target_stage = condition_values.get(expected_target_key)
        target_stage = target_metadata.get("visit_stage")
        if expected_target_stage and target_stage and str(expected_target_stage) != str(target_stage):
            raise click.ClickException(
                f"prediction expects target stage {expected_target_stage}, got {target_stage}"
            )
        source_affine = source_metadata.get("affine_lps")
        target_affine = target_metadata.get("affine_lps")
        source_spacing = source_metadata.get("spacing_dhw")
        target_spacing = target_metadata.get("spacing_dhw")
        comparable = all(
            value is not None
            for value in (source_affine, target_affine, source_spacing, target_spacing)
        )
        if comparable:
            geometry_verified = bool(
                np.allclose(source_affine, target_affine, rtol=1e-5, atol=1e-4)
                and np.allclose(source_spacing, target_spacing, rtol=1e-5, atol=1e-4)
            )
            geometry_status = "matched_source_target_physical_grid" if geometry_verified else "mismatch"
        if not geometry_verified and not allow_unverified_geometry:
            raise click.ClickException(
                "source and target physical grids are absent or differ; voxelwise metrics require "
                "validated alignment (or explicit exploratory --allow-unverified-geometry)"
            )
    report: dict[str, Any] = {
        "candidate_set": candidate_set,
        "data_range": data_range,
        "foreground_rule": f"target > {foreground_threshold} (evaluation only)",
        "patient_id": patient_id,
        "target_visit_stage": target_metadata.get("visit_stage"),
        "sampling_record": sampling_record,
        "prediction_integrity_verified": prediction_integrity_verified,
        "prediction_integrity_status": prediction_integrity_status,
        "source_integrity_verified": source_integrity_verified,
        "source_integrity_status": source_integrity_status,
        "voxelwise_geometry_verified": geometry_verified,
        "geometry_status": geometry_status,
        "patient_level_candidate_mae_ci": patient_bootstrap_mean_ci(
            {patient_id: float(candidate_set["candidate_mae_mean"].mean())}
        ),
    }
    if source is not None:
        source_image, explicit_source_metadata = load_prepared_image(source)
        if sampling_record is not None:
            if source_binding_error is None:
                try:
                    validate_source_archive_binding(
                        source, sampling_record.get("metadata", {})
                    )
                except (OSError, ValueError) as exc:
                    source_binding_error = str(exc)
                    if not allow_unverified_geometry:
                        raise click.ClickException(
                            "source archive integrity verification failed: "
                            f"{source_binding_error}; use --allow-unverified-geometry only for "
                            "explicitly exploratory scoring"
                        ) from exc
                else:
                    source_integrity_verified = True
                    source_integrity_status = "matched_sampling_sidecar_sha256"
            explicit_patient = str(explicit_source_metadata.get("patient_id", "unknown"))
            if patient_id != "unknown" and explicit_patient != patient_id:
                raise click.ClickException("copy-source baseline patient differs from target patient")
        if source_binding_error is not None:
            prediction_integrity_verified = False
            prediction_integrity_status = f"unverified: {source_binding_error}"
            source_integrity_verified = False
            source_integrity_status = prediction_integrity_status
        report["copy_source_no_change"] = evaluate_prediction(
            source_image[None],
            target_batch,
            data_range=data_range,
            foreground_mask=foreground,
        )
        report["prediction_integrity_verified"] = prediction_integrity_verified
        report["prediction_integrity_status"] = prediction_integrity_status
        report["source_integrity_verified"] = source_integrity_verified
        report["source_integrity_status"] = source_integrity_status
    destination = _write_json(output, report)
    click.echo(str(destination))


@main.command("evaluate-cohort")
@_config_options
@click.option(
    "--manifest",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    required=True,
    help="JSONL rows with prediction/target and optional source or provenance hints.",
)
@click.option(
    "--pair-manifest",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    required=True,
    help="Trusted cached pairs.jsonl used to train the evaluated model.",
)
@click.option("--output", type=click.Path(dir_okay=False, path_type=str), required=True)
@click.option("--data-range", type=click.FloatRange(min=0, min_open=True), default=None)
@click.option("--foreground-threshold", type=float, default=None)
@click.option("--bootstrap-samples", type=click.IntRange(min=1), default=None)
@click.option(
    "--bootstrap-confidence",
    type=click.FloatRange(min=0, max=1, min_open=True, max_open=True),
    default=0.95,
    show_default=True,
)
@click.option("--bootstrap-seed", type=int, default=None)
def evaluate_cohort_command(
    config_path: str,
    overrides: tuple[str, ...],
    manifest: str,
    pair_manifest: str,
    output: str,
    data_range: float | None,
    foreground_threshold: float | None,
    bootstrap_samples: int | None,
    bootstrap_confidence: float,
    bootstrap_seed: int | None,
) -> None:
    """Evaluate a provenance-checked held-out JSONL prediction cohort."""

    from ispy2_symmflow.evaluation import evaluate_cohort

    config = _configuration(config_path, overrides)
    evaluation = config.get("evaluation", {})
    try:
        report = evaluate_cohort(
            manifest,
            pair_manifest=pair_manifest,
            data_range=float(
                data_range if data_range is not None else evaluation.get("data_range", 2.0)
            ),
            foreground_threshold=float(
                foreground_threshold
                if foreground_threshold is not None
                else evaluation.get("foreground_threshold", -0.95)
            ),
            bootstrap_samples=int(
                bootstrap_samples
                if bootstrap_samples is not None
                else evaluation.get("bootstrap_samples", 1000)
            ),
            bootstrap_confidence=float(bootstrap_confidence),
            bootstrap_seed=int(
                bootstrap_seed
                if bootstrap_seed is not None
                else config.get("project", {}).get("seed", 0)
            ),
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    destination = _write_json(output, report)
    click.echo(str(destination))


@main.command("evaluate-autoencoder")
@_config_options
@click.option(
    "--checkpoint",
    "--autoencoder-checkpoint",
    "checkpoint",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    required=True,
)
@click.option(
    "--visit-manifest",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    required=True,
)
@click.option("--split", type=click.Choice(["val", "test"]), required=True)
@click.option("--output", type=click.Path(dir_okay=False, path_type=str), required=True)
@click.option("--device", type=click.Choice(["auto", "cpu", "cuda"]), default="auto")
@click.option("--data-range", type=click.FloatRange(min=0, min_open=True), default=None)
@click.option("--foreground-threshold", type=float, default=None)
@click.option("--bootstrap-samples", type=click.IntRange(min=1), default=None)
@click.option(
    "--bootstrap-confidence",
    type=click.FloatRange(min=0, max=1, min_open=True, max_open=True),
    default=0.95,
    show_default=True,
)
@click.option("--bootstrap-seed", type=int, default=None)
def evaluate_autoencoder_command(
    config_path: str,
    overrides: tuple[str, ...],
    checkpoint: str,
    visit_manifest: str,
    split: str,
    output: str,
    device: str,
    data_range: float | None,
    foreground_threshold: float | None,
    bootstrap_samples: int | None,
    bootstrap_confidence: float,
    bootstrap_seed: int | None,
) -> None:
    """Score posterior-mean autoencoder reconstructions on val or test visits."""

    from ispy2_symmflow.evaluation import evaluate_autoencoder_reconstruction

    config = _configuration(config_path, overrides)
    evaluation = config.get("evaluation", {})
    try:
        report = evaluate_autoencoder_reconstruction(
            checkpoint,
            visit_manifest,
            split=split,
            data_range=float(
                data_range if data_range is not None else evaluation.get("data_range", 2.0)
            ),
            foreground_threshold=float(
                foreground_threshold
                if foreground_threshold is not None
                else evaluation.get("foreground_threshold", -0.95)
            ),
            bootstrap_samples=int(
                bootstrap_samples
                if bootstrap_samples is not None
                else evaluation.get("bootstrap_samples", 1000)
            ),
            bootstrap_confidence=float(bootstrap_confidence),
            bootstrap_seed=int(
                bootstrap_seed
                if bootstrap_seed is not None
                else config.get("project", {}).get("seed", 0)
            ),
            device=device,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    destination = _write_json(output, report)
    click.echo(str(destination))


@main.command("make-synthetic")
@click.option("--output-dir", type=click.Path(file_okay=False, path_type=str), required=True)
@click.option("--patients", type=click.IntRange(min=3), default=6, show_default=True)
@click.option("--seed", type=int, default=1234, show_default=True)
def make_synthetic_command(output_dir: str, patients: int, seed: int) -> None:
    """Create labeled synthetic paired volumes for software smoke testing."""

    from ispy2_symmflow.data.synthetic import create_synthetic_dataset

    result = create_synthetic_dataset(output_dir, patient_count=patients, seed=seed)
    click.echo(json.dumps(result, indent=2, sort_keys=True))


# Underscore aliases match the pipeline stage names in experiment plans.
main.add_command(audit_data, "audit_data")
main.add_command(prepare_data, "prepare_data")
main.add_command(train_autoencoder_command, "train_autoencoder")
main.add_command(cache_latents_command, "cache_latents")
main.add_command(train_symmflow_command, "train_symmflow")
main.add_command(train_baseline_command, "train_baseline")
main.add_command(sample_baseline_command, "sample_baseline")
main.add_command(evaluate_cohort_command, "evaluate_cohort")
main.add_command(evaluate_autoencoder_command, "evaluate_autoencoder")


if __name__ == "__main__":
    main()
