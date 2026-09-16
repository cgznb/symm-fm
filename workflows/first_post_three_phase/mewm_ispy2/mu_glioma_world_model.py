from __future__ import annotations

import hashlib
import math
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


MU_MODALITY_COUNT = 4
MU_LATENT_CHANNELS = 8
MU_PRODUCTION_LATENT_SHAPE = (40, 64, 64)
MU_FM_PATCH_GRID = (10, 8, 8)
MU_IMAGE_GRID = (2, 4, 4)
MU_IMAGE_TOKENS_PER_MODALITY = math.prod(MU_IMAGE_GRID)
MU_IMAGE_TOKEN_COUNT = MU_MODALITY_COUNT * MU_IMAGE_TOKENS_PER_MODALITY
MU_CONTEXT_TOKEN_COUNT = MU_IMAGE_TOKEN_COUNT + 4
MU_CONTEXT_DIM = 768
MU_MEDGEMMA_MODEL_ID = "google/medgemma-4b-it"
MU_MEDGEMMA_REVISION = "290cda5eeccbee130f987c4ad74a59ae6f196408"
MU_MEDGEMMA_LORA_TARGET = (
    r".*language_model\.layers\.\d+\.self_attn\.(q_proj|v_proj)$"
)
MU_EMPTY_CLINICAL_TEXT = "Clinical context is unavailable."


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(root: Path) -> str:
    try:
        result = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={root}",
                "-C",
                str(root),
                "rev-parse",
                "HEAD",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        raise ValueError("FM-BCMRI Git revision is unavailable") from None
    return result.stdout.strip()


@dataclass(frozen=True)
class MUFMLoadReport:
    revision: str
    checkpoint_sha256: str
    loaded_parameters: int
    missing_keys: tuple[str, ...]


def _adapt_fm_patch_projection(
    body: nn.Module,
    source_weight: torch.Tensor,
    source_bias: torch.Tensor,
) -> None:
    if tuple(source_weight.shape) != (768, 1, 8, 8, 8):
        raise ValueError("FM-BCMRI source patch projection shape changed")
    target = body.patch_embed.proj
    if tuple(target.weight.shape) != (768, 8, 4, 8, 8):
        raise ValueError("MU FM-BCMRI patch projection shape changed")
    resized = F.interpolate(
        source_weight.float(),
        size=(4, 8, 8),
        mode="trilinear",
        align_corners=False,
    )
    source_norm = source_weight.float().flatten(1).norm(dim=1).clamp_min(1e-12)
    resized_norm = resized.flatten(1).norm(dim=1).clamp_min(1e-12)
    resized = resized * (source_norm / resized_norm).reshape(-1, 1, 1, 1, 1)
    adapted = resized.repeat(1, MU_LATENT_CHANNELS, 1, 1, 1) / math.sqrt(
        MU_LATENT_CHANNELS
    )
    with torch.no_grad():
        target.weight.copy_(adapted)
        target.bias.copy_(source_bias.float())


def load_mu_fmbcmri_body(
    *,
    root: str | Path,
    revision: str,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
) -> tuple[nn.Module, MUFMLoadReport]:
    """Load FM-BCMRI and adapt only its patch stem to 8-channel MU latents."""

    fm_root = Path(root).expanduser().resolve()
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    if _git_revision(fm_root) != revision:
        raise ValueError("FM-BCMRI Git revision mismatch")
    actual_sha256 = _sha256_file(checkpoint_file)
    if actual_sha256 != checkpoint_sha256:
        raise ValueError("FM-BCMRI checkpoint SHA256 mismatch")
    if str(fm_root) not in sys.path:
        sys.path.insert(0, str(fm_root))
    try:
        from fmbcmri.lib.layers.patch_embed import PatchEmbed3D  # type: ignore
        from fmbcmri.lib.models.vision_transformer_moco import (  # type: ignore
            VisionTransformerMoCo3D,
        )
    except Exception as error:
        raise ImportError(f"could not import FM-BCMRI from {fm_root}") from error

    body = VisionTransformerMoCo3D(
        img_size=MU_PRODUCTION_LATENT_SHAPE,
        patch_size=(4, 8, 8),
        in_chans=MU_LATENT_CHANNELS,
        num_classes=0,
        embed_dim=MU_CONTEXT_DIM,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        embed_layer=PatchEmbed3D,
    )
    if tuple(body.patch_embed.grid_size) != MU_FM_PATCH_GRID:
        raise RuntimeError("MU FM-BCMRI patch grid changed")
    payload = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    raw_state = payload.get("state_dict", payload)
    if not isinstance(raw_state, dict):
        raise ValueError("FM-BCMRI checkpoint state_dict is invalid")
    source = {
        key.removeprefix("base_encoder."): value
        for key, value in raw_state.items()
        if key.startswith("base_encoder.") and isinstance(value, torch.Tensor)
    }
    patch_weight = source.get("patch_embed.proj.weight")
    patch_bias = source.get("patch_embed.proj.bias")
    if not isinstance(patch_weight, torch.Tensor) or not isinstance(
        patch_bias, torch.Tensor
    ):
        raise ValueError("FM-BCMRI checkpoint is missing its patch projection")
    target_state = body.state_dict()
    compatible = {
        key: value
        for key, value in source.items()
        if key not in {"patch_embed.proj.weight", "pos_embed"}
        and key in target_state
        and tuple(value.shape) == tuple(target_state[key].shape)
    }
    incompatible = body.load_state_dict(compatible, strict=False)
    expected_missing = ("patch_embed.proj.weight", "pos_embed")
    missing = tuple(sorted(incompatible.missing_keys))
    if missing != expected_missing or incompatible.unexpected_keys:
        raise ValueError("FM-BCMRI checkpoint incompatibility exceeds stem and position")
    _adapt_fm_patch_projection(body, patch_weight, patch_bias)
    return body, MUFMLoadReport(
        revision=revision,
        checkpoint_sha256=actual_sha256,
        loaded_parameters=sum(value.numel() for value in compatible.values()),
        missing_keys=missing,
    )


@runtime_checkable
class MUTextTower(Protocol):
    hidden_size: int

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        """Encode a text batch into `[B,H]`."""


@runtime_checkable
class MUImageTower(Protocol):
    output_dim: int

    def encode(self, source_latent: torch.Tensor) -> torch.Tensor:
        """Encode `[B*4,8,40,64,64]` into `[B*4,32,768]`."""


class MUFMImageTower(nn.Module):
    """Shared, fully trainable FM-BCMRI image tower for all four modalities."""

    output_dim = MU_CONTEXT_DIM

    def __init__(self, body: nn.Module, *, checkpoint_blocks: bool = True) -> None:
        super().__init__()
        self.body = body
        self.checkpoint_blocks = bool(checkpoint_blocks)

    def encode(self, source_latent: torch.Tensor) -> torch.Tensor:
        if tuple(source_latent.shape[1:]) != (
            MU_LATENT_CHANNELS,
            *MU_PRODUCTION_LATENT_SHAPE,
        ):
            raise ValueError("MU FM input must be [B*4,8,40,64,64]")
        batch = source_latent.shape[0]
        tokens = self.body.patch_embed(source_latent)
        if tuple(tokens.shape) != (batch, math.prod(MU_FM_PATCH_GRID), MU_CONTEXT_DIM):
            raise RuntimeError("MU FM patch token shape changed")
        position = self.body.pos_embed.to(device=tokens.device, dtype=tokens.dtype)
        if tuple(position.shape) != (
            1,
            math.prod(MU_FM_PATCH_GRID) + 1,
            MU_CONTEXT_DIM,
        ):
            raise RuntimeError("MU FM fixed 3D position embedding shape changed")
        cls = self.body.cls_token.expand(batch, -1, -1).to(dtype=tokens.dtype)
        tokens = torch.cat((cls, tokens), dim=1) + position
        tokens = self.body.pos_drop(tokens)
        tokens = self.body.norm_pre(tokens)
        for block in self.body.blocks:
            if self.checkpoint_blocks and self.training and torch.is_grad_enabled():
                tokens = checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
        tokens = self.body.norm(tokens)[:, 1:]
        volume = tokens.transpose(1, 2).reshape(
            batch, MU_CONTEXT_DIM, *MU_FM_PATCH_GRID
        )
        pooled = F.adaptive_avg_pool3d(volume, MU_IMAGE_GRID)
        return pooled.flatten(2).transpose(1, 2).contiguous()

    def forward(self, source_latent: torch.Tensor) -> torch.Tensor:
        return self.encode(source_latent)


class MUMedicalTextTower(nn.Module):
    """Revision-locked MedGemma text tower with NF4 QLoRA adapters."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        *,
        hidden_size: int,
        max_length: int = 512,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or max_length <= 0:
            raise ValueError("MU text tower dimensions must be positive")
        self.model = model
        self.tokenizer = tokenizer
        self.hidden_size = int(hidden_size)
        self.max_length = int(max_length)

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_id: str = MU_MEDGEMMA_MODEL_ID,
        revision: str = MU_MEDGEMMA_REVISION,
        max_length: int = 512,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        local_files_only: bool = True,
    ) -> "MUMedicalTextTower":
        if model_id != MU_MEDGEMMA_MODEL_ID or revision != MU_MEDGEMMA_REVISION:
            raise ValueError("MU MedGemma identity must match the locked revision")
        try:
            from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
            from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig
        except Exception as error:
            raise ImportError("Transformers, bitsandbytes, and PEFT are required") from error
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, local_files_only=local_files_only
        )
        base = AutoModelForImageTextToText.from_pretrained(
            model_id,
            revision=revision,
            quantization_config=quantization,
            device_map="auto",
            local_files_only=local_files_only,
        )
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
        model = get_peft_model(
            base,
            LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=MU_MEDGEMMA_LORA_TARGET,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            ),
        )
        text_config = getattr(model.config, "text_config", model.config)
        return cls(
            model,
            tokenizer,
            hidden_size=int(getattr(text_config, "hidden_size")),
            max_length=max_length,
        )

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        if isinstance(texts, (str, bytes)) or not texts or any(
            not isinstance(text, str) or not text.strip() for text in texts
        ):
            raise ValueError("MU text batches must contain nonempty strings")
        tokens = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        tokens = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in tokens.items()
        }
        outputs = self.model(
            **tokens,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            language = getattr(outputs, "language_model_output", None)
            hidden_states = getattr(language, "hidden_states", None)
        if not hidden_states:
            raise RuntimeError("MU MedGemma did not return language hidden states")
        hidden = hidden_states[-1]
        attention_mask = tokens.get("attention_mask")
        if not isinstance(attention_mask, torch.Tensor) or tuple(hidden.shape) != (
            len(texts),
            attention_mask.shape[1],
            self.hidden_size,
        ):
            raise RuntimeError("MU MedGemma hidden-state shape changed")
        mask = attention_mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def forward(self, texts: Sequence[str]) -> torch.Tensor:
        return self.encode(texts)

    def lora_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(
            parameter
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad and "lora_" in name
        )


class MUFourierDays(nn.Module):
    def __init__(self, output_dim: int = 128) -> None:
        super().__init__()
        if output_dim <= 0 or output_dim % 2:
            raise ValueError("MU Fourier day dimension must be positive and even")
        frequencies = torch.exp(
            torch.linspace(math.log(1.0), math.log(1.0 / 10000.0), output_dim // 2)
        )
        self.register_buffer("frequencies", frequencies)

    def forward(self, days: torch.Tensor) -> torch.Tensor:
        if days.ndim != 1 or not bool(torch.isfinite(days).all()):
            raise ValueError("MU delta days must be a finite [B] tensor")
        angles = days.float().reshape(-1, 1) * self.frequencies.reshape(1, -1)
        return torch.cat((angles.sin(), angles.cos()), dim=1)


@dataclass(frozen=True)
class MUMultimodalContext:
    tokens: torch.Tensor


class MUMultimodalConditioner(nn.Module):
    """Build one 132-token condition sequence for each target modality."""

    def __init__(
        self,
        image_tower: MUImageTower,
        text_tower: MUTextTower,
        *,
        text_hidden_size: int | None = None,
        context_dim: int = MU_CONTEXT_DIM,
        time_fourier_dim: int = 128,
        empty_clinical_text: str = MU_EMPTY_CLINICAL_TEXT,
    ) -> None:
        super().__init__()
        if not callable(getattr(image_tower, "encode", None)):
            raise TypeError("MU conditioner image tower must implement encode")
        if not callable(getattr(text_tower, "encode", None)):
            raise TypeError("MU conditioner text tower must implement encode")
        hidden_size = text_hidden_size or getattr(text_tower, "hidden_size", None)
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError("MU conditioner requires a positive text hidden size")
        if image_tower.output_dim != context_dim:
            raise ValueError("MU FM output and context dimensions must match")
        self.image_tower = image_tower
        self.text_tower = text_tower
        self.text_hidden_size = hidden_size
        self.context_dim = int(context_dim)
        self.empty_clinical_text = empty_clinical_text
        self.source_modality_embedding = nn.Embedding(MU_MODALITY_COUNT, context_dim)
        self.target_modality_embedding = nn.Embedding(MU_MODALITY_COUNT, context_dim)
        self.text_projection = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Linear(hidden_size, context_dim)
        )
        self.text_role_embedding = nn.Embedding(2, context_dim)
        self.days_encoder = MUFourierDays(time_fourier_dim)
        self.days_projection = nn.Sequential(
            nn.LayerNorm(time_fourier_dim), nn.Linear(time_fourier_dim, context_dim)
        )

    @property
    def token_count(self) -> int:
        return MU_CONTEXT_TOKEN_COUNT

    def _text_values(
        self,
        clinical_text: Sequence[str],
        treatment_text: Sequence[str],
        batch: int,
    ) -> tuple[list[str], list[str]]:
        if (
            isinstance(clinical_text, (str, bytes))
            or isinstance(treatment_text, (str, bytes))
            or len(clinical_text) != batch
            or len(treatment_text) != batch
        ):
            raise ValueError("MU text conditions must contain one value per patient")
        clinical = [
            value if isinstance(value, str) and value.strip() else self.empty_clinical_text
            for value in clinical_text
        ]
        if any(not isinstance(value, str) or not value.strip() for value in treatment_text):
            raise ValueError("MU treatment text values must be nonempty")
        return clinical, list(treatment_text)

    def forward(
        self,
        source_latent: torch.Tensor,
        *,
        clinical_text: Sequence[str],
        treatment_text: Sequence[str],
        delta_days: torch.Tensor,
    ) -> MUMultimodalContext:
        if tuple(source_latent.shape[1:]) != (
            MU_MODALITY_COUNT,
            MU_LATENT_CHANNELS,
            *MU_PRODUCTION_LATENT_SHAPE,
        ):
            raise ValueError("MU source latent must be [B,4,8,40,64,64]")
        batch = source_latent.shape[0]
        if delta_days.shape != (batch,) or not bool(torch.isfinite(delta_days).all()):
            raise ValueError("MU delta days must be a finite [B] tensor")
        clinical, treatment = self._text_values(
            clinical_text, treatment_text, batch
        )
        flat_source = source_latent.reshape(
            batch * MU_MODALITY_COUNT,
            MU_LATENT_CHANNELS,
            *MU_PRODUCTION_LATENT_SHAPE,
        )
        image = self.image_tower.encode(flat_source)
        if tuple(image.shape) != (
            batch * MU_MODALITY_COUNT,
            MU_IMAGE_TOKENS_PER_MODALITY,
            self.context_dim,
        ):
            raise RuntimeError("MU FM pooled token shape changed")
        image = image.reshape(
            batch,
            MU_MODALITY_COUNT,
            MU_IMAGE_TOKENS_PER_MODALITY,
            self.context_dim,
        )
        modality_ids = torch.arange(MU_MODALITY_COUNT, device=image.device)
        image = image + self.source_modality_embedding(modality_ids).reshape(
            1, MU_MODALITY_COUNT, 1, self.context_dim
        ).to(dtype=image.dtype)
        image = image.flatten(1, 2)

        hidden = self.text_tower.encode([*clinical, *treatment])
        if tuple(hidden.shape) != (2 * batch, self.text_hidden_size):
            raise RuntimeError("MU shared text tower output shape changed")
        projection = self.text_projection[1]
        hidden = hidden.to(device=projection.weight.device, dtype=projection.weight.dtype)
        text = F.normalize(self.text_projection(hidden), p=2, dim=-1)
        text = text.to(device=image.device, dtype=image.dtype).reshape(2, batch, -1)
        roles = self.text_role_embedding(
            torch.arange(2, device=image.device)
        ).to(dtype=image.dtype)
        clinical_token = text[0] + roles[0]
        treatment_token = text[1] + roles[1]

        days_device = self.days_projection[1].weight.device
        days = self.days_projection(self.days_encoder(delta_days.to(days_device)))
        days = days.to(device=image.device, dtype=image.dtype)
        shared = torch.cat(
            (
                image,
                clinical_token.unsqueeze(1),
                treatment_token.unsqueeze(1),
                days.unsqueeze(1),
            ),
            dim=1,
        )
        shared = shared.unsqueeze(1).expand(-1, MU_MODALITY_COUNT, -1, -1)
        target = self.target_modality_embedding(modality_ids).reshape(
            1, MU_MODALITY_COUNT, 1, self.context_dim
        ).expand(batch, -1, -1, -1)
        tokens = torch.cat((shared, target.to(dtype=image.dtype)), dim=2)
        return MUMultimodalContext(
            tokens=tokens.reshape(
                batch * MU_MODALITY_COUNT,
                MU_CONTEXT_TOKEN_COUNT,
                self.context_dim,
            )
        )


@dataclass(frozen=True)
class MUPreparedContext:
    tokens: torch.Tensor


@dataclass(frozen=True)
class MUWorldModelOutput:
    velocity: torch.Tensor


class MUGliomaWorldModel(nn.Module):
    """Single-branch four-modality rectified-flow dynamics model."""

    def __init__(
        self,
        *,
        conditioner: MUMultimodalConditioner,
        flow_unet: nn.Module,
        expected_spatial_shape: tuple[int, int, int] = MU_PRODUCTION_LATENT_SHAPE,
    ) -> None:
        super().__init__()
        self.conditioner = conditioner
        self.flow_unet = flow_unet
        self.expected_spatial_shape = tuple(expected_spatial_shape)

    def _validate_source(self, source: torch.Tensor) -> tuple[int, tuple[int, int, int]]:
        if source.ndim != 6 or source.shape[1:3] != (
            MU_MODALITY_COUNT,
            MU_LATENT_CHANNELS,
        ):
            raise ValueError("MU world-model source must be [B,4,8,D,H,W]")
        spatial = tuple(int(value) for value in source.shape[3:])
        if spatial != self.expected_spatial_shape:
            raise ValueError("MU production latent spatial shape changed")
        return int(source.shape[0]), spatial

    def prepare_context(
        self,
        source_latent: torch.Tensor,
        *,
        clinical_text: Sequence[str],
        treatment_text: Sequence[str],
        delta_days: torch.Tensor,
    ) -> MUPreparedContext:
        self._validate_source(source_latent)
        context = self.conditioner(
            source_latent,
            clinical_text=clinical_text,
            treatment_text=treatment_text,
            delta_days=delta_days,
        ).tokens
        return MUPreparedContext(tokens=context)

    def velocity_from_context(
        self,
        *,
        source_latent: torch.Tensor,
        flow_state: torch.Tensor,
        flow_time: torch.Tensor,
        prepared: MUPreparedContext,
    ) -> MUWorldModelOutput:
        batch, spatial = self._validate_source(source_latent)
        if flow_state.shape != source_latent.shape:
            raise ValueError("MU source and flow states must have matching shapes")
        if flow_time.shape != (batch,) or not bool(torch.isfinite(flow_time).all()):
            raise ValueError("MU flow time must be a finite [B] tensor")
        flat_batch = batch * MU_MODALITY_COUNT
        if tuple(prepared.tokens.shape) != (
            flat_batch,
            MU_CONTEXT_TOKEN_COUNT,
            MU_CONTEXT_DIM,
        ):
            raise ValueError("MU prepared context must be [B*4,132,768]")
        flat_source = source_latent.reshape(
            flat_batch, MU_LATENT_CHANNELS, *spatial
        )
        flat_state = flow_state.reshape(flat_batch, MU_LATENT_CHANNELS, *spatial)
        sample = torch.cat((flat_state, flat_source), dim=1)
        timesteps = (
            flow_time.to(device=sample.device, dtype=sample.dtype)
            .reshape(batch, 1)
            .expand(-1, MU_MODALITY_COUNT)
            .reshape(flat_batch)
            * 1000.0
        )
        velocity = self.flow_unet(sample, timesteps, context=prepared.tokens)
        velocity = getattr(velocity, "sample", velocity)
        if not isinstance(velocity, torch.Tensor) or tuple(velocity.shape) != (
            flat_batch,
            MU_LATENT_CHANNELS,
            *spatial,
        ):
            raise RuntimeError("MU dynamics velocity shape changed")
        return MUWorldModelOutput(
            velocity=velocity.reshape(
                batch, MU_MODALITY_COUNT, MU_LATENT_CHANNELS, *spatial
            )
        )

    def forward(
        self,
        *,
        source_latent: torch.Tensor,
        flow_state: torch.Tensor,
        flow_time: torch.Tensor,
        clinical_text: Sequence[str],
        treatment_text: Sequence[str],
        delta_days: torch.Tensor,
    ) -> MUWorldModelOutput:
        prepared = self.prepare_context(
            source_latent,
            clinical_text=clinical_text,
            treatment_text=treatment_text,
            delta_days=delta_days,
        )
        return self.velocity_from_context(
            source_latent=source_latent,
            flow_state=flow_state,
            flow_time=flow_time,
            prepared=prepared,
        )


def build_mu_glioma_world_model(
    *,
    image_tower: MUImageTower,
    text_tower: MUTextTower,
    text_hidden_size: int | None = None,
) -> MUGliomaWorldModel:
    try:
        from generative.networks.nets import DiffusionModelUNet
    except Exception as error:
        raise ImportError("MONAI Generative is required for MU dynamics") from error
    conditioner = MUMultimodalConditioner(
        image_tower,
        text_tower,
        text_hidden_size=text_hidden_size,
    )
    flow = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=16,
        out_channels=MU_LATENT_CHANNELS,
        num_res_blocks=2,
        num_channels=(64, 128, 256),
        attention_levels=(False, False, True),
        norm_num_groups=32,
        num_head_channels=32,
        with_conditioning=True,
        cross_attention_dim=MU_CONTEXT_DIM,
    )
    return MUGliomaWorldModel(conditioner=conditioner, flow_unet=flow)


__all__ = [
    "MU_CONTEXT_DIM",
    "MU_CONTEXT_TOKEN_COUNT",
    "MU_EMPTY_CLINICAL_TEXT",
    "MU_FM_PATCH_GRID",
    "MU_IMAGE_GRID",
    "MU_IMAGE_TOKEN_COUNT",
    "MU_MEDGEMMA_LORA_TARGET",
    "MU_MEDGEMMA_MODEL_ID",
    "MU_MEDGEMMA_REVISION",
    "MUFMLoadReport",
    "MUFMImageTower",
    "MUFourierDays",
    "MUGliomaWorldModel",
    "MUMedicalTextTower",
    "MUMultimodalConditioner",
    "MUMultimodalContext",
    "MUPreparedContext",
    "MUTextTower",
    "MUWorldModelOutput",
    "build_mu_glioma_world_model",
    "load_mu_fmbcmri_body",
]
