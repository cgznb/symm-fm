from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn


MEDGEMMA_MODEL_ID = "google/medgemma-4b-it"
MEDGEMMA_REVISION = "290cda5eeccbee130f987c4ad74a59ae6f196408"


class SharedMedGemmaTower(nn.Module):
    """One revision-locked NF4/LoRA language tower shared by both text fields."""

    def __init__(self, model: nn.Module, tokenizer: Any, *, hidden_size: int) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.hidden_size = int(hidden_size)

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_id: str = MEDGEMMA_MODEL_ID,
        revision: str = MEDGEMMA_REVISION,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        local_files_only: bool = False,
    ) -> "SharedMedGemmaTower":
        if model_id != MEDGEMMA_MODEL_ID or revision != MEDGEMMA_REVISION:
            raise ValueError("MedGemma base identity must match the locked model revision")
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForImageTextToText,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, local_files_only=local_files_only
        )
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            revision=revision,
            quantization_config=quantization,
            device_map="auto",
            local_files_only=local_files_only,
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(
            model,
            LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "v_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
        config = model.config
        text_config = getattr(config, "text_config", config)
        return cls(model, tokenizer, hidden_size=int(text_config.hidden_size))

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        if not texts or not all(isinstance(text, str) and text.strip() for text in texts):
            raise ValueError("text batches must contain nonempty strings")
        tokens = self.tokenizer(
            list(texts), padding=True, truncation=True, return_tensors="pt"
        )
        device = next(self.model.parameters()).device
        tokens = {key: value.to(device) for key, value in tokens.items()}
        outputs = self.model(
            **tokens, output_hidden_states=True, return_dict=True, use_cache=False
        )
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            language_output = getattr(outputs, "language_model_output", None)
            hidden_states = getattr(language_output, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError("MedGemma did not return language hidden states")
        hidden = hidden_states[-1]
        mask = tokens["attention_mask"].to(hidden.dtype).unsqueeze(-1)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


class FourierDays(nn.Module):
    def __init__(self, output_dim: int = 128) -> None:
        super().__init__()
        if output_dim % 2:
            raise ValueError("Fourier day dimension must be even")
        frequencies = torch.exp(
            torch.linspace(math.log(1.0), math.log(1.0 / 10000.0), output_dim // 2)
        )
        self.register_buffer("frequencies", frequencies)

    def forward(self, days: torch.Tensor) -> torch.Tensor:
        angles = days.float().reshape(-1, 1) * self.frequencies.reshape(1, -1)
        return torch.cat((angles.sin(), angles.cos()), dim=1)


class ISPY2Conditioner(nn.Module):
    def __init__(self, text_tower: nn.Module, *, text_hidden_size: int | None = None) -> None:
        super().__init__()
        self.text_tower = text_tower
        hidden_size = int(
            text_hidden_size
            if text_hidden_size is not None
            else getattr(text_tower, "hidden_size")
        )
        self.text_projection = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 768)
        )
        self.days_encoder = FourierDays(128)
        self.stage_embedding = nn.Embedding(4, 64)
        self.fusion = nn.Sequential(
            nn.LayerNorm(1728),
            nn.Linear(1728, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
        )

    def forward(
        self,
        action_text: Sequence[str],
        clinical_text: Sequence[str],
        delta_days: torch.Tensor,
        stage_id: torch.Tensor,
    ) -> torch.Tensor:
        if len(action_text) != len(clinical_text):
            raise ValueError("action and clinical text batch sizes must match")
        device = self.text_projection[1].weight.device
        action_hidden = self.text_tower.encode(list(action_text)).to(device)
        clinical_hidden = self.text_tower.encode(list(clinical_text)).to(device)
        action_embedding = self.text_projection(action_hidden)
        clinical_embedding = self.text_projection(clinical_hidden)
        days_embedding = self.days_encoder(delta_days.to(device))
        stages = stage_id.to(device=device, dtype=torch.long)
        if torch.any((stages < 1) | (stages > 3)):
            raise ValueError("stage IDs must be 1, 2, or 3")
        stage_embedding = self.stage_embedding(stages)
        combined = torch.cat(
            (action_embedding, clinical_embedding, days_embedding, stage_embedding), dim=1
        )
        if combined.shape[1] != 1728:
            raise RuntimeError("FiLM fusion input contract is not 1728D")
        return self.fusion(combined)
