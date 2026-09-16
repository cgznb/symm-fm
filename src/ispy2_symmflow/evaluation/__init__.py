"""MRI-aware image and optional physical-volume evaluation."""

from .aggregate import aggregate_by_patient, patient_bootstrap_mean_ci
from .autoencoder import evaluate_autoencoder_reconstruction
from .cohort import evaluate_cohort
from .diagnostics import DiagnosticCase, patient_derangement, run_usage_diagnostics
from .metrics import evaluate_prediction, evaluate_sample_set

__all__ = [
    "DiagnosticCase",
    "aggregate_by_patient",
    "evaluate_autoencoder_reconstruction",
    "evaluate_cohort",
    "evaluate_prediction",
    "evaluate_sample_set",
    "patient_bootstrap_mean_ci",
    "patient_derangement",
    "run_usage_diagnostics",
]
