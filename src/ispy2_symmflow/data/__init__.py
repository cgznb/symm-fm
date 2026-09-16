"""DICOM indexing, QC, manifests, and patient-level splitting."""

from .audit import DataAudit, audit_dataset
from .manifest import (
    build_pair_manifest,
    build_visit_manifest,
    read_jsonl,
    read_visit_manifest,
    visit_from_dict,
    write_jsonl,
)
from .mewm import (
    MewmConnectedPairGeometry,
    MewmImportResult,
    MewmMetadataValidationResult,
    MewmVisitGeometry,
    import_mewm_roi_cache,
    validate_mewm_bundle_metadata,
)
from .schema import PairRecord, PhaseRef, QCEvent, VisitRecord
from .split import assign_patient_splits

__all__ = [
    "DataAudit",
    "MewmConnectedPairGeometry",
    "MewmImportResult",
    "MewmMetadataValidationResult",
    "MewmVisitGeometry",
    "PairRecord",
    "PhaseRef",
    "QCEvent",
    "VisitRecord",
    "assign_patient_splits",
    "audit_dataset",
    "build_pair_manifest",
    "build_visit_manifest",
    "import_mewm_roi_cache",
    "read_jsonl",
    "read_visit_manifest",
    "visit_from_dict",
    "validate_mewm_bundle_metadata",
    "write_jsonl",
]
