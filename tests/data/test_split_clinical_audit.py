from __future__ import annotations

import csv
from pathlib import Path

from ispy2_symmflow.data.audit import audit_dataset
from ispy2_symmflow.data.clinical import load_clinical_records
from ispy2_symmflow.data.split import assign_patient_splits


def test_patient_split_is_exact_and_deterministic() -> None:
    patients = [f"P{index:03d}" for index in range(58)]
    first = assign_patient_splits(patients, seed=17)
    second = assign_patient_splits(list(reversed(patients)), seed=17)
    assert first == second
    assert list(first.values()).count("train") == 46
    assert list(first.values()).count("val") == 6
    assert list(first.values()).count("test") == 6


def test_outcomes_are_quarantined_from_conditions(tmp_path: Path) -> None:
    path = tmp_path / "clinical.csv"
    path.write_text(
        "Subject ID,Age,HER2 Status,Treatment Arm,pCR,RCB Class\n"
        "P1,51,positive,arm-a,1,0\n",
        encoding="utf-8",
    )
    record = load_clinical_records(path)["P1"]
    assert record.baseline == {"age": "51", "her2_status": "positive"}
    assert record.treatment == {"treatment_arm": "arm-a"}
    assert record.evaluation_only == {"pcr": "1", "rcb_class": "0"}


def test_fast_audit_uses_series_metadata_without_pixels(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.csv"
    fields = [
        "Series UID",
        "Collection",
        "Subject ID",
        "Study UID",
        "Study Description",
        "Study Date",
        "Series Description",
        "Manufacturer",
        "Modality",
        "SOP Class UID",
        "Number of Images",
        "File Size",
    ]
    rows = [
        {
            "Series UID": "1.2.3.1",
            "Collection": "ISPY2",
            "Subject ID": "P1",
            "Study UID": "1.2.3",
            "Study Description": "ISPY2MRIT0",
            "Study Date": "01-01-2020",
            "Series Description": "ISPY2 VOLSER original DCE",
            "Manufacturer": "TEST",
            "Modality": "MR",
            "SOP Class UID": "1.2.840.10008.5.1.4.1.1.4",
            "Number of Images": "12",
            "File Size": "1024",
        },
        {
            "Series UID": "1.2.3.2",
            "Collection": "ISPY2",
            "Subject ID": "P1",
            "Study UID": "1.2.3",
            "Study Description": "ISPY2MRIT0",
            "Study Date": "01-01-2020",
            "Series Description": "ISPY2 VOLSER Analysis Mask",
            "Manufacturer": "TEST",
            "Modality": "SEG",
            "SOP Class UID": "1.2.840.10008.5.1.4.1.1.66.4",
            "Number of Images": "1",
            "File Size": "512",
        },
    ]
    with metadata.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    audit = audit_dataset(tmp_path, header_sample_limit=0)
    assert audit.patient_count == 1
    assert audit.study_count == 1
    assert audit.dicom_instance_count == 13
    assert audit.dce_candidate_studies == 1
    assert audit.segmentation_studies == 1


def test_fast_audit_honors_explicit_metadata_csv(tmp_path: Path) -> None:
    data_root = tmp_path / "dicom-root"
    data_root.mkdir()
    metadata = tmp_path / "custom-series-index.csv"
    metadata.write_text(
        "Series UID,Collection,Subject ID,Study UID,Study Description,Study Date,"
        "Series Description,Manufacturer,Modality,SOP Class UID,Number of Images,File Size\n"
        "1.2.3.1,ISPY2,P1,1.2.3,ISPY2MRIT0,01-01-2020,"
        "ISPY2 VOLSER original DCE,TEST,MR,1.2.840.10008.5.1.4.1.1.4,12,1024\n",
        encoding="utf-8",
    )

    audit = audit_dataset(
        data_root,
        metadata_csv=metadata,
        header_sample_limit=0,
    )

    assert audit.patient_count == 1
    assert audit.series_count == 1
