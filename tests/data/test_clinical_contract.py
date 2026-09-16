from __future__ import annotations

import pytest

from ispy2_symmflow.data.clinical import load_clinical_records


def test_clinical_aliases_and_receptors_are_canonicalized(tmp_path) -> None:
    table = tmp_path / "clinical.csv"
    table.write_text(
        "Subject ID,Age at Diagnosis,Arm,ER Status,PR Status\n"
        "P1,51,arm-a,Positive,Negative\n"
        "P2,49,arm-b,Negative,Negative\n",
        encoding="utf-8",
    )

    records = load_clinical_records(table)
    assert records["P1"].baseline == {"age": "51", "hr_status": "positive"}
    assert records["P1"].treatment == {"treatment_arm": "arm-a"}
    assert records["P2"].baseline == {"age": "49", "hr_status": "negative"}


def test_single_negative_receptor_is_retained_as_missing_hr_status(tmp_path) -> None:
    table = tmp_path / "clinical.csv"
    table.write_text(
        "Subject ID,ER Status,PR Status\nP1,Negative,\n",
        encoding="utf-8",
    )

    record = load_clinical_records(table)["P1"]
    assert "hr_status" not in record.baseline


def test_explicit_unavailable_hr_status_is_retained_as_missing(tmp_path) -> None:
    table = tmp_path / "clinical.csv"
    table.write_text(
        "Subject ID,HR Status\nP1,Not assessed\n",
        encoding="utf-8",
    )

    record = load_clinical_records(table)["P1"]
    assert "hr_status" not in record.baseline


def test_unrecognized_hr_status_still_fails_closed(tmp_path) -> None:
    table = tmp_path / "clinical.csv"
    table.write_text(
        "Subject ID,HR Status\nP1,maybe positive\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unrecognized hr_status"):
        load_clinical_records(table)


def test_her2_status_is_canonicalized_and_missing_is_omitted(tmp_path) -> None:
    table = tmp_path / "clinical.csv"
    table.write_text(
        "Subject ID,HER2 Status\nP1,POS\nP2,Equivocal\n",
        encoding="utf-8",
    )

    records = load_clinical_records(table)
    assert records["P1"].baseline["her2_status"] == "positive"
    assert "her2_status" not in records["P2"].baseline


def test_explicit_hr_status_must_agree_with_receptors(tmp_path) -> None:
    table = tmp_path / "clinical.csv"
    table.write_text(
        "Subject ID,HR Status,ER Status,PR Status\n"
        "P1,Negative,Positive,Negative\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="conflicts with ER/PR"):
        load_clinical_records(table)
