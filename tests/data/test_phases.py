from __future__ import annotations

from pathlib import Path

from ispy2_symmflow.data.dicom import DicomHeader, SeriesSummary
from ispy2_symmflow.data.phases import identify_temporal_dce_phases


def _header(
    temporal_position: int,
    z: float,
    *,
    acquisition_time: str = "071702",
    bolus_time: str | None = None,
) -> DicomHeader:
    return DicomHeader(
        path=Path(f"tp{temporal_position}-z{z}.dcm"),
        patient_id="ISPY2-TEST",
        study_uid="1.2.3",
        series_uid="1.2.3.4",
        sop_instance_uid=f"1.2.3.4.{temporal_position}.{z}",
        sop_class_uid="1.2.840.10008.5.1.4.1.1.4",
        modality="MR",
        study_description="ISPY2MRIT0",
        series_description="ISPY2: VOLSER: bi-lateral: original DCE",
        study_date="20200101",
        series_number="61800",
        instance_number=int(z + 1),
        acquisition_number=None,
        temporal_position=temporal_position,
        number_of_temporal_positions=3,
        acquisition_time=acquisition_time,
        acquisition_datetime=None,
        contrast_bolus_start_time=bolus_time,
        rows=4,
        columns=5,
        number_of_frames=None,
        pixel_spacing=(1.0, 1.0),
        slice_thickness=1.0,
        spacing_between_slices=1.0,
        orientation_lps=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        position_lps=(0.0, 0.0, z),
        transfer_syntax_uid="1.2.840.10008.1.2.1",
        transfer_syntax_compressed=False,
        image_type=("ORIGINAL", "PRIMARY", "DYNAMIC"),
        rescale_slope=None,
        rescale_intercept=None,
    )


def _summary(headers: list[DicomHeader]) -> SeriesSummary:
    return SeriesSummary(
        path=Path("/dataset/series"),
        collection="ISPY2",
        patient_directory="ISPY2-TEST",
        study_directory="study",
        header=headers[0],
        instance_count=len(headers),
    )


def test_identical_times_require_verified_volser_convention() -> None:
    headers = [_header(tp, z) for tp in (2, 0, 1) for z in (2.0, 0.0, 1.0)]
    rejected = identify_temporal_dce_phases(_summary(headers), headers)
    assert not rejected.usable
    assert any(event.code == "ambiguous_phase_time_order" for event in rejected.qc)

    accepted = identify_temporal_dce_phases(
        _summary(headers), headers, derived_phase_convention_verified=True
    )
    assert accepted.usable
    assert accepted.phases["pre"].temporal_position == 0
    assert accepted.phases["early"].temporal_position == 1
    assert accepted.phases["late"].temporal_position == 2
    assert "ordered_by_verified_volser_temporal_position_convention" in accepted.phases[
        "pre"
    ].evidence


def test_bolus_time_can_establish_phase_semantics() -> None:
    times = {0: "095900", 1: "100100", 2: "100500"}
    headers = [
        _header(tp, z, acquisition_time=times[tp], bolus_time="100000")
        for tp in (1, 2, 0)
        for z in (0.0, 1.0)
    ]
    assessment = identify_temporal_dce_phases(_summary(headers), headers)
    assert assessment.usable
    assert assessment.phases["pre"].temporal_position == 0
    assert assessment.phases["early"].temporal_position == 1
    assert assessment.phases["late"].temporal_position == 2
    assert assessment.phases["early"].reliability == "high"


def test_partial_acquisition_times_do_not_crash_bolus_comparison() -> None:
    headers = [
        _header(
            tp,
            z,
            acquisition_time="100100" if tp != 1 else "",
            bolus_time="100000",
        )
        for tp in (0, 1, 2)
        for z in (0.0, 1.0)
    ]
    assessment = identify_temporal_dce_phases(
        _summary(headers), headers, derived_phase_convention_verified=True
    )
    assert assessment.usable
    assert any(event.code == "bolus_comparison_unavailable" for event in assessment.qc)
