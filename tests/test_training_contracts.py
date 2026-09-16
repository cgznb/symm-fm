from __future__ import annotations

import pytest

from ispy2_symmflow.training.contracts import (
    require_configured_time_pair,
    require_configured_time_pairs,
)


def test_all_qc_passing_pairs_must_match_configured_time_pair() -> None:
    records = [
        {
            "pair_id": "patient-a-T0-T1",
            "earlier_stage": "T0",
            "later_stage": "T1",
        },
        {
            "pair_id": "patient-b-T0-T3",
            "earlier_stage": "T0",
            "later_stage": "T3",
        },
    ]

    with pytest.raises(ValueError, match="patient-b-T0-T3"):
        require_configured_time_pair(records, ["T0", "T1"])


def test_configured_time_pair_contract_accepts_one_exact_interval() -> None:
    pair = require_configured_time_pair(
        [{"pair_id": "p", "earlier_stage": "T1", "later_stage": "T2"}],
        ["T1", "T2"],
    )
    assert pair == ("T1", "T2")


def test_configured_time_pairs_accept_all_declared_intervals() -> None:
    records = [
        {"pair_id": "p-a", "earlier_stage": "T0", "later_stage": "T2"},
        {"pair_id": "p-b", "earlier_stage": "T1", "later_stage": "T3"},
    ]
    pairs = require_configured_time_pairs(
        records,
        {"time_pair": None, "time_pairs": [["T0", "T2"], ["T1", "T3"]]},
    )
    assert pairs == (("T0", "T2"), ("T1", "T3"))


def test_configured_time_pairs_require_each_declared_interval() -> None:
    records = [{"pair_id": "p-a", "earlier_stage": "T0", "later_stage": "T1"}]
    with pytest.raises(ValueError, match="T1->T2"):
        require_configured_time_pairs(
            records,
            {"time_pair": None, "time_pairs": [["T0", "T1"], ["T1", "T2"]]},
        )
