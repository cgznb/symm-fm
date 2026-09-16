# 12 drugs = the union of every token across the 13 I-SPY2 arms (Arm split on " + ").
DRUGS = [
    "Paclitaxel", "ABT 888", "Carboplatin", "AMG 386", "Trastuzumab", "Ganetespib",
    "Ganitumab", "MK-2206", "Neratinib", "Pembrolizumab", "Pertuzumab", "T-DM1",
]


def drug_col(d):
    """Column name for a drug: 'ABT 888' -> 'drug_abt_888', 'T-DM1' -> 'drug_t_dm1'."""
    return "drug_" + d.lower().replace(" ", "_").replace("-", "_")


DRUG_COLS = [drug_col(d) for d in DRUGS]


def arm_binaries(arm):
    """{drug_name: 0/1} for an Arm string ('Paclitaxel + AMG 386'); non-str (ACRIN/NaN) -> all 0."""
    parts = {p.strip() for p in str(arm).split("+")} if isinstance(arm, str) else set()
    return {d: (1 if d in parts else 0) for d in DRUGS}


def arm_column_row(arm):
    """{drug_col: 0/1} ready to write as CSV columns."""
    return {drug_col(d): v for d, v in arm_binaries(arm).items()}
