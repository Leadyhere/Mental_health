"""Load the raw xlsx dataset, apply the label fixes documented in
docs/data_notes.md, encode categorical fields, and write a stratified
train/eval split as parquet files plus a label_maps.json.

Findings this script encodes (see docs/data_notes.md for full detail):
  - Rows with Q2-Q15 all null are mislabeled "Low" in the raw sheet; they are
    the true "N/A-Feeling Fine" cohort and are relabeled here.
  - The raw `Risk Level` column (not the stale `Summary Statistics` sheet)
    is the authoritative label, corroborated by the independent `Risk Counts`
    sheet.
  - Q9/Q10/Q11 raw values are mapped onto the live verbatim answer options.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_XLSX = REPO_ROOT / "data" / "raw" / "mental_health_final_USER_TONE_dataset (1).xlsx"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

Q_COLUMNS = [f"Q{i}" for i in range(2, 16)]  # Q2..Q15, raw-fix null-check columns
COLUMN_RENAME = {
    "User #": "user_id",
    "Timestamp": "timestamp",
    "Q1: Reason for Opening Form": "q1_reason",
    "Q2: When Does It Feel Hardest": "q2_when_hardest",
    "Q3: How Is It Affecting Daily Life": "q3_daily_impact",
    "Q4: How Long Has This Been Going On": "q4_duration",
    "Q5: Current Emotions": "q5_emotions",
    "Q6: Coping Behaviors": "q6_coping",
    "Q7: Feels Supported": "q7_supported",
    "Q8: Intensity (0-10)": "q8_intensity",
    "Q9: Passive SI": "q9_passive_si",
    "Q10: Active SI / SH Thoughts": "q10_active_si",
    "Q11: Feels Safe": "q11_feels_safe",
    "Q12: Self-Label": "q12_self_label",
    "Q13: Impact on Sleep / Appetite / Energy": "q13_sleep_appetite_energy",
    "Q14: Support Person Available": "q14_support_person",
    "Q15: Open to Getting Support": "q15_open_to_support",
    "Risk Level": "risk_level_raw",
    "Suggested Next Steps": "suggested_next_steps",
}

# Q2..Q15 renamed columns, used to detect the all-null N/A-Feeling-Fine cohort.
Q_RENAMED = [
    "q2_when_hardest", "q3_daily_impact", "q4_duration", "q5_emotions",
    "q6_coping", "q7_supported", "q8_intensity", "q9_passive_si",
    "q10_active_si", "q11_feels_safe", "q12_self_label",
    "q13_sleep_appetite_energy", "q14_support_person", "q15_open_to_support",
]

Q9_MAP = {"No": "No", "Rarely": "Occasionally", "Sometimes": "Occasionally", "Often": "Often"}
Q10_MAP = {
    "No": "No",
    "Brief thoughts, but I don't want to act on them": "Yes but wouldn't act",
    "Ongoing thoughts, but no plan": "Yes but wouldn't act",
    "Yes, I may act on these thoughts": "Yes and it feels risky right now",
}

RISK_LEVELS = ["N/A-Feeling Fine", "Low", "Mild", "Moderate", "Severe", "Emergency"]


def load_raw() -> pd.DataFrame:
    df = pd.read_excel(RAW_XLSX, sheet_name="Combined Data")
    df = df.rename(columns=COLUMN_RENAME)
    return df


def apply_label_fix(df: pd.DataFrame) -> pd.DataFrame:
    is_na_cohort = df[Q_RENAMED].isna().all(axis=1)
    df["risk_level"] = df["risk_level_raw"]
    df.loc[is_na_cohort, "risk_level"] = "N/A-Feeling Fine"
    return df


def apply_answer_mappings(df: pd.DataFrame) -> pd.DataFrame:
    df["q9_passive_si_mapped"] = df["q9_passive_si"].map(Q9_MAP)
    df["q10_active_si_mapped"] = df["q10_active_si"].map(Q10_MAP)
    return df


def derive_sub_labels(df: pd.DataFrame) -> pd.DataFrame:
    # suicide_risk: directly reuses the mapped Q10 (falls back to "No" for the
    # N/A cohort where Q10 is null, since no distress was reported).
    df["suicide_risk"] = df["q10_active_si_mapped"].fillna("No")

    # functional_impairment: derived from Q3 (daily-life impact) free text
    # presence + Q13 categorical severity; kept coarse (low/some/marked) since
    # both source columns are free-text/categorical, not numeric.
    def impairment(row):
        if pd.isna(row["q13_sleep_appetite_energy"]):
            return "none"
        val = str(row["q13_sleep_appetite_energy"]).lower()
        if "noticeable disruption" in val:
            return "marked"
        if "slight" in val:
            return "some"
        return "some"

    df["functional_impairment"] = df.apply(impairment, axis=1)

    # support_level: from Q7 (feels supported) + Q14 (support person available)
    def support(row):
        if pd.isna(row["q7_supported"]) and pd.isna(row["q14_support_person"]):
            return "unknown"
        supported = str(row.get("q7_supported", "")).lower()
        has_person = str(row.get("q14_support_person", "")).lower()
        if "yes" in has_person and ("somewhat" in supported or "yes" in supported):
            return "supported"
        if "not" in has_person or "no" in has_person:
            return "isolated"
        return "mixed"

    df["support_level"] = df.apply(support, axis=1)

    # help_readiness: directly reuses Q15 (already phrased as readiness).
    df["help_readiness"] = df["q15_open_to_support"].fillna("N/A")
    return df


def run_consistency_assertions(df: pd.DataFrame) -> None:
    total = len(df)
    assert total == 11500, f"expected 11500 rows, got {total}"

    na_count = (df["risk_level"] == "N/A-Feeling Fine").sum()
    assert na_count == 3222, f"expected 3222 N/A-Feeling Fine rows, got {na_count}"

    low_count = (df["risk_level"] == "Low").sum()
    assert low_count == 200, f"expected 200 true Low rows, got {low_count}"

    # Emergency <-> Q10/Q11 boundary check (see docs/data_notes.md Finding 3):
    # every Emergency-labeled row must have the highest-risk Q10 answer AND
    # the unsafe-right-now Q11 answer -- this is the passive-vs-active-intent
    # boundary that determines Severe vs Emergency and must never drift.
    emergency_rows = df[df["risk_level"] == "Emergency"]
    assert len(emergency_rows) == 1918, f"expected 1918 Emergency rows, got {len(emergency_rows)}"
    assert (emergency_rows["q10_active_si"] == "Yes, I may act on these thoughts").all(), (
        "found an Emergency row whose Q10 answer is not the highest-risk option"
    )
    assert (emergency_rows["q11_feels_safe"] == "No, I feel unsafe").all(), (
        "found an Emergency row whose Q11 answer is not 'No, I feel unsafe'"
    )

    # Severe rows must never carry the Emergency-defining Q10/Q11 combination
    # -- this is the critical "passive SI without intent = Severe, not
    # Emergency" boundary from the product spec.
    severe_rows = df[df["risk_level"] == "Severe"]
    severe_with_emergency_signal = severe_rows[
        (severe_rows["q10_active_si"] == "Yes, I may act on these thoughts")
        & (severe_rows["q11_feels_safe"] == "No, I feel unsafe")
    ]
    assert len(severe_with_emergency_signal) == 0, (
        f"{len(severe_with_emergency_signal)} Severe rows carry the Emergency-defining "
        "Q10/Q11 combination -- Severe/Emergency boundary has drifted"
    )

    print("All consistency assertions passed.")


def build_label_maps(df: pd.DataFrame) -> dict:
    return {
        "risk_level": RISK_LEVELS,
        "suicide_risk": sorted(df["suicide_risk"].dropna().unique().tolist()),
        "functional_impairment": sorted(df["functional_impairment"].dropna().unique().tolist()),
        "support_level": sorted(df["support_level"].dropna().unique().tolist()),
        "help_readiness": sorted(df["help_readiness"].dropna().unique().tolist()),
        "q9_options_live": ["No", "Occasionally", "Often", "prefer not to answer"],
        "q10_options_live": ["No", "Yes but wouldn't act", "Yes and it feels risky right now"],
    }


def stratified_split(df: pd.DataFrame, eval_frac: float = 0.15, seed: int = 42):
    eval_parts = []
    train_parts = []
    for level in RISK_LEVELS:
        subset = df[df["risk_level"] == level].sample(frac=1.0, random_state=seed)
        n_eval = max(1, int(round(len(subset) * eval_frac)))
        eval_parts.append(subset.iloc[:n_eval])
        train_parts.append(subset.iloc[n_eval:])
    eval_df = pd.concat(eval_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    train_df = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train_df, eval_df


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw()
    df = apply_label_fix(df)
    df = apply_answer_mappings(df)
    df = derive_sub_labels(df)

    run_consistency_assertions(df)

    label_maps = build_label_maps(df)
    (PROCESSED_DIR / "label_maps.json").write_text(json.dumps(label_maps, indent=2), encoding="utf-8")

    train_df, eval_df = stratified_split(df)
    train_df.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    eval_df.to_parquet(PROCESSED_DIR / "eval.parquet", index=False)

    print(f"Wrote {len(train_df)} train rows, {len(eval_df)} eval rows.")
    print("Risk level distribution (full dataset):")
    print(df["risk_level"].value_counts())


if __name__ == "__main__":
    sys.exit(main())
