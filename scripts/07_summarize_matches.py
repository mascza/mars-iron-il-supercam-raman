"""Summarize 06 outputs into per-(observation x compound) match records.

Reads the three output parquets from 06 (raw_scores, cosine_diagnostic,
bump_indicator) plus the catalog (peak_catalog), aggregates per
(observation x compound), computes cosine percentiles in two scopes
(full corpus and science-only), annotates mineral-confound overlaps,
counts quality flags above 3 sigma, applies the v4 tier rules
(strong_candidate / candidate / no_evidence), and writes
results/matches.parquet (5,744 rows).

Tier names changed from the v4 handoff's "strong indicator" / "weak
indicator" / "no evidence" to "strong_candidate" / "candidate" /
"no_evidence" for reviewer-defensible language. Cosine percentile is
exposed as two columns: full-corpus (used by tier logic) and
science-only (excluding target_name == 'scct_diamond' for reviewer
defense). The new column tier_eligibility_explanation reports the first
unmet criterion when tier != strong_candidate.

Tier is determined by SNR and cosine alone. Confound overlaps and
quality-flag counts are annotation; they do not gate tier. ARTIFACT-class
peaks in raw_scores are ignored by the count and max filters because
peak_class is neither A nor B; no explicit ARTIFACT filter is needed.

Note on evidence_pattern priority: when n_a_primary_above_3sigma == 0
but n_a_secondary_above_3sigma >= 1, the row's tier is no_evidence and
the evidence_pattern rules priority places the row in 'no_evidence' or
'cation_only' depending on n_b_above_3sigma, regardless of the secondary
A signal. The implicit anion-via-secondary-only case is captured via
the n_a_secondary_above_3sigma count column. This priority interaction
matches the v4 handoff Q6 wording and is intentional.

EMIM2-FeSO4 cannot reach strong_candidate under v4 because the catalog
has zero EMIM2-FeSO4 A-secondary peaks; the explanation column reports
"compound has zero A-secondaries in catalog" for FeSO4 candidate rows
that pass the A-primary 5-sigma gate.

References:
  See docs/figures/d_design/D_handoff_v4.md Q5 (tier thresholds), Q6
  (matches.parquet schema), Q7 (mineral confound annotation).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _processing import (  # noqa: E402
    MARS_MINERAL_CONFOUNDS,
    check_confound_overlap,
)

# Inputs from 06 plus the catalog.
RAW_SCORES_PATH = REPO_ROOT / "results" / "raw_scores.parquet"
COSINE_PATH = REPO_ROOT / "results" / "cosine_diagnostic.parquet"
BUMP_PATH = REPO_ROOT / "results" / "bump_indicator.parquet"
CATALOG_PATH = (
    REPO_ROOT / "data" / "reference_library" / "peak_catalog.parquet"
)

# Outputs (gitignored via results/*.parquet and results/.provenance.json).
RESULTS_DIR = REPO_ROOT / "results"
MATCHES_PATH = RESULTS_DIR / "matches.parquet"
PROVENANCE_PATH = RESULTS_DIR / ".provenance.json"

COMPOUND_ORDER = (
    "EMIM-FeCl4",
    "EMIM-FeBr4",
    "EMIM2-Fe2Cl7",
    "EMIM2-FeSO4",
    "EMIM-cation",
)

# Tier thresholds (Q5).
THRESH_PRIMARY = 3.0
THRESH_STRONG = 5.0
THRESH_SECONDARY = 3.0
COSINE_TOP_PERCENTILE = 90.0

# Cation-corroboration threshold for evidence_pattern (Q6).
THRESH_CATION_CORROB = 2

# Confound matching tolerance (Q7); fixed at +/-5 cm-1 regardless of
# per-peak doublet-partition tightening upstream.
CONFOUND_TOL_CM = 5.0

# SCCT identification: target_name is the canonical filter (29 unique
# observations across 23 sols). target_classification == 'SCCT' covers a
# wider set of calibration targets and is not used here.
SCCT_DIAMOND_TARGET_NAME = "scct_diamond"


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def load_catalog_a_peaks() -> pd.DataFrame:
    """Load aggregate-convolved catalog rows, parse peak_subclass from
    class_notes prefix, return only Class A peaks (primary + secondary).

    Used both to enumerate diagnostic peaks per compound for the confound
    check and to derive COMPOUNDS_WITH_NO_A_SECONDARY for the explanation
    logic.
    """
    df = pd.read_parquet(CATALOG_PATH)
    df = df[
        (df["lab_or_convolved"] == "convolved") & (df["is_aggregate"] == True)
    ].copy()

    def parse_subclass(label: object, notes: object) -> str:
        if label != "A":
            return ""
        s = "" if notes is None else str(notes)
        if s.startswith("primary;"):
            return "primary"
        if s.startswith("secondary;"):
            return "secondary"
        return ""

    df["peak_subclass"] = [
        parse_subclass(lbl, n)
        for lbl, n in zip(df["class_label"], df["class_notes"])
    ]
    return df[df["class_label"] == "A"].reset_index(drop=True)


def compute_compounds_with_no_a_secondary(
    catalog_a: pd.DataFrame,
) -> list[str]:
    """List of compounds in COMPOUND_ORDER that have zero A-secondary peaks
    in the catalog. Used by the explanation logic when a candidate has
    A-primary >= 5 sigma but the catalog provides no secondary corroborator.
    """
    out = []
    for c in COMPOUND_ORDER:
        sub = catalog_a[
            (catalog_a["compound"] == c)
            & (catalog_a["peak_subclass"] == "secondary")
        ]
        if len(sub) == 0:
            out.append(c)
    return out


def aggregate_raw_scores(raw: pd.DataFrame) -> pd.DataFrame:
    """Per (sol, sclk_int, seqid, point_number, compound), compute the five
    count columns and three max columns from raw_scores. Counts use
    snr_detrended thresholds; max columns return NaN when no peaks of
    that class exist for the compound.
    """
    keys = ["sol", "sclk_int", "seqid", "point_number", "compound"]
    all_keys = (
        raw[keys].drop_duplicates().sort_values(keys).reset_index(drop=True)
    )
    idx = pd.MultiIndex.from_frame(all_keys)

    a_prim = (raw["peak_class"] == "A") & (
        raw["peak_subclass"] == "primary"
    )
    a_sec = (raw["peak_class"] == "A") & (
        raw["peak_subclass"] == "secondary"
    )
    b_mask = raw["peak_class"] == "B"

    def count_above(mask: pd.Series, threshold: float) -> pd.Series:
        sub = raw[mask & (raw["snr_detrended"] >= threshold)]
        if sub.empty:
            return pd.Series(0, index=idx, dtype="int64")
        s = sub.groupby(keys).size()
        return s.reindex(idx, fill_value=0).astype("int64")

    def max_within(mask: pd.Series) -> pd.Series:
        sub = raw[mask]
        if sub.empty:
            return pd.Series(np.nan, index=idx, dtype="float64")
        s = sub.groupby(keys)["snr_detrended"].max()
        return s.reindex(idx)

    out = all_keys.copy()
    out["n_a_primary_above_3sigma"] = count_above(a_prim, THRESH_PRIMARY).to_numpy()
    out["n_a_primary_above_5sigma"] = count_above(a_prim, THRESH_STRONG).to_numpy()
    out["n_a_secondary_above_3sigma"] = count_above(a_sec, THRESH_SECONDARY).to_numpy()
    out["n_a_secondary_above_5sigma"] = count_above(a_sec, THRESH_STRONG).to_numpy()
    out["n_b_above_3sigma"] = count_above(b_mask, THRESH_PRIMARY).to_numpy()
    out["max_a_primary_snr"] = max_within(a_prim).to_numpy()
    out["max_a_secondary_snr"] = max_within(a_sec).to_numpy()
    out["max_b_snr"] = max_within(b_mask).to_numpy()
    return out


def aggregate_quality_flags(
    raw: pd.DataFrame, all_keys_idx: pd.MultiIndex
) -> pd.Series:
    """Per (obs, compound), count of A peaks where snr_detrended >= 3 AND
    (position_offset_normalized > 0.8 OR peak_curvature_sign != -1).
    Quality flags on noise-level peaks (snr < 3) are not informative and
    are excluded.
    """
    a_above_3 = (raw["peak_class"] == "A") & (
        raw["snr_detrended"] >= THRESH_PRIMARY
    )
    flag = (raw["position_offset_normalized"] > 0.8) | (
        raw["peak_curvature_sign"] != -1
    )
    sub = raw[a_above_3 & flag]
    if sub.empty:
        return pd.Series(0, index=all_keys_idx, dtype="int64")
    keys = ["sol", "sclk_int", "seqid", "point_number", "compound"]
    s = sub.groupby(keys).size()
    return s.reindex(all_keys_idx, fill_value=0).astype("int64")


def compound_confound_annotations(
    catalog_a: pd.DataFrame,
) -> pd.DataFrame:
    """Per compound, walk that compound's A peaks (primary + secondary)
    and check each catalog center against MARS_MINERAL_CONFOUNDS within
    +/-CONFOUND_TOL_CM. Returns one row per compound with
    n_confound_overlapped_diagnostic_peaks and confound_labels.

    Confound info is a catalog property (peak positions vs mineral
    positions), not a Mars-spectrum property; the resulting per-compound
    values broadcast unchanged to every observation row of that compound.
    """
    rows = []
    for c in COMPOUND_ORDER:
        peaks = catalog_a[catalog_a["compound"] == c].sort_values(
            "position_cm_inv"
        )
        n_overlapped = 0
        labels: list[str] = []
        for _, p in peaks.iterrows():
            ctr = float(p["position_cm_inv"])
            matches = check_confound_overlap(
                ctr, CONFOUND_TOL_CM, MARS_MINERAL_CONFOUNDS
            )
            if matches:
                n_overlapped += 1
                labels.extend(f"{ctr:.2f}:{m}" for m in matches)
        rows.append(
            {
                "compound": c,
                "n_confound_overlapped_diagnostic_peaks": n_overlapped,
                "confound_labels": ", ".join(labels),
            }
        )
    return pd.DataFrame(rows)


def assign_tier_and_explanation(
    row: pd.Series, compounds_with_no_secondary: set[str]
) -> tuple[str, str]:
    """Return (tier, tier_eligibility_explanation) for one row.

    Tier: strong_candidate / candidate / no_evidence (Q5).
    Explanation: reports the first unmet criterion in the order
      [A-primary >= 5 sigma, A-secondary corroboration, cosine >= 90th].
    """
    n_prim_3 = int(row["n_a_primary_above_3sigma"])
    n_prim_5 = int(row["n_a_primary_above_5sigma"])
    n_sec_3 = int(row["n_a_secondary_above_3sigma"])
    cos_pct_full = row["cosine_percentile_within_compound_full_corpus"]
    max_prim = row["max_a_primary_snr"]
    compound = row["compound"]

    if (
        n_prim_5 >= 1
        and n_sec_3 >= 1
        and pd.notna(cos_pct_full)
        and float(cos_pct_full) >= COSINE_TOP_PERCENTILE
    ):
        return (
            "strong_candidate",
            "strong_candidate: passes all three criteria",
        )
    if n_prim_3 >= 1:
        if pd.isna(max_prim) or float(max_prim) < THRESH_STRONG:
            return (
                "candidate",
                f"candidate: A-primary {float(max_prim):.2f} below 5sigma "
                f"threshold",
            )
        if compound in compounds_with_no_secondary:
            return (
                "candidate",
                "candidate: compound has zero A-secondaries in catalog",
            )
        if n_sec_3 == 0:
            return (
                "candidate",
                "candidate: zero A-secondaries above 3sigma",
            )
        pct = float(cos_pct_full) if pd.notna(cos_pct_full) else float("nan")
        return (
            "candidate",
            f"candidate: cosine percentile {pct:.1f} below 90th percentile",
        )
    return "no_evidence", "no_evidence: no A-primary above 3sigma"


def assign_evidence_pattern(row: pd.Series) -> str:
    """Categorical evidence pattern (Q6). Apply rules in order; first
    match wins.
    """
    n_prim_3 = int(row["n_a_primary_above_3sigma"])
    n_sec_3 = int(row["n_a_secondary_above_3sigma"])
    n_b_3 = int(row["n_b_above_3sigma"])
    tier = row["tier"]
    if tier == "no_evidence":
        if n_b_3 < THRESH_CATION_CORROB:
            return "no_evidence"
        return "cation_only"
    if (n_prim_3 >= 1 or n_sec_3 >= 1) and n_b_3 == 0:
        return "anion_only"
    if (n_prim_3 >= 1 or n_sec_3 >= 1) and n_b_3 >= THRESH_CATION_CORROB:
        return "anion+cation"
    return "mixed"


def main() -> None:
    if not RAW_SCORES_PATH.is_file():
        raise FileNotFoundError(f"Missing input: {RAW_SCORES_PATH}")
    if not COSINE_PATH.is_file():
        raise FileNotFoundError(f"Missing input: {COSINE_PATH}")
    if not BUMP_PATH.is_file():
        raise FileNotFoundError(f"Missing input: {BUMP_PATH}")
    if not CATALOG_PATH.is_file():
        raise FileNotFoundError(f"Missing input: {CATALOG_PATH}")
    RESULTS_DIR.mkdir(exist_ok=True)

    t0 = time.perf_counter()
    raw = pd.read_parquet(RAW_SCORES_PATH)
    cosine = pd.read_parquet(COSINE_PATH)
    bump = pd.read_parquet(BUMP_PATH)
    catalog_a = load_catalog_a_peaks()

    n_input_raw = len(raw)
    n_input_cos = len(cosine)
    n_input_bump = len(bump)

    compounds_with_no_secondary_list = compute_compounds_with_no_a_secondary(
        catalog_a
    )
    compounds_with_no_secondary = set(compounds_with_no_secondary_list)

    # Per (obs, compound) aggregation from raw_scores.
    keys = ["sol", "sclk_int", "seqid", "point_number", "compound"]
    agg = aggregate_raw_scores(raw)
    all_keys_idx = pd.MultiIndex.from_frame(agg[keys])
    qf = aggregate_quality_flags(raw, all_keys_idx).rename("n_quality_flags")
    agg["n_quality_flags"] = qf.to_numpy()

    # Per-observation metadata broadcast onto the (obs, compound) frame.
    obs_keys = ["sol", "sclk_int", "seqid", "point_number"]
    obs_meta = (
        raw[obs_keys + ["target_name", "target_classification"]]
        .drop_duplicates(subset=obs_keys)
        .reset_index(drop=True)
    )
    agg = agg.merge(obs_meta, on=obs_keys, how="left")

    # Cosine percentile in two scopes.
    cosine_full = cosine.copy()
    cosine_full["cosine_percentile_within_compound_full_corpus"] = (
        cosine_full.groupby("compound")["cosine_diagnostic"].rank(pct=True)
        * 100.0
    )
    science_obs = obs_meta.loc[
        obs_meta["target_name"] != SCCT_DIAMOND_TARGET_NAME, obs_keys
    ]
    cosine_science = cosine.merge(
        science_obs, on=obs_keys, how="inner"
    )[keys + ["cosine_diagnostic"]].copy()
    cosine_science["cosine_percentile_within_compound_science_only"] = (
        cosine_science.groupby("compound")["cosine_diagnostic"].rank(
            pct=True
        )
        * 100.0
    )
    cosine_full = cosine_full.merge(
        cosine_science[
            keys + ["cosine_percentile_within_compound_science_only"]
        ],
        on=keys,
        how="left",
    )

    # Bump indicator (one value per observation, broadcast to 4 compounds).
    bump_join = bump[
        obs_keys + ["bump_region_structure_indicator"]
    ]

    # Confound annotations (catalog property; broadcast per compound).
    confound = compound_confound_annotations(catalog_a)

    # Compose the matches frame.
    out = agg.merge(
        cosine_full[
            keys
            + [
                "cosine_diagnostic",
                "cosine_percentile_within_compound_full_corpus",
                "cosine_percentile_within_compound_science_only",
            ]
        ],
        on=keys,
        how="left",
    )
    out = out.merge(bump_join, on=obs_keys, how="left")
    out = out.merge(confound, on="compound", how="left")
    out["confound_labels"] = out["confound_labels"].fillna("")
    out["n_confound_overlapped_diagnostic_peaks"] = (
        out["n_confound_overlapped_diagnostic_peaks"].fillna(0).astype("int64")
    )

    # Tier + explanation (sequential; small enough to stay row-wise).
    tiers: list[str] = []
    explanations: list[str] = []
    for _, row in out.iterrows():
        t, e = assign_tier_and_explanation(row, compounds_with_no_secondary)
        tiers.append(t)
        explanations.append(e)
    out["tier"] = tiers
    out["tier_eligibility_explanation"] = explanations
    out["evidence_pattern"] = [
        assign_evidence_pattern(row) for _, row in out.iterrows()
    ]

    # Final column order.
    column_order = [
        "sol",
        "sclk_int",
        "seqid",
        "point_number",
        "target_name",
        "target_classification",
        "compound",
        "n_a_primary_above_3sigma",
        "n_a_primary_above_5sigma",
        "n_a_secondary_above_3sigma",
        "n_a_secondary_above_5sigma",
        "n_b_above_3sigma",
        "max_a_primary_snr",
        "max_a_secondary_snr",
        "max_b_snr",
        "cosine_diagnostic",
        "cosine_percentile_within_compound_full_corpus",
        "cosine_percentile_within_compound_science_only",
        "bump_region_structure_indicator",
        "n_quality_flags",
        "n_confound_overlapped_diagnostic_peaks",
        "confound_labels",
        "evidence_pattern",
        "tier",
        "tier_eligibility_explanation",
    ]
    out = out[column_order]
    out = (
        out.sort_values(
            ["sol", "sclk_int", "seqid", "point_number", "compound"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    # Cast integer-valued columns to int64 for parquet stability.
    for col in ["sol", "sclk_int", "point_number"]:
        out[col] = out[col].astype("int64")
    for col in [
        "n_a_primary_above_3sigma",
        "n_a_primary_above_5sigma",
        "n_a_secondary_above_3sigma",
        "n_a_secondary_above_5sigma",
        "n_b_above_3sigma",
        "n_quality_flags",
        "n_confound_overlapped_diagnostic_peaks",
    ]:
        out[col] = out[col].astype("int64")

    pq.write_table(
        pa.Table.from_pandas(out, preserve_index=False),
        MATCHES_PATH,
        compression="zstd",
    )

    elapsed = time.perf_counter() - t0
    provenance = {
        "script_path": str(Path("scripts") / Path(__file__).name),
        "git_commit_hash": git_head(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_input_rows": {
            "raw_scores.parquet": int(n_input_raw),
            "cosine_diagnostic.parquet": int(n_input_cos),
            "bump_indicator.parquet": int(n_input_bump),
        },
        "n_output_rows": {
            "matches.parquet": int(len(out)),
        },
        "design_constants": {
            "THRESH_PRIMARY": THRESH_PRIMARY,
            "THRESH_STRONG": THRESH_STRONG,
            "THRESH_SECONDARY": THRESH_SECONDARY,
            "COSINE_TOP_PERCENTILE": COSINE_TOP_PERCENTILE,
            "THRESH_CATION_CORROB": THRESH_CATION_CORROB,
            "COMPOUND_ORDER": list(COMPOUND_ORDER),
            "COMPOUNDS_WITH_NO_A_SECONDARY": compounds_with_no_secondary_list,
            "MARS_MINERAL_CONFOUNDS_KEYS": sorted(
                MARS_MINERAL_CONFOUNDS.keys()
            ),
        },
    }
    PROVENANCE_PATH.write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Wall clock: {elapsed:.2f} s")
    print(f"Wrote {MATCHES_PATH.name}: {len(out)} rows")


if __name__ == "__main__":
    main()
