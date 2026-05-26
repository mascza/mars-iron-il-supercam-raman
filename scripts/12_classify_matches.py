"""v3 weighted-fingerprint scoring + per-target replicate reproducibility.

Replaces the v1/v2 gated tier promotion. Reads `results/raw_scores.parquet`
(produced by 06; per (shot x catalog peak) SNR / curvature / position
data) and `data/mars_processed/mars_processed.parquet` (for per-target
mean spectra). Writes:

  results/candidates_ranked_v3.csv
  results/targets_ranked_v3.csv
  results/headline_targets_v3.json
  results/screen_summary_v3.json
  results/target_mean_spectra/<compound>_<target>.csv  (top 3 per compound)

The v1/v2 outputs (`screen_summary_post_audit.json`,
`tier2_candidates_post_audit.csv`, `headline_candidates.json`) are NOT
overwritten; they remain in place for traceability of the prior
gated approach.

v3 SCORING

Per-peak weight = chemical_specificity * confounder_penalty:

  chemical_specificity from class_label:
    A          1.0
    B          1.0
    C          0.5
    ARTIFACT   0.0  (excluded)
    cation A   0.7  (under compound EMIM-cation)

  confounder_penalty from position vs the v3 task's Mars-mineral list
  (hematite 290; goethite 385; olivine 823, 855; perchlorate 932-962;
  apatite 960-965):
    1.0  if 0 confounders within +/-10 cm-1
    0.5  if exactly 1
    0.25 if >=2

Resulting per-peak weights match the user spec EXCEPT for FeBr4 391 C:
goethite 385 sits 6.14 cm-1 from this peak (within +/-10), so the
strict rule gives weight 0.25, not the 0.5 listed in the spec table.
The spec said "do not tune" the weights, so the strict rule is applied
and the discrepancy is reported in docs/screen_rerun_summary.md v3.

Per-shot composite_score = sum across all anion-class catalog peaks
of the compound (A + B + C, ARTIFACT excluded by zero weight) PLUS
all EMIM-cation A peaks of:
    peak_weight * min(SNR / 3.0, 10.0)   if SNR >= 3 AND curvature == -1
                                          AND argmax within +/- tolerance
    0                                     otherwise
The SNR/3 cap at 10 (= SNR up to 30 sigma counted) prevents one
extremely strong peak from saturating the score.

PER-TARGET REPLICATE REPRODUCIBILITY

For each anion compound, take the top 20 highest-scoring shots; group
by target_name; for any target with >=2 of its shots in the top 20,
retrieve ALL shots SuperCam acquired at that target and compute:
  per A-class peak:
    n_shots_with_peak       (SNR >= 3 within +/-5, curv -1, argmax ok)
    replicate_fraction      (n_shots_with_peak / n_shots_at_target)
    mean_SNR_when_present   (mean SNR across shots where peak is present)
  composite_target_score    (weighted mean of replicate_fractions
                              across A-class peaks, weights = peak_weight)
  list of constituent shot composite_scores

HEADLINE TARGETS

For each compound: filter to targets where at least one A-class peak
has replicate_fraction >= 0.3, then rank by composite_target_score
and pick the top three (preferring sol-distance >= 30 from the
headline for the 2nd and 3rd targets when alternatives exist).

If fewer than 3 targets meet the >=0.3 replicate threshold for a
compound, fall back to the highest composite_target_score regardless
of the threshold; document the fallback in screen_summary_v3.json.

PER-TARGET MEAN SPECTRA

For each compound's top 3 targets, average the normalized Mars spectra
across all shots at the target (intensity_normalized_mean from
mars_processed.parquet on the standard 150-1700 cm-1 grid). Saved
under results/target_mean_spectra/.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

RAW_SCORES_PATH = REPO_ROOT / "results" / "raw_scores.parquet"
REF_SPECTRA_PATH = REPO_ROOT / "data" / "reference_library" / "reference_spectra.parquet"
MARS_PROCESSED_PATH = REPO_ROOT / "data" / "mars_processed" / "mars_processed.parquet"
CALCITE_PATH = REPO_ROOT / "results" / "calcite_control_v4.json"

RESULTS_DIR = REPO_ROOT / "results"
OUT_CANDIDATES = RESULTS_DIR / "candidates_ranked_v4.csv"
OUT_REPLICATE = RESULTS_DIR / "replicate_diagnostic_v4.csv"
OUT_HEADLINES = RESULTS_DIR / "headline_shots_v4.json"
OUT_SUMMARY = RESULTS_DIR / "screen_summary_v4.json"
TARGET_SPECTRA_DIR = RESULTS_DIR / "target_mean_spectra"
HEADLINE_SPECTRA_DIR = RESULTS_DIR / "headline_spectra_v4"

ANION_COMPOUND_ORDER = (
    "EMIM-FeCl4", "EMIM-FeBr4", "EMIM2-Fe2Cl7", "EMIM2-FeSO4",
)
CATION_COMPOUND = "EMIM-cation"
COMPOUND_SHORT = {
    "EMIM-FeCl4": "FeCl4",
    "EMIM-FeBr4": "FeBr4",
    "EMIM2-Fe2Cl7": "Fe2Cl7",
    "EMIM2-FeSO4": "FeSO4",
}

# Scoring constants.
SNR_THRESH = 3.0
SNR_CAP = 10.0  # SNR/3 capped at 10
POSITION_OFFSET_LIMIT = 1.0  # argmax must be within +/- tolerance

# SCCT exclusion.
SCCT_CLASS = "SCCT"
SCCT_PREFIX = "scct_"

# Top-N for per-target replicate analysis.
TOP_N_FOR_REPLICATE = 20
REPLICATE_FRACTION_THRESHOLD = 0.3
HEADLINE_TOP_N = 3
SOL_DISTANCE_PREF = 30  # preferred sol distance for 2nd/3rd target selection

# Mars-mineral confounders (v3 task spec).
CONF_POINTS = {"hematite": 290.0, "goethite": 385.0}
CONF_RANGES = {
    "olivine": [(823.0, 823.0), (855.0, 855.0)],
    "perchlorate": [(932.0, 962.0)],
    "apatite": [(960.0, 965.0)],
}
CONFOUNDER_WINDOW = 10.0

# Chemical specificity per class.
CHEM_SPEC = {"A": 1.0, "B": 1.0, "C": 0.5, "ARTIFACT": 0.0}
# v4: cation specificity reduced from v3's 0.7 -> 0.4 to prevent the
# 22-cation pool from dominating per-shot composite_score over the
# 1-3 anion peaks per compound. All other weights unchanged.
CATION_CHEM_SPEC = 0.4
SOL_DISTANCE_HEADLINE = 30  # required sol distance between consecutive headlines

# Confounder strings keyed by (compound, class) and (compound, position).
def confounder_string_for(compound: str, peak_class: str, position: float) -> str | None:
    if compound == "EMIM2-FeSO4" and peak_class == "A":
        return ("FeSO4 nu1 at 958 cm-1 overlaps perchlorate (932-962 cm-1) "
                "and apatite (960-965 cm-1) ranges, both present on the "
                "Martian surface")
    if compound == "EMIM-FeCl4" and peak_class == "B":
        return ("FeCl4 nu3 corroboration at 386 cm-1 overlaps goethite "
                "Raman mode at 385 cm-1")
    if compound == "EMIM-FeBr4" and peak_class == "B":
        return ("FeBr4 nu3 corroboration at 293 cm-1 overlaps hematite "
                "Raman mode at 290 cm-1")
    if compound == "EMIM-FeBr4" and peak_class == "C" and abs(position - 391.14) < 1.0:
        return ("FeBr4 391 cm-1 overtone overlaps goethite Raman mode at "
                "385 cm-1 (within +/-10 cm-1; weight reduced)")
    return None


def n_confounders_within(pos: float, window: float = CONFOUNDER_WINDOW) -> int:
    n = 0
    for _name, p in CONF_POINTS.items():
        if abs(p - pos) <= window:
            n += 1
    for _name, ranges in CONF_RANGES.items():
        for lo, hi in ranges:
            if not (hi < pos - window or lo > pos + window):
                n += 1
                break
    return n


def confounder_penalty(pos: float) -> float:
    n = n_confounders_within(pos)
    if n == 0:
        return 1.0
    if n == 1:
        return 0.5
    return 0.25


def is_scct(target_class, target_name) -> bool:
    if isinstance(target_class, str) and target_class == SCCT_CLASS:
        return True
    if isinstance(target_name, str) and target_name.lower().startswith(SCCT_PREFIX):
        return True
    return False


def safe_target_name(name: str) -> str:
    if name is None:
        return "unknown"
    s = "".join(c if c.isalnum() else "_" for c in str(name))
    return s.strip("_") or "unknown"


def passes_test(snr, curv, offset) -> bool:
    if not np.isfinite(snr):
        return False
    if float(snr) < SNR_THRESH:
        return False
    if int(curv) != -1:
        return False
    if float(offset) > POSITION_OFFSET_LIMIT:
        return False
    return True


def load_catalog_peak_weights(raw: pd.DataFrame) -> dict:
    """Return {(compound, position): {peak_class, peak_weight}} for every
    distinct catalog peak in raw_scores.
    """
    distinct = raw[
        ["compound", "peak_center_catalog", "peak_class"]
    ].drop_duplicates(subset=["compound", "peak_center_catalog"]).reset_index(drop=True)
    out = {}
    for _, r in distinct.iterrows():
        compound = r["compound"]
        pos = float(r["peak_center_catalog"])
        cls = r["peak_class"]
        if compound == CATION_COMPOUND:
            chem = CATION_CHEM_SPEC
            cp = 1.0  # cation peaks: confounder_penalty default 1.0 per spec
        else:
            chem = CHEM_SPEC.get(cls, 0.0)
            cp = confounder_penalty(pos)
        weight = chem * cp
        out[(compound, pos)] = {
            "peak_class": cls,
            "chem_specificity": chem,
            "confounder_penalty": cp,
            "peak_weight": weight,
        }
    return out


def main() -> int:
    if not RAW_SCORES_PATH.is_file():
        raise FileNotFoundError(RAW_SCORES_PATH)
    if not MARS_PROCESSED_PATH.is_file():
        raise FileNotFoundError(MARS_PROCESSED_PATH)
    RESULTS_DIR.mkdir(exist_ok=True)
    TARGET_SPECTRA_DIR.mkdir(exist_ok=True)

    t0 = time.perf_counter()
    raw = pd.read_parquet(RAW_SCORES_PATH)
    weights = load_catalog_peak_weights(raw)

    # Print per-peak weights for transparency.
    print("Per-peak weights (anion peaks):")
    for compound in ANION_COMPOUND_ORDER:
        for (cmp, pos), w in sorted(weights.items()):
            if cmp != compound:
                continue
            print(f"  {COMPOUND_SHORT[compound]:7s}  {pos:7.2f}  "
                  f"class={w['peak_class']}  chem={w['chem_specificity']:.2f}  "
                  f"conf={w['confounder_penalty']:.2f}  weight={w['peak_weight']:.3f}")

    # Build per-shot lookup of raw-score rows.
    keys = ["sol", "sclk_int", "seqid", "point_number"]
    shots = raw[keys + ["target_name", "target_classification"]].drop_duplicates(
        subset=keys
    ).reset_index(drop=True)

    # Per (shot, compound) candidate row construction.
    candidate_rows: list[dict] = []
    raw_indexed = raw.set_index(keys + ["compound"]).sort_index()

    for _, shot in shots.iterrows():
        target_class = (
            "" if pd.isna(shot["target_classification"])
            else str(shot["target_classification"])
        )
        target_name = (
            "" if pd.isna(shot["target_name"]) else str(shot["target_name"])
        )
        if is_scct(target_class, target_name):
            continue
        shot_key = (
            int(shot["sol"]), int(shot["sclk_int"]),
            str(shot["seqid"]), int(shot["point_number"]),
        )

        # Cation peaks at this shot (shared across all anion compounds).
        cation_idx = shot_key + (CATION_COMPOUND,)
        try:
            cation_grp = raw_indexed.loc[cation_idx]
            if isinstance(cation_grp, pd.Series):
                cation_grp = cation_grp.to_frame().T
        except KeyError:
            cation_grp = pd.DataFrame(columns=raw.columns)

        cation_matched = []
        cation_score_contrib = 0.0
        for _, r in cation_grp.iterrows():
            pos = float(r["peak_center_catalog"])
            w_info = weights.get((CATION_COMPOUND, pos), None)
            if w_info is None:
                continue
            if not passes_test(r["snr_detrended"], r["peak_curvature_sign"],
                               r["position_offset_normalized"]):
                continue
            snr = float(r["snr_detrended"])
            contrib = w_info["peak_weight"] * min(snr / 3.0, SNR_CAP)
            cation_score_contrib += contrib
            cation_matched.append({
                "position": pos,
                "class": "cation_A",
                "SNR": snr,
                "weight": w_info["peak_weight"],
                "contribution_to_score": contrib,
            })

        for compound in ANION_COMPOUND_ORDER:
            anion_idx = shot_key + (compound,)
            try:
                anion_grp = raw_indexed.loc[anion_idx]
                if isinstance(anion_grp, pd.Series):
                    anion_grp = anion_grp.to_frame().T
            except KeyError:
                continue

            anion_score = 0.0
            n_anion_matched = 0
            anion_matched_records = []
            confounder_flag_set: set[str] = set()
            for _, r in anion_grp.iterrows():
                pos = float(r["peak_center_catalog"])
                w_info = weights.get((compound, pos), None)
                if w_info is None or w_info["peak_weight"] == 0.0:
                    continue  # ARTIFACT class or unknown.
                if not passes_test(r["snr_detrended"], r["peak_curvature_sign"],
                                   r["position_offset_normalized"]):
                    continue
                snr = float(r["snr_detrended"])
                contrib = w_info["peak_weight"] * min(snr / 3.0, SNR_CAP)
                anion_score += contrib
                n_anion_matched += 1
                anion_matched_records.append({
                    "position": pos,
                    "class": w_info["peak_class"],
                    "SNR": snr,
                    "weight": w_info["peak_weight"],
                    "contribution_to_score": contrib,
                })
                cs = confounder_string_for(compound, w_info["peak_class"], pos)
                if cs:
                    confounder_flag_set.add(cs)

            composite_score = anion_score + cation_score_contrib

            # Anion confounders are reported even when score == 0 only if A
            # peak passes; if no anion peak passes, no candidate row.
            if n_anion_matched == 0 and composite_score == 0.0:
                continue
            matched_peaks = anion_matched_records + cation_matched

            row = {
                "sol": shot_key[0],
                "sclk_int": shot_key[1],
                "seqid": shot_key[2],
                "point_number": shot_key[3],
                "target_name": target_name,
                "target_classification": target_class,
                "compound": COMPOUND_SHORT[compound],
                "composite_score": composite_score,
                "anion_score_component": anion_score,
                "cation_score_component": cation_score_contrib,
                "n_anion_peaks_matched": n_anion_matched,
                "n_cation_peaks_matched": len(cation_matched),
                "matched_peaks": json.dumps(matched_peaks),
                "confounder_flags": json.dumps(sorted(confounder_flag_set)),
            }
            candidate_rows.append(row)

    cand_df = pd.DataFrame(candidate_rows)
    if cand_df.empty:
        print("WARNING: no candidates produced; exiting.")
        return 1

    # Add spectrum_path from mars_processed.
    mars_paths = pd.read_parquet(
        MARS_PROCESSED_PATH,
        columns=["sol", "sclk_int", "seqid", "point", "source_relpath"],
    ).drop_duplicates(subset=["sol", "sclk_int", "seqid", "point"])
    mars_paths = mars_paths.rename(columns={"point": "point_number"})
    cand_df = cand_df.merge(
        mars_paths, on=["sol", "sclk_int", "seqid", "point_number"], how="left"
    )
    cand_df["spectrum_path"] = cand_df["source_relpath"]
    cand_df = cand_df.drop(columns=["source_relpath"])

    cand_df = cand_df.sort_values(
        ["compound", "composite_score"], ascending=[True, False]
    ).reset_index(drop=True)
    cand_df.to_csv(OUT_CANDIDATES, index=False)

    # Per-compound: top 20 -> group by target -> replicate analysis.
    target_records: list[dict] = []
    headline_records: dict = {}

    # Anion peaks per compound (A-class only) for replicate analysis.
    anion_a_positions: dict[str, list[tuple[float, float]]] = {}
    for compound in ANION_COMPOUND_ORDER:
        a_peaks = []
        for (cmp, pos), w in weights.items():
            if cmp == compound and w["peak_class"] == "A":
                a_peaks.append((pos, w["peak_weight"]))
        anion_a_positions[compound] = sorted(a_peaks)

    # For each target's per-A-peak replicate analysis, query raw_scores.
    raw_by_compound_target = {
        compound: raw[raw["compound"] == compound]
        for compound in ANION_COMPOUND_ORDER
    }

    for compound in ANION_COMPOUND_ORDER:
        short = COMPOUND_SHORT[compound]
        sub = cand_df[cand_df["compound"] == short]
        if sub.empty:
            print(f"WARNING: no candidates for {short}")
            continue
        top_n = sub.head(TOP_N_FOR_REPLICATE)
        target_counts = top_n.groupby("target_name").size()
        eligible_targets = target_counts[target_counts >= 2].index.tolist()
        compound_target_records: list[dict] = []
        for target in eligible_targets:
            if not target:
                continue
            anion_at_target = raw_by_compound_target[compound][
                raw_by_compound_target[compound]["target_name"] == target
            ]
            if anion_at_target.empty:
                continue
            shots_at_target = anion_at_target[
                ["sol", "sclk_int", "seqid", "point_number"]
            ].drop_duplicates().reset_index(drop=True)
            n_shots_at_target = int(len(shots_at_target))
            per_peak: dict[str, dict] = {}
            weighted_num = 0.0
            weighted_den = 0.0
            for pos, w in anion_a_positions[compound]:
                # Per shot-at-target check whether this A peak passes.
                rows_for_peak = anion_at_target[
                    (anion_at_target["peak_center_catalog"] - pos).abs() < 0.05
                ]
                n_with = 0
                snr_vals = []
                for _, r in rows_for_peak.iterrows():
                    if passes_test(r["snr_detrended"], r["peak_curvature_sign"],
                                   r["position_offset_normalized"]):
                        n_with += 1
                        snr_vals.append(float(r["snr_detrended"]))
                frac = n_with / n_shots_at_target if n_shots_at_target else 0.0
                mean_snr = float(np.mean(snr_vals)) if snr_vals else float("nan")
                per_peak[f"{pos:.2f}"] = {
                    "replicate_fraction": float(frac),
                    "n_shots_with_peak": int(n_with),
                    "mean_SNR_when_present": mean_snr,
                }
                weighted_num += w * frac
                weighted_den += w
            composite_target_score = (
                weighted_num / weighted_den if weighted_den > 0 else 0.0
            )
            shot_scores = sub[sub["target_name"] == target]["composite_score"].tolist()
            sol_first = int(shots_at_target["sol"].iloc[0])
            compound_target_records.append({
                "compound": short,
                "target": target,
                "sol": sol_first,
                "n_shots_at_target": n_shots_at_target,
                "n_shots_in_top20": int(target_counts[target]),
                "per_a_peak": per_peak,
                "composite_target_score": float(composite_target_score),
                "max_replicate_fraction": float(max(
                    (v["replicate_fraction"] for v in per_peak.values()), default=0.0
                )),
                "shot_composite_scores": [float(s) for s in shot_scores],
                "shot_sols": sub[sub["target_name"] == target]["sol"].tolist(),
            })
        compound_target_records.sort(
            key=lambda r: r["composite_target_score"], reverse=True
        )
        target_records.extend(compound_target_records)

        # v4: per-shot headline selection (replaces per-target headline).
        # 1st = highest composite_score; 2nd = next-highest with sol diff
        # >=30 from 1st; 3rd = next-highest with sol diff >=30 from both.
        # Replicate analysis stays in compound_target_records but is now
        # auxiliary; not used to gate the headline.
        sub_sorted = sub.sort_values("composite_score", ascending=False).reset_index(drop=True)
        per_shot_picks = []
        for _, row in sub_sorted.iterrows():
            if not per_shot_picks:
                per_shot_picks.append(row)
                continue
            if all(abs(int(row["sol"]) - int(p["sol"])) >= SOL_DISTANCE_HEADLINE
                   for p in per_shot_picks):
                per_shot_picks.append(row)
            if len(per_shot_picks) >= 3:
                break
        # Lookup helper: replicate_fraction at primary A peak for the
        # shot's target, if that target appears in compound_target_records.
        def replicate_lookup(target_name: str) -> dict:
            for r in compound_target_records:
                if r["target"] == target_name:
                    primary_pos = anion_a_positions[compound][0][0]
                    pp = r["per_a_peak"].get(f"{primary_pos:.2f}", {})
                    return {
                        "target_in_replicate_pool": True,
                        "n_shots_at_target": r["n_shots_at_target"],
                        "primary_peak_position": primary_pos,
                        "replicate_fraction_at_primary": pp.get(
                            "replicate_fraction", 0.0
                        ),
                        "n_shots_with_primary": pp.get("n_shots_with_peak", 0),
                        "mean_SNR_at_primary_when_present": pp.get(
                            "mean_SNR_when_present", float("nan")
                        ),
                    }
            return {
                "target_in_replicate_pool": False,
                "note": ("target has fewer than 2 shots in this compound's "
                         "top 20; replicate analysis not run for this "
                         "target."),
            }

        slot_records = {}
        for slot, pick in zip(("headline_shot", "second_shot", "third_shot"),
                              per_shot_picks):
            slot_records[slot] = {
                "sol": int(pick["sol"]),
                "sclk_int": int(pick["sclk_int"]),
                "seqid": str(pick["seqid"]),
                "point_number": int(pick["point_number"]),
                "target_name": str(pick["target_name"]),
                "target_classification": str(pick["target_classification"]),
                "spectrum_path": str(pick["spectrum_path"]),
                "compound": short,
                "composite_score": float(pick["composite_score"]),
                "anion_score_component": float(pick["anion_score_component"]),
                "cation_score_component": float(pick["cation_score_component"]),
                "n_anion_peaks_matched": int(pick["n_anion_peaks_matched"]),
                "n_cation_peaks_matched": int(pick["n_cation_peaks_matched"]),
                "matched_peaks": json.loads(pick["matched_peaks"]),
                "confounder_flags": json.loads(pick["confounder_flags"]),
                "replicate_at_target": replicate_lookup(str(pick["target_name"])),
            }
        headline_records[short] = slot_records

    # Write replicate diagnostic CSV (v4: auxiliary, not used for headline
    # selection).
    if target_records:
        rows_for_csv = []
        for r in target_records:
            row = {
                "compound": r["compound"],
                "target": r["target"],
                "sol": r["sol"],
                "n_shots_at_target": r["n_shots_at_target"],
                "n_shots_in_top20": r["n_shots_in_top20"],
                "composite_target_score": r["composite_target_score"],
                "max_replicate_fraction": r["max_replicate_fraction"],
                "per_a_peak": json.dumps(r["per_a_peak"]),
                "shot_composite_scores": json.dumps(r["shot_composite_scores"]),
                "shot_sols": json.dumps(r["shot_sols"]),
            }
            rows_for_csv.append(row)
        targets_df = pd.DataFrame(rows_for_csv).sort_values(
            ["compound", "composite_target_score"], ascending=[True, False]
        ).reset_index(drop=True)
        targets_df.to_csv(OUT_REPLICATE, index=False)

    # Write headline_shots_v4.json.
    OUT_HEADLINES.write_text(json.dumps(headline_records, indent=2) + "\n", encoding="utf-8")

    # Per-target mean spectra (kept for the auxiliary replicate
    # diagnostic; covers the top targets in the replicate pool, NOT the
    # per-shot headline targets).
    needed_targets: set[tuple[str, str]] = set()
    for r in target_records[:12]:  # top targets across all compounds
        needed_targets.add((r["compound"], r["target"]))
    if needed_targets:
        all_target_names = sorted({n for _, n in needed_targets})
        mars_cols = [
            "sol", "sclk_int", "seqid", "point", "target_name",
            "wavenumber_cm_inv", "intensity_normalized_mean",
        ]
        mars_df = pd.read_parquet(MARS_PROCESSED_PATH, columns=mars_cols)
        mars_df = mars_df[mars_df["target_name"].isin(all_target_names)]
        for short, target in needed_targets:
            sub = mars_df[mars_df["target_name"] == target]
            if sub.empty:
                continue
            mean_spec = (
                sub.groupby("wavenumber_cm_inv")["intensity_normalized_mean"]
                .mean()
                .reset_index()
                .sort_values("wavenumber_cm_inv")
            )
            out = TARGET_SPECTRA_DIR / f"{short}_{safe_target_name(target)}.csv"
            mean_spec.rename(columns={
                "wavenumber_cm_inv": "wavenumber_cm",
                "intensity_normalized_mean": "intensity_normalized",
            }).to_csv(out, index=False)

    # v4: export per-shot headline spectra (12 mars + 4 lab = 16 files).
    HEADLINE_SPECTRA_DIR.mkdir(exist_ok=True)
    headline_shots_to_export: list[tuple[str, dict]] = []
    for short, rec in headline_records.items():
        for slot in ("headline_shot", "second_shot", "third_shot"):
            s = rec.get(slot)
            if s:
                headline_shots_to_export.append((short, s))
    if headline_shots_to_export:
        mars_cols = [
            "sol", "sclk_int", "seqid", "point",
            "wavenumber_cm_inv", "intensity_normalized_mean",
        ]
        mars_df = pd.read_parquet(MARS_PROCESSED_PATH, columns=mars_cols)
        for short, s in headline_shots_to_export:
            sub = mars_df[
                (mars_df["sol"] == s["sol"])
                & (mars_df["sclk_int"] == s["sclk_int"])
                & (mars_df["seqid"] == s["seqid"])
                & (mars_df["point"] == s["point_number"])
            ].sort_values("wavenumber_cm_inv")
            if sub.empty:
                continue
            tname = safe_target_name(s["target_name"])
            out = HEADLINE_SPECTRA_DIR / f"mars_{short}_sol{s['sol']:04d}_{tname}.csv"
            sub[["wavenumber_cm_inv", "intensity_normalized_mean"]].rename(
                columns={
                    "wavenumber_cm_inv": "wavenumber_cm",
                    "intensity_normalized_mean": "intensity_normalized",
                }
            ).to_csv(out, index=False)
        # Lab convolved means per anion compound.
        ref = pd.read_parquet(REF_SPECTRA_PATH)
        ref = ref[ref["usable_for_supercam"] == True]
        for compound in ANION_COMPOUND_ORDER:
            short = COMPOUND_SHORT[compound]
            sub = ref[ref["compound"] == compound]
            agg = (
                sub.groupby("wavenumber_cm_inv")["intensity_normalized_convolved"]
                .mean().reset_index().sort_values("wavenumber_cm_inv")
            )
            out = HEADLINE_SPECTRA_DIR / f"lab_{short}_reference.csv"
            agg.rename(columns={
                "wavenumber_cm_inv": "wavenumber_cm",
                "intensity_normalized_convolved": "intensity_normalized",
            }).to_csv(out, index=False)

    # Build screen_summary_v4.json.
    md = shots.copy()
    n_sols = int(md["sol"].nunique())
    n_targets = int(md["target_name"].dropna().nunique())
    sol_min = int(md["sol"].min())
    sol_max = int(md["sol"].max())
    corpus = int(len(md))

    def slot_payload(s: dict) -> dict:
        rep = s.get("replicate_at_target", {})
        return {
            "sol": s["sol"],
            "sclk_int": s["sclk_int"],
            "seqid": s["seqid"],
            "point_number": s["point_number"],
            "target": s["target_name"],
            "target_classification": s["target_classification"],
            "spectrum_path": s["spectrum_path"],
            "composite_score": s["composite_score"],
            "anion_score_component": s["anion_score_component"],
            "cation_score_component": s["cation_score_component"],
            "n_anion_peaks_matched": s["n_anion_peaks_matched"],
            "n_cation_peaks_matched": s["n_cation_peaks_matched"],
            "replicate_fraction_at_primary_peak_at_this_target":
                rep.get("replicate_fraction_at_primary",
                        "single-shot or <2 in top20"),
            "n_shots_at_target": rep.get("n_shots_at_target", 1),
        }

    per_compound_summary: dict[str, dict] = {}
    for compound in ANION_COMPOUND_ORDER:
        short = COMPOUND_SHORT[compound]
        sub = cand_df[cand_df["compound"] == short]
        n_above_zero = int((sub["composite_score"] > 0).sum())
        top_shot_score = float(sub["composite_score"].max()) if not sub.empty else 0.0
        rec = headline_records.get(short, {})
        per_compound_summary[short] = {
            "n_shots_with_score_above_0": n_above_zero,
            "top_shot_composite_score": top_shot_score,
            "headline_shot": slot_payload(rec["headline_shot"]) if rec.get("headline_shot") else None,
            "second_shot":   slot_payload(rec["second_shot"])   if rec.get("second_shot")   else None,
            "third_shot":    slot_payload(rec["third_shot"])    if rec.get("third_shot")    else None,
        }

    calcite_block = {}
    if CALCITE_PATH.is_file():
        try:
            calcite_data = json.loads(CALCITE_PATH.read_text(encoding="utf-8"))
            calcite_block = {
                "top_shot_composite_score": calcite_data.get("top_shot_composite_score"),
                "top_target_composite_score": calcite_data.get("top_target_composite_score"),
                "median_score_distribution": calcite_data.get("median_score_distribution"),
                "p95_score_distribution": calcite_data.get("p95_score_distribution"),
                "n_eligible_targets_top20": calcite_data.get("n_eligible_targets_top20"),
            }
        except (OSError, json.JSONDecodeError):
            calcite_block = {"warning": f"could not parse {CALCITE_PATH.name}"}
    else:
        calcite_block = {"warning": (f"{CALCITE_PATH.name} not yet generated; "
                                     "run scripts/_control_calcite_screen.py first")}

    summary = {
        "pipeline_version": "v4_rebalanced_cation_0.4",
        "corpus_size": corpus,
        "n_sols": n_sols,
        "n_targets": n_targets,
        "sol_range": [sol_min, sol_max],
        "scoring_constants": {
            "SNR_threshold": SNR_THRESH,
            "SNR_cap": SNR_CAP,
            "position_offset_limit": POSITION_OFFSET_LIMIT,
            "confounder_window_cm": CONFOUNDER_WINDOW,
            "chemical_specificity": CHEM_SPEC,
            "cation_chemical_specificity": CATION_CHEM_SPEC,
            "TOP_N_FOR_REPLICATE": TOP_N_FOR_REPLICATE,
            "REPLICATE_FRACTION_THRESHOLD": REPLICATE_FRACTION_THRESHOLD,
        },
        "anion_peak_weights": {
            COMPOUND_SHORT[cmp] + " " + f"{pos:.2f}": {
                "class": w["peak_class"],
                "chem_specificity": w["chem_specificity"],
                "confounder_penalty": w["confounder_penalty"],
                "weight": w["peak_weight"],
            }
            for (cmp, pos), w in sorted(weights.items())
            if cmp in ANION_COMPOUND_ORDER
        },
        "per_compound": per_compound_summary,
        "calcite_control": calcite_block,
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    elapsed = time.perf_counter() - t0
    print(f"\nWall clock: {elapsed:.2f} s")
    print()
    print("Per-compound v4 results (per-shot headlines):")
    for short, rec in per_compound_summary.items():
        h = rec.get("headline_shot") or {}
        s2 = rec.get("second_shot") or {}
        s3 = rec.get("third_shot") or {}
        print(f"  {short:7s}  top_shot={rec['top_shot_composite_score']:6.2f}")
        for tag, slot in (("1st", h), ("2nd", s2), ("3rd", s3)):
            if not slot:
                print(f"    {tag}: <none>")
                continue
            print(f"    {tag}: sol{slot.get('sol')} {slot.get('target')}  "
                  f"score={slot.get('composite_score'):.2f}  "
                  f"n_anion={slot.get('n_anion_peaks_matched')}  "
                  f"n_cation={slot.get('n_cation_peaks_matched')}")
    print()
    print(f"Wrote {OUT_CANDIDATES.name}: {len(cand_df)} rows")
    if target_records:
        print(f"Wrote {OUT_REPLICATE.name}: {len(target_records)} rows (auxiliary)")
    print(f"Wrote {OUT_HEADLINES.name}")
    print(f"Wrote {OUT_SUMMARY.name}")
    if headline_shots_to_export:
        print(f"Wrote {HEADLINE_SPECTRA_DIR.name}/ with "
              f"{len(headline_shots_to_export)} mars + 4 lab CSVs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
