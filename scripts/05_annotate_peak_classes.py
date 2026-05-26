"""Annotate peak_catalog.parquet with the post-audit three-tier A/B/C
schema and move EMIM cation peaks under a separate compound identifier.

Reads `data/reference_library/peak_catalog.parquet`, applies the rules
below to aggregate-convolved rows, and writes the catalog back. Per-
spectrum and lab rows are not annotated and are passed through unchanged.

Three actions per rule (see RULES below):
  KEEP : assign class_label and class_notes; row remains under its
         current compound.
  MOVE : assign class_label and class_notes; row's compound column is
         rewritten to MOVE_COMPOUND.
  DROP : row is removed from the aggregate-convolved subset (it is not
         written back).

Class definitions (post-audit):
  A         primary anion diagnostic OR cation peak (under compound
            EMIM-cation). class_notes prefix begins with 'primary;'
            (preserved for the existing peak_subclass parser in 07).
  B         anion corroboration peak (chemically real but compromised
            by Mars-mineral overlap; useful only at the same shot as
            an A-class hit). 2 rows: FeCl4 386 (goethite overlap),
            FeBr4 293 (hematite overlap).
  C         observed lab feature with no chemical assignment as a
            fundamental mode (overtone, combination, compound-specific
            without a named mode). Recorded for completeness; the
            matcher writes raw intensity but no tier promotion uses
            these. 4 rows: FeBr4 391/442/490 + Fe2Cl7 460.
  ARTIFACT  trim-edge baseline residual at the 150 cm-1 boundary.

The 11 existing single-spectrum FeSO4 class-C rows (n=1/2 fragile
evidence) are dropped — they were retained pre-audit "for completeness,
not weighted heavily in matching"; the new schema has no slot for them
because they are neither anion-diagnostic nor cation-corroboration.

Coverage requirement: every aggregate-convolved row must match exactly
one rule. Unmatched rows fail the run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = REPO_ROOT / "data" / "reference_library" / "peak_catalog.parquet"
README_PATH = REPO_ROOT / "data" / "reference_library" / "README.md"

ANION_COMPOUND_ORDER = ("EMIM-FeCl4", "EMIM-FeBr4", "EMIM2-Fe2Cl7", "EMIM2-FeSO4")
CATION_COMPOUND = "EMIM-cation"
COMPOUND_ORDER = ANION_COMPOUND_ORDER + (CATION_COMPOUND,)

# Half-width of the position-match window applied around each rule center.
TOLERANCE_CM = 5.0

# Per-rule tolerance overrides (the FeSO4 doublet partition rules at 1022/1027
# and 1086/1091, which sit ~5.2-5.4 cm-1 apart and would otherwise share an
# overlap zone). After the audit the FeSO4 doublet rows are dropped, so the
# overrides are effectively unused; kept here so the entry shape is stable
# if the catalog is rebuilt with those rows.
TOLERANCE_OVERRIDES: dict[tuple[str, float], float] = {
    ("EMIM2-FeSO4", 1022.01): 2.5,
    ("EMIM2-FeSO4", 1027.38): 2.5,
    ("EMIM2-FeSO4", 1086.60): 2.5,
    ("EMIM2-FeSO4", 1091.81): 2.5,
}

# Action tags.
KEEP = "KEEP"
MOVE = "MOVE"
DROP = "DROP"

# Rules: (compound, center_cm_inv, action, class_label_or_blank, notes).
# A row matches a rule when its (current) compound equals rule.compound and
# its position falls inside [center - tol, center + tol]. When multiple rules
# match a row, the first listed wins.
RULES: list[tuple[str, float, str, str, str]] = [
    # --- ARTIFACT: trim-edge baseline residuals at 150 cm-1 boundary ---
    ("EMIM2-Fe2Cl7", 146.83, KEEP, "ARTIFACT",
     "trim-edge artifact (ALS baseline residual)"),
    ("EMIM2-FeSO4",  148.56, KEEP, "ARTIFACT",
     "trim-edge artifact (ALS baseline residual)"),

    # --- Anion class A (primary diagnostic) ---
    ("EMIM-FeCl4",   330.52, KEEP, "A",
     "primary; [FeCl4]- nu1 symmetric stretch; "
     "collides with Fe2Cl7 terminal mode"),
    ("EMIM-FeBr4",   218.40, KEEP, "A",
     "primary; [FeBr4]- nu1 symmetric stretch"),
    ("EMIM2-Fe2Cl7", 190.94, KEEP, "A",
     "primary; Fe-Cl-Fe bridging mode; definitive dinuclear marker; "
     "no major Mars-mineral confounder"),
    ("EMIM2-Fe2Cl7", 326.10, KEEP, "A",
     "primary; terminal Fe-Cl mode; collides with FeCl4 nu1"),
    ("EMIM2-FeSO4",  958.72, KEEP, "A",
     "primary; coordinated SO4(2-) nu1 symmetric stretch; "
     "downshifted from free-ion 983 cm-1 by Fe(III) chelation; "
     "overlaps perchlorate (932-962) and apatite (960-965)"),

    # --- Anion class B (corroboration; Mars-mineral overlap) ---
    ("EMIM-FeCl4",   386.31, KEEP, "B",
     "corroboration; [FeCl4]- nu3 asymmetric stretch; "
     "overlaps goethite 385 cm-1"),
    ("EMIM-FeBr4",   293.64, KEEP, "B",
     "corroboration; [FeBr4]- nu3 asymmetric stretch; "
     "overlaps hematite 290 cm-1"),

    # --- Anion class C (observed; not used in screen promotion) ---
    ("EMIM-FeBr4",   391.14, KEEP, "C",
     "observed; overtone/combination; not used in screening"),
    ("EMIM-FeBr4",   441.93, KEEP, "C",
     "observed; compound-specific; not used in screening"),
    ("EMIM-FeBr4",   489.96, KEEP, "C",
     "observed; compound-specific; not used in screening"),
    ("EMIM2-Fe2Cl7", 460.31, KEEP, "C",
     "observed; compound-specific; not used in screening"),

    # --- Existing class B (EMIM cation) rows: MOVE to compound=EMIM-cation
    # and label class A under that compound (per user spec: cation entries
    # are class A for cation purposes). class_notes prefixed with
    # "primary;" so the existing peak_subclass parser in 07 picks them
    # up; this is conventional, all cation-A rows are "primary" cation
    # peaks under the EMIM-cation compound.
    ("EMIM-FeCl4",    594.48, MOVE, "A",
     "primary; EMIM cation 593 family"),
    ("EMIM-FeCl4",   1023.58, MOVE, "A",
     "primary; EMIM ring breathing (~1020); pairs with Fe2Cl7 1020.69"),
    ("EMIM-FeCl4",   1088.77, MOVE, "A",
     "primary; EMIM cation; pairs with Fe2Cl7 1086.71"),
    ("EMIM-FeCl4",   1335.04, MOVE, "A", "primary; EMIM cation"),
    ("EMIM-FeCl4",   1417.90, MOVE, "A", "primary; EMIM cation"),
    ("EMIM-FeBr4",    596.38, MOVE, "A",
     "primary; EMIM cation 593 family"),
    ("EMIM-FeBr4",   1416.15, MOVE, "A", "primary; EMIM cation"),
    ("EMIM2-Fe2Cl7",  592.02, MOVE, "A", "primary; EMIM cation"),
    ("EMIM2-Fe2Cl7",  849.66, MOVE, "A",
     "primary; EMIM C-N stretch; pairs with FeSO4 856.43"),
    ("EMIM2-Fe2Cl7", 1020.69, MOVE, "A",
     "primary; EMIM ring breathing (~1020); pairs with FeCl4 1023.58"),
    ("EMIM2-Fe2Cl7", 1086.71, MOVE, "A",
     "primary; EMIM cation; pairs with FeCl4 1088.77"),
    ("EMIM2-Fe2Cl7", 1331.57, MOVE, "A", "primary; EMIM cation"),
    ("EMIM2-Fe2Cl7", 1415.74, MOVE, "A", "primary; EMIM cation"),
    ("EMIM2-Fe2Cl7", 1598.38, MOVE, "A",
     "primary; EMIM cation (likely ring mode)"),
    ("EMIM2-FeSO4",   593.27, MOVE, "A", "primary; EMIM cation"),
    ("EMIM2-FeSO4",   699.57, MOVE, "A", "primary; EMIM cation"),
    ("EMIM2-FeSO4",   856.43, MOVE, "A",
     "primary; EMIM C-N stretch; pairs with Fe2Cl7 849.66"),
    ("EMIM2-FeSO4",  1250.17, MOVE, "A", "primary; EMIM cation"),
    ("EMIM2-FeSO4",  1335.45, MOVE, "A", "primary; EMIM cation"),
    ("EMIM2-FeSO4",  1419.27, MOVE, "A", "primary; EMIM cation"),
    ("EMIM2-FeSO4",  1451.54, MOVE, "A", "primary; EMIM cation"),
    ("EMIM2-FeSO4",  1565.12, MOVE, "A", "primary; EMIM cation"),

    # --- Existing class C (single-spectrum FeSO4) rows: DROP. The new
    # schema has no slot for these "fragile evidence" peaks; they are
    # neither anion-diagnostic nor cation corroboration. Pre-audit
    # comment: "fragile evidence retained for completeness, not weighted
    # heavily in matching" -- removing now per the post-audit redesign.
    ("EMIM2-FeSO4",   238.07, DROP, "", ""),
    ("EMIM2-FeSO4",   418.42, DROP, "", ""),
    ("EMIM2-FeSO4",   429.09, DROP, "", ""),
    ("EMIM2-FeSO4",  1022.01, DROP, "", ""),
    ("EMIM2-FeSO4",  1027.38, DROP, "", ""),
    ("EMIM2-FeSO4",  1086.60, DROP, "", ""),
    ("EMIM2-FeSO4",  1091.81, DROP, "", ""),
    ("EMIM2-FeSO4",  1169.60, DROP, "", ""),
    ("EMIM2-FeSO4",  1187.22, DROP, "", ""),
    ("EMIM2-FeSO4",  1390.67, DROP, "", ""),
    ("EMIM2-FeSO4",  1667.13, DROP, "", ""),
]


CLASS_SECTION_MARKER = "## Class label system"

CLASS_SECTION_TEXT = """## Class label system (post-audit, three-tier)

Aggregate convolved rows in `peak_catalog.parquet` carry a `class_label` plus a `class_notes` string. Per-spectrum (raw) and lab rows are not annotated. Built by `scripts/05_annotate_peak_classes.py`.

The post-audit catalog uses a three-tier A/B/C scheme for anion peaks plus a separate compound identifier `EMIM-cation` for shared cation peaks. The audit (see `docs/figures/catalog_audit/catalog_integrity_audit.md`) confirmed that several pre-audit "secondary" peaks were chemically meaningless or sat on Mars-mineral confounders; this schema removes them from any tier-promotion role.

- `A` -- primary anion diagnostic (under an anion compound) OR cation peak (under `EMIM-cation`). Required for tier promotion. `class_notes` begins with `primary;` so the existing `peak_subclass` parser in 07 reads them as primaries.
- `B` -- anion corroboration peak. Chemically real fundamental mode but compromised by Mars-mineral overlap (FeCl4 386 / goethite 385; FeBr4 293 / hematite 290). Useful only at the same shot as an A-class hit; cannot promote on its own.
- `C` -- observed lab feature with no chemical assignment as a fundamental mode. Recorded for completeness; raw intensity is reported in candidate output for reviewer inspection but does not enter promotion logic.
- `ARTIFACT` -- trim-edge baseline residual at the 150 cm-1 boundary. 2 rows.

Catalog row counts (aggregate-convolved):

- 5 anion class A: FeCl4 330.52; FeBr4 218.40; Fe2Cl7 190.94 + 326.10; FeSO4 958.72.
- 2 anion class B: FeCl4 386.31; FeBr4 293.64.
- 4 anion class C: FeBr4 391.14, 441.93, 489.96; Fe2Cl7 460.31.
- 22 cation class A under compound `EMIM-cation` (moved from the pre-audit per-anion class B): EMIM cation peaks measured on the WITec instrument at the listed positions. Tier-2 cation corroboration tests for at least one of these above SNR 3sigma at the same Mars shot.
- 2 ARTIFACT: trim-edge residuals at 147/149.

Annotation is rule-based and hard-coded in `scripts/05_annotate_peak_classes.py`. A row matches a rule when its compound equals the rule compound and its `position_cm_inv` falls within +/- 5 cm-1 of the rule center. Three actions per rule: KEEP (assign class label, leave compound unchanged), MOVE (assign class label, rewrite compound column to `EMIM-cation`), DROP (remove from the aggregate-convolved subset).

Coverage requirement: every aggregate-convolved row matches exactly one rule. The script fails loudly and refuses to write the parquet if any row is unmatched.

Two collisions deserve specific notes:

- The 326-330 cm-1 region carries the FeCl4 nu1 symmetric stretch (330.52) and the Fe2Cl7 terminal Fe-Cl mode (326.10). Both are class A primary in their respective compounds. Co-detection of the 191 cm-1 Fe-Cl-Fe bridging mode (Fe2Cl7 only, class A) is required to call a target dinuclear; absence of 191 is consistent with mononuclear FeCl4. The new Fe2Cl7 Tier-2 rule encodes this requirement.
- The 1020 cm-1 line is `EMIM-cation` class A (EMIM ring breathing), present in all four lab spectra. The audit confirmed that the FeSO4 sulfate nu1 marker is the 958 cm-1 row (coordinated SO4(2-) downshifted from free-ion 983 by Fe(III) chelation); the previous FeSO4 single-spectrum entry near 1022 was an EMIM cation feature, not sulfate, and has been dropped.
"""


KNOWN_ISSUE_MARKER = "EMIM-FeBr4 lacks the 1335 cm-1"
KNOWN_ISSUE_TEXT = (
    "EMIM-FeBr4 lacks the 1335 cm-1 EMIM ring mode aggregate. The feature "
    "is physically present in the spectra (visible in "
    "`reference_spectra.parquet`) but its convolved peak height falls below "
    "the catalog's prominence floor and so does not enter the aggregate set."
)


def apply_rules(
    columns: dict[str, list],
) -> tuple[
    int,
    int,
    int,
    int,
    dict[str, dict[str, int]],
    list[str],
    list[tuple[str, float, int]],
    list[int],
]:
    """Mutate `columns` to assign class_label/class_notes (and rewrite
    compound for MOVE rows). Returns:
      n_kept         rows where action == KEEP
      n_moved        rows where action == MOVE
      n_dropped      rows where action == DROP
      n_modified     n_kept + n_moved (rows whose label/notes were touched)
      counts         per-compound class-label tallies (keyed by post-action
                     compound)
      multi_warnings strings logged when a row matched >1 rule
      unmatched      list of (compound, position, row_index) for unmatched
                     aggregate-convolved rows
      drop_indices   sorted list of row indices to remove from `columns`
                     (only aggregate-convolved rows can be dropped)
    """
    n_rows = len(columns["compound"])
    counts: dict[str, dict[str, int]] = {
        c: {"A": 0, "B": 0, "C": 0, "ARTIFACT": 0, "unclassified": 0}
        for c in COMPOUND_ORDER
    }
    multi_warnings: list[str] = []
    unmatched: list[tuple[str, float, int]] = []
    drop_indices: list[int] = []
    n_kept = n_moved = n_dropped = 0

    for i in range(n_rows):
        if not columns["is_aggregate"][i]:
            continue
        if columns["lab_or_convolved"][i] != "convolved":
            continue

        compound = columns["compound"][i]
        pos = float(columns["position_cm_inv"][i])

        matches = []
        for r in RULES:
            if r[0] != compound:
                continue
            tol = TOLERANCE_OVERRIDES.get((r[0], r[1]), TOLERANCE_CM)
            if (r[1] - tol) <= pos <= (r[1] + tol):
                matches.append((r, tol))
        if not matches:
            counts.setdefault(compound, {"A": 0, "B": 0, "C": 0,
                                        "ARTIFACT": 0, "unclassified": 0})
            counts[compound]["unclassified"] += 1
            unmatched.append((compound, pos, i))
            continue
        if len(matches) > 1:
            multi_warnings.append(
                f"{compound} pos={pos:.2f} matched {len(matches)} rules: "
                + ", ".join(
                    f"center={r[1]:.2f}+-{t} -> {r[2]}/{r[3]}"
                    for r, t in matches
                )
                + "; using first"
            )

        chosen, _chosen_tol = matches[0]
        action = chosen[2]
        label = chosen[3]
        notes = chosen[4]

        if action == DROP:
            drop_indices.append(i)
            n_dropped += 1
            continue

        columns["class_label"][i] = label
        columns["class_notes"][i] = notes
        if action == MOVE:
            columns["compound"][i] = CATION_COMPOUND
            n_moved += 1
        else:
            n_kept += 1
        post_compound = columns["compound"][i]
        counts.setdefault(post_compound, {"A": 0, "B": 0, "C": 0,
                                         "ARTIFACT": 0, "unclassified": 0})
        if label in ("A", "B", "C", "ARTIFACT"):
            counts[post_compound][label] += 1

    return (
        n_kept,
        n_moved,
        n_dropped,
        n_kept + n_moved,
        counts,
        multi_warnings,
        unmatched,
        sorted(drop_indices),
    )


def filter_columns(
    columns: dict[str, list], drop_indices: list[int]
) -> dict[str, list]:
    """Return a new columns dict with `drop_indices` rows removed."""
    if not drop_indices:
        return columns
    keep_mask = [True] * len(columns["compound"])
    for i in drop_indices:
        keep_mask[i] = False
    return {col: [v for v, keep in zip(vals, keep_mask) if keep]
            for col, vals in columns.items()}


def write_back(table: pa.Table, columns: dict[str, list]) -> None:
    """Write the catalog. On first run, insert class_notes right after
    class_label; on rerun keep existing field order so writes are
    idempotent.
    """
    old_fields = list(table.schema)
    field_names = [f.name for f in old_fields]
    if "class_notes" in field_names:
        new_fields = old_fields
    else:
        label_idx = field_names.index("class_label")
        new_fields = (
            old_fields[: label_idx + 1]
            + [pa.field("class_notes", pa.string())]
            + old_fields[label_idx + 1 :]
        )
    new_schema = pa.schema(new_fields)
    new_table = pa.Table.from_pydict(columns, schema=new_schema)
    pq.write_table(new_table, CATALOG_PATH, compression="zstd")


def update_readme() -> tuple[bool, bool]:
    """Replace the Class label system section with the post-audit text;
    keep the FeBr4 1335 known-issue note. Returns (replaced_class_section,
    added_known_issue).
    """
    text = README_PATH.read_text(encoding="utf-8")
    replaced_class = False
    added_issue = False

    # Replace existing class section (or append if absent).
    start = text.find(CLASS_SECTION_MARKER)
    if start == -1:
        provenance_idx = text.find("## Provenance")
        if provenance_idx == -1:
            text = text.rstrip() + "\n\n" + CLASS_SECTION_TEXT + "\n"
        else:
            text = (
                text[:provenance_idx]
                + CLASS_SECTION_TEXT
                + "\n"
                + text[provenance_idx:]
            )
        replaced_class = True
    else:
        # find the next top-level heading after the class section
        end = text.find("\n## ", start + 1)
        if end == -1:
            text = text[:start] + CLASS_SECTION_TEXT
        else:
            text = text[:start] + CLASS_SECTION_TEXT + text[end:]
        replaced_class = True

    if KNOWN_ISSUE_MARKER not in text:
        ki_idx = text.find("## Known issues")
        if ki_idx != -1:
            per_run_idx = text.find("Per-run warnings:", ki_idx)
            if per_run_idx != -1:
                text = (
                    text[:per_run_idx]
                    + KNOWN_ISSUE_TEXT
                    + "\n\n"
                    + text[per_run_idx:]
                )
            else:
                next_section = text.find("\n## ", ki_idx + 1)
                insert_pos = (
                    next_section if next_section != -1 else len(text)
                )
                text = (
                    text[:insert_pos]
                    + "\n"
                    + KNOWN_ISSUE_TEXT
                    + "\n"
                    + text[insert_pos:]
                )
            added_issue = True

    README_PATH.write_text(text, encoding="utf-8")
    return replaced_class, added_issue


def main() -> None:
    if not CATALOG_PATH.is_file():
        raise FileNotFoundError(f"Missing catalog: {CATALOG_PATH}")
    if not README_PATH.is_file():
        raise FileNotFoundError(f"Missing README: {README_PATH}")

    table = pq.read_table(CATALOG_PATH)
    columns: dict[str, list] = {
        col: table.column(col).to_pylist() for col in table.column_names
    }
    columns["class_label"] = [
        v if isinstance(v, str) else "" for v in columns["class_label"]
    ]
    if "class_notes" not in columns:
        columns["class_notes"] = [""] * len(columns["class_label"])
    else:
        columns["class_notes"] = [
            v if isinstance(v, str) else "" for v in columns["class_notes"]
        ]

    (
        n_kept,
        n_moved,
        n_dropped,
        n_modified,
        counts,
        multi_warnings,
        unmatched,
        drop_indices,
    ) = apply_rules(columns)

    for w in multi_warnings:
        print(f"WARNING: {w}")

    if unmatched:
        print(file=sys.stderr)
        print(
            f"FAIL: {len(unmatched)} aggregate-convolved row(s) matched "
            f"no rule:",
            file=sys.stderr,
        )
        for compound, pos, _idx in unmatched:
            print(f"  {compound:<14}  {pos:7.2f} cm-1", file=sys.stderr)
        print(
            "Refusing to write parquet/README. Add or widen rules in "
            "scripts/05_annotate_peak_classes.py.",
            file=sys.stderr,
        )
        sys.exit(1)

    columns = filter_columns(columns, drop_indices)
    write_back(table, columns)
    replaced_class, added_issue = update_readme()

    print()
    print("Annotation summary (post-audit three-tier schema):")
    print(f"  KEEP rows: {n_kept}")
    print(f"  MOVE rows (relabelled to compound={CATION_COMPOUND}): {n_moved}")
    print(f"  DROP rows (single-spectrum FeSO4, removed): {n_dropped}")
    print(f"  Total aggregate-convolved rows touched: {n_modified}")
    print(f"  Total aggregate-convolved rows after drop: "
          f"{sum(int(columns['is_aggregate'][i] and columns['lab_or_convolved'][i] == 'convolved') for i in range(len(columns['compound'])))}")

    print()
    print("Per compound (post-action):")
    for c in COMPOUND_ORDER:
        if c not in counts:
            continue
        d = counts[c]
        print(f"  {c:<14}  A={d['A']}  B={d['B']}  C={d['C']}  "
              f"ARTIFACT={d['ARTIFACT']}")

    print()
    if replaced_class:
        print("README: replaced 'Class label system' section.")
    if added_issue:
        print("README: appended FeBr4 1335 known-issue note.")
    else:
        print("README: FeBr4 1335 known-issue note already present.")


if __name__ == "__main__":
    main()
