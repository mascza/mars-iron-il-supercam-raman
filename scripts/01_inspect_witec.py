"""Inspect a WITec Raman acquisition folder.

Supports two WITec export layouts:

* split format: separate `*Export File (X-Axis).txt`, `(Y-Axis).txt`,
  `(Header).txt`, and `*Information.txt` files (one acquisition per folder)
* combined format: each acquisition is a single `*Spec.Data N.txt` file
  with header + data inline, paired with `*Information.txt`
  (multiple acquisitions per folder)

The format is auto-detected from the folder contents.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

UNITS_RE = re.compile(r"\s*\[[^\]]*\]\s*$")
SPECTRUM_NUM_RE = re.compile(r"Spectrum--(\d+)--")


@dataclass(frozen=True)
class Acquisition:
    """A single WITec acquisition with its parsed metadata and arrays."""

    folder: Path
    spectrum_number: str
    sample_name: str
    sample_inferred: bool
    x: np.ndarray
    y: np.ndarray
    metadata: dict[str, dict[str, str]]
    header: dict[str, str]


def read_axis(path: Path) -> np.ndarray:
    """Load a one-value-per-line float file into a 1-D ndarray."""
    return np.loadtxt(path)


def parse_header(path: Path) -> dict[str, str]:
    """Parse a WITec *Export File (Header).txt into a flat {key: value} dict.

    Format is 'Key = Value' per line. Comment lines starting with '//' and
    INI-style section markers like '[Header]' are ignored.
    """
    header: dict[str, str] = {}
    with path.open(encoding="latin-1") as f:
        for raw in f:
            stripped = raw.strip()
            if not stripped or stripped.startswith("//") or stripped.startswith("["):
                continue
            if "=" not in stripped:
                continue
            key, _, value = stripped.partition("=")
            header[key.strip()] = value.strip()
    return header


def parse_information(path: Path) -> dict[str, dict[str, str]]:
    """Parse a WITec *Information.txt file into {section: {key: value}}.

    Section headers are lines ending in ':' with nothing after.
    Key-value lines are 'Key:<sep>Value'; we split on the first ':'.
    Trailing unit annotations like ' [nm]' are stripped from keys.
    Orphan key-value pairs before the first section land in '_root'.
    """
    sections: dict[str, dict[str, str]] = {"_root": {}}
    current = "_root"

    with path.open(encoding="latin-1") as f:
        for raw in f:
            stripped = raw.strip()
            if not stripped or ":" not in stripped:
                continue

            key_part, _, value_part = stripped.partition(":")
            key = UNITS_RE.sub("", key_part.strip())
            value = value_part.strip()

            if not value:
                current = key
                sections.setdefault(current, {})
            else:
                sections[current][key] = value

    return sections


def lookup(metadata: dict[str, dict[str, str]], key: str) -> str | None:
    """Return the value for `key` from any section, or None."""
    for section in metadata.values():
        if key in section:
            return section[key]
    return None


def parse_spec_data(
    path: Path,
) -> tuple[dict[str, str], np.ndarray, np.ndarray]:
    """Parse a combined-format Spec.Data file into (header, x, y).

    Layout: header lines (Key = Value) terminated by a blank line,
    then a `[Data]` marker, then non-numeric label rows, then
    comma-separated `x, y` numeric rows.
    """
    with path.open(encoding="latin-1") as f:
        lines = f.read().splitlines()

    try:
        data_idx = next(
            i for i, ln in enumerate(lines) if ln.strip() == "[Data]"
        )
    except StopIteration as exc:
        raise ValueError(f"No [Data] marker in {path}") from exc

    header: dict[str, str] = {}
    for line in lines[:data_idx]:
        s = line.strip()
        if not s or s.startswith("//") or s.startswith("["):
            continue
        if "=" in s:
            key, _, value = s.partition("=")
            header[key.strip()] = value.strip()

    rows: list[tuple[float, float]] = []
    for line in lines[data_idx + 1 :]:
        s = line.strip()
        if not s:
            continue
        parts = s.split(",")
        try:
            x_val = float(parts[0])
            y_val = float(parts[1])
        except (ValueError, IndexError):
            # Label rows like "X-Axis,..." fall through here and are skipped.
            continue
        rows.append((x_val, y_val))

    if not rows:
        raise ValueError(f"No numeric x,y rows after [Data] in {path}")

    arr = np.asarray(rows)
    return header, arr[:, 0], arr[:, 1]


def find_spec_data_pairs(
    folder: Path,
) -> list[tuple[Path, Path, str]]:
    """Pair each `*Spec.Data*.txt` with its `*Spectrum--NNN--Information.txt`.

    Returns a list of (spec_data_path, info_path, spectrum_number) tuples,
    sorted by spectrum number. Raises if a partner Information file is
    missing or duplicated.
    """
    spec_files = sorted(folder.glob("*Spec.Data*.txt"))
    if not spec_files:
        raise FileNotFoundError(f"No *Spec.Data*.txt files in {folder}")

    pairs: list[tuple[Path, Path, str]] = []
    for spec_path in spec_files:
        m = SPECTRUM_NUM_RE.search(spec_path.name)
        if not m:
            raise RuntimeError(
                f"Cannot extract spectrum number from filename: {spec_path.name}"
            )
        num = m.group(1)
        info_candidates = sorted(
            folder.glob(f"*Spectrum--{num}--Information.txt")
        )
        if not info_candidates:
            raise FileNotFoundError(
                f"No matching Information.txt for spectrum {num} in {folder}"
            )
        if len(info_candidates) > 1:
            raise RuntimeError(
                f"Multiple Information files for spectrum {num}: "
                f"{[p.name for p in info_candidates]}"
            )
        pairs.append((spec_path, info_candidates[0], num))
    return pairs


def resolve_sample_name(
    metadata: dict[str, dict[str, str]], fallback_folder: Path
) -> tuple[str, bool]:
    """Return (sample_name, was_inferred).

    If `Sample Name` is missing or blank in metadata, infer it from
    `fallback_folder.name` with underscores converted to hyphens.
    """
    name = lookup(metadata, "Sample Name")
    if name:
        return name, False
    return fallback_folder.name.replace("_", "-"), True


def load_acquisition(folder: Path) -> list[Acquisition]:
    """Load every acquisition in `folder`, regardless of export format.

    Returns a list of `Acquisition` records: length 1 for split format,
    length N for combined format with N spectra. Each record has been
    validated against its header's SizeGraph and has its sample name
    resolved (with the underscore-to-hyphen folder-name fallback).
    """
    fmt = detect_format(folder)

    if fmt == "split":
        info_path = find_one(folder, "*Information.txt")
        header_path = find_one(folder, "*Export File (Header).txt")
        x_path = find_one(folder, "*Export File (X-Axis).txt")
        y_path = find_one(folder, "*Export File (Y-Axis).txt")

        metadata = parse_information(info_path)
        header = parse_header(header_path)
        x = read_axis(x_path)
        y = read_axis(y_path)

        validate_sizes(x, y, header)
        sample_name, inferred = resolve_sample_name(metadata, folder.parent)

        m = SPECTRUM_NUM_RE.search(info_path.name)
        if m:
            spec_num = m.group(1)
        else:
            print(
                f"WARNING: no 'Spectrum--NNN--' pattern in {info_path.name}; "
                "falling back to spectrum_number='01'"
            )
            spec_num = "01"

        return [
            Acquisition(
                folder=folder,
                spectrum_number=spec_num,
                sample_name=sample_name,
                sample_inferred=inferred,
                x=x,
                y=y,
                metadata=metadata,
                header=header,
            )
        ]

    pairs = find_spec_data_pairs(folder)
    out: list[Acquisition] = []
    for spec_path, info_path, num in pairs:
        metadata = parse_information(info_path)
        header, x, y = parse_spec_data(spec_path)
        validate_sizes(x, y, header)
        sample_name, inferred = resolve_sample_name(metadata, folder)
        out.append(
            Acquisition(
                folder=folder,
                spectrum_number=num,
                sample_name=sample_name,
                sample_inferred=inferred,
                x=x,
                y=y,
                metadata=metadata,
                header=header,
            )
        )
    return out


def detect_format(folder: Path) -> str:
    """Return 'split' or 'combined' based on which export files are present."""
    if list(folder.glob("*Export File (X-Axis).txt")):
        return "split"
    if list(folder.glob("*Spec.Data*.txt")):
        return "combined"
    raise FileNotFoundError(
        f"Cannot detect WITec export format in {folder}: "
        "expected '*Export File (X-Axis).txt' (split) or "
        "'*Spec.Data*.txt' (combined)"
    )


def find_one(folder: Path, pattern: str) -> Path:
    """Return the single file in `folder` matching `pattern`, else raise."""
    matches = sorted(folder.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern!r} in {folder}")
    if len(matches) > 1:
        raise RuntimeError(
            f"Expected one file matching {pattern!r} in {folder}, got {len(matches)}: "
            f"{[m.name for m in matches]}"
        )
    return matches[0]


def validate_sizes(
    x: np.ndarray, y: np.ndarray, header: dict[str, str]
) -> None:
    """Check that X and Y agree in length, and match the header's SizeGraph."""
    if x.shape != y.shape:
        raise ValueError(
            f"X and Y arrays have different shapes: {x.shape} vs {y.shape}"
        )
    size_graph_str = header.get("SizeGraph")
    if size_graph_str is None:
        raise ValueError("Header is missing required field 'SizeGraph'")
    try:
        size_graph = int(size_graph_str)
    except ValueError as exc:
        raise ValueError(
            f"Header SizeGraph is not an integer: {size_graph_str!r}"
        ) from exc
    if x.size != size_graph:
        raise ValueError(
            f"Array length mismatch: len(x)=len(y)={x.size} but "
            f"header SizeGraph={size_graph}"
        )


def print_summary(
    *,
    acquisition_label: str,
    sample_name: str,
    sample_inferred: bool,
    x: np.ndarray,
    y: np.ndarray,
    metadata: dict[str, dict[str, str]],
    header: dict[str, str],
) -> None:
    """Print a key-value summary of the acquisition to stdout."""
    sample_display = (
        f"{sample_name}  (inferred from folder)"
        if sample_inferred
        else sample_name
    )
    excitation = lookup(metadata, "Excitation Wavelength")
    laser_power = lookup(metadata, "Laser Power")
    integration = lookup(metadata, "Integration Time")
    accumulations = lookup(metadata, "Number Of Accumulations")
    sensor_temp = lookup(metadata, "Sensor Temperature")

    x_unit = header.get("XAxisUnit", "")
    data_unit = header.get("DataUnit", "")
    pos_unit = header.get("PositionUnit", "")
    pos_x = header.get("PositionX")
    pos_y = header.get("PositionY")
    pos_z = header.get("PositionZ")

    peak_idx = int(np.argmax(y))
    peak_x = float(x[peak_idx])
    peak_y = float(y[peak_idx])

    rows: list[tuple[str, str]] = [
        ("Acquisition", acquisition_label),
        ("Sample name", sample_display),
        ("Excitation wavelength (nm)", excitation or "n/a"),
        ("Laser power (mW)", laser_power or "n/a"),
        (
            "Integration x accumulations",
            f"{integration or 'n/a'} s x {accumulations or 'n/a'}",
        ),
        ("Sensor temperature (C)", sensor_temp or "n/a"),
        (
            f"Stage position ({pos_unit})" if pos_unit else "Stage position",
            f"X={pos_x}, Y={pos_y}, Z={pos_z}"
            if (pos_x is not None and pos_y is not None and pos_z is not None)
            else "n/a",
        ),
        (
            f"X range ({x_unit})" if x_unit else "X range",
            f"{x.min():.2f} -> {x.max():.2f}  (n={x.size})",
        ),
        (
            f"Y range ({data_unit})" if data_unit else "Y range",
            f"{y.min():.1f} -> {y.max():.1f}",
        ),
        (
            "Max-intensity peak",
            f"{peak_x:.2f} {x_unit}  ({peak_y:.1f} {data_unit})".strip(),
        ),
    ]

    label_w = max(len(k) for k, _ in rows)
    print()
    print("WITec acquisition summary")
    print("-" * (label_w + 40))
    for k, v in rows:
        print(f"{k:<{label_w}}  {v}")
    print()


def make_plot(
    *,
    sample_name: str,
    x: np.ndarray,
    y: np.ndarray,
    metadata: dict[str, dict[str, str]],
    header: dict[str, str],
    out_path: Path,
) -> None:
    """Render a single-spectrum plot and save it as PNG."""
    excitation = lookup(metadata, "Excitation Wavelength")
    power = lookup(metadata, "Laser Power")
    data_unit = header.get("DataUnit", "counts")

    title = sample_name
    if excitation and power:
        title = f"{sample_name}  -  {excitation} nm, {power} mW"

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(x, y)
    ax.set_xlabel("Raman shift (cm$^{-1}$)")
    ax.set_ylabel(f"Intensity ({data_unit})")
    ax.set_title(title)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a WITec acquisition folder."
    )
    parser.add_argument(
        "folder", type=Path, help="Path to a WITec acquisition folder"
    )
    args = parser.parse_args()

    folder: Path = args.folder.resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    repo_root = Path(__file__).resolve().parent.parent
    figures_dir = repo_root / "results" / "figures"

    acquisitions = load_acquisition(folder)
    multi = len(acquisitions) > 1

    for acq in acquisitions:
        if multi:
            label = f"{folder.name} / spectrum {acq.spectrum_number}"
            out_path = (
                figures_dir
                / f"inspect_{folder.name}_spectrum_{acq.spectrum_number}.png"
            )
        else:
            label = folder.name
            out_path = figures_dir / f"inspect_{folder.name}.png"

        print_summary(
            acquisition_label=label,
            sample_name=acq.sample_name,
            sample_inferred=acq.sample_inferred,
            x=acq.x,
            y=acq.y,
            metadata=acq.metadata,
            header=acq.header,
        )
        make_plot(
            sample_name=acq.sample_name,
            x=acq.x,
            y=acq.y,
            metadata=acq.metadata,
            header=acq.header,
            out_path=out_path,
        )
        print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    main()
