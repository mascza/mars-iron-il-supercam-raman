"""Shared spectrum processing primitives used by the lab and Mars pipelines.

The four-step recipe (cosmic ray removal, ALS baseline, trim, normalize) plus a
resample-to-uniform-grid step are factored here so 02_process_witec.py and
03_ingest_supercam.py share a single implementation.

References:
  Eilers, P.H.C., Boelens, H.F.M. (2005). Baseline correction with asymmetric
  least squares smoothing.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import spsolve

SUPERCAM_FWHM_CM = 12.0
SUPERCAM_SIGMA_CM = SUPERCAM_FWHM_CM / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def cosmic_ray_remove(y: np.ndarray, max_iter: int = 3) -> np.ndarray:
    """Replace cosmic ray spikes with the median of a 5-point neighborhood.

    A point is flagged when its discrete second-derivative residual exceeds
    8.0 * MAD of the second derivative. Boundary points cannot be flagged
    because the second derivative is undefined there. Iterates until no
    points are flagged or `max_iter` is reached.
    """
    y = y.astype(float).copy()
    n = y.size
    for _ in range(max_iter):
        d2 = y[2:] - 2.0 * y[1:-1] + y[:-2]
        med = np.median(d2)
        mad = np.median(np.abs(d2 - med))
        if mad == 0.0:
            break
        flagged_inner = np.abs(d2 - med) > 8.0 * mad
        flagged = np.zeros(n, dtype=bool)
        flagged[1:-1] = flagged_inner
        if not flagged.any():
            break
        for i in np.flatnonzero(flagged):
            lo = max(0, i - 2)
            hi = min(n, i + 3)
            y[i] = np.median(y[lo:hi])
    return y


def als_baseline(
    y: np.ndarray,
    lam: float = 1e5,
    p: float = 0.001,
    niter: int = 10,
) -> np.ndarray:
    """Asymmetric least squares baseline (Eilers and Boelens 2005)."""
    n = y.size
    D = diags([1.0, -2.0, 1.0], offsets=[0, 1, 2], shape=(n - 2, n)).tocsc()
    DtD = D.T @ D
    w = np.ones(n)
    z = np.zeros(n)
    for _ in range(niter):
        W = diags(w, 0, shape=(n, n)).tocsc()
        Z = csc_matrix(W + lam * DtD)
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1.0 - p) * (y < z)
    return z


def trim_to_range(
    x: np.ndarray,
    y: np.ndarray,
    x_min: float = 150.0,
    x_max: float = 1700.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (x, y) restricted to x_min <= x <= x_max."""
    mask = (x >= x_min) & (x <= x_max)
    return x[mask], y[mask]


def resample_to_grid(
    x: np.ndarray, y: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    """Linear interpolation onto `grid`. Returns NaN outside the input range.

    `np.interp` clamps to the endpoint values outside the input range; this
    wrapper instead writes NaN at those positions to make missing data
    explicit downstream.
    """
    if x.size == 0:
        return np.full(grid.shape, np.nan, dtype=float)
    out = np.interp(grid, x, y).astype(float)
    out[(grid < x[0]) | (grid > x[-1])] = np.nan
    return out


def normalize_to_max(y: np.ndarray) -> np.ndarray:
    """Divide by the NaN-ignoring max. Returns y unchanged if max is 0/NaN."""
    if y.size == 0:
        return y
    peak = np.nanmax(y)
    if not np.isfinite(peak) or peak <= 0.0:
        return y
    return y / peak


def gaussian_convolve(
    intensity: np.ndarray, sigma_px: float
) -> np.ndarray:
    """Gaussian convolve `intensity`; preserve original NaN positions.

    Operates on the contiguous finite block; pads the result back to full
    length with NaN at the original NaN positions. The output therefore has
    the same shape and the same NaN mask as the input.
    """
    finite = np.isfinite(intensity)
    if not finite.any():
        return intensity.astype(float).copy()
    idx = np.where(finite)[0]
    lo, hi = int(idx[0]), int(idx[-1] + 1)
    sub = intensity[lo:hi].astype(float)
    sub = np.where(np.isfinite(sub), sub, 0.0)
    convolved = gaussian_filter1d(sub, sigma=sigma_px, mode="nearest")
    out = np.full(intensity.shape, np.nan)
    out[lo:hi] = convolved
    out[~finite] = np.nan
    return out


def lorentzian(
    x: np.ndarray, x0: float, gamma: float, A: float, c: float
) -> np.ndarray:
    """Lorentzian profile with HWHM gamma, peak height A, baseline offset c.

    L(x) = A * gamma^2 / ((x - x0)^2 + gamma^2) + c

    FWHM = 2 * gamma. Analytical area above baseline = pi * A * gamma.
    """
    return A * gamma * gamma / ((x - x0) ** 2 + gamma * gamma) + c


# Numerical-noise floor for the curvature-sign branch in
# compute_peak_quality_flags. The three-point parabola coefficient `a` is
# rounded to sign 0 when |a| is at or below this threshold. Set to a pure
# numerical floor: downstream consumers (07's n_quality_flags) count peaks
# where peak_curvature_sign != -1, so signs 0 and +1 collapse to the same
# behaviour and the threshold value is not load-bearing on tier or
# evidence_pattern. A larger value would risk false-zero on marginal real
# peaks; raising it would be a future calibration choice.
CURVATURE_ZERO_THRESHOLD = 1e-12


def compute_local_polynomial_detrending(
    wavenumbers: np.ndarray,
    intensities: np.ndarray,
    center: float,
    tolerance: float,
    peer_peak_centers,
    *,
    half_window: float = 30.0,
    peer_tol: float = 5.0,
    poly_degree: int = 2,
) -> dict:
    """Local polynomial detrending around `center` with peer-peak fit exclusion.

    The polynomial is fit in the variable (lambda - center) on points inside
    [center - half_window, center + half_window], minus the fit-exclusion set:
    [center - tolerance, center + tolerance] (the "own" window) plus
    [pc - peer_tol, pc + peer_tol] for every pc in `peer_peak_centers` other
    than the target itself. The caller decides which peers are relevant: in
    06_match_mars.py the caller supplies cross-compound peers whose tolerance
    window overlaps the fit window.

    Returns a dict with these keys:
      peak_height          max of the detrended residual in [center +/- tolerance]
      peak_position        wavenumber at that maximum
      local_mad            MAD of the detrended residuals over fit-region points
      snr                  peak_height / local_mad (nan if local_mad == 0)
      poly_coeffs          (c0, c1, c2): polynomial in (lambda - center),
                           coefficients listed low-to-high power
      n_fit_points         number of points used in the polynomial fit
      window_lo, window_hi the +/- half_window bounds, as floats
      detrended_window     full detrended residual array over the in-window points
      wavenumbers_window   wavenumbers corresponding to detrended_window

    `detrended_window` and `wavenumbers_window` extend the handoff signature so
    the caller can forward them to compute_peak_quality_flags without
    recomputing the detrended values.

    Negative SNR is reported honestly when the catalog position sits in a
    trough relative to the local trend; downstream tier logic does not promote
    on negative SNR.
    """
    wn = np.asarray(wavenumbers, dtype=float)
    yv = np.asarray(intensities, dtype=float)
    W_lo = float(center) - float(half_window)
    W_hi = float(center) + float(half_window)
    in_W = (wn >= W_lo) & (wn <= W_hi)
    x = wn[in_W]
    y = yv[in_W]
    finite = np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size == 0:
        raise ValueError(
            f"compute_local_polynomial_detrending: no finite samples in window "
            f"around center={center}"
        )
    own_mask = (x >= float(center) - float(tolerance)) & (
        x <= float(center) + float(tolerance)
    )
    fit_mask = ~own_mask
    for pc in peer_peak_centers:
        pc = float(pc)
        if abs(pc - float(center)) < 1e-3:
            continue
        if pc + peer_tol < W_lo or pc - peer_tol > W_hi:
            continue
        fit_mask &= ~((x >= pc - peer_tol) & (x <= pc + peer_tol))
    if int(fit_mask.sum()) < poly_degree + 1:
        raise ValueError(
            f"compute_local_polynomial_detrending: insufficient fit points "
            f"around center={center}: {int(fit_mask.sum())} after exclusion"
        )
    if not own_mask.any():
        raise ValueError(
            f"compute_local_polynomial_detrending: no grid points in tolerance "
            f"window for center={center}"
        )
    xc = x - float(center)
    coeffs_desc = np.polyfit(xc[fit_mask], y[fit_mask], poly_degree)
    poly_at = np.polyval(coeffs_desc, xc)
    detrended = y - poly_at
    in_tol_d = detrended[own_mask]
    in_tol_x = x[own_mask]
    arg = int(np.argmax(in_tol_d))
    peak_height = float(in_tol_d[arg])
    peak_position = float(in_tol_x[arg])
    fit_residuals = detrended[fit_mask]
    local_mad = float(
        np.median(np.abs(fit_residuals - np.median(fit_residuals)))
    )
    snr = peak_height / local_mad if local_mad > 0.0 else float("nan")
    if poly_degree == 2:
        c2 = float(coeffs_desc[0])
        c1 = float(coeffs_desc[1])
        c0 = float(coeffs_desc[2])
        poly_coeffs = (c0, c1, c2)
    else:
        poly_coeffs = tuple(float(v) for v in coeffs_desc[::-1])
    return {
        "peak_height": peak_height,
        "peak_position": peak_position,
        "local_mad": local_mad,
        "snr": snr,
        "poly_coeffs": poly_coeffs,
        "n_fit_points": int(fit_mask.sum()),
        "window_lo": W_lo,
        "window_hi": W_hi,
        "detrended_window": detrended,
        "wavenumbers_window": x,
    }


def compute_peak_quality_flags(
    detrended_window: np.ndarray,
    wavenumbers: np.ndarray,
    peak_position: float,
    catalog_center: float,
    tolerance: float,
) -> dict:
    """Two peak-quality metrics derived from the detrended window.

    position_offset_normalized = abs(peak_position - catalog_center) / tolerance
      Range [0, 1] when peak_position lies within +/- tolerance of catalog_center.
      Edge values (>= 0.8) often indicate the metric is reading off a shoulder
      rather than a centered peak.

    peak_curvature_sign: int in {-1, 0, +1}
      Quadratic coefficient `a` of a parabola fit to three detrended points
      at peak_position and at peak_position +/- 1 cm-1. With grid spacing 1
      cm-1 this reduces to a = (y(p-1) - 2*y(p) + y(p+1)) / 2. np.interp
      handles the case where a query wavenumber does not fall exactly on a
      grid point.
        a < -CURVATURE_ZERO_THRESHOLD -> -1 (real local maximum)
        a > +CURVATURE_ZERO_THRESHOLD -> +1 (valley or rising/falling edge)
        |a| <= CURVATURE_ZERO_THRESHOLD -> 0 (effectively flat)

    See CURVATURE_ZERO_THRESHOLD docstring for the rationale on why the
    threshold value is not load-bearing.
    """
    offset = abs(float(peak_position) - float(catalog_center)) / float(tolerance)
    wn = np.asarray(wavenumbers, dtype=float)
    d = np.asarray(detrended_window, dtype=float)
    y_minus = float(np.interp(float(peak_position) - 1.0, wn, d))
    y_zero = float(np.interp(float(peak_position), wn, d))
    y_plus = float(np.interp(float(peak_position) + 1.0, wn, d))
    a = (y_minus - 2.0 * y_zero + y_plus) / 2.0
    if abs(a) <= CURVATURE_ZERO_THRESHOLD:
        sign = 0
    elif a < 0.0:
        sign = -1
    else:
        sign = 1
    return {
        "position_offset_normalized": float(offset),
        "peak_curvature_sign": int(sign),
    }


def compute_cosine_in_diagnostic_windows(
    mars_intensities: np.ndarray,
    ref_intensities: np.ndarray,
    wavenumbers: np.ndarray,
    peak_centers,
    *,
    peer_peak_centers=(),
    tolerance: float = 5.0,
    half_window: float = 15.0,
    peer_tol: float = 5.0,
    poly_degree: int = 2,
) -> float:
    """Cosine similarity over compound-specific diagnostic windows.

    Per window center c in `peak_centers`:
      W = [c - half_window, c + half_window].
      Fit-exclusion = [c - tolerance, c + tolerance] union the
      [pc - peer_tol, pc + peer_tol] intervals for every pc in
      peer_peak_centers, pc != c, that overlaps W.
      Detrend the Mars and reference intensities within W using a
      degree-`poly_degree` polynomial fit on the non-excluded points.
      The detrended residuals from each window are concatenated, in the
      order of `peak_centers`, into running Mars and reference vectors.

    Cosine is computed on the two concatenated vectors. Detrending is
    applied to both; the lab reference is on a flat post-ALS baseline so its
    polynomial is effectively zero, but the call is kept for symmetry with
    Mars.

    `peer_peak_centers` extends the v4 handoff signature. The handoff text
    did not name a peer-list parameter for cosine; the parameter is added
    here so cross-compound peer-fit-exclusion can be applied inside cosine
    windows for the same reason it is applied in
    compute_local_polynomial_detrending. Pass an empty iterable to disable.

    Returns nan when no window contributes any data, or when one of the
    concatenated vectors has zero norm.
    """
    wn = np.asarray(wavenumbers, dtype=float)
    m = np.asarray(mars_intensities, dtype=float)
    r = np.asarray(ref_intensities, dtype=float)
    peers = [float(p) for p in peer_peak_centers]
    mars_concat = []
    ref_concat = []
    for c in peak_centers:
        c = float(c)
        W_lo = c - float(half_window)
        W_hi = c + float(half_window)
        in_W = (wn >= W_lo) & (wn <= W_hi)
        x = wn[in_W]
        ym = m[in_W]
        yr = r[in_W]
        finite = np.isfinite(ym) & np.isfinite(yr)
        x = x[finite]
        ym = ym[finite]
        yr = yr[finite]
        if x.size == 0:
            continue
        own = (x >= c - float(tolerance)) & (x <= c + float(tolerance))
        fit_mask = ~own
        for pc in peers:
            if abs(pc - c) < 1e-3:
                continue
            if pc + peer_tol < W_lo or pc - peer_tol > W_hi:
                continue
            fit_mask &= ~((x >= pc - peer_tol) & (x <= pc + peer_tol))
        if int(fit_mask.sum()) < poly_degree + 1:
            continue
        xc = x - c
        cm = np.polyfit(xc[fit_mask], ym[fit_mask], poly_degree)
        cr = np.polyfit(xc[fit_mask], yr[fit_mask], poly_degree)
        mars_d = ym - np.polyval(cm, xc)
        ref_d = yr - np.polyval(cr, xc)
        mars_concat.append(mars_d)
        ref_concat.append(ref_d)
    if not mars_concat:
        return float("nan")
    mv = np.concatenate(mars_concat)
    rv = np.concatenate(ref_concat)
    nm = float(np.linalg.norm(mv))
    nr = float(np.linalg.norm(rv))
    if nm == 0.0 or nr == 0.0:
        return float("nan")
    return float(np.dot(mv, rv) / (nm * nr))


# Mars mineral confound reference data for Q7 confound annotation in 07.
#
# Citations:
#
#   Lafuente B, Downs RT, Yang H, Stone N (2015). The power of databases:
#   the RRUFF project. In: Highlights in Mineralogical Crystallography,
#   Armbruster T, Danisi RM (eds). De Gruyter, pp 1-30.
#   RRUFF entry R040024 (hematite): https://rruff.info/Hematite/R040024
#
#   de Faria DLA, Venancio Silva S, de Oliveira MT (1997). Raman
#   microspectroscopy of some iron oxides and oxyhydroxides.
#   J Raman Spectrosc 28:873-878.
#   Reports hematite Raman modes at 226, 245, 293, 298, 411, 497, 612 cm-1
#   plus 1322 cm-1 second-order two-magnon mode at low laser power.
#   Catalog values rounded to integers for +/-5 cm-1 overlap matching.
#
# Future expansion: NASA Ames Raman Spectroscopic Database (RAMdb,
# Mattioda et al. 2024 Icarus 208:115769) is the appropriate source for
# astrochemistry-relevant confounds (PAHs, amino acids, ice analogs) if
# the manuscript candidate set motivates adding them.
#
# The initial entry is hematite-only: that is the only confound for which
# we have empirical evidence in the 10-spectrum design inspection. Other
# minerals (apatite, jarosite, olivine, pyroxene, calcite) have known
# peaks in the diagnostic windows but should be added when specific
# candidate hits motivate them, not speculatively.
MARS_MINERAL_CONFOUNDS: dict[str, tuple[int, ...]] = {
    "hematite": (225, 245, 293, 411, 497, 612, 1322),
}


def check_confound_overlap(
    catalog_center: float,
    tolerance: float,
    mineral_dict: dict[str, tuple[int, ...]],
) -> list[str]:
    """Return labels for any mineral peak within +/- tolerance of
    catalog_center, formatted as "{mineral}_{mineral_peak}".

    `mineral_dict` is keyed by mineral name; values are iterables of
    integer peak positions in cm-1. Iteration order follows the dict's
    insertion order and the tuples' element order, both deterministic
    in Python 3.7+.

    Used by 07 to annotate which catalog A-peaks fall in mineral confound
    windows. Confound annotation is informational; tier promotion in 07
    is determined by SNR and cosine alone (handoff Q5).
    """
    matches: list[str] = []
    for mineral, peaks in mineral_dict.items():
        for p in peaks:
            if abs(int(p) - float(catalog_center)) <= float(tolerance):
                matches.append(f"{mineral}_{int(p)}")
    return matches
