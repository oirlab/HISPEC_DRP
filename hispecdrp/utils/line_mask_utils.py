"""
Build SPIROU-style ``.mas`` line masks from stellar template spectra.

Loads a template (BT-Settl HDF5 via :mod:`hispecdrp.utils.spectral_templates_utils`),
detects absorption minima, assigns segment widths from a linear law matched to the
bundled ``K2_all.mas`` geometry, applies depth / deblending / telluric filters, and
writes a three-column mask (vacuum nm: start, end, weight). Generation options are
serialized as JSON on a ``# HISPEC_DRP_MASK_META`` line for parsing
(:func:`read_mask_generation_meta`).

See ``hispecdrp/data/masks/README.md`` for mask format notes and
``hispecdrp/data/atmosphere/README.md`` for the transmission table used in telluric
filtering.
"""
from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.signal import argrelextrema

from hispecdrp.utils.spectral_templates_utils import load_template_from_path

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_WIDTH_CALIBRATION_K2_ALL",
    "LINE_MASK_META_PREFIX",
    "LineMaskEntry",
    "build_line_mask_from_template",
    "fit_half_width_linear_from_mas",
    "load_atmospheric_transmission_um",
    "mean_transmission_over_velocity_interval_um",
    "parse_mask_generation_meta_line",
    "read_mask_generation_meta",
]

# Prefix for a single-line JSON blob of generation parameters (after ``# ``).
LINE_MASK_META_PREFIX = "# HISPEC_DRP_MASK_META "

# Half-width (nm) vs line-center vacuum wavelength (nm): half_width ≈ slope * λ_nm + intercept.
# Coefficients from a least-squares fit to bundled ``K2_all.mas`` (2797 segments); intercept
# is numerically ~0 because mask segments are an almost fixed Δln λ (see data/masks/README.md).
DEFAULT_WIDTH_CALIBRATION_K2_ALL: tuple[float, float] = (
    7.033348083502036e-06,
    0.0,
)

_DEFAULT_ATMOSPHERE_REL = os.path.join(
    os.path.dirname(__file__), "..", "data", "atmosphere", "mktrans_zm_10_10.dat"
)


def parse_mask_generation_meta_line(line: str) -> dict | None:
    """
    If ``line`` starts with :data:`LINE_MASK_META_PREFIX`, return the parsed JSON object.

    The remainder of the line after the prefix must be a single JSON object.
    """
    s = line.strip()
    if not s.startswith(LINE_MASK_META_PREFIX):
        return None
    payload = s[len(LINE_MASK_META_PREFIX) :].strip()
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def read_mask_generation_meta(path: str) -> dict | None:
    """Scan the start of a ``.mas`` file for :func:`parse_mask_generation_meta_line`."""
    with open(path, encoding="utf-8") as f:
        for _ in range(200):
            line = f.readline()
            if not line:
                break
            meta = parse_mask_generation_meta_line(line)
            if meta is not None:
                return meta
    return None


@dataclass(frozen=True)
class LineMaskEntry:
    """One SPIROU-style mask segment after detection (vacuum nanometers)."""

    start_nm: float
    end_nm: float
    weight: float
    center_nm: float
    depth: float


def fit_half_width_linear_from_mas(mas_path: str) -> tuple[float, float]:
    """
    Fit ``half_width_nm = slope * center_nm + intercept`` from a SPIROU ``.mas`` file.

    Comment lines (``#``, ``;``, ``//``) are skipped; centers and half-widths are
    computed from the first two columns (start_nm, end_nm).
    """
    centers: list[float] = []
    halfw: list[float] = []
    with open(mas_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(("#", ";", "//")):
                continue
            parts = re.split(r"\s+", line)
            if len(parts) < 2:
                continue
            try:
                a = float(parts[0])
                b = float(parts[1])
            except ValueError:
                continue
            centers.append(0.5 * (a + b))
            halfw.append(0.5 * (b - a))
    if len(centers) < 2:
        raise ValueError(f"Too few numeric rows to fit widths: {mas_path!r}")
    x = np.asarray(centers, dtype=np.float64)
    y = np.asarray(halfw, dtype=np.float64)
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def load_atmospheric_transmission_um(
    path: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load Gemini ``mktrans`` two-column ASCII: wavelength (µm), transmission (0–1).

    Comment lines (``#``, ``;``, ``//``) are skipped. The bundled file under
    ``data/atmosphere/`` ships with a documented header; data rows are fixed-format
    decimals. External copies that omit a leading ``0`` on the first column
    (e.g. ``.900180``) are still accepted.
    """
    p = path if path is not None else os.path.normpath(_DEFAULT_ATMOSPHERE_REL)
    rows: list[tuple[float, float]] = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(("#", ";", "//")):
                continue
            if line[0] == ".":
                line = "0" + line
            parts = re.split(r"\s+", line)
            if len(parts) < 2:
                continue
            try:
                w = float(parts[0])
                t = float(parts[1])
            except ValueError:
                continue
            rows.append((w, t))
    if len(rows) < 8:
        raise ValueError(f"Too few transmission samples in {p!r}")
    w = np.asarray([r[0] for r in rows], dtype=np.float64)
    t = np.asarray([r[1] for r in rows], dtype=np.float64)
    order = np.argsort(w)
    return w[order], t[order]


def mean_transmission_over_velocity_interval_um(
    line_wavelength_um: float,
    trans_wavelength_um: np.ndarray,
    transmission: np.ndarray,
    half_width_km_s: float,
) -> float:
    """
    Mean atmospheric transmission over ``line_wavelength_um * (1 ± v/c)`` (vacuum, µm).

    ``half_width_km_s`` is a **half-width** in km/s; the interval is
    :math:`\\pm` that value (total span :math:`2 v / c` in fractional wavelength).
    Sampling uses a fixed number of points on the interval; values are linearly
    interpolated in the transmission table and clipped to the table bounds.
    """
    lam = float(line_wavelength_um)
    if not np.isfinite(lam) or lam <= 0:
        return float("nan")
    c_m_s = 299792458.0
    v_m_s = abs(float(half_width_km_s)) * 1000.0
    beta = min(v_m_s / c_m_s, 0.2)
    lo = lam * (1.0 - beta)
    hi = lam * (1.0 + beta)
    tw0 = float(trans_wavelength_um[0])
    tw1 = float(trans_wavelength_um[-1])
    lo = float(np.clip(lo, tw0, tw1))
    hi = float(np.clip(hi, tw0, tw1))
    if hi <= lo:
        return float(np.interp(lam, trans_wavelength_um, transmission))
    grid = np.linspace(lo, hi, num=32, dtype=np.float64)
    samp = np.interp(grid, trans_wavelength_um, transmission)
    return float(np.mean(samp))


def _continuum_envelope(flux: np.ndarray, size: int) -> np.ndarray:
    if size < 3:
        size = 3
    if size % 2 == 0:
        size += 1
    env = maximum_filter1d(flux, size=size, mode="nearest")
    return np.maximum(env, 1.0e-30)


def _local_minima_indices(flux: np.ndarray, order: int) -> np.ndarray:
    if order < 1:
        order = 1
    idx = argrelextrema(flux, np.less, order=order)[0]
    return np.asarray(idx, dtype=np.int64)


def _cluster_max_depth(
    centers_nm: np.ndarray,
    merge_radii_nm: np.ndarray,
    depths: np.ndarray,
) -> np.ndarray:
    """Union lines with ``|Δλ| < r_i + r_j``; keep deepest per component (bucketed union-find)."""
    n = int(centers_nm.shape[0])
    if n == 0:
        return np.array([], dtype=np.int64)
    parent = np.arange(n, dtype=np.int64)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    max_r = float(np.max(merge_radii_nm))
    bin_w = max(2.0 * max_r, 1.0e-9)
    bins: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        lam = float(centers_nm[i])
        b0 = int(np.floor(lam / bin_w))
        for bb in (b0 - 1, b0, b0 + 1):
            for j in bins[bb]:
                if abs(centers_nm[i] - centers_nm[j]) < merge_radii_nm[i] + merge_radii_nm[j]:
                    union(i, j)
        bins[b0].append(i)

    roots: dict[int, list[int]] = {}
    for i in range(n):
        r = find(i)
        roots.setdefault(r, []).append(i)
    keep: list[int] = []
    for members in roots.values():
        best = max(members, key=lambda k: float(depths[k]))
        keep.append(best)
    return np.sort(np.asarray(keep, dtype=np.int64))


def build_line_mask_from_template(
    template_path: str,
    *,
    wl_min_um: float = 0.8,
    wl_max_um: float = 3.0,
    width_slope_nm_per_nm: float | None = None,
    width_intercept_nm: float = 0.0,
    min_continuum_normalized_depth: float = 0.1,
    low_pass_width_multiplier: float = 5.0,
    telluric_velocity_half_width_km_s: float = 500.0,
    telluric_min_mean_transmission: float = 0.5,
    transmission_file: str | None = None,
    continuum_filter_size: int = 51,
    local_min_order: int = 2,
    output_path: str | None = None,
    masks_directory: str | None = None,
) -> tuple[str, list[LineMaskEntry]]:
    """
    Detect lines in a template spectrum and write a filtered ``.mas`` mask.

    Parameters
    ----------
    template_path
        Path to a supported template (BT-Settl ``.h5``).
    wl_min_um, wl_max_um
        Wavelength clip in **vacuum** microns before minima search.
    width_slope_nm_per_nm, width_intercept_nm
        Half-width (nm) = ``slope * center_vac_nm + intercept``. If ``slope`` is
        ``None``, uses :data:`DEFAULT_WIDTH_CALIBRATION_K2_ALL` (fit to ``K2_all.mas``).
    min_continuum_normalized_depth
        Keep lines with ``depth >=`` this value, where
        ``depth = 1 - flux/continuum`` and ``continuum`` is a local maximum envelope.
    low_pass_width_multiplier
        Deblending radius uses ``0.5 * multiplier * full_width_nm`` per line (overlap
        if separation < sum of radii); in each merged group the deepest line is kept.
    telluric_velocity_half_width_km_s
        Half-width in km/s for the telluric averaging interval (± this value).
    telluric_min_mean_transmission
        Lines with mean transmission below this value are removed.
    transmission_file
        Two-column ``mktrans``-style path (µm, transmission). Default: bundled
        ``data/atmosphere/mktrans_zm_10_10.dat``.
    continuum_filter_size
        Odd-ish pixel size for the maximum-filter continuum envelope.
    local_min_order
        ``order`` passed to :func:`scipy.signal.argrelextrema` for local minima.
    output_path
        Full path for the output ``.mas``. If ``None``, writes under
        ``masks_directory`` as ``<template_basename>.mas`` (same stem as the template,
        extension ``.mas``). A JSON metadata line (:data:`LINE_MASK_META_PREFIX`) records
        all generation parameters for reproducibility.
    masks_directory
        Default output directory (package ``data/masks`` if ``None``).

    Returns
    -------
    path_written, entries
        Output file path and in-memory segment list.
    """
    slope = (
        float(DEFAULT_WIDTH_CALIBRATION_K2_ALL[0])
        if width_slope_nm_per_nm is None
        else float(width_slope_nm_per_nm)
    )
    icept = float(width_intercept_nm)
    trans_path_resolved = os.path.normpath(
        transmission_file if transmission_file is not None else _DEFAULT_ATMOSPHERE_REL
    )
    trans_path_abs = os.path.abspath(trans_path_resolved)

    wave_um, flux = load_template_from_path(
        template_path, wl_min_um=wl_min_um, wl_max_um=wl_max_um
    )
    ok = np.isfinite(wave_um) & np.isfinite(flux) & (wave_um > 0)
    wave_um = wave_um[ok]
    flux = flux[ok]
    if wave_um.size < 16:
        raise ValueError("Template wavelength/flux too short after clipping")

    cont = _continuum_envelope(flux, continuum_filter_size)
    norm = flux / cont
    depth = 1.0 - norm

    idx = _local_minima_indices(flux, local_min_order)
    idx = idx[(idx > 0) & (idx < wave_um.size - 1)]

    centers_um = wave_um[idx].astype(np.float64)
    depths = depth[idx].astype(np.float64)
    centers_nm = centers_um * 1000.0

    half_width_nm = slope * centers_nm + icept
    half_width_nm = np.maximum(half_width_nm, 1.0e-6)

    # --- (1) depth threshold ---
    m = depths >= float(min_continuum_normalized_depth)
    centers_nm = centers_nm[m]
    depths = depths[m]
    half_width_nm = half_width_nm[m]

    if centers_nm.size == 0:
        raise ValueError("No lines left after depth threshold; relax min depth or continuum size")

    # --- (2) deblend / low-pass style merge ---
    full_width_nm = 2.0 * half_width_nm
    mult = float(low_pass_width_multiplier)
    merge_radii_nm = 0.5 * mult * full_width_nm
    order = np.argsort(centers_nm)
    centers_nm = centers_nm[order]
    depths = depths[order]
    half_width_nm = half_width_nm[order]
    merge_radii_nm = merge_radii_nm[order]

    keep = _cluster_max_depth(centers_nm, merge_radii_nm, depths)
    centers_nm = centers_nm[keep]
    depths = depths[keep]
    half_width_nm = half_width_nm[keep]

    # --- (3) telluric filter ---
    tw, tt = load_atmospheric_transmission_um(transmission_file)
    vhalf = float(telluric_velocity_half_width_km_s)
    tmin = float(telluric_min_mean_transmission)
    keep_t: list[int] = []
    for i in range(centers_nm.size):
        lam_um = float(centers_nm[i] * 1.0e-3)
        mean_t = mean_transmission_over_velocity_interval_um(lam_um, tw, tt, vhalf)
        if np.isfinite(mean_t) and mean_t >= tmin:
            keep_t.append(i)
    if not keep_t:
        raise ValueError(
            "No lines left after telluric filter; raise telluric_min_mean_transmission "
            "or widen/change the transmission file"
        )
    keep_t_arr = np.asarray(keep_t, dtype=np.int64)
    centers_nm = centers_nm[keep_t_arr]
    depths = depths[keep_t_arr]
    half_width_nm = half_width_nm[keep_t_arr]

    entries: list[LineMaskEntry] = []
    for i in range(centers_nm.size):
        c = float(centers_nm[i])
        hw = float(half_width_nm[i])
        d = float(depths[i])
        entries.append(
            LineMaskEntry(
                start_nm=c - hw,
                end_nm=c + hw,
                weight=d,
                center_nm=c,
                depth=d,
            )
        )

    # --- (4) save ---
    if masks_directory is None:
        masks_directory = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "data", "masks")
        )
    os.makedirs(masks_directory, exist_ok=True)

    base = os.path.splitext(os.path.basename(template_path))[0]
    auto_name = f"{base}.mas"
    out = output_path if output_path is not None else os.path.join(masks_directory, auto_name)

    meta: dict = {
        "schema_version": 1,
        "generator": "hispecdrp.utils.line_mask_utils.build_line_mask_from_template",
        "template_path": os.path.abspath(template_path),
        "wl_min_um": float(wl_min_um),
        "wl_max_um": float(wl_max_um),
        "width_slope_nm_per_nm_resolved": float(slope),
        "width_slope_nm_per_nm_user": None
        if width_slope_nm_per_nm is None
        else float(width_slope_nm_per_nm),
        "width_intercept_nm": float(icept),
        "min_continuum_normalized_depth": float(min_continuum_normalized_depth),
        "low_pass_width_multiplier": float(low_pass_width_multiplier),
        "telluric_velocity_half_width_km_s": float(vhalf),
        "telluric_min_mean_transmission": float(tmin),
        "transmission_file": trans_path_abs,
        "continuum_filter_size": int(continuum_filter_size),
        "local_min_order": int(local_min_order),
        "n_lines": int(len(entries)),
        "half_width_law_nm": "half_width_nm = width_slope_nm_per_nm_resolved * lambda_vac_nm + width_intercept_nm",
    }
    meta_json = json.dumps(meta, sort_keys=True, separators=(",", ":"))

    header = (
        "# HISPEC DRP template-derived SPIROU-style CCF line mask "
        "(columns: start_nm, end_nm, weight).\n"
        f"# Template basename: {os.path.basename(template_path)}\n"
        "# Wavelength columns: vacuum nanometers; weights: continuum-normalized depth.\n"
        "# Generation parameters: single-line JSON on the next line "
        "(parse with read_mask_generation_meta).\n"
        f"{LINE_MASK_META_PREFIX}{meta_json}\n"
    )
    with open(out, "w", encoding="utf-8") as f:
        f.write(header)
        for e in entries:
            f.write(f"  {e.start_nm: .16e}        {e.end_nm: .16e}       {e.weight: .16e}\n")

    logger.info("Wrote %d mask lines to %s", len(entries), out)
    return out, entries
