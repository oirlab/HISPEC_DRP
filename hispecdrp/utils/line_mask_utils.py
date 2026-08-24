"""Line-mask utilities (mask I/O, MaskLines container, template-built masks).

The canonical overview of mask formats and generation lives in the documentation:
``docs/source/algorithms/line_mask.rst``.
"""
from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from astropy import units as u
from scipy.ndimage import maximum_filter1d
from scipy.signal import argrelextrema

from hispecdrp.utils.physical_constants import C_M_S
from hispecdrp.utils.spectral_templates_utils import load_template_from_path

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_WIDTH_CALIBRATION_K2_ALL",
    "MASK_LINE_HALF_WIDTH_SLOPE_NM_PER_NM",
    "MASK_STYLE_DELTA",
    "MASK_STYLE_TOPHAT",
    "MASK_STYLE_KPF",
    "LINE_MASK_META_PREFIX",
    "MaskLineStyle",
    "MaskLines",
    "masklines_wavelength_unit_meta",
    "package_data_masks_dir",
    "OPTIMAL_TEMPLATE_MASK_BUILD_KW",
    "RELAXED_TEMPLATE_MASK_BUILD_KW",
    "CCFOPT_MASK_BUILD_KW",
    "CCFOPT_SINGLE_LINE_STELLAR_RV_KW",
    "IDEAL_SPECTRUM_STELLAR_RV_KW",
    "STELLAR_RV_WIDE_WINDOW_UM",
    "apply_ccfopt_defaults_for_line_by_line",
    "E2ERV_MASK_BUILD_KW",
    "LineMaskEntry",
    "build_line_mask_from_template",
    "build_ccfopt_line_mask_from_template_and_seed",
    "build_ccfopt_line_mask_from_rvccf_table",
    "build_ccfopt_line_mask_from_l2_line_by_line_rv",
    "build_ccfopt_line_mask_from_l2_single_line_ccf",
    "build_e2erv_line_mask_from_template",
    "compare_line_center_lists_um",
    "detect_template_line_centers_gradient_um",
    "detect_template_line_centers_parabolic_um",
    "fit_half_width_linear_from_mas",
    "DEFAULT_PACKAGED_LINE_MASK_BASENAME",
    "packaged_default_line_mask_path",
    "paired_line_mask_path_for_template",
    "pixel_edges_from_wavelength",
    "resolve_mask_path",
    "restrict_mask_to_chunk",
    "stellar_rv_ccf_mask_lines",
    "load_atmospheric_transmission_um",
    "mean_transmission_over_velocity_interval_um",
    "parse_mask_generation_meta_line",
    "read_mask_generation_meta",
]

# Prefix for a single-line JSON blob of generation parameters (after ``# ``).
LINE_MASK_META_PREFIX = "# HISPEC_DRP_MASK_META "

# Values for ``mask_style`` in :func:`read_mask_generation_meta` / written ``.mas`` JSON.
MASK_STYLE_DELTA: str = "delta"
MASK_STYLE_TOPHAT: str = "tophat"
# Deprecated alias for :data:`MASK_STYLE_TOPHAT` (legacy on-disk value ``"kpf"``).
MASK_STYLE_KPF: str = MASK_STYLE_TOPHAT


def _normalize_mask_style(raw: str) -> str:
    """Map legacy ``kpf`` mask_style labels to canonical ``tophat``."""
    style = str(raw).strip().lower()
    if style == "kpf":
        return MASK_STYLE_TOPHAT
    return style

# Half-width (nm) vs line-center vacuum wavelength (nm): half_width ≈ slope * λ_nm + intercept.
# Coefficients from a least-squares fit to bundled ``K2_all.mas`` (2797 segments); intercept
# is numerically ~0 because mask segments are an almost fixed Δln λ (see data/masks/README.md).
DEFAULT_WIDTH_CALIBRATION_K2_ALL: tuple[float, float] = (
    7.033348083502036e-06,
    0.0,
)

# Half-width law (nm): ``half_width_nm = slope * center_vac_nm + intercept`` for
# template-built SPIROU ``.mas`` files. This scale (nm per nm of line center) sets how
# wide each segment is before deblending / telluric filters.
MASK_LINE_HALF_WIDTH_SLOPE_NM_PER_NM: float = 1.0e-6

# Defaults for template-derived **delta** masks (gradient centroids, no segment width).
# Historical “optimal” path: depth cut + deblend + telluric gate + saturated-core floor.
OPTIMAL_TEMPLATE_MASK_BUILD_KW: dict[str, Any] = {
    "min_continuum_normalized_depth": 0.1,
    "low_pass_width_multiplier": 7.283985083415576,
    "telluric_velocity_half_width_km_s": 288.2674944075802,
    "telluric_min_mean_transmission": 0.0,
    "continuum_filter_size": 181,
    "apply_deblend": True,
    "apply_telluric_filter": True,
    "min_continuum_normalized_flux_floor": 1.0e-4,
}

# Relaxed seed masks: maximize line count (depth ≥ 10%; no deblend / telluric /
# saturated-core rejection). Used by ``regenerate_template_masks.py`` and the
# order-scan ccfopt bias-cull notebook.
RELAXED_TEMPLATE_MASK_BUILD_KW: dict[str, Any] = {
    "min_continuum_normalized_depth": 0.1,
    "low_pass_width_multiplier": 0.0,
    "telluric_velocity_half_width_km_s": 288.2674944075802,
    "telluric_min_mean_transmission": 0.0,
    "continuum_filter_size": 181,
    "apply_deblend": False,
    "apply_telluric_filter": False,
    "min_continuum_normalized_flux_floor": None,
}

# Defaults for CCF-contrast–optimized delta masks (seed refine at injected RV = 0).
CCFOPT_MASK_BUILD_KW: dict[str, Any] = {
    "spectrum_window_half_um": 0.01,
    "ccf_v_half_m_s": 8000.0,
    "ccf_v_step_m_s": 25.0,
    "search_half_m_s": 2500.0,
    "coarse_step_m_s": 250.0,
    "fine_half_m_s": 300.0,
    "fine_step_m_s": 25.0,
    "rv_gate_m_s": 2000.0,
    "min_good_pixels": 32,
}

# Stellar-RV kwargs for the E2E single-line ccfopt builder **and** the matching
# runtime estimator in ``rv_recovery_sim.ipynb``.
#
# Line-by-line CCF (not order-window ensembles): each mask line is a rest-frame
# λ calibrated once at injected RV=0 so that line's CCF → 0. The packaged
# ``*.ccfopt.delta.mas`` is then static across bulk RV; only ``ccf_input_velocity``
# moves at runtime. Velocity grid (``v_half_window_m_s``, ``v_step_m_s``) is shared
# with the ideal-spectrum RV recovery and build-ccfopt-mask simulation examples.
CCFOPT_MAX_ABS_RV_M_S = 500.0

# Wide wavelength-chunk default on :class:`~hispecdrp.primitives.stellar_rv.stellar_rv_primitive.StellarRVPrimitive`
# (``window_line_by_line=False``).  Line-by-line ccfopt uses :data:`CCFOPT_SINGLE_LINE_STELLAR_RV_KW`.
STELLAR_RV_WIDE_WINDOW_UM = 0.25

CCFOPT_SINGLE_LINE_STELLAR_RV_KW: dict[str, Any] = {
    "window_size_um": 5.65e-5,
    "window_line_by_line": True,
    "v_half_window_m_s": 20_000.0,
    "v_step_m_s": 10.0,
    "ccf_input_velocity_m_s": 0.0,
    "mask_ccf_pixel_interp_order": 3,
    "ccf_min_good_pixels": 9,
    "max_abs_rv_m_s": CCFOPT_MAX_ABS_RV_M_S,
    "min_contrast": 0.0,
    # After λ' shift, re-run single-line CCF and drop lines with |RV − center|
    # at or above this (m/s). Default 0.001 m/s = 0.1 cm/s.
    "rezero_verify_abs_rv_m_s": 0.001,
    # Drop lines whose formal single-line RV uncertainty is at or above this (m/s).
    # Removes aliases / shallow lines with tens-of-m/s (or worse) error bars.
    "max_rv_err_m_s": 10.0,
}


def apply_ccfopt_defaults_for_line_by_line(
    *,
    window_size_um: float,
    v_half_window: float,
    v_step_coarse: float,
) -> tuple[float, float, float]:
    """
    Replace wide ``StellarRVPrimitive`` defaults with ccfopt mask-build settings.

    When ``window_line_by_line=True``, line-mask CCF should use the same narrow
    ``window_size_um`` and velocity grid as :data:`CCFOPT_SINGLE_LINE_STELLAR_RV_KW`
    unless the caller explicitly set non-default values.
    """
    kw = CCFOPT_SINGLE_LINE_STELLAR_RV_KW
    if float(window_size_um) == float(STELLAR_RV_WIDE_WINDOW_UM):
        window_size_um = float(kw["window_size_um"])
    if float(v_step_coarse) == 25.0:
        v_step_coarse = float(kw["v_step_m_s"])
    if float(v_half_window) == 15000.0:
        v_half_window = float(kw["v_half_window_m_s"])
    return float(window_size_um), float(v_half_window), float(v_step_coarse)


#: Defaults for ideal-spectrum paint + line-mask RV (alias for notebooks/tests).
IDEAL_SPECTRUM_STELLAR_RV_KW = CCFOPT_SINGLE_LINE_STELLAR_RV_KW

#: Defaults for :func:`build_e2erv_line_mask_from_template` (RV-aligned e2e mask).
E2ERV_MASK_BUILD_KW: dict[str, Any] = {
    "min_continuum_normalized_depth": 0.20,
    "deblend_sep_resels": 1.5,
    "resel_um": 1.5e-4,  # ~R=8000 at 1.2 µm
    "continuum_filter_size": 181,
    "min_continuum_normalized_flux_floor": 1.0e-3,
}

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


def package_data_masks_dir() -> Path:
    """Absolute path to ``hispecdrp/data/masks``."""
    return Path(__file__).resolve().parents[1] / "data" / "masks"


#: Packaged default **delta** line mask basename under :func:`package_data_masks_dir`.
#: Prefer the e2e-validated mask used to bake ``HB.sim_hdf_wavesol.line_zp.fits``.
DEFAULT_PACKAGED_LINE_MASK_BASENAME: str = (
    "lte049.0-2.5-0.0a+0.0.BT-Settl.spec.7.ccfopt.delta.mas"
)
#: Fallback when the CCF-opt product is absent (older checkouts / seed delta).
_DEFAULT_PACKAGED_LINE_MASK_FALLBACK: str = (
    "lte049.0-2.5-0.0a+0.0.BT-Settl.spec.7.delta.mas"
)


def packaged_default_line_mask_path() -> str:
    """Absolute path to the packaged default **delta** mask (same as ``resolve_mask_path(None)``)."""
    masks_dir = package_data_masks_dir()
    preferred = masks_dir / DEFAULT_PACKAGED_LINE_MASK_BASENAME
    if preferred.is_file():
        return str(preferred.resolve())
    fallback = masks_dir / _DEFAULT_PACKAGED_LINE_MASK_FALLBACK
    return str(fallback.resolve())


def paired_line_mask_path_for_template(
    template_filename: str,
    *,
    mask_style: str = MASK_STYLE_DELTA,
    masks_directory: Path | str | None = None,
) -> str:
    """
    Resolve the packaged line mask paired with a template basename or path.

    For template ``lte049.0-2.5-0.0a+0.0.BT-Settl.spec.7.h5``, returns
    ``.../lte049.0-2.5-0.0a+0.0.BT-Settl.spec.7.delta.mas`` when ``mask_style`` is
    ``delta``.
    """
    stem = Path(template_filename).stem
    masks_dir = (
        Path(masks_directory) if masks_directory is not None else package_data_masks_dir()
    )
    suffix = _normalize_mask_style(mask_style)
    path = masks_dir / f"{stem}.{suffix}.mas"
    if not path.is_file():
        raise FileNotFoundError(
            f"Paired line mask not found for template {template_filename!r}: {path}"
        )
    return str(path.resolve())


def resolve_mask_path(mask_path: str | None) -> str:
    """
    Resolve ``mask_path`` to an absolute filesystem path, or the packaged default when ``None``.

    Raises
    ------
    FileNotFoundError
        If ``mask_path`` is ``None`` and the default mask file is missing.
    """
    if mask_path:
        p = Path(mask_path).expanduser()
        return str(p if p.is_absolute() else (Path.cwd() / p).resolve())
    default_file = Path(packaged_default_line_mask_path())
    if not default_file.is_file():
        raise FileNotFoundError(
            f"Default mask not found: {default_file}. "
            "Pass StellarRVPrimitive(mask_path=...) to use a custom mask."
        )
    return str(default_file)


def masklines_wavelength_unit_meta(unit: u.UnitBase) -> dict[str, str]:
    """Serializable Astropy/FITS unit strings for :attr:`MaskLines.metadata` wavelength keys."""
    return {
        "astropy_unit_string": unit.to_string(),
        "fits_unit_string": unit.to_string("fits"),
    }


MaskLineStyle = Literal["delta", "tophat"]


def _assert_mas_mask_style_matches_columns(
    mask_file: str, ncols: int, meta: dict | None
) -> None:
    """
    If ``.mas`` header JSON (``# HISPEC_DRP_MASK_META``) includes ``mask_style``, require
    that it matches the numeric layout: ``delta`` → 2 columns (vacuum nm, weight);
    ``tophat`` → ≥3 columns (start_nm, end_nm, weight, …). Legacy ``mask_style='kpf'``
    is treated as ``tophat``.
    """
    if not meta:
        return
    raw = meta.get("mask_style")
    if raw is None:
        return
    if not isinstance(raw, str):
        raise ValueError(
            f"mask_style in mask metadata ({mask_file!r}) must be a string, "
            f"got {type(raw).__name__!r}"
        )
    style = _normalize_mask_style(raw)
    if style in ("", "none", "null"):
        return
    if style == "delta":
        if ncols != 2:
            raise ValueError(
                f"Mask metadata declares mask_style='delta' (wavelength_nm, weight) but "
                f"{mask_file!r} has {ncols} numeric column(s); expected 2."
            )
        return
    if style == MASK_STYLE_TOPHAT:
        if ncols < 3:
            raise ValueError(
                f"Mask metadata declares mask_style='tophat' (start_nm, end_nm, weight) but "
                f"{mask_file!r} has {ncols} numeric column(s); expected ≥3."
            )
        return
    raise ValueError(
        f"Unknown mask_style {raw!r} in mask metadata for {mask_file!r} "
        "(expected 'delta' or 'tophat')"
    )


def _infer_wavelength_disk_to_um(values: np.ndarray) -> np.ndarray:
    """
    Infer wavelength column scale for legacy two-column text masks: median of positive finite
    values → Å (>1e3) /1e4, nm (>10) /1e3, else assume µm. Not used for ``.mas`` (vacuum nm).
    """
    v = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(v) & (v > 0)
    if not np.any(finite):
        return v
    med = float(np.nanmedian(v[finite]))
    if med > 1e3:
        return v / 1e4
    if med > 10:
        return v / 1e3
    return v


@dataclass(eq=False)
class MaskLines:
    """
    Tabulated CCF line mask (**delta** zero-width lines or **tophat** finite segments).

    **delta** — two on-disk columns (vacuum line-center nm, weight): ``line_left_edge``,
    ``line_right_edge``, and ``line_center`` coincide; ``line_width`` is 0.
    **tophat** — ≥3 columns (vacuum segment start nm, end nm, weight): ``line_center`` is the
    segment midpoint; ``line_width = line_right_edge - line_left_edge``.

    Wavelengths are **vacuum**; internal arrays use **microns** (nm÷1000 on read/write
    for ``.mas``). Use :meth:`load` and :meth:`save` for disk I/O with
    ``# HISPEC_DRP_MASK_META`` JSON (:func:`read_mask_generation_meta`).

    CCF hot paths use :attr:`line_lambda_um` (centers, µm) and :attr:`weight`.
    """

    mask_style: MaskLineStyle
    line_center: u.Quantity
    line_left_edge: u.Quantity
    line_right_edge: u.Quantity
    line_width: u.Quantity
    weight: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.line_center, u.Quantity):
            self.line_center = u.Quantity(
                np.asarray(self.line_center, dtype=np.float64).ravel(), u.um
            )
        if not isinstance(self.line_left_edge, u.Quantity):
            self.line_left_edge = u.Quantity(
                np.asarray(self.line_left_edge, dtype=np.float64).ravel(), u.um
            )
        if not isinstance(self.line_right_edge, u.Quantity):
            self.line_right_edge = u.Quantity(
                np.asarray(self.line_right_edge, dtype=np.float64).ravel(), u.um
            )
        if not isinstance(self.line_width, u.Quantity):
            self.line_width = u.Quantity(
                np.asarray(self.line_width, dtype=np.float64).ravel(), u.um
            )

        lc = self.line_center.to(u.um)
        ll = self.line_left_edge.to(u.um)
        lr = self.line_right_edge.to(u.um)
        lw = self.line_width.to(u.um)
        self.line_center = lc
        self.line_left_edge = ll
        self.line_right_edge = lr
        self.line_width = lw

        w = np.asarray(self.weight, dtype=np.float64).ravel()
        n = int(lc.size)
        if ll.size != n or lr.size != n or lw.size != n:
            raise ValueError(
                "MaskLines wavelength quantities must all have the same shape as line_center."
            )
        if w.size != n:
            raise ValueError("MaskLines.weight must match line_center length.")
        if w.ndim != 1:
            raise ValueError("MaskLines.weight must be 1D.")
        self.weight = w

        diff = (lr - ll).value
        if not np.allclose(diff, lw.value, rtol=0.0, atol=1.0e-12):
            raise ValueError(
                "MaskLines.line_width must equal line_right_edge - line_left_edge (µm)."
            )

        if self.mask_style not in ("delta", MASK_STYLE_TOPHAT):
            raise ValueError(
                f"Unknown mask_style {self.mask_style!r} (expected 'delta' or 'tophat')."
            )

        self.metadata = dict(self.metadata or {})

    @property
    def line_lambda_um(self) -> np.ndarray:
        """Rest line centers in µm (``line_center`` in µm)."""
        return np.asarray(self.line_center.to_value(u.um), dtype=np.float64)

    @classmethod
    def _from_mas_path(cls, mask_file: str) -> MaskLines:
        meta = read_mask_generation_meta(mask_file)
        try:
            data = np.loadtxt(mask_file, comments="#")
        except OSError as e:
            raise ValueError(f"Cannot read mask file: {mask_file}") from e
        if data.ndim == 1:
            data = data.reshape(1, -1)
        ncols = int(data.shape[1])
        _assert_mas_mask_style_matches_columns(mask_file, ncols, meta)
        md: dict[str, Any] = dict(meta or {})
        md["source_file"] = str(Path(mask_file).resolve())
        md["disk_wavelength_unit"] = "vacuum_nanometer"
        md["stored_wavelength_unit"] = masklines_wavelength_unit_meta(u.um)
        if ncols == 2:
            w_um = data[:, 0].astype(np.float64) / 1e3
            weight = data[:, 1].astype(np.float64)
            if "mask_style" not in md:
                md["mask_style"] = "delta"
                md["mask_style_inferred_from_columns"] = True
            return cls.from_delta_centers_um(w_um, weight, metadata=md)
        if ncols < 3:
            raise ValueError(
                f"Need 2 columns (wavelength_nm, weight) or ≥3 (start_nm, end_nm, weight): {mask_file}"
            )
        start = data[:, 0].astype(np.float64) / 1e3
        end = data[:, 1].astype(np.float64) / 1e3
        weight = data[:, 2].astype(np.float64)
        lo = np.minimum(start, end)
        hi = np.maximum(start, end)
        if "mask_style" not in md:
            md["mask_style"] = MASK_STYLE_TOPHAT
            md["mask_style_inferred_from_columns"] = True
        else:
            md["mask_style"] = _normalize_mask_style(str(md["mask_style"]))
        return cls.from_tophat_segment_edges_um(lo, hi, weight, metadata=md)

    @classmethod
    def _from_two_column_path(cls, mask_file: str) -> MaskLines:
        try:
            data = np.loadtxt(mask_file, comments="#")
        except OSError as e:
            raise ValueError(f"Cannot read mask file: {mask_file}") from e
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] < 2:
            raise ValueError(f"Need ≥2 columns (wavelength, weight): {mask_file}")
        w = _infer_wavelength_disk_to_um(data[:, 0].astype(np.float64))
        weight = data[:, 1].astype(np.float64)
        md: dict[str, Any] = {
            "source_file": str(Path(mask_file).resolve()),
            "mask_style": "delta",
            "disk_wavelength_column0": "inferred_angstrom_or_nm_or_micron_via_median_heuristic",
            "stored_wavelength_unit": masklines_wavelength_unit_meta(u.um),
        }
        return cls.from_delta_centers_um(w, weight, metadata=md)

    @classmethod
    def load(cls, path: str) -> MaskLines:
        """
        Load a mask from disk: ``.mas`` (vacuum nm on disk → µm) or two-column text
        (``.txt`` / ``.dat`` / ``.csv``; wavelength column Å/nm/µm via median heuristic).
        Other extensions try ``.mas`` parsing first, then two-column.
        """
        p = Path(path)
        suf = p.suffix.lower()
        if suf == ".mas":
            return cls._from_mas_path(path)
        if suf in (".txt", ".dat", ".csv"):
            return cls._from_two_column_path(path)
        try:
            return cls._from_mas_path(path)
        except Exception:
            return cls._from_two_column_path(path)

    @classmethod
    def from_delta_centers_um(
        cls,
        line_lambda_um: np.ndarray,
        weight: np.ndarray,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> MaskLines:
        """Build a **delta** mask: zero width at each center (µm)."""
        x = np.asarray(line_lambda_um, dtype=np.float64).ravel()
        wt = np.asarray(weight, dtype=np.float64).ravel()
        center = x * u.um
        z = np.zeros_like(x, dtype=np.float64) * u.um
        md: dict[str, Any] = dict(metadata or {})
        md.setdefault(
            "hispecdrp_wavelength_columns",
            {
                "line_center": masklines_wavelength_unit_meta(u.um),
                "line_left_edge": masklines_wavelength_unit_meta(u.um),
                "line_right_edge": masklines_wavelength_unit_meta(u.um),
                "line_width": masklines_wavelength_unit_meta(u.um),
            },
        )
        md.setdefault("mask_style", "delta")
        return cls(
            mask_style="delta",
            line_center=center,
            line_left_edge=center,
            line_right_edge=center,
            line_width=z,
            weight=wt,
            metadata=md,
        )

    @classmethod
    def from_tophat_segment_edges_um(
        cls,
        lo_um: np.ndarray,
        hi_um: np.ndarray,
        weight: np.ndarray,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> MaskLines:
        """Build a **tophat** (finite-width) mask from rest segment bounds in µm."""
        lo = np.asarray(lo_um, dtype=np.float64).ravel()
        hi = np.asarray(hi_um, dtype=np.float64).ravel()
        wt = np.asarray(weight, dtype=np.float64).ravel()
        lo_q = lo * u.um
        hi_q = hi * u.um
        center = 0.5 * (lo_q + hi_q)
        width = hi_q - lo_q
        md: dict[str, Any] = dict(metadata or {})
        md.setdefault(
            "hispecdrp_wavelength_columns",
            {
                "line_center": masklines_wavelength_unit_meta(u.um),
                "line_left_edge": masklines_wavelength_unit_meta(u.um),
                "line_right_edge": masklines_wavelength_unit_meta(u.um),
                "line_width": masklines_wavelength_unit_meta(u.um),
            },
        )
        md.setdefault("mask_style", MASK_STYLE_TOPHAT)
        return cls(
            mask_style=MASK_STYLE_TOPHAT,
            line_center=center,
            line_left_edge=lo_q,
            line_right_edge=hi_q,
            line_width=width,
            weight=wt,
            metadata=md,
        )

    @classmethod
    def from_kpf_segment_edges_um(
        cls,
        lo_um: np.ndarray,
        hi_um: np.ndarray,
        weight: np.ndarray,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> MaskLines:
        """Deprecated alias for :meth:`from_tophat_segment_edges_um`."""
        return cls.from_tophat_segment_edges_um(lo_um, hi_um, weight, metadata=metadata)

    @classmethod
    def empty_like_style(
        cls, mask_style: MaskLineStyle, *, metadata: dict[str, Any] | None = None
    ) -> MaskLines:
        """Zero-line mask preserving ``mask_style`` (e.g. after wavelength restriction)."""
        z = np.array([], dtype=np.float64) * u.um
        md = dict(metadata or {})
        md["n_lines"] = 0
        md.setdefault(
            "hispecdrp_wavelength_columns",
            {
                "line_center": masklines_wavelength_unit_meta(u.um),
                "line_left_edge": masklines_wavelength_unit_meta(u.um),
                "line_right_edge": masklines_wavelength_unit_meta(u.um),
                "line_width": masklines_wavelength_unit_meta(u.um),
            },
        )
        return cls(
            mask_style=mask_style,
            line_center=z,
            line_left_edge=z,
            line_right_edge=z,
            line_width=z,
            weight=np.array([], dtype=np.float64),
            metadata=md,
        )

    def metadata_dict(self) -> dict[str, Any]:
        """Shallow copy of :attr:`metadata` for inspection or JSON serialization."""
        return dict(self.metadata)

    def save(
        self,
        output_path: str | None = None,
        *,
        masks_directory: str | None = None,
        basename: str | None = None,
    ) -> str:
        """
        Write this mask to a ``.mas`` file.

        **On-disk layout:** ``delta`` → vacuum line-center (nm), weight; ``tophat`` →
        vacuum segment start (nm), end (nm), weight. One ``# HISPEC_DRP_MASK_META`` JSON
        line (:func:`read_mask_generation_meta`).

        When ``output_path`` is ``None``, writes
        ``{masks_directory}/{basename}.{mask_style}.mas`` with ``basename`` from
        ``metadata["template_path"]`` when set, else ``\"mask\"``. ``masks_directory``
        defaults to :func:`package_data_masks_dir`.
        """
        style = str(self.mask_style)
        if basename is not None:
            stem = Path(basename).stem
        else:
            tp = self.metadata.get("template_path")
            if isinstance(tp, str) and tp.strip():
                stem = Path(tp.strip()).stem
            else:
                stem = "mask"

        if masks_directory is None:
            masks_directory = str(package_data_masks_dir())

        if output_path is None:
            out_path = Path(masks_directory).expanduser()
            out_path.mkdir(parents=True, exist_ok=True)
            out_path = out_path / f"{stem}.{style}.mas"
        else:
            out_path = Path(output_path).expanduser()
            if out_path.parent != Path("."):
                out_path.parent.mkdir(parents=True, exist_ok=True)

        meta: dict[str, Any] = dict(self.metadata)
        meta["mask_style"] = style
        meta.setdefault("schema_version", 2)
        meta["n_lines"] = int(self.line_center.size)
        if style == "delta":
            meta.setdefault(
                "disk_wavelength_columns",
                {
                    "column0_line_center": "vacuum_nanometer",
                    "column1_weight": "dimensionless",
                },
            )
        else:
            meta.setdefault(
                "disk_wavelength_columns",
                {
                    "column0_segment_start": "vacuum_nanometer",
                    "column1_segment_end": "vacuum_nanometer",
                    "column2_weight": "dimensionless",
                },
            )

        meta_json = json.dumps(meta, sort_keys=True, separators=(",", ":"))
        if style == "delta":
            col_desc = "wavelength_vac_nm, weight"
        else:
            col_desc = "start_nm, end_nm, weight"
        header = (
            f"# HISPEC DRP CCF line mask (columns: {col_desc}).\n"
            f"# mask_style: {style}\n"
            "# Metadata JSON on the next line (parse with read_mask_generation_meta).\n"
            f"{LINE_MASK_META_PREFIX}{meta_json}\n"
        )

        wts = np.asarray(self.weight, dtype=np.float64)
        parts: list[str] = [header]
        if style == "delta":
            lc_nm = self.line_center.to_value(u.nm)
            for i in range(int(lc_nm.size)):
                parts.append(
                    f"  {float(lc_nm[i]): .16e}       {float(wts[i]): .16e}\n"
                )
        else:
            lo_nm = self.line_left_edge.to_value(u.nm)
            hi_nm = self.line_right_edge.to_value(u.nm)
            for i in range(int(lo_nm.size)):
                parts.append(
                    f"  {float(lo_nm[i]): .16e}       {float(hi_nm[i]): .16e}       {float(wts[i]): .16e}\n"
                )

        out_path.write_text("".join(parts), encoding="utf-8")
        return str(out_path.resolve())


def pixel_edges_from_wavelength(wave_um: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Pixel bin left/right edges (µm) from a sorted wavelength array (midpoints + end padding).
    """
    w = np.asarray(wave_um, dtype=np.float64).ravel()
    if w.size < 2:
        raise ValueError("Need at least 2 wavelength points to build pixel edges.")
    x_start = (w + np.roll(w, 1)) / 2.0
    x_end = np.roll(x_start, -1)
    x_start[0] = w[0] - (w[1] - w[0]) / 2.0
    x_end[-1] = w[-1] + (w[-1] - w[-2]) / 2.0
    return x_start, x_end


def restrict_mask_to_chunk(
    mask_lines: MaskLines,
    w_min_um: float,
    w_max_um: float,
    *,
    v_min_m_s: float,
    v_max_m_s: float,
    zb: float = 0.0,
) -> MaskLines:
    """
    Keep lines whose rest **centers** can Doppler-shift into ``[w_min_um, w_max_um]`` for
    some velocity in ``[v_min_m_s, v_max_m_s]`` (m/s) with denominator ``(1+zb)``.
    """
    zb1 = 1.0 + zb
    sh_a = (1.0 + v_min_m_s / C_M_S) / zb1
    sh_b = (1.0 + v_max_m_s / C_M_S) / zb1
    sh_lo = min(sh_a, sh_b)
    sh_hi = max(sh_a, sh_b)
    lam = mask_lines.line_lambda_um
    keep = (lam * sh_hi >= w_min_um) & (lam * sh_lo <= w_max_um)
    if not np.any(keep):
        return MaskLines.empty_like_style(
            mask_lines.mask_style,
            metadata={
                **mask_lines.metadata,
                "restrict_mask_to_chunk": "empty",
                "w_min_um": float(w_min_um),
                "w_max_um": float(w_max_um),
            },
        )
    k = np.asarray(keep, dtype=bool)
    md = {
        **mask_lines.metadata,
        "restrict_mask_to_chunk": True,
        "n_lines_before": int(mask_lines.line_center.size),
        "n_lines_after": int(np.count_nonzero(k)),
        "w_min_um": float(w_min_um),
        "w_max_um": float(w_max_um),
    }
    return MaskLines(
        mask_style=mask_lines.mask_style,
        line_center=mask_lines.line_center[k],
        line_left_edge=mask_lines.line_left_edge[k],
        line_right_edge=mask_lines.line_right_edge[k],
        line_width=mask_lines.line_width[k],
        weight=mask_lines.weight[k],
        metadata=md,
    )


def stellar_rv_ccf_mask_lines(
    *,
    mask_path: str | None,
    mask_wavelength=None,
    mask_weight=None,
) -> tuple[MaskLines, str]:
    """Resolve mask lines for line-mask CCF (path, default package mask, or in-memory arrays)."""
    has_wl = mask_wavelength is not None
    has_wt = mask_weight is not None
    if has_wl or has_wt:
        if not (has_wl and has_wt):
            raise ValueError(
                "mask_wavelength and mask_weight must both be provided when supplying an in-memory mask"
            )
        wl = np.asarray(mask_wavelength, dtype=np.float64).ravel()
        wt = np.asarray(mask_weight, dtype=np.float64).ravel()
        if wl.size != wt.size:
            raise ValueError(
                "mask_wavelength and mask_weight must have the same length "
                f"(got {wl.size} vs {wt.size})"
            )
        if wl.size == 0:
            raise ValueError("in-memory mask has zero lines")
        return (
            MaskLines.from_delta_centers_um(
                wl,
                wt,
                metadata={
                    "source": "in_memory_mask",
                    "mask_style": "delta",
                    "stored_wavelength_unit": masklines_wavelength_unit_meta(u.um),
                },
            ),
            "in_memory_mask",
        )
    resolved = resolve_mask_path(mask_path)
    return MaskLines.load(resolved), resolved


@dataclass(frozen=True)
class LineMaskEntry:
    """One template-built mask row: vacuum wavelength (nm) and CCF weight."""

    wavelength_nm: float
    weight: float


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
    hw = np.asarray(halfw, dtype=np.float64)
    finite = np.asarray(centers, dtype=np.float64)
    m = hw > 1.0e-9
    if int(np.sum(m)) < 2:
        raise ValueError(
            f"Too few finite-width rows (need ≥2 with half-width > 1e-9 nm) to fit: {mas_path!r}"
        )
    x = finite[m]
    y = hw[m]
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
    c_m_s = C_M_S
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


def _parabolic_minimum_wavelength_um(
    wave_um: np.ndarray,
    flux: np.ndarray,
    idx: int,
) -> float:
    """
    Sub-pixel wavelength of a local absorption minimum.

    Fits :math:`F(\\lambda) \\approx a\\lambda^2 + b\\lambda + c` to three consecutive
    samples bracketing ``idx``; returns the vertex if it is a **minimum** with
    :math:`a > 0` and lies inside the bracketed interval. Otherwise returns
    ``wave_um[idx]``.

    This aligns the stored segment midpoint ``(start_nm + end_nm)/2`` with the same
    convention :meth:`MaskLines.load` uses for
    three-column SPIROU masks (``line_lambda_um``), while improving precision over using
    the grid sample at ``idx`` alone.
    """
    n = int(wave_um.size)
    if n < 3 or idx <= 0 or idx >= n - 1:
        return float(wave_um[idx])
    x = np.asarray(wave_um[idx - 1 : idx + 2], dtype=np.float64)
    yv = np.asarray(flux[idx - 1 : idx + 2], dtype=np.float64)
    if not np.all(np.isfinite(x) & np.isfinite(yv)):
        return float(wave_um[idx])
    # F = a x^2 + b x + c; minimum at x = -b/(2a) when a > 0
    a_mat = np.column_stack([x**2, x, np.ones(3, dtype=np.float64)])
    try:
        coef, _res, rank, _s = np.linalg.lstsq(a_mat, yv, rcond=None)
    except np.linalg.LinAlgError:
        return float(wave_um[idx])
    if rank < 3:
        return float(wave_um[idx])
    a, b, _c = (float(coef[0]), float(coef[1]), float(coef[2]))
    scale = max(np.max(np.abs(yv)), 1.0)
    if abs(a) < 1e-18 * scale:
        return float(wave_um[idx])
    if a <= 0.0:
        return float(wave_um[idx])
    xv = -b / (2.0 * a)
    lo, hi = float(np.min(x)), float(np.max(x))
    if xv < lo or xv > hi:
        return float(wave_um[idx])
    return float(xv)


def detect_template_line_centers_parabolic_um(
    wave_um: np.ndarray,
    flux: np.ndarray,
    *,
    local_min_order: int = 2,
    continuum_filter_size: int = 181,
    min_continuum_normalized_depth: float | None = None,
) -> np.ndarray:
    """
    Line centers (vacuum µm) from discrete local flux minima + parabolic refinement.

    This matches the **centroid** step in :func:`build_line_mask_from_template` before
    depth / deblend / telluric filters: :func:`_local_minima_indices` on ``flux`` and
    :func:`_parabolic_minimum_wavelength_um` at each index.

    Parameters
    ----------
    wave_um, flux
        Same sorted vacuum-µm grid as ``load_template_from_path`` (ascending λ).
    local_min_order
        ``order`` for :func:`scipy.signal.argrelextrema`.
    continuum_filter_size
        Used only if ``min_continuum_normalized_depth`` is set.
    min_continuum_normalized_depth
        If not ``None``, drop centers with ``1 - F/F_c`` below this threshold at the
        refined λ (``F``, ``F_c`` interpolated), matching the spirit of the mask depth
        cut in :func:`build_line_mask_from_template`.

    Returns
    -------
    np.ndarray
        1-D float64 centers in **vacuum microns**, sorted ascending.
    """
    w = np.asarray(wave_um, dtype=np.float64).ravel()
    f = np.asarray(flux, dtype=np.float64).ravel()
    if w.size != f.size or w.size < 3:
        return np.array([], dtype=np.float64)
    idx = _local_minima_indices(f, int(local_min_order))
    idx = idx[(idx > 0) & (idx < w.size - 1)]
    if idx.size == 0:
        return np.array([], dtype=np.float64)
    centers = np.array(
        [_parabolic_minimum_wavelength_um(w, f, int(i)) for i in idx],
        dtype=np.float64,
    )
    centers.sort()
    if min_continuum_normalized_depth is None:
        return centers
    cont = _continuum_envelope(f, int(continuum_filter_size))
    fv = np.interp(centers, w, f)
    cv = np.interp(centers, w, cont)
    depths = 1.0 - fv / np.maximum(cv, 1.0e-30)
    m = depths >= float(min_continuum_normalized_depth)
    return np.sort(centers[m])


def _merge_sorted_centers_um(centers_um: np.ndarray, min_sep_um: float) -> np.ndarray:
    """Drop duplicates closer than ``min_sep_um`` (keep first)."""
    c = np.sort(np.asarray(centers_um, dtype=np.float64).ravel())
    c = c[np.isfinite(c)]
    if c.size == 0:
        return c
    sep = float(max(min_sep_um, 0.0))
    out: list[float] = [float(c[0])]
    for x in c[1:]:
        if x - out[-1] >= sep:
            out.append(float(x))
    return np.asarray(out, dtype=np.float64)


def _absorption_centers_from_flux_gradients_um(
    wave_um: np.ndarray,
    flux: np.ndarray,
    cont: np.ndarray,
    *,
    min_continuum_normalized_depth: float,
) -> np.ndarray:
    """
    Interpolated λ where ``dF/dλ = 0`` between samples, keeping minima (``d²F/dλ² > 0``).

    Uses :func:`numpy.gradient` with the wavelength coordinate (non-uniform spacing
    supported). Each bracket ``[i-1, i]`` where ``d1`` changes sign yields a linearly
    interpolated root; roots with non-positive second derivative (local maxima of
    flux) are discarded. Shallow points fail ``min_continuum_normalized_depth``.
    """
    w = np.asarray(wave_um, dtype=np.float64).ravel()
    f = np.asarray(flux, dtype=np.float64).ravel()
    c = np.asarray(cont, dtype=np.float64).ravel()
    if w.size < 3 or w.size != f.size or f.size != c.size:
        return np.array([], dtype=np.float64)
    d1 = np.gradient(f, w, edge_order=1)
    d2 = np.gradient(d1, w, edge_order=1)
    centers: list[float] = []
    for i in range(1, w.size):
        a, b = float(d1[i - 1]), float(d1[i])
        wa, wb = float(w[i - 1]), float(w[i])
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        if a == 0.0:
            lam = wa
        elif b == 0.0:
            lam = wb
        elif a * b < 0.0:
            t = a / (a - b)
            lam = (1.0 - t) * wa + t * wb
        else:
            continue
        d2v = float(np.interp(lam, w, d2))
        if d2v <= 0.0:
            continue
        fv = float(np.interp(lam, w, f))
        cv = float(np.interp(lam, w, c))
        depth = 1.0 - fv / max(cv, 1.0e-30)
        if depth < float(min_continuum_normalized_depth):
            continue
        centers.append(lam)
    return np.asarray(centers, dtype=np.float64)


def detect_template_line_centers_gradient_um(
    wave_um: np.ndarray,
    flux: np.ndarray,
    *,
    continuum_filter_size: int = 181,
    min_continuum_normalized_depth: float = 0.04,
    merge_min_sep_um: float | None = None,
) -> np.ndarray:
    """
    Absorption line centers (vacuum µm) from **gradient zeros** with positive curvature.

    Intervals where the first derivative of flux w.r.t. λ (``numpy.gradient``) crosses
    zero are refined to a wavelength; the point is kept only if the second derivative
    of flux w.r.t. λ is **positive** (flux minimum, not maximum) and the usual
    continuum-normalized depth exceeds ``min_continuum_normalized_depth``.

    This is an alternative to chaining :func:`_local_minima_indices` with
    :func:`_parabolic_minimum_wavelength_um`; lists can be compared with
    :func:`compare_line_center_lists_um`.

    Parameters
    ----------
    wave_um, flux
        Sorted vacuum-µm template grid (as from ``load_template_from_path``).
    continuum_filter_size
        Odd-ish size for :func:`_continuum_envelope` (depth definition).
    min_continuum_normalized_depth
        Minimum ``1 - F/F_c`` at the interpolated stationary point.
    merge_min_sep_um
        Merge roots closer than this (µm). Default: ``0.25 * median(Δλ)`` on the grid.

    Returns
    -------
    np.ndarray
        Centers in **vacuum microns**, sorted ascending.
    """
    w = np.asarray(wave_um, dtype=np.float64).ravel()
    f = np.asarray(flux, dtype=np.float64).ravel()
    if w.size < 8:
        return np.array([], dtype=np.float64)
    cont = _continuum_envelope(f, int(continuum_filter_size))
    raw = _absorption_centers_from_flux_gradients_um(
        w,
        f,
        cont,
        min_continuum_normalized_depth=float(min_continuum_normalized_depth),
    )
    dw = np.diff(w)
    dw = dw[np.isfinite(dw) & (dw > 0)]
    med = float(np.median(dw)) if dw.size else 1.0e-6
    sep = float(0.25 * med) if merge_min_sep_um is None else float(merge_min_sep_um)
    return _merge_sorted_centers_um(raw, sep)


def compare_line_center_lists_um(
    centers_a: np.ndarray,
    centers_b: np.ndarray,
    *,
    match_tol_nm: float = 0.05,
) -> dict[str, Any]:
    """
    Match two line-center lists (vacuum µm) within ``match_tol_nm`` (nanometers).

    Returns counts and boolean masks over the **sorted** union of inputs for which
    centers fall in only A, only B, or both (matched pairs use greedy nearest-neighbor
    within tolerance on the merged sorted list — for diagnostics only).

    Notes
    -----
    Greedy matching: sort all centers from A and B with tags; sweep λ increasing,
    pairing unmatched A with next unmatched B if ``|Δλ|`` ≤ tol (in µm: ``tol_nm * 1e-3``).
    Simpler approach: for each a in A, match if any b in B with ``|a-b|`` <= tol_um.

    Returns
    -------
    dict
        Keys: ``n_a``, ``n_b``, ``n_match``, ``n_a_only``, ``n_b_only``,
        ``match_tol_nm``, ``matched_pairs`` (list of (a_um, b_um)), ``centers_a_only_um``,
        ``centers_b_only_um``.
    """
    tol_um = abs(float(match_tol_nm)) * 1.0e-3
    a = np.sort(np.asarray(centers_a, dtype=np.float64).ravel())
    b = np.sort(np.asarray(centers_b, dtype=np.float64).ravel())
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    used_b = np.zeros(b.size, dtype=bool)
    pairs: list[tuple[float, float]] = []
    a_only: list[float] = []
    for x in a:
        if b.size == 0:
            a_only.append(float(x))
            continue
        free = np.flatnonzero(~used_b)
        if free.size == 0:
            a_only.append(float(x))
            continue
        dist = np.abs(b[free] - x)
        k = int(np.argmin(dist))
        if float(dist[k]) <= tol_um:
            j = int(free[k])
            used_b[j] = True
            pairs.append((float(x), float(b[j])))
        else:
            a_only.append(float(x))
    b_only = [float(b[k]) for k in range(b.size) if not used_b[k]]
    n_match = len(pairs)
    return {
        "n_a": int(a.size),
        "n_b": int(b.size),
        "n_match": n_match,
        "n_a_only": len(a_only),
        "n_b_only": len(b_only),
        "match_tol_nm": float(match_tol_nm),
        "matched_pairs": pairs,
        "centers_a_only_um": np.asarray(a_only, dtype=np.float64),
        "centers_b_only_um": np.asarray(b_only, dtype=np.float64),
    }


def _cluster_max_depth(
    centers_nm: np.ndarray,
    merge_radii_nm: np.ndarray,
    depths: np.ndarray,
) -> np.ndarray:
    """Union lines with ``|Δλ| < r_i + r_j``; keep largest per-line metric per component (union-find).

    ``depths`` is the per-line scalar used to break ties (continuum-normalized depth
    or another positive strength measure).
    """
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


def _segment_mean_continuum_normalized_depth_um(
    wave_um: np.ndarray,
    flux: np.ndarray,
    cont: np.ndarray,
    lo_um: float,
    hi_um: float,
) -> float:
    """
    Mean :math:`1 - F/F_c` over discrete template samples inside ``[lo_um, hi_um]``.

    Helper for segment-averaged depth (e.g. comparisons); template-mask **weights** in
    :func:`build_line_mask_from_template` use line-center :math:`1 - F/F_c` only.
    """
    lo = float(lo_um)
    hi = float(hi_um)
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return float("nan")
    m = (wave_um >= lo) & (wave_um <= hi)
    if not np.any(m):
        return float("nan")
    c = np.maximum(cont[m], 1.0e-30)
    d = 1.0 - flux[m] / c
    v = float(np.mean(d))
    if not np.isfinite(v):
        return float("nan")
    return float(np.clip(v, 0.0, 1.0))


def _local_min_continuum_normalized_flux_um(
    wave_um: np.ndarray,
    flux: np.ndarray,
    cont: np.ndarray,
    centers_um: np.ndarray,
    *,
    half_width_um: float,
) -> np.ndarray:
    """
    Per centroid, minimum of :math:`F/F_c` over discrete samples in
    ``[λ - half_width, λ + half_width]``.

    Used to reject saturated / unreliable line cores (very small :math:`F/F_c`).
    """
    w = np.asarray(wave_um, dtype=np.float64).ravel()
    f = np.asarray(flux, dtype=np.float64).ravel()
    c = np.maximum(np.asarray(cont, dtype=np.float64).ravel(), 1.0e-30)
    lam = np.asarray(centers_um, dtype=np.float64).ravel()
    if w.size < 1 or lam.size == 0:
        return np.array([], dtype=np.float64)
    f_over_c = f / c
    hw = float(half_width_um)
    if not np.isfinite(hw) or hw <= 0.0:
        return np.full(lam.size, np.nan, dtype=np.float64)
    out = np.empty(lam.size, dtype=np.float64)
    for i in range(lam.size):
        lo = float(lam[i]) - hw
        hi = float(lam[i]) + hw
        m = (w >= lo) & (w <= hi)
        if np.any(m):
            out[i] = float(np.min(f_over_c[m]))
        else:
            out[i] = float(np.interp(float(lam[i]), w, f_over_c))
    return out


def _continuum_normalized_d2_flux_dlam2_at_um(
    wave_um: np.ndarray,
    flux: np.ndarray,
    cont: np.ndarray,
    centers_um: np.ndarray,
) -> np.ndarray:
    """
    Second derivative w.r.t. λ of continuum-normalized flux ``F/F_c`` at ``centers_um``.

    Uses :func:`numpy.gradient` twice on ``F/F_c`` with coordinate ``wave_um``. At an
    absorption minimum, this is positive (upward curvature); larger values indicate
    sharper line cores for a given width.
    """
    w = np.asarray(wave_um, dtype=np.float64).ravel()
    f = np.asarray(flux, dtype=np.float64).ravel()
    c = np.maximum(np.asarray(cont, dtype=np.float64).ravel(), 1.0e-30)
    if w.size < 3 or w.size != f.size or f.size != c.size:
        return np.array([], dtype=np.float64)
    f_norm = f / c
    d1 = np.gradient(f_norm, w, edge_order=1)
    d2 = np.gradient(d1, w, edge_order=1)
    lam = np.asarray(centers_um, dtype=np.float64).ravel()
    return np.interp(lam, w, d2).astype(np.float64)


def build_line_mask_from_template(
    template_path: str,
    *,
    wl_min_um: float = 0.8,
    wl_max_um: float = 3.0,
    min_continuum_normalized_depth: float = OPTIMAL_TEMPLATE_MASK_BUILD_KW[
        "min_continuum_normalized_depth"
    ],
    low_pass_width_multiplier: float = OPTIMAL_TEMPLATE_MASK_BUILD_KW[
        "low_pass_width_multiplier"
    ],
    telluric_velocity_half_width_km_s: float = OPTIMAL_TEMPLATE_MASK_BUILD_KW[
        "telluric_velocity_half_width_km_s"
    ],
    telluric_min_mean_transmission: float = OPTIMAL_TEMPLATE_MASK_BUILD_KW[
        "telluric_min_mean_transmission"
    ],
    transmission_file: str | None = None,
    continuum_filter_size: int = OPTIMAL_TEMPLATE_MASK_BUILD_KW["continuum_filter_size"],
    min_continuum_normalized_flux_floor: float | None = OPTIMAL_TEMPLATE_MASK_BUILD_KW[
        "min_continuum_normalized_flux_floor"
    ],
    apply_deblend: bool = OPTIMAL_TEMPLATE_MASK_BUILD_KW["apply_deblend"],
    apply_telluric_filter: bool = OPTIMAL_TEMPLATE_MASK_BUILD_KW["apply_telluric_filter"],
    local_min_fn_window_half_width_nm: float | None = None,
    output_path: str | None = None,
    masks_directory: str | None = None,
) -> tuple[str, list[LineMaskEntry]]:
    """
    Detect lines in a template spectrum and write a filtered ``.mas`` mask.

    Centroids come from :func:`detect_template_line_centers_gradient_um`. Each output
    row is ``(wavelength_vacuum_nm, weight)`` with no segment edges. Raw weights are
    continuum-normalized depth :math:`1 - F/F_c` at each line center (interpolated),
    then are divided by their sum so **output weights sum to 1**.

    Deblending (when ``apply_deblend``) uses
    ``0.5 * low_pass_width_multiplier * median(Δλ_nm)`` as the merge radius per line
    (deepest line kept in each cluster), where ``median(Δλ_nm)`` is from the sorted
    template grid. Pass ``apply_deblend=False`` / ``apply_telluric_filter=False`` (see
    :data:`RELAXED_TEMPLATE_MASK_BUILD_KW`) to maximize line count.

    Parameters
    ----------
    template_path
        Path to a supported template (BT-Settl ``.h5``).
    wl_min_um, wl_max_um
        Wavelength clip in **vacuum** microns before line search.
    min_continuum_normalized_depth
        Passed to :func:`detect_template_line_centers_gradient_um` and re-applied after
        interpolation to centers.
    low_pass_width_multiplier
        Scales the deblending radius vs. the template median wavelength step (nm).
        Ignored when ``apply_deblend`` is false.
    telluric_velocity_half_width_km_s
        Half-width in km/s for the telluric averaging interval (± this value).
        Ignored when ``apply_telluric_filter`` is false.
    telluric_min_mean_transmission
        Lines with mean transmission below this value are removed (telluric path only).
    transmission_file
        Two-column ``mktrans``-style path (µm, transmission). Default: bundled
        ``data/atmosphere/mktrans_zm_10_10.dat``.
    continuum_filter_size
        Odd-ish pixel size for the maximum-filter continuum envelope.
    min_continuum_normalized_flux_floor
        If not ``None``, drop candidate lines whose **local minimum** of :math:`F/F_c`
        (over ``±local_min_fn_window_half_width_nm`` around the centroid) is **below**
        this value. Saturated cores have :math:`F/F_c \\to 0` and are skipped.
        Set ``None`` to keep saturated cores (relaxed seed).
    apply_deblend
        If false, skip nearest-neighbor deblend clustering (keep all depth-passing lines).
    apply_telluric_filter
        If false, skip the atmospheric-transmission gate.
    local_min_fn_window_half_width_nm
        Half-width of the wavelength interval (nm, vacuum) used for the local
        :math:`\\min(F/F_c)` test. If ``None``, uses ``5 ×`` the template median step (nm).
    output_path
        Full path for the output ``.mas``. If ``None``, writes under
        ``masks_directory`` as ``<template_basename>.delta.mas`` (``delta`` is the
        on-disk :data:`MASK_STYLE_DELTA` layout; distinct stems per style match
        :meth:`MaskLines.save`).
    masks_directory
        Default output directory (package ``data/masks`` if ``None``).

    Returns
    -------
    path_written, entries
        Output file path and in-memory rows (``wavelength_nm``, ``weight``).
    """
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
    _ord = np.argsort(wave_um)
    wave_um = wave_um[_ord]
    flux = flux[_ord]
    if wave_um.size < 16:
        raise ValueError("Template wavelength/flux too short after clipping")

    cont = _continuum_envelope(flux, continuum_filter_size)

    dw_um = np.diff(wave_um)
    dw_um = dw_um[np.isfinite(dw_um) & (dw_um > 0)]
    med_dw_um = float(np.median(dw_um)) if dw_um.size else 1.0e-6
    median_dw_nm = med_dw_um * 1000.0

    centers_um = detect_template_line_centers_gradient_um(
        wave_um,
        flux,
        continuum_filter_size=int(continuum_filter_size),
        min_continuum_normalized_depth=float(min_continuum_normalized_depth),
        merge_min_sep_um=None,
    )
    if centers_um.size == 0:
        raise ValueError(
            "No gradient line centers; relax min_continuum_normalized_depth or continuum_filter_size"
        )

    flux_c = np.interp(centers_um, wave_um, flux)
    cont_c = np.interp(centers_um, wave_um, cont)
    depths = 1.0 - flux_c / np.maximum(cont_c, 1.0e-30)
    depths = np.clip(np.asarray(depths, dtype=np.float64), 0.0, 1.0)

    if local_min_fn_window_half_width_nm is not None:
        half_win_um = float(local_min_fn_window_half_width_nm) * 1.0e-3
    else:
        half_win_um = max(5.0 * med_dw_um, 1.0e-9)
    min_fn_local = _local_min_continuum_normalized_flux_um(
        wave_um, flux, cont, centers_um, half_width_um=half_win_um
    )

    centers_nm = centers_um * 1000.0

    m = depths >= float(min_continuum_normalized_depth)
    if min_continuum_normalized_flux_floor is not None:
        floor = float(min_continuum_normalized_flux_floor)
        m &= np.isfinite(min_fn_local) & (min_fn_local >= floor)
    centers_nm = centers_nm[m]
    depths = depths[m]

    if centers_nm.size == 0:
        raise ValueError("No lines left after depth threshold; relax min depth or continuum size")

    weights = np.asarray(depths, dtype=np.float64)
    if not np.any(weights > 0.0):
        raise ValueError(
            "No lines with positive continuum-normalized depth; check template sampling"
        )

    merge_r_nm = 0.0
    if bool(apply_deblend):
        merge_r_nm = 0.5 * float(low_pass_width_multiplier) * float(median_dw_nm)
        merge_radii_nm = np.full(centers_nm.shape[0], merge_r_nm, dtype=np.float64)

        order = np.argsort(centers_nm)
        centers_nm = centers_nm[order]
        weights = weights[order]

        keep = _cluster_max_depth(centers_nm, merge_radii_nm, weights)
        centers_nm = centers_nm[keep]
        weights = weights[keep]
    else:
        order = np.argsort(centers_nm)
        centers_nm = centers_nm[order]
        weights = weights[order]

    vhalf = float(telluric_velocity_half_width_km_s)
    tmin = float(telluric_min_mean_transmission)
    if bool(apply_telluric_filter):
        tw, tt = load_atmospheric_transmission_um(transmission_file)
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
        weights = weights[keep_t_arr]

    w_sum = float(np.sum(weights))
    if not np.isfinite(w_sum) or w_sum <= 0.0:
        raise ValueError("Cannot normalize mask weights: non-positive or non-finite sum")
    weights = weights / w_sum

    entries: list[LineMaskEntry] = []
    for i in range(centers_nm.size):
        lam_nm = float(centers_nm[i])
        wt = float(weights[i])
        entries.append(LineMaskEntry(wavelength_nm=lam_nm, weight=wt))

    if masks_directory is None:
        masks_directory = str(package_data_masks_dir())
    os.makedirs(masks_directory, exist_ok=True)

    base = os.path.splitext(os.path.basename(template_path))[0]
    auto_name = f"{base}.{MASK_STYLE_DELTA}.mas"
    out = output_path if output_path is not None else os.path.join(masks_directory, auto_name)

    meta: dict[str, Any] = {
        "schema_version": 2,
        "mask_style": MASK_STYLE_DELTA,
        "generator": "hispecdrp.utils.line_mask_utils.build_line_mask_from_template",
        "template_path": os.path.abspath(template_path),
        "wl_min_um": float(wl_min_um),
        "wl_max_um": float(wl_max_um),
        "min_continuum_normalized_depth": float(min_continuum_normalized_depth),
        "low_pass_width_multiplier": float(low_pass_width_multiplier),
        "telluric_velocity_half_width_km_s": float(vhalf),
        "telluric_min_mean_transmission": float(tmin),
        "apply_deblend": bool(apply_deblend),
        "apply_telluric_filter": bool(apply_telluric_filter),
        "transmission_file": trans_path_abs,
        "continuum_filter_size": int(continuum_filter_size),
        "centroid_method": "gradient",
        "mask_line_geometry": "delta_two_column",
        "mask_weight_definition": "continuum_normalized_depth_1_minus_F_over_Fc",
        "weights_sum_normalized_to": 1.0,
        "deblend_merge_radius_nm": float(merge_r_nm),
        "median_template_step_nm": float(median_dw_nm),
        "local_min_fn_window_half_width_nm": float(half_win_um * 1000.0),
        "min_continuum_normalized_flux_floor": (
            None
            if min_continuum_normalized_flux_floor is None
            else float(min_continuum_normalized_flux_floor)
        ),
        "n_lines": int(len(entries)),
    }
    from astropy import units as _u

    def _disk_u(unit: _u.UnitBase) -> dict[str, str]:
        return {
            "astropy_unit_string": unit.to_string(),
            "fits_unit_string": unit.to_string("fits"),
        }

    meta["disk_wavelength_columns"] = {
        "column0_line_center": "vacuum_nanometer",
        "column1_weight": "dimensionless",
    }
    meta["hispecdrp_wavelength_columns"] = {
        "line_center": masklines_wavelength_unit_meta(u.nm),
        "line_left_edge": masklines_wavelength_unit_meta(u.nm),
        "line_right_edge": masklines_wavelength_unit_meta(u.nm),
        "line_width": masklines_wavelength_unit_meta(u.nm),
    }

    centers_um_arr = (centers_nm * 1.0e-3).astype(np.float64)
    ml = MaskLines.from_delta_centers_um(centers_um_arr, weights, metadata=meta)
    path_written = ml.save(output_path=out)

    logger.info("Wrote %d mask lines to %s", len(entries), path_written)
    return path_written, entries

def _ccfopt_single_line_gaussian_contrast(
    wave_um: np.ndarray,
    flux: np.ndarray,
    err: np.ndarray,
    dq: np.ndarray,
    lam_um: float,
    *,
    v_grid: np.ndarray,
    v_step_m_s: float,
    rv_gate_m_s: float,
) -> tuple[float, float]:
    """
    Single-line spline CCF + Gaussian fit at injected RV ≈ 0.

    Returns ``(|contrast|, rv_m_s)``; either may be NaN when the fit fails the RV gate.
    """
    from hispecdrp.primitives.utils.ccf.ccf_fitting import calc_ccf_metrics
    from hispecdrp.primitives.utils.ccf.ccf_utils import (
        LINE_MASK_CCF_STORED_NEGATED_FROM_RAW,
    )
    from hispecdrp.primitives.utils.ccf.ccf_spline import compute_ccf_spline

    if not np.isfinite(lam_um) or lam_um <= 0.0:
        return float("nan"), float("nan")
    _vg, ccf_raw, ccf_err = compute_ccf_spline(
        wave_um,
        flux,
        err,
        dq.astype(np.int32),
        np.asarray([lam_um], dtype=np.float64),
        np.asarray([1.0], dtype=np.float64),
        vel_grid=v_grid,
        zb=0.0,
        use_variance=True,
        line_set_velocity=0.0,
    )
    stored = -ccf_raw if LINE_MASK_CCF_STORED_NEGATED_FROM_RAW else ccf_raw
    if not np.any(np.isfinite(stored)):
        return float("nan"), float("nan")
    metrics = calc_ccf_metrics(
        v_grid,
        stored,
        ccf_err,
        pix_size=float(v_step_m_s),
        absorption=not LINE_MASK_CCF_STORED_NEGATED_FROM_RAW,
        calc_bis=False,
        mu_guess=0.0,
    )
    rv = float(metrics.rv)
    err_rv = float(metrics.rv_err)
    contrast = float(metrics.contrast)
    if (
        not np.isfinite(rv)
        or not np.isfinite(err_rv)
        or err_rv <= 0.0
        or not np.isfinite(contrast)
        or abs(rv) > float(rv_gate_m_s)
    ):
        return float("nan"), rv if np.isfinite(rv) else float("nan")
    return abs(contrast), rv


def _ccfopt_search_line_center_um(
    wave_um: np.ndarray,
    flux: np.ndarray,
    err: np.ndarray,
    dq: np.ndarray,
    lam0_um: float,
    *,
    v_grid: np.ndarray,
    v_step_m_s: float,
    search_half_m_s: float,
    coarse_step_m_s: float,
    fine_half_m_s: float,
    fine_step_m_s: float,
    rv_gate_m_s: float,
) -> tuple[float, float]:
    """
    Two-stage grid search over mask λ near ``lam0_um`` maximizing ``|Gaussian contrast|``.

    Trial centers are ``λ = λ₀ (1 + δv/c)``. Returns ``(best_lam_um, best_contrast)``.
    """

    def _trials(half_m_s: float, step_m_s: float, center_dv_m_s: float = 0.0) -> np.ndarray:
        n = int(np.floor(float(half_m_s) / float(step_m_s) + 0.5))
        if n < 0:
            n = 0
        return center_dv_m_s + np.arange(-n, n + 1, dtype=np.float64) * float(step_m_s)

    def _best_on(dvs: np.ndarray) -> tuple[float, float, float]:
        best_c = float("nan")
        best_lam = float("nan")
        best_dv = 0.0
        for dv in dvs:
            lam = float(lam0_um) * (1.0 + float(dv) / C_M_S)
            c_abs, _rv = _ccfopt_single_line_gaussian_contrast(
                wave_um,
                flux,
                err,
                dq,
                lam,
                v_grid=v_grid,
                v_step_m_s=v_step_m_s,
                rv_gate_m_s=rv_gate_m_s,
            )
            if not np.isfinite(c_abs) or c_abs <= 0.0:
                continue
            if not np.isfinite(best_c) or c_abs > best_c:
                best_c = float(c_abs)
                best_lam = float(lam)
                best_dv = float(dv)
        return best_lam, best_c, best_dv

    lam_c, con_c, dv_c = _best_on(_trials(search_half_m_s, coarse_step_m_s))
    if not np.isfinite(lam_c):
        return float("nan"), float("nan")
    lam_f, con_f, _dv_f = _best_on(_trials(fine_half_m_s, fine_step_m_s, center_dv_m_s=dv_c))
    if np.isfinite(lam_f) and np.isfinite(con_f):
        return float(lam_f), float(con_f)
    return float(lam_c), float(con_c)


def build_ccfopt_line_mask_from_template_and_seed(
    template_path: str,
    seed_mask_path: str,
    *,
    wl_min_um: float = 0.8,
    wl_max_um: float = 3.0,
    spectrum_window_half_um: float = CCFOPT_MASK_BUILD_KW["spectrum_window_half_um"],
    ccf_v_half_m_s: float = CCFOPT_MASK_BUILD_KW["ccf_v_half_m_s"],
    ccf_v_step_m_s: float = CCFOPT_MASK_BUILD_KW["ccf_v_step_m_s"],
    search_half_m_s: float = CCFOPT_MASK_BUILD_KW["search_half_m_s"],
    coarse_step_m_s: float = CCFOPT_MASK_BUILD_KW["coarse_step_m_s"],
    fine_half_m_s: float = CCFOPT_MASK_BUILD_KW["fine_half_m_s"],
    fine_step_m_s: float = CCFOPT_MASK_BUILD_KW["fine_step_m_s"],
    rv_gate_m_s: float = CCFOPT_MASK_BUILD_KW["rv_gate_m_s"],
    min_good_pixels: int = CCFOPT_MASK_BUILD_KW["min_good_pixels"],
    output_path: str | None = None,
    masks_directory: str | None = None,
    wave_um: np.ndarray | None = None,
    flux: np.ndarray | None = None,
) -> tuple[str, list[LineMaskEntry]]:
    """
    Refine a seed delta mask so each line maximizes single-line Gaussian CCF contrast at RV=0.

    For every seed center ``λ₀``, search trial centers
    ``λ = λ₀ (1 + δv/c)`` on the **template spectrum** (injected RV = 0) and keep the
    wavelength that maximizes ``|Gaussian contrast|`` from
    :func:`~hispecdrp.primitives.utils.ccf.ccf_fitting.calc_ccf_metrics`. Output
    weights are proportional to that contrast and are renormalized to sum to 1.

    Parameters
    ----------
    template_path
        Path to a BT-Settl ``.h5`` template (used for I/O naming / metadata; also loaded
        unless ``wave_um`` / ``flux`` are provided).
    seed_mask_path
        Path to an existing delta ``.mas`` seed (typically the packaged depth-weighted mask).
    wave_um, flux
        Optional in-memory vacuum-µm spectrum (for tests). When either is set, both must
        be set and ``template_path`` is used only for metadata / default output naming.
    output_path
        Full output path. Default: ``<stem>.ccfopt.delta.mas`` under ``masks_directory``.

    Returns
    -------
    path_written, entries
        Output ``.mas`` path and in-memory rows.
    """
    seed = MaskLines.load(seed_mask_path)
    if str(seed.mask_style) != MASK_STYLE_DELTA:
        raise ValueError(
            f"CCF-opt builder requires a delta seed mask; got style={seed.mask_style!r}"
        )
    seed_lam = np.asarray(seed.line_lambda_um, dtype=np.float64).ravel()
    seed_w = np.asarray(seed.weight, dtype=np.float64).ravel()
    keep_seed = np.isfinite(seed_lam) & (seed_lam > 0.0) & np.isfinite(seed_w) & (seed_w > 0.0)
    seed_lam = seed_lam[keep_seed]
    if seed_lam.size == 0:
        raise ValueError(f"No positive-weight seed lines in {seed_mask_path}")

    if (wave_um is None) ^ (flux is None):
        raise ValueError("wave_um and flux must both be provided or both omitted")
    if wave_um is None:
        wave_um, flux = load_template_from_path(
            template_path, wl_min_um=wl_min_um, wl_max_um=wl_max_um
        )
    else:
        wave_um = np.asarray(wave_um, dtype=np.float64).ravel()
        flux = np.asarray(flux, dtype=np.float64).ravel()

    ok = np.isfinite(wave_um) & np.isfinite(flux) & (wave_um > 0)
    wave_um = wave_um[ok]
    flux = flux[ok]
    order = np.argsort(wave_um)
    wave_um = wave_um[order]
    flux = flux[order]
    if wave_um.size < int(min_good_pixels):
        raise ValueError("Template spectrum too short after clipping")

    # Relative flux error proxy (template has no formal σ).
    med_abs = float(np.nanmedian(np.abs(flux)))
    err_floor = max(1.0e-6 * med_abs, 1.0e-30)
    err_all = np.maximum(np.abs(flux) * 0.01, err_floor).astype(np.float64)
    dq_all = np.zeros(flux.shape[0], dtype=np.int32)

    v_half = float(ccf_v_half_m_s)
    v_step = float(ccf_v_step_m_s)
    v_grid = np.arange(-v_half, v_half + 0.5 * v_step, v_step, dtype=np.float64)
    win_half = float(spectrum_window_half_um)

    out_lam: list[float] = []
    out_contrast: list[float] = []
    n_seed = int(seed_lam.size)
    n_drop = 0
    log_every = max(1, n_seed // 20)
    for i, lam0 in enumerate(seed_lam):
        if i % log_every == 0:
            logger.info(
                "CCF-opt mask: line %d/%d (λ₀=%.6f µm)",
                i + 1,
                n_seed,
                float(lam0),
            )
        lam0_f = float(lam0)
        if lam0_f < float(wl_min_um) or lam0_f > float(wl_max_um):
            n_drop += 1
            continue
        # Local spectrum window large enough for CCF ± search Doppler span.
        doppler_pad = (
            (v_half + float(search_half_m_s)) / C_M_S * lam0_f * 3.0
        )
        half = max(win_half, doppler_pad, 1.0e-6)
        mwin = (wave_um >= lam0_f - half) & (wave_um <= lam0_f + half)
        if int(np.count_nonzero(mwin)) < int(min_good_pixels):
            n_drop += 1
            continue
        w_loc = wave_um[mwin]
        f_loc = flux[mwin]
        e_loc = err_all[mwin]
        d_loc = dq_all[mwin]
        lam_best, con_best = _ccfopt_search_line_center_um(
            w_loc,
            f_loc,
            e_loc,
            d_loc,
            lam0_f,
            v_grid=v_grid,
            v_step_m_s=v_step,
            search_half_m_s=float(search_half_m_s),
            coarse_step_m_s=float(coarse_step_m_s),
            fine_half_m_s=float(fine_half_m_s),
            fine_step_m_s=float(fine_step_m_s),
            rv_gate_m_s=float(rv_gate_m_s),
        )
        if not np.isfinite(lam_best) or not np.isfinite(con_best) or con_best <= 0.0:
            n_drop += 1
            continue
        out_lam.append(float(lam_best))
        out_contrast.append(float(con_best))

    if not out_lam:
        raise ValueError(
            "CCF-opt builder kept no lines; relax rv_gate_m_s / search windows or check seed"
        )

    centers_um_arr = np.asarray(out_lam, dtype=np.float64)
    contrasts = np.asarray(out_contrast, dtype=np.float64)
    order_out = np.argsort(centers_um_arr)
    centers_um_arr = centers_um_arr[order_out]
    contrasts = contrasts[order_out]
    w_sum = float(np.sum(contrasts))
    if not np.isfinite(w_sum) or w_sum <= 0.0:
        raise ValueError("Cannot normalize CCF-opt weights: non-positive contrast sum")
    weights = contrasts / w_sum

    entries: list[LineMaskEntry] = [
        LineMaskEntry(wavelength_nm=float(centers_um_arr[i] * 1000.0), weight=float(weights[i]))
        for i in range(centers_um_arr.size)
    ]

    if masks_directory is None:
        masks_directory = str(package_data_masks_dir())
    os.makedirs(masks_directory, exist_ok=True)
    base = os.path.splitext(os.path.basename(template_path))[0]
    auto_name = f"{base}.ccfopt.{MASK_STYLE_DELTA}.mas"
    out = output_path if output_path is not None else os.path.join(masks_directory, auto_name)

    meta: dict[str, Any] = {
        "schema_version": 2,
        "mask_style": MASK_STYLE_DELTA,
        "generator": (
            "hispecdrp.utils.line_mask_utils.build_ccfopt_line_mask_from_template_and_seed"
        ),
        "template_path": os.path.abspath(template_path),
        "seed_mask_path": os.path.abspath(seed_mask_path),
        "wl_min_um": float(wl_min_um),
        "wl_max_um": float(wl_max_um),
        "spectrum_window_half_um": float(spectrum_window_half_um),
        "ccf_v_half_m_s": float(ccf_v_half_m_s),
        "ccf_v_step_m_s": float(ccf_v_step_m_s),
        "search_half_m_s": float(search_half_m_s),
        "coarse_step_m_s": float(coarse_step_m_s),
        "fine_half_m_s": float(fine_half_m_s),
        "fine_step_m_s": float(fine_step_m_s),
        "rv_gate_m_s": float(rv_gate_m_s),
        "injected_rv_m_s": 0.0,
        "mask_line_geometry": "delta_two_column",
        "mask_weight_definition": "abs_gaussian_ccf_contrast",
        "weights_sum_normalized_to": 1.0,
        "n_seed_lines": int(n_seed),
        "n_dropped_lines": int(n_drop),
        "n_lines": int(len(entries)),
        "disk_wavelength_columns": {
            "column0_line_center": "vacuum_nanometer",
            "column1_weight": "dimensionless",
        },
        "hispecdrp_wavelength_columns": {
            "line_center": masklines_wavelength_unit_meta(u.nm),
            "line_left_edge": masklines_wavelength_unit_meta(u.nm),
            "line_right_edge": masklines_wavelength_unit_meta(u.nm),
            "line_width": masklines_wavelength_unit_meta(u.nm),
        },
    }
    ml = MaskLines.from_delta_centers_um(centers_um_arr, weights, metadata=meta)
    path_written = ml.save(output_path=out)
    logger.info(
        "Wrote %d CCF-opt mask lines (%d dropped) to %s",
        len(entries),
        n_drop,
        path_written,
    )
    return path_written, entries


def build_ccfopt_line_mask_from_rvccf_table(
    seed_mask_path: str,
    rvccf_table,
    *,
    template_path: str | None = None,
    output_path: str | None = None,
    masks_directory: str | None = None,
    match_half_m_s: float = 5000.0,
    max_abs_rv_m_s: float = CCFOPT_MAX_ABS_RV_M_S,
    min_contrast: float = 0.0,
) -> tuple[str, list[LineMaskEntry]]:
    """
    Build a CCF-opt delta mask from a line-by-line stellar-RV ``RVCCF`` table.

    For each seed line, gather all ``RVCCF`` rows within ``match_half_m_s`` (any
    echelle order). Drop the seed when **any** matching row has
    ``|RV| > max_abs_rv_m_s`` (failed / aliased calibration in at least one order).
    Otherwise pick the nearest unused matching row, shift the mask center so that
    chunk's measured RV becomes 0:

    ``λ' = λ_seed × (1 + RV / c)``,

    and set the weight proportional to ``|Gaussian contrast|`` of that chunk's
    stored CCF. Weights are renormalized to sum to 1.

    Parameters
    ----------
    seed_mask_path
        Path to the seed ``*.delta.mas`` (typically the packaged depth-weighted mask).
    rvccf_table
        Astropy table from ``StellarRVPrimitive`` with ``window_line_by_line=True``
        (columns ``RV``, ``WAVELENGTH``, ``CCF``, ``V_MIN``, ``V_MAX``, ``V_STEP``).
    template_path
        Optional template path for metadata / default output naming.
    output_path, masks_directory
        Output ``.mas`` path (default ``<stem>.ccfopt.delta.mas`` under
        ``masks_directory``).
    match_half_m_s
        Maximum ``|Δv|`` between seed λ and chunk ``WAVELENGTH`` for a match.
    max_abs_rv_m_s
        Drop a seed line when any matching ``RVCCF`` row (across orders) has
        ``|RV|`` above this gate (failed / aliased per-line calibration).
    min_contrast
        Drop lines whose absolute Gaussian contrast is at or below this floor.
    """
    from hispecdrp.primitives.utils.ccf.ccf_fitting import calc_ccf_metrics
    from hispecdrp.primitives.utils.ccf.ccf_utils import (
        LINE_MASK_CCF_STORED_NEGATED_FROM_RAW,
    )

    seed = MaskLines.load(seed_mask_path)
    if str(seed.mask_style) != MASK_STYLE_DELTA:
        raise ValueError(
            f"CCF-opt RVCCF builder requires a delta seed mask; got style={seed.mask_style!r}"
        )
    seed_lam = np.asarray(seed.line_lambda_um, dtype=np.float64).ravel()
    seed_w = np.asarray(seed.weight, dtype=np.float64).ravel()
    keep_seed = np.isfinite(seed_lam) & (seed_lam > 0.0) & np.isfinite(seed_w) & (seed_w > 0.0)
    seed_lam = seed_lam[keep_seed]
    if seed_lam.size == 0:
        raise ValueError(f"No positive-weight seed lines in {seed_mask_path}")

    if rvccf_table is None or len(rvccf_table) == 0:
        raise ValueError("rvccf_table is empty")
    colnames = set(getattr(rvccf_table, "colnames", []) or [])
    for req in ("RV", "WAVELENGTH", "CCF", "V_MIN", "V_MAX", "V_STEP"):
        if req not in colnames:
            raise ValueError(f"rvccf_table missing required column {req!r}")

    chunk_wl = np.asarray(rvccf_table["WAVELENGTH"], dtype=np.float64).ravel()
    chunk_rv = np.asarray(rvccf_table["RV"], dtype=np.float64).ravel()
    n_chunk = int(chunk_wl.size)
    contrasts = np.full(n_chunk, np.nan, dtype=np.float64)
    for i in range(n_chunk):
        rv_i = float(chunk_rv[i])
        if not np.isfinite(rv_i):
            continue
        ccf = np.asarray(rvccf_table["CCF"][i], dtype=np.float64).ravel()
        if ccf.size < 5 or not np.any(np.isfinite(ccf)):
            continue
        v_min = float(rvccf_table["V_MIN"][i])
        v_max = float(rvccf_table["V_MAX"][i])
        v_step = float(rvccf_table["V_STEP"][i])
        if not (np.isfinite(v_min) and np.isfinite(v_max) and np.isfinite(v_step) and v_step > 0.0):
            continue
        v_grid = v_min + np.arange(ccf.size, dtype=np.float64) * v_step
        # Guard against stale V_MAX metadata vs actual CCF length.
        if abs(float(v_grid[-1]) - v_max) > 2.0 * v_step:
            v_grid = np.linspace(v_min, v_max, ccf.size, dtype=np.float64)
            v_step = float(v_grid[1] - v_grid[0]) if ccf.size > 1 else v_step
        metrics = calc_ccf_metrics(
            v_grid,
            ccf,
            None,
            pix_size=float(v_step),
            absorption=not LINE_MASK_CCF_STORED_NEGATED_FROM_RAW,
            calc_bis=False,
            mu_guess=rv_i if np.isfinite(rv_i) else 0.0,
        )
        con = float(metrics.contrast)
        if np.isfinite(con):
            contrasts[i] = abs(con)

    used_chunk = np.zeros(n_chunk, dtype=bool)
    out_lam: list[float] = []
    out_contrast: list[float] = []
    n_drop = 0
    match_half = float(match_half_m_s)
    for lam0 in seed_lam:
        lam0_f = float(lam0)
        # Match in velocity space relative to the seed center.
        dv = C_M_S * (chunk_wl - lam0_f) / lam0_f
        match = (
            np.isfinite(dv)
            & np.isfinite(chunk_rv)
            & np.isfinite(contrasts)
            & (np.abs(dv) <= match_half)
            & (contrasts > float(min_contrast))
        )
        if not np.any(match):
            n_drop += 1
            continue
        if float(np.nanmax(np.abs(chunk_rv[match]))) > float(max_abs_rv_m_s):
            n_drop += 1
            continue
        ok = match & (~used_chunk)
        if not np.any(ok):
            n_drop += 1
            continue
        idx_cand = np.flatnonzero(ok)
        j = int(idx_cand[np.argmin(np.abs(dv[idx_cand]))])
        used_chunk[j] = True
        rv_j = float(chunk_rv[j])
        lam_new = lam0_f * (1.0 + rv_j / C_M_S)
        if not np.isfinite(lam_new) or lam_new <= 0.0:
            n_drop += 1
            continue
        out_lam.append(float(lam_new))
        out_contrast.append(float(contrasts[j]))

    if not out_lam:
        raise ValueError(
            "CCF-opt RVCCF builder kept no lines; check seed/RVCCF match gates"
        )

    centers_um_arr = np.asarray(out_lam, dtype=np.float64)
    contrasts_a = np.asarray(out_contrast, dtype=np.float64)
    order_out = np.argsort(centers_um_arr)
    centers_um_arr = centers_um_arr[order_out]
    contrasts_a = contrasts_a[order_out]
    w_sum = float(np.sum(contrasts_a))
    if not np.isfinite(w_sum) or w_sum <= 0.0:
        raise ValueError("Cannot normalize CCF-opt weights: non-positive contrast sum")
    weights = contrasts_a / w_sum

    entries: list[LineMaskEntry] = [
        LineMaskEntry(wavelength_nm=float(centers_um_arr[i] * 1000.0), weight=float(weights[i]))
        for i in range(centers_um_arr.size)
    ]

    if masks_directory is None:
        masks_directory = str(package_data_masks_dir())
    os.makedirs(masks_directory, exist_ok=True)
    if template_path is not None:
        base = os.path.splitext(os.path.basename(template_path))[0]
    else:
        base = Path(seed_mask_path).name
        for suffix in (".ccfopt.delta.mas", ".delta.mas", ".mas"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
    auto_name = f"{base}.ccfopt.{MASK_STYLE_DELTA}.mas"
    out = output_path if output_path is not None else os.path.join(masks_directory, auto_name)

    meta: dict[str, Any] = {
        "schema_version": 2,
        "mask_style": MASK_STYLE_DELTA,
        "generator": (
            "hispecdrp.utils.line_mask_utils.build_ccfopt_line_mask_from_rvccf_table"
        ),
        "seed_mask_path": os.path.abspath(seed_mask_path),
        "template_path": (
            os.path.abspath(template_path) if template_path is not None else None
        ),
        "match_half_m_s": float(match_half_m_s),
        "max_abs_rv_m_s": float(max_abs_rv_m_s),
        "min_contrast": float(min_contrast),
        "injected_rv_m_s": 0.0,
        "mask_line_geometry": "delta_two_column",
        "mask_weight_definition": "abs_gaussian_ccf_contrast",
        "wavelength_offset_definition": "lambda_prime = lambda_seed * (1 + RV/c)",
        "weights_sum_normalized_to": 1.0,
        "n_seed_lines": int(seed_lam.size),
        "n_rvccf_rows": int(n_chunk),
        "n_dropped_lines": int(n_drop),
        "n_lines": int(len(entries)),
        "disk_wavelength_columns": {
            "column0_line_center": "vacuum_nanometer",
            "column1_weight": "dimensionless",
        },
        "hispecdrp_wavelength_columns": {
            "line_center": masklines_wavelength_unit_meta(u.nm),
            "line_left_edge": masklines_wavelength_unit_meta(u.nm),
            "line_right_edge": masklines_wavelength_unit_meta(u.nm),
            "line_width": masklines_wavelength_unit_meta(u.nm),
        },
    }
    ml = MaskLines.from_delta_centers_um(centers_um_arr, weights, metadata=meta)
    path_written = ml.save(output_path=out)
    logger.info(
        "Wrote %d CCF-opt mask lines from RVCCF (%d seed lines dropped) to %s",
        len(entries),
        n_drop,
        path_written,
    )
    return path_written, entries


def build_ccfopt_line_mask_from_l2_line_by_line_rv(
    seed_mask_path: str,
    input_model,
    *,
    fiber: str | int = 1,
    template_path: str | None = None,
    output_path: str | None = None,
    masks_directory: str | None = None,
    ccf_input_velocity_m_s: float = 0.0,
    outlier_sigma: float = 1.0e6,
    use_single_line_builder: bool = False,
    match_half_m_s: float = 5000.0,
    max_abs_rv_m_s: float = CCFOPT_MAX_ABS_RV_M_S,
    min_contrast: float = 0.0,
    window_size_um: float = CCFOPT_SINGLE_LINE_STELLAR_RV_KW["window_size_um"],
    window_line_by_line: bool = CCFOPT_SINGLE_LINE_STELLAR_RV_KW["window_line_by_line"],
    v_half_window_m_s: float = CCFOPT_SINGLE_LINE_STELLAR_RV_KW["v_half_window_m_s"],
    v_step_m_s: float = CCFOPT_SINGLE_LINE_STELLAR_RV_KW["v_step_m_s"],
    mask_ccf_pixel_interp_order: int = CCFOPT_SINGLE_LINE_STELLAR_RV_KW[
        "mask_ccf_pixel_interp_order"
    ],
    zb: float = 0.0,
    max_seed_lines: int | None = None,
) -> tuple[str, list[LineMaskEntry]]:
    """
    Build a CCF-opt delta mask from an injected-RV=0 L2 via line-by-line stellar RV.

    Two-step workflow (default ``use_single_line_builder=False``):

    1. Run ``StellarRVPrimitive(algorithm="ccf_line_mask")`` on ``input_model`` with
       the seed mask and **line-by-line** windowing at ``ccf_input_velocity_m_s``
       (typically 0). Set ``outlier_sigma`` very large (default ``1e6``) so chunk
       outlier rejection does not drop mask lines before aggregation.
    2. Pass the per-line ``RVCCF`` table to
       :func:`build_ccfopt_line_mask_from_rvccf_table`, which drops seed lines when
       any matching per-order calibration RV exceeds ``max_abs_rv_m_s`` (default
       :data:`CCFOPT_MAX_ABS_RV_M_S`), then shifts each kept line
       ``λ' = λ × (1 + RV/c)`` and sets weights proportional to ``|Gaussian contrast|``.

    When ``use_single_line_builder=True``, delegate to the slower per-line CCF path
    :func:`build_ccfopt_line_mask_from_l2_single_line_ccf` (gold-standard).
    """
    if use_single_line_builder:
        return build_ccfopt_line_mask_from_l2_single_line_ccf(
            seed_mask_path,
            input_model,
            fiber=fiber,
            template_path=template_path,
            output_path=output_path,
            masks_directory=masks_directory,
            window_size_um=window_size_um,
            window_line_by_line=window_line_by_line,
            v_half_window_m_s=v_half_window_m_s,
            v_step_m_s=v_step_m_s,
            ccf_input_velocity_m_s=ccf_input_velocity_m_s,
            mask_ccf_pixel_interp_order=mask_ccf_pixel_interp_order,
            max_abs_rv_m_s=max_abs_rv_m_s,
            min_contrast=min_contrast,
            zb=zb,
            max_seed_lines=max_seed_lines,
        )

    from hispecdrp.primitives import StellarRVPrimitive

    fiber_s = str(fiber)
    ost = int(getattr(input_model.meta, "order_start", 0))
    oen = int(getattr(input_model.meta, "order_end", ost))
    do_orders = list(range(ost, oen + 1))

    out = StellarRVPrimitive.call(
        input_model,
        algorithm="ccf_line_mask",
        mask_path=str(seed_mask_path),
        ccf_input_velocity=float(ccf_input_velocity_m_s),
        do_fibers=[fiber_s],
        do_orders=do_orders,
        window_line_by_line=bool(window_line_by_line),
        window_size_um=float(window_size_um),
        v_half_window=float(v_half_window_m_s),
        v_step_coarse=float(v_step_m_s),
        mask_ccf_pixel_interp_order=int(mask_ccf_pixel_interp_order),
        ccf_zb=float(zb),
        outlier_sigma=float(outlier_sigma),
        save_result=False,
        save_preview=False,
        show_preview=False,
    )
    tbl = getattr(out, f"rvccf_fib{fiber_s}", None)
    if tbl is None or len(tbl) == 0:
        raise ValueError(
            "line-by-line StellarRV produced empty RVCCF; check seed mask and L2"
        )
    return build_ccfopt_line_mask_from_rvccf_table(
        seed_mask_path,
        tbl,
        template_path=template_path,
        output_path=output_path,
        masks_directory=masks_directory,
        match_half_m_s=match_half_m_s,
        max_abs_rv_m_s=max_abs_rv_m_s,
        min_contrast=min_contrast,
    )


def build_ccfopt_line_mask_from_l2_single_line_ccf(
    seed_mask_path: str,
    input_model,
    *,
    fiber: str | int = 2,
    template_path: str | None = None,
    output_path: str | None = None,
    masks_directory: str | None = None,
    window_size_um: float = CCFOPT_SINGLE_LINE_STELLAR_RV_KW["window_size_um"],
    window_line_by_line: bool = CCFOPT_SINGLE_LINE_STELLAR_RV_KW["window_line_by_line"],
    v_half_window_m_s: float = CCFOPT_SINGLE_LINE_STELLAR_RV_KW["v_half_window_m_s"],
    v_step_m_s: float = CCFOPT_SINGLE_LINE_STELLAR_RV_KW["v_step_m_s"],
    ccf_input_velocity_m_s: float = CCFOPT_SINGLE_LINE_STELLAR_RV_KW[
        "ccf_input_velocity_m_s"
    ],
    mask_ccf_pixel_interp_order: int = CCFOPT_SINGLE_LINE_STELLAR_RV_KW[
        "mask_ccf_pixel_interp_order"
    ],
    ccf_min_good_pixels: int = CCFOPT_SINGLE_LINE_STELLAR_RV_KW["ccf_min_good_pixels"],
    max_abs_rv_m_s: float = CCFOPT_SINGLE_LINE_STELLAR_RV_KW["max_abs_rv_m_s"],
    min_contrast: float = CCFOPT_SINGLE_LINE_STELLAR_RV_KW["min_contrast"],
    rezero_verify_abs_rv_m_s: float | None = CCFOPT_SINGLE_LINE_STELLAR_RV_KW[
        "rezero_verify_abs_rv_m_s"
    ],
    max_rv_err_m_s: float | None = CCFOPT_SINGLE_LINE_STELLAR_RV_KW["max_rv_err_m_s"],
    zb: float = 0.0,
    max_seed_lines: int | None = None,
) -> tuple[str, list[LineMaskEntry]]:
    """
    Build a static rest-frame CCF-opt delta mask by zeroing single-line CCF bias.

    For each seed line, run the same line-mask CCF path as
    ``StellarRVPrimitive(algorithm="ccf_line_mask")`` with **line-by-line**
    settings (:data:`CCFOPT_SINGLE_LINE_STELLAR_RV_KW`, matching
    ``rv_recovery_sim.ipynb``) but with a mask containing **only that line**.
    Shift the line center so the recovered RV equals the injected prior
    (``ccf_input_velocity_m_s``, typically 0):

    ``λ' = λ_seed × (1 + (RV − ccf_input_velocity) / c)``

    The output ``.mas`` is a fixed rest-frame list (does not vary with bulk RV).
    Weights are proportional to ``|Gaussian CCF contrast|`` at the measured
    (pre-shift) CCF and are renormalized to sum to 1.

    Lines are dropped when:

    - no valid chunk RV, ``|RV|`` above ``max_abs_rv_m_s``, or contrast at/below
      ``min_contrast``;
    - formal RV uncertainty ``≥ max_rv_err_m_s`` (default 10 m/s);
    - after the λ shift, re-measured ``|RV − center| ≥ rezero_verify_abs_rv_m_s``
      (default 0.001 m/s = 0.1 cm/s), or the re-measured uncertainty fails the
      same ``max_rv_err_m_s`` gate.

    Parameters
    ----------
    seed_mask_path
        Packaged ``*.delta.mas`` seed.
    input_model
        Blaze-corrected L2-like model (``WAVE_FIB*`` / ``SPEC_FIB*``).
    fiber
        Fiber label matching the L2 attributes (default ``2``).
    """
    from hispecdrp.primitives.utils.ccf.ccf_common import resolve_ccf_broad_grid_m_s
    from hispecdrp.primitives.utils.ccf.ccf_fitting import calc_ccf_metrics
    from hispecdrp.primitives.utils.ccf.ccf_utils import (
        LINE_MASK_CCF_STORED_NEGATED_FROM_RAW,
        run_ccf_line_mask_for_fiber,
    )

    seed = MaskLines.load(seed_mask_path)
    if str(seed.mask_style) != MASK_STYLE_DELTA:
        raise ValueError(
            f"CCF-opt single-line builder requires a delta seed; got style={seed.mask_style!r}"
        )
    seed_lam = np.asarray(seed.line_lambda_um, dtype=np.float64).ravel()
    seed_w = np.asarray(seed.weight, dtype=np.float64).ravel()
    keep_seed = (
        np.isfinite(seed_lam)
        & (seed_lam > 0.0)
        & np.isfinite(seed_w)
        & (seed_w > 0.0)
    )
    seed_lam = seed_lam[keep_seed]
    if seed_lam.size == 0:
        raise ValueError(f"No positive-weight seed lines in {seed_mask_path}")
    if max_seed_lines is not None:
        seed_lam = seed_lam[: int(max_seed_lines)]

    fiber_s = str(fiber)
    order_vec = list(input_model.make_order_vec())
    do_orders = list(order_vec)
    vm, vx, vs = resolve_ccf_broad_grid_m_s(None, None, float(v_step_m_s))
    v_step = float(vs)
    center = float(ccf_input_velocity_m_s)
    half = float(v_half_window_m_s)

    out_lam: list[float] = []
    out_contrast: list[float] = []
    out_rv: list[float] = []
    n_seed = int(seed_lam.size)
    n_drop = 0
    log_every = max(1, n_seed // 20)

    for i, lam0 in enumerate(seed_lam):
        if i % log_every == 0:
            logger.info(
                "CCF-opt single-line: %d/%d (λ₀=%.6f µm)",
                i + 1,
                n_seed,
                float(lam0),
            )
        lam0_f = float(lam0)
        _wl, rvs, errs, _ords, _prev, supp = run_ccf_line_mask_for_fiber(
            input_model=input_model,
            fiber=fiber_s,
            do_orders=do_orders,
            order_vec=order_vec,
            window_size_um=float(window_size_um),
            mask_path=None,
            mask_wavelength=np.asarray([lam0_f], dtype=np.float64),
            mask_weight=np.asarray([1.0], dtype=np.float64),
            v_min_m_s=float(vm),
            v_max_m_s=float(vx),
            v_step_m_s=v_step,
            ccf_center_m_s=center,
            ccf_half_window_m_s=half,
            zb=float(zb),
            want_preview=False,
            min_good_pixels=int(ccf_min_good_pixels),
            mask_ccf_pixel_interp_order=int(mask_ccf_pixel_interp_order),
            window_line_by_line=bool(window_line_by_line),
        )
        rvs_a = np.asarray(rvs, dtype=np.float64).ravel()
        errs_a = np.asarray(errs, dtype=np.float64).ravel()
        valid = np.isfinite(rvs_a) & np.isfinite(errs_a) & (errs_a > 0.0)
        if not np.any(valid):
            n_drop += 1
            continue
        # Single-line masks typically yield one chunk; use inverse-variance mean.
        w_inv = 1.0 / np.maximum(errs_a[valid], 1.0e-30) ** 2
        rv = float(np.average(rvs_a[valid], weights=w_inv))
        rv_err = float(np.sqrt(1.0 / np.sum(w_inv)))
        if not np.isfinite(rv) or abs(rv) > float(max_abs_rv_m_s):
            n_drop += 1
            continue
        if (
            max_rv_err_m_s is not None
            and np.isfinite(float(max_rv_err_m_s))
            and float(max_rv_err_m_s) > 0.0
            and (not np.isfinite(rv_err) or rv_err >= float(max_rv_err_m_s))
        ):
            n_drop += 1
            continue

        idx = int(np.flatnonzero(valid)[0])
        ccf = np.asarray(supp["CCF"][idx], dtype=np.float64).ravel()
        if ccf.size < 5 or not np.any(np.isfinite(ccf)):
            n_drop += 1
            continue
        v_min = float(supp["V_MIN"][idx])
        v_step_i = float(supp["V_STEP"][idx])
        if not (np.isfinite(v_min) and np.isfinite(v_step_i) and v_step_i > 0.0):
            n_drop += 1
            continue
        v_grid = v_min + np.arange(ccf.size, dtype=np.float64) * v_step_i
        metrics = calc_ccf_metrics(
            v_grid,
            ccf,
            None,
            pix_size=float(v_step_i),
            absorption=not LINE_MASK_CCF_STORED_NEGATED_FROM_RAW,
            calc_bis=False,
            mu_guess=rv,
        )
        contrast = abs(float(metrics.contrast))
        if not np.isfinite(contrast) or contrast <= float(min_contrast):
            n_drop += 1
            continue

        # Zero the measured bias relative to the injected CCF center.
        lam_new = lam0_f * (1.0 + (rv - center) / C_M_S)
        if not np.isfinite(lam_new) or lam_new <= 0.0:
            n_drop += 1
            continue

        # Require post-shift re-zero (default 0.1 cm/s) and uncertainty gate.
        need_rezero = (
            rezero_verify_abs_rv_m_s is not None
            and np.isfinite(float(rezero_verify_abs_rv_m_s))
            and float(rezero_verify_abs_rv_m_s) > 0.0
        )
        need_err_gate = (
            max_rv_err_m_s is not None
            and np.isfinite(float(max_rv_err_m_s))
            and float(max_rv_err_m_s) > 0.0
        )
        if need_rezero or need_err_gate:
            _wl2, rvs2, errs2, _o2, _p2, _s2 = run_ccf_line_mask_for_fiber(
                input_model=input_model,
                fiber=fiber_s,
                do_orders=do_orders,
                order_vec=order_vec,
                window_size_um=float(window_size_um),
                mask_path=None,
                mask_wavelength=np.asarray([lam_new], dtype=np.float64),
                mask_weight=np.asarray([1.0], dtype=np.float64),
                v_min_m_s=float(vm),
                v_max_m_s=float(vx),
                v_step_m_s=v_step,
                ccf_center_m_s=center,
                ccf_half_window_m_s=half,
                zb=float(zb),
                want_preview=False,
                min_good_pixels=int(ccf_min_good_pixels),
                mask_ccf_pixel_interp_order=int(mask_ccf_pixel_interp_order),
                window_line_by_line=bool(window_line_by_line),
            )
            rvs2_a = np.asarray(rvs2, dtype=np.float64).ravel()
            errs2_a = np.asarray(errs2, dtype=np.float64).ravel()
            valid2 = np.isfinite(rvs2_a) & np.isfinite(errs2_a) & (errs2_a > 0.0)
            if not np.any(valid2):
                n_drop += 1
                continue
            w2 = 1.0 / np.maximum(errs2_a[valid2], 1.0e-30) ** 2
            rv_chk = float(np.average(rvs2_a[valid2], weights=w2))
            rv_err_chk = float(np.sqrt(1.0 / np.sum(w2)))
            if need_rezero and (
                not np.isfinite(rv_chk)
                or abs(rv_chk - center) >= float(rezero_verify_abs_rv_m_s)
            ):
                n_drop += 1
                continue
            if need_err_gate and (
                not np.isfinite(rv_err_chk) or rv_err_chk >= float(max_rv_err_m_s)
            ):
                n_drop += 1
                continue

        out_lam.append(float(lam_new))
        out_contrast.append(float(contrast))
        out_rv.append(float(rv))

    if not out_lam:
        raise ValueError(
            "CCF-opt single-line builder kept no lines; check L2 coverage / seed / gates"
        )

    centers_um_arr = np.asarray(out_lam, dtype=np.float64)
    contrasts_a = np.asarray(out_contrast, dtype=np.float64)
    rvs_kept = np.asarray(out_rv, dtype=np.float64)
    order_out = np.argsort(centers_um_arr)
    centers_um_arr = centers_um_arr[order_out]
    contrasts_a = contrasts_a[order_out]
    rvs_kept = rvs_kept[order_out]
    n_after_single = int(centers_um_arr.size)

    # Full-mask line-by-line pass: drop lines that fail the same re-zero / err
    # gates when all surviving lines are used together (catches residual aliases).
    n_fullmask_drop = 0
    if (
        bool(window_line_by_line)
        and centers_um_arr.size > 0
        and (
            (
                rezero_verify_abs_rv_m_s is not None
                and np.isfinite(float(rezero_verify_abs_rv_m_s))
                and float(rezero_verify_abs_rv_m_s) > 0.0
            )
            or (
                max_rv_err_m_s is not None
                and np.isfinite(float(max_rv_err_m_s))
                and float(max_rv_err_m_s) > 0.0
            )
        )
    ):
        w_tmp = contrasts_a / float(np.sum(contrasts_a))
        logger.info(
            "CCF-opt: full-mask quality filter on %d lines (rezero=%s m/s, max_err=%s m/s)",
            n_after_single,
            rezero_verify_abs_rv_m_s,
            max_rv_err_m_s,
        )
        wl_f, rvs_f, errs_f, _o_f, _p_f, _s_f = run_ccf_line_mask_for_fiber(
            input_model=input_model,
            fiber=fiber_s,
            do_orders=do_orders,
            order_vec=order_vec,
            window_size_um=float(window_size_um),
            mask_path=None,
            mask_wavelength=centers_um_arr,
            mask_weight=w_tmp,
            v_min_m_s=float(vm),
            v_max_m_s=float(vx),
            v_step_m_s=v_step,
            ccf_center_m_s=center,
            ccf_half_window_m_s=half,
            zb=float(zb),
            want_preview=False,
            min_good_pixels=int(ccf_min_good_pixels),
            mask_ccf_pixel_interp_order=int(mask_ccf_pixel_interp_order),
            window_line_by_line=True,
        )
        wl_f = np.asarray(wl_f, dtype=np.float64).ravel()
        rvs_f = np.asarray(rvs_f, dtype=np.float64).ravel()
        errs_f = np.asarray(errs_f, dtype=np.float64).ravel()
        keep_full = np.zeros(centers_um_arr.size, dtype=bool)
        for i_line, lc in enumerate(centers_um_arr):
            dv = C_M_S * (wl_f - float(lc)) / max(float(lc), 1.0e-30)
            j = int(np.nanargmin(np.abs(dv))) if dv.size else -1
            if j < 0 or not np.isfinite(dv[j]) or abs(float(dv[j])) > 500.0:
                n_fullmask_drop += 1
                continue
            rv_j = float(rvs_f[j])
            err_j = float(errs_f[j])
            if (
                rezero_verify_abs_rv_m_s is not None
                and np.isfinite(float(rezero_verify_abs_rv_m_s))
                and float(rezero_verify_abs_rv_m_s) > 0.0
                and (not np.isfinite(rv_j) or abs(rv_j - center) >= float(rezero_verify_abs_rv_m_s))
            ):
                n_fullmask_drop += 1
                continue
            if (
                max_rv_err_m_s is not None
                and np.isfinite(float(max_rv_err_m_s))
                and float(max_rv_err_m_s) > 0.0
                and (not np.isfinite(err_j) or err_j <= 0.0 or err_j >= float(max_rv_err_m_s))
            ):
                n_fullmask_drop += 1
                continue
            keep_full[i_line] = True
        if not np.any(keep_full):
            raise ValueError(
                "CCF-opt full-mask filter kept no lines; relax rezero/err gates"
            )
        centers_um_arr = centers_um_arr[keep_full]
        contrasts_a = contrasts_a[keep_full]
        rvs_kept = rvs_kept[keep_full]
        n_drop += int(n_fullmask_drop)

    w_sum = float(np.sum(contrasts_a))
    if not np.isfinite(w_sum) or w_sum <= 0.0:
        raise ValueError("Cannot normalize CCF-opt weights: non-positive contrast sum")
    weights = contrasts_a / w_sum

    entries: list[LineMaskEntry] = [
        LineMaskEntry(
            wavelength_nm=float(centers_um_arr[i] * 1000.0),
            weight=float(weights[i]),
        )
        for i in range(centers_um_arr.size)
    ]

    if masks_directory is None:
        masks_directory = str(package_data_masks_dir())
    os.makedirs(masks_directory, exist_ok=True)
    if template_path is not None:
        base = os.path.splitext(os.path.basename(template_path))[0]
    else:
        base = Path(seed_mask_path).name
        for suffix in (".ccfopt.delta.mas", ".delta.mas", ".mas"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
    auto_name = f"{base}.ccfopt.{MASK_STYLE_DELTA}.mas"
    out = output_path if output_path is not None else os.path.join(masks_directory, auto_name)

    meta: dict[str, Any] = {
        "schema_version": 2,
        "mask_style": MASK_STYLE_DELTA,
        "generator": (
            "hispecdrp.utils.line_mask_utils.build_ccfopt_line_mask_from_l2_single_line_ccf"
        ),
        "seed_mask_path": os.path.abspath(seed_mask_path),
        "template_path": (
            os.path.abspath(template_path) if template_path is not None else None
        ),
        "fiber": fiber_s,
        "window_size_um": float(window_size_um),
        "window_line_by_line": bool(window_line_by_line),
        "v_half_window_m_s": float(v_half_window_m_s),
        "v_step_m_s": float(v_step_m_s),
        "ccf_input_velocity_m_s": float(ccf_input_velocity_m_s),
        "mask_ccf_pixel_interp_order": int(mask_ccf_pixel_interp_order),
        "ccf_min_good_pixels": int(ccf_min_good_pixels),
        "max_abs_rv_m_s": float(max_abs_rv_m_s),
        "min_contrast": float(min_contrast),
        "rezero_verify_abs_rv_m_s": (
            float(rezero_verify_abs_rv_m_s)
            if rezero_verify_abs_rv_m_s is not None
            else None
        ),
        "max_rv_err_m_s": (
            float(max_rv_err_m_s) if max_rv_err_m_s is not None else None
        ),
        "zb": float(zb),
        "injected_rv_m_s": float(ccf_input_velocity_m_s),
        "mask_line_geometry": "delta_two_column",
        "mask_weight_definition": "abs_gaussian_ccf_contrast",
        "wavelength_offset_definition": (
            "lambda_prime = lambda_seed * (1 + (RV - ccf_input_velocity) / c); "
            "single-line line-by-line CCF, then full-mask rezero/err filter"
        ),
        "weights_sum_normalized_to": 1.0,
        "n_seed_lines": int(n_seed),
        "n_dropped_lines": int(n_drop),
        "n_lines_after_single_line_pass": int(n_after_single),
        "n_dropped_fullmask_filter": int(n_fullmask_drop),
        "n_lines": int(len(entries)),
        "seed_single_line_rv_median_m_s": float(np.nanmedian(rvs_kept)),
        "seed_single_line_rv_mad_m_s": float(
            np.nanmedian(np.abs(rvs_kept - np.nanmedian(rvs_kept)))
        ),
        "disk_wavelength_columns": {
            "column0_line_center": "vacuum_nanometer",
            "column1_weight": "dimensionless",
        },
        "hispecdrp_wavelength_columns": {
            "line_center": masklines_wavelength_unit_meta(u.nm),
            "line_left_edge": masklines_wavelength_unit_meta(u.nm),
            "line_right_edge": masklines_wavelength_unit_meta(u.nm),
            "line_width": masklines_wavelength_unit_meta(u.nm),
        },
    }
    ml = MaskLines.from_delta_centers_um(centers_um_arr, weights, metadata=meta)
    path_written = ml.save(output_path=out)
    logger.info(
        "Wrote %d CCF-opt single-line mask lines (%d seed dropped) to %s",
        len(entries),
        n_drop,
        path_written,
    )
    return path_written, entries


def build_e2erv_line_mask_from_template(
    template_path: str,
    *,
    wl_min_um: float = 0.8,
    wl_max_um: float = 3.0,
    min_continuum_normalized_depth: float = E2ERV_MASK_BUILD_KW[
        "min_continuum_normalized_depth"
    ],
    deblend_sep_resels: float = E2ERV_MASK_BUILD_KW["deblend_sep_resels"],
    resel_um: float = E2ERV_MASK_BUILD_KW["resel_um"],
    continuum_filter_size: int = E2ERV_MASK_BUILD_KW["continuum_filter_size"],
    min_continuum_normalized_flux_floor: float | None = E2ERV_MASK_BUILD_KW[
        "min_continuum_normalized_flux_floor"
    ],
    output_path: str | None = None,
    masks_directory: str | None = None,
) -> tuple[str, list[LineMaskEntry]]:
    """
    Build a delta line mask aligned with e2e WAVE / RV truth (template centers).

    Uses parabolic absorption centroids (same family as the sim-HDF line WAVE
    correction), a resolution-element deblend that keeps the deepest line in each
    blend, and continuum-normalized depth weights (sum → 1). No CCF-opt λ shifts —
    centers stay at template truth so CCF at injected RV=0 is unbiased when WAVE
    labels apparent line centers.

    Default output: ``<template_stem>.e2erv.delta.mas``.
    """
    wave_um, flux = load_template_from_path(
        template_path, wl_min_um=wl_min_um, wl_max_um=wl_max_um
    )
    ok = np.isfinite(wave_um) & np.isfinite(flux) & (wave_um > 0)
    wave_um = np.asarray(wave_um, dtype=np.float64)[ok]
    flux = np.asarray(flux, dtype=np.float64)[ok]
    order = np.argsort(wave_um)
    wave_um, flux = wave_um[order], flux[order]
    if wave_um.size < 16:
        raise ValueError("Template too short after clipping")

    cont = _continuum_envelope(flux, int(continuum_filter_size))
    centers = detect_template_line_centers_parabolic_um(
        wave_um,
        flux,
        min_continuum_normalized_depth=float(min_continuum_normalized_depth),
    )
    if centers.size == 0:
        raise ValueError("No parabolic line centers for e2erv mask")

    depths = 1.0 - np.interp(centers, wave_um, flux) / np.maximum(
        np.interp(centers, wave_um, cont), 1e-30
    )
    fn = np.interp(centers, wave_um, flux) / np.maximum(
        np.interp(centers, wave_um, cont), 1e-30
    )
    keep = np.isfinite(depths) & (depths >= float(min_continuum_normalized_depth))
    if min_continuum_normalized_flux_floor is not None:
        keep &= fn >= float(min_continuum_normalized_flux_floor)
    centers = centers[keep]
    depths = depths[keep]
    if centers.size == 0:
        raise ValueError("All e2erv candidates failed depth/floor cuts")

    # Deblend: merge radius = deblend_sep_resels * resel (nm on each side → sum).
    merge_r_nm = 0.5 * float(deblend_sep_resels) * float(resel_um) * 1000.0
    centers_nm = centers * 1000.0
    merge_radii = np.full(centers_nm.shape, merge_r_nm, dtype=np.float64)
    keep_idx = _cluster_max_depth(centers_nm, merge_radii, depths)
    centers = centers[keep_idx]
    depths = depths[keep_idx]
    weights = depths / float(np.sum(depths))

    if masks_directory is None:
        masks_directory = str(package_data_masks_dir())
    os.makedirs(masks_directory, exist_ok=True)
    base = os.path.splitext(os.path.basename(template_path))[0]
    out = (
        output_path
        if output_path is not None
        else os.path.join(masks_directory, f"{base}.e2erv.delta.mas")
    )
    meta: dict[str, Any] = {
        "schema_version": 2,
        "mask_style": MASK_STYLE_DELTA,
        "generator": "hispecdrp.utils.line_mask_utils.build_e2erv_line_mask_from_template",
        "template_path": os.path.abspath(template_path),
        "wl_min_um": float(wl_min_um),
        "wl_max_um": float(wl_max_um),
        "min_continuum_normalized_depth": float(min_continuum_normalized_depth),
        "deblend_sep_resels": float(deblend_sep_resels),
        "resel_um": float(resel_um),
        "deblend_merge_radius_nm": float(merge_r_nm),
        "centroid_method": "parabolic",
        "mask_weight_definition": "continuum_normalized_depth_1_minus_F_over_Fc",
        "weights_sum_normalized_to": 1.0,
        "n_lines": int(centers.size),
        "purpose": "e2e_rv_aligned_with_sim_hdf_line_wave",
    }
    ml = MaskLines.from_delta_centers_um(centers, weights, metadata=meta)
    path_written = ml.save(output_path=out)
    entries = [
        LineMaskEntry(wavelength_nm=float(c * 1000.0), weight=float(w))
        for c, w in zip(centers, weights)
    ]
    logger.info("Wrote %d e2erv mask lines to %s", len(entries), path_written)
    return path_written, entries
