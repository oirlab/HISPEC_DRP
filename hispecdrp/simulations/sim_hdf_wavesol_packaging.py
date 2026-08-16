"""
Packaged deposit-inverse + absorption-line-ZP WAVESOL for sim-HDF e2e runs.

The geometric HDF WAVE matches paint centers, but ZEMAX PSF stamps shift apparent
absorption minima by ~50–60 m/s.  :func:`refine_sim_hdf_wavesol_line_zeropoint`
measures that shift on a reference rate map and rescales ``WAVE_FIB`` so labeled
λ matches line centers. Optional L2-spectrum poly + CCF-order refine (see the
regenerator ``--spectrum`` / ``--ccf-refine`` flags) folds residual
order-dependent CCF systematics into WAVE for sub-10 cm/s e2e RV.

Building that cal once from an e2e rate map and shipping it under
``hispecdrp/data/wavesol/`` lets other users skip the runtime ZP step while
recovering the same bias-free WAVE.

Regenerate (from repo root)::

    python hispecdrp/data/wavesol/regenerate_sim_hdf_line_zp_wavesol.py \\
        --rate-map test_data/e2e_rv0_hdf_trace_linezp/rate_map.fits \\
        --spectrum 'test_data/e2e_rv0_hdf_trace_linezp/*_spec-blaze_correct.fits' \\
        --mask 'hispecdrp/data/masks/lte049*.ccfopt.delta.mas' \\
        --ccf-refine
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .. import datamodels
from ..datamodels.lfcwavesol import LFCWavesolModel

logger = logging.getLogger(__name__)

__all__ = [
    "PACKAGED_SIM_HDF_LINE_ZP_NOTE",
    "packaged_sim_hdf_wavesol_dir",
    "packaged_sim_hdf_wavesol_basename",
    "packaged_sim_hdf_wavesol_path",
    "packaged_sim_hdf_wavesol_meta_path",
    "is_packaged_sim_hdf_line_zp_record",
    "build_packaged_sim_hdf_line_zp_wavesol",
    "save_packaged_sim_hdf_line_zp_wavesol",
    "load_packaged_sim_hdf_line_zp_wavesol",
]

# Persisted on ``meta.note`` / caldb ``note`` so Stage-1 WavelengthSolutionSelector
# prefers this product over a competing geometric ``lfc_wavesol`` for the same night.
PACKAGED_SIM_HDF_LINE_ZP_NOTE = "sim_hdf_wavesol.line_zp"


def is_packaged_sim_hdf_line_zp_record(record: dict | None) -> bool:
    """True when a caldb / to_record dict is the packaged sim-HDF line-ZP WAVE."""
    if not record:
        return False
    note = str(record.get("note") or "")
    filename = str(record.get("filename") or "")
    return (
        PACKAGED_SIM_HDF_LINE_ZP_NOTE in note
        or "sim_hdf_wavesol.line_zp" in filename
    )


def packaged_sim_hdf_wavesol_dir() -> Path:
    """Directory for packaged sim-HDF line-ZP wavesol FITS (+ JSON sidecars)."""
    import hispecdrp

    return Path(hispecdrp.__file__).resolve().parent / "data" / "wavesol"


def packaged_sim_hdf_wavesol_basename(spectrograph: str) -> str:
    spec = spectrograph.upper()
    prefix = "HB" if spec == "BSPEC" else "HR" if spec == "RSPEC" else spec
    return f"{prefix}.sim_hdf_wavesol.line_zp.fits"


def packaged_sim_hdf_wavesol_path(spectrograph: str) -> Path:
    return packaged_sim_hdf_wavesol_dir() / packaged_sim_hdf_wavesol_basename(spectrograph)


def packaged_sim_hdf_wavesol_meta_path(spectrograph: str) -> Path:
    return packaged_sim_hdf_wavesol_path(spectrograph).with_suffix(".json")


def build_packaged_sim_hdf_line_zp_wavesol(
    spectrograph: str,
    rate_map: np.ndarray,
    *,
    known_rv_m_s: float = 0.0,
    fibers: list[int] | tuple[int, ...] | None = None,
    meta: dict[str, Any] | None = None,
    poly_deg: int = 2,
    n_lines_per_order: int = 40,
) -> tuple[LFCWavesolModel, dict[str, Any]]:
    """
    Deposit-inverse HDF WAVE + absorption-line correction measured on ``rate_map``.

    Default ``poly_deg=2`` applies a per-order pixel-dependent Doppler scale so
    labeled λ matches apparent line centers along the order (not only a constant ZP).

    Returns the mutated :class:`~hispecdrp.datamodels.lfcwavesol.LFCWavesolModel`
    and the info dict from
    :func:`~hispecdrp.simulations.e2e_pipeline.refine_sim_hdf_wavesol_line_zeropoint`.
    """
    from .e2e_pipeline import (
        create_wavelength_solution_model_from_sim_hdf,
        refine_sim_hdf_wavesol_line_zeropoint,
    )

    spectrograph = spectrograph.upper()
    wavesol = create_wavelength_solution_model_from_sim_hdf(
        spectrograph, meta=meta
    )
    zp_info = refine_sim_hdf_wavesol_line_zeropoint(
        wavesol,
        np.asarray(rate_map, dtype=np.float64),
        spectrograph=spectrograph,
        fibers=fibers,
        known_rv_m_s=float(known_rv_m_s),
        poly_deg=int(poly_deg),
        n_lines_per_order=int(n_lines_per_order),
    )
    return wavesol, zp_info


def save_packaged_sim_hdf_line_zp_wavesol(
    wavesol: LFCWavesolModel,
    spectrograph: str,
    *,
    zp_info: dict[str, Any] | None = None,
    rate_map_path: str | Path | None = None,
    overwrite: bool = True,
) -> Path:
    """Write packaged FITS (+ JSON provenance) under ``hispecdrp/data/wavesol/``."""
    out_dir = packaged_sim_hdf_wavesol_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = packaged_sim_hdf_wavesol_path(spectrograph)
    if out_path.is_file() and not overwrite:
        raise FileExistsError(out_path)
    wavesol.save(str(out_path))
    meta = {
        "spectrograph": spectrograph.upper(),
        "wavesol_path": str(out_path),
        "method": "deposit_inverse_hdf + absorption_line_zeropoint",
        "known_rv_m_s": None if zp_info is None else zp_info.get("known_rv_m_s"),
        "median_dv_m_s": None if zp_info is None else zp_info.get("median_dv_m_s"),
        "rate_map_path": None if rate_map_path is None else str(rate_map_path),
        "zp_info": zp_info,
        "note": (
            "WAVE labels apparent absorption centers on the simulated detector "
            "(cancels ~55 m/s ZEMAX PSF deposition shift vs geometric deposit-inverse WAVE). "
            "Geometry inject/recover at deposit centers remains a separate test."
        ),
    }
    packaged_sim_hdf_wavesol_meta_path(spectrograph).write_text(
        json.dumps(meta, indent=2, default=str) + "\n"
    )
    logger.info(
        "Wrote packaged sim-HDF line-ZP wavesol (%s): %s (median_dv=%s m/s)",
        spectrograph.upper(),
        out_path,
        None if zp_info is None else zp_info.get("median_dv_m_s"),
    )
    return out_path


def load_packaged_sim_hdf_line_zp_wavesol(
    spectrograph: str,
    *,
    meta: dict[str, Any] | None = None,
) -> LFCWavesolModel:
    """Load the packaged line-ZP wavesol for ``spectrograph``."""
    path = packaged_sim_hdf_wavesol_path(spectrograph)
    if not path.is_file():
        raise FileNotFoundError(
            f"Packaged sim-HDF line-ZP wavesol not found: {path}. "
            "Regenerate with "
            "python hispecdrp/data/wavesol/regenerate_sim_hdf_line_zp_wavesol.py "
            "--rate-map <e2e_rate_map.fits>"
        )
    wavesol = datamodels.open(str(path))
    if not isinstance(wavesol, LFCWavesolModel):
        # Some open paths return a generic CalibrationModel; re-wrap if needed.
        loaded = wavesol
        wavesol = LFCWavesolModel(config=loaded.config, meta=dict(loaded.meta))
        for fib in (1, 2, 3):
            w = getattr(loaded, f"wave_fib{fib}", None)
            if w is not None:
                setattr(wavesol, f"wave_fib{fib}", np.asarray(w, dtype=np.float64))
            dq = getattr(loaded, f"spec_dq_fib{fib}", None)
            if dq is not None:
                setattr(wavesol, f"spec_dq_fib{fib}", np.asarray(dq))
    if meta:
        for k, v in meta.items():
            try:
                wavesol.meta[k] = v
            except Exception:
                pass
    # Tag for caldb / WavelengthSolutionSelector preference over geometric LFC WAVE.
    try:
        wavesol.meta.note = PACKAGED_SIM_HDF_LINE_ZP_NOTE
    except Exception:
        pass
    # Packaged FITS ships with mjd_start=0 / 1858-11-17; sync datetime_obs when the
    # caller supplied a science MJD so registration joins the same-night version family.
    mjd = getattr(wavesol.meta, "mjd_start", None)
    dt = getattr(wavesol.meta, "datetime_obs", None)
    if mjd is not None and dt in (None, "", "1858-11-17T00:00:00.000"):
        import astropy.time

        try:
            wavesol.meta.datetime_obs = astropy.time.Time(
                float(mjd), format="mjd", scale="utc"
            ).isot
        except Exception:
            pass
    return wavesol
