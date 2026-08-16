"""Matched-line differential RV helpers (joint inliers + line QC).

Used by recovery / bulk differential tests and by multi-bulk ccfopt mask culls.
The differential estimator is intentionally *not* ``agg(B) − agg(A)``: it matches
lines by wavelength and requires joint inlier membership (or applies QC gates)
so epoch-asymmetric clipping cannot create a spurious ΔRV.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from hispecdrp.utils.physical_constants import C_M_S

__all__ = [
    "MATCHED_LINE_DIFF_KW",
    "match_rvccf_rows_by_wavelength",
    "ccf_peak_velocity_m_s",
    "line_passes_hard_qc",
    "estimate_matched_line_differential_rv",
]


# Defaults for matched-line differential RV and hard QC (recommendation #1–#4).
MATCHED_LINE_DIFF_KW: dict[str, Any] = {
    "match_half_m_s": 5000.0,
    # Joint inliers: require INLIER>0 at both epochs when the column exists.
    "require_joint_inliers": True,
    # Hard QC (applied to each epoch before joint membership).
    "max_rv_err_m_s": 10.0,
    "max_abs_rv_from_center_m_s": 5000.0,
    "peak_hop_max_m_s": 100.0,
    # Per-line differential residual gate relative to expected drv (m/s).
    "max_abs_line_diff_resid_m_s": 2.0,
    # Drop lines whose inv-var weight would exceed this fraction of Σw.
    "max_weight_fraction": 0.05,
}


def match_rvccf_rows_by_wavelength(
    tbl_a,
    tbl_b,
    *,
    match_half_m_s: float = MATCHED_LINE_DIFF_KW["match_half_m_s"],
) -> dict[str, np.ndarray]:
    """
    Greedy nearest-wavelength match of RVCCF rows (velocity half-window).

    Returns arrays aligned on matched pairs: wavelengths, RVs, errors, orders,
    inlier flags, and optional CCF metadata for peak-hop QC.
    """
    if tbl_a is None or tbl_b is None or len(tbl_a) == 0 or len(tbl_b) == 0:
        raise ValueError("match_rvccf_rows_by_wavelength: empty RVCCF table")
    for req in ("RV", "RV_ERR", "WAVELENGTH"):
        if req not in tbl_a.colnames or req not in tbl_b.colnames:
            raise ValueError(f"RVCCF missing column {req!r}")

    wla = np.asarray(tbl_a["WAVELENGTH"], dtype=np.float64).ravel()
    wlb = np.asarray(tbl_b["WAVELENGTH"], dtype=np.float64).ravel()
    rva = np.asarray(tbl_a["RV"], dtype=np.float64).ravel()
    rvb = np.asarray(tbl_b["RV"], dtype=np.float64).ravel()
    ea = np.asarray(tbl_a["RV_ERR"], dtype=np.float64).ravel()
    eb = np.asarray(tbl_b["RV_ERR"], dtype=np.float64).ravel()
    oda = (
        np.asarray(tbl_a["ORDER"], dtype=np.int32).ravel()
        if "ORDER" in tbl_a.colnames
        else np.full(wla.size, -1, dtype=np.int32)
    )
    ina = (
        np.asarray(tbl_a["INLIER"], dtype=np.float64).ravel() > 0
        if "INLIER" in tbl_a.colnames
        else np.ones(wla.size, dtype=bool)
    )
    inb = (
        np.asarray(tbl_b["INLIER"], dtype=np.float64).ravel() > 0
        if "INLIER" in tbl_b.colnames
        else np.ones(wlb.size, dtype=bool)
    )
    has_ccf = all(c in tbl_a.colnames and c in tbl_b.colnames for c in ("CCF", "V_MIN", "V_STEP"))

    used = np.zeros(wlb.size, dtype=bool)
    out_la: list[float] = []
    out_lb: list[float] = []
    out_rva: list[float] = []
    out_rvb: list[float] = []
    out_ea: list[float] = []
    out_eb: list[float] = []
    out_od: list[int] = []
    out_ina: list[bool] = []
    out_inb: list[bool] = []
    out_peak_a: list[float] = []
    out_peak_b: list[float] = []
    half = float(match_half_m_s)

    for i, lam in enumerate(wla):
        lam_f = float(lam)
        if not np.isfinite(lam_f) or lam_f <= 0 or not np.isfinite(rva[i]):
            continue
        dv = C_M_S * (wlb - lam_f) / lam_f
        ok = np.isfinite(dv) & (np.abs(dv) <= half) & (~used) & np.isfinite(rvb)
        if not np.any(ok):
            continue
        idx = np.flatnonzero(ok)
        j = int(idx[np.argmin(np.abs(dv[idx]))])
        used[j] = True
        out_la.append(lam_f)
        out_lb.append(float(wlb[j]))
        out_rva.append(float(rva[i]))
        out_rvb.append(float(rvb[j]))
        out_ea.append(float(ea[i]))
        out_eb.append(float(eb[j]))
        out_od.append(int(oda[i]))
        out_ina.append(bool(ina[i]))
        out_inb.append(bool(inb[j]))
        if has_ccf:
            out_peak_a.append(
                ccf_peak_velocity_m_s(
                    tbl_a["CCF"][i],
                    float(tbl_a["V_MIN"][i]),
                    float(tbl_a["V_STEP"][i]),
                )
            )
            out_peak_b.append(
                ccf_peak_velocity_m_s(
                    tbl_b["CCF"][j],
                    float(tbl_b["V_MIN"][j]),
                    float(tbl_b["V_STEP"][j]),
                )
            )
        else:
            out_peak_a.append(float("nan"))
            out_peak_b.append(float("nan"))

    return {
        "wavelength_a_um": np.asarray(out_la, dtype=np.float64),
        "wavelength_b_um": np.asarray(out_lb, dtype=np.float64),
        "rv_a_m_s": np.asarray(out_rva, dtype=np.float64),
        "rv_b_m_s": np.asarray(out_rvb, dtype=np.float64),
        "rv_err_a_m_s": np.asarray(out_ea, dtype=np.float64),
        "rv_err_b_m_s": np.asarray(out_eb, dtype=np.float64),
        "order": np.asarray(out_od, dtype=np.int32),
        "inlier_a": np.asarray(out_ina, dtype=bool),
        "inlier_b": np.asarray(out_inb, dtype=bool),
        "ccf_peak_a_m_s": np.asarray(out_peak_a, dtype=np.float64),
        "ccf_peak_b_m_s": np.asarray(out_peak_b, dtype=np.float64),
    }


def ccf_peak_velocity_m_s(ccf, v_min: float, v_step: float) -> float:
    """Velocity of the discrete CCF maximum (stored peak-up convention)."""
    y = np.asarray(ccf, dtype=np.float64).ravel()
    if y.size < 3 or not np.any(np.isfinite(y)):
        return float("nan")
    if not (np.isfinite(v_min) and np.isfinite(v_step) and v_step > 0):
        return float("nan")
    i = int(np.nanargmax(y))
    return float(v_min) + float(i) * float(v_step)


def line_passes_hard_qc(
    rv_m_s: float,
    rv_err_m_s: float,
    *,
    center_m_s: float,
    ccf_peak_m_s: float | None = None,
    max_rv_err_m_s: float = MATCHED_LINE_DIFF_KW["max_rv_err_m_s"],
    max_abs_rv_from_center_m_s: float = MATCHED_LINE_DIFF_KW["max_abs_rv_from_center_m_s"],
    peak_hop_max_m_s: float = MATCHED_LINE_DIFF_KW["peak_hop_max_m_s"],
) -> bool:
    """Hard single-line QC: finite, err/center gates, optional Gaussian≠peak hop."""
    if not (np.isfinite(rv_m_s) and np.isfinite(rv_err_m_s) and rv_err_m_s > 0):
        return False
    if float(rv_err_m_s) >= float(max_rv_err_m_s):
        return False
    if abs(float(rv_m_s) - float(center_m_s)) >= float(max_abs_rv_from_center_m_s):
        return False
    if ccf_peak_m_s is not None and np.isfinite(ccf_peak_m_s) and float(peak_hop_max_m_s) > 0:
        if abs(float(rv_m_s) - float(ccf_peak_m_s)) >= float(peak_hop_max_m_s):
            return False
    return True


def estimate_matched_line_differential_rv(
    tbl_a,
    tbl_b,
    *,
    inject_a_m_s: float,
    inject_b_m_s: float,
    match_half_m_s: float = MATCHED_LINE_DIFF_KW["match_half_m_s"],
    require_joint_inliers: bool = MATCHED_LINE_DIFF_KW["require_joint_inliers"],
    max_rv_err_m_s: float = MATCHED_LINE_DIFF_KW["max_rv_err_m_s"],
    max_abs_rv_from_center_m_s: float = MATCHED_LINE_DIFF_KW["max_abs_rv_from_center_m_s"],
    peak_hop_max_m_s: float = MATCHED_LINE_DIFF_KW["peak_hop_max_m_s"],
    max_abs_line_diff_resid_m_s: float = MATCHED_LINE_DIFF_KW["max_abs_line_diff_resid_m_s"],
    max_weight_fraction: float = MATCHED_LINE_DIFF_KW["max_weight_fraction"],
) -> dict[str, Any]:
    """
    Estimate ``drv = inject_b − inject_a`` from matched line-by-line RVs.

    Uses joint inliers (recommendation #2), hard QC / peak-hop / leverage caps
    (#4), and returns the inverse-variance mean of ``RV_b − RV_a`` among kept
    lines (#1) — not ``agg(B) − agg(A)``.
    """
    matched = match_rvccf_rows_by_wavelength(tbl_a, tbl_b, match_half_m_s=match_half_m_s)
    n = int(matched["rv_a_m_s"].size)
    drv = float(inject_b_m_s) - float(inject_a_m_s)
    if n == 0:
        return {
            "estimated_drv_m_s": float("nan"),
            "estimated_drv_err_m_s": float("nan"),
            "residual_m_s": float("nan"),
            "n_matched": 0,
            "n_kept": 0,
            "drv_m_s": drv,
            "keep": np.zeros(0, dtype=bool),
            "matched": matched,
        }

    rva = matched["rv_a_m_s"]
    rvb = matched["rv_b_m_s"]
    ea = matched["rv_err_a_m_s"]
    eb = matched["rv_err_b_m_s"]
    d_rv = rvb - rva
    line_resid = d_rv - drv
    err_d = np.sqrt(np.maximum(ea, 0.0) ** 2 + np.maximum(eb, 0.0) ** 2)

    qc_a = np.array(
        [
            line_passes_hard_qc(
                float(rva[i]),
                float(ea[i]),
                center_m_s=float(inject_a_m_s),
                ccf_peak_m_s=float(matched["ccf_peak_a_m_s"][i]),
                max_rv_err_m_s=max_rv_err_m_s,
                max_abs_rv_from_center_m_s=max_abs_rv_from_center_m_s,
                peak_hop_max_m_s=peak_hop_max_m_s,
            )
            for i in range(n)
        ],
        dtype=bool,
    )
    qc_b = np.array(
        [
            line_passes_hard_qc(
                float(rvb[i]),
                float(eb[i]),
                center_m_s=float(inject_b_m_s),
                ccf_peak_m_s=float(matched["ccf_peak_b_m_s"][i]),
                max_rv_err_m_s=max_rv_err_m_s,
                max_abs_rv_from_center_m_s=max_abs_rv_from_center_m_s,
                peak_hop_max_m_s=peak_hop_max_m_s,
            )
            for i in range(n)
        ],
        dtype=bool,
    )
    keep = qc_a & qc_b
    if require_joint_inliers:
        keep &= matched["inlier_a"] & matched["inlier_b"]
    if float(max_abs_line_diff_resid_m_s) > 0:
        keep &= np.isfinite(line_resid) & (np.abs(line_resid) < float(max_abs_line_diff_resid_m_s))
    keep &= np.isfinite(err_d) & (err_d > 0)

    # Leverage cap: iteratively drop lines with weight fraction above the ceiling.
    w = np.zeros(n, dtype=np.float64)
    w[keep] = 1.0 / np.maximum(err_d[keep], 1e-30) ** 2
    if float(max_weight_fraction) > 0 and np.any(keep):
        for _ in range(8):
            w_sum = float(np.sum(w[keep]))
            if w_sum <= 0:
                break
            frac = w / w_sum
            heavy = keep & (frac > float(max_weight_fraction))
            if not np.any(heavy):
                break
            # Drop the single heaviest offender each pass.
            i_drop = int(np.flatnonzero(heavy)[np.argmax(frac[heavy])])
            keep[i_drop] = False
            w[i_drop] = 0.0

    n_kept = int(np.sum(keep))
    if n_kept == 0:
        est = float("nan")
        err = float("nan")
    else:
        est = float(np.average(d_rv[keep], weights=w[keep]))
        err = float(math.sqrt(1.0 / np.sum(w[keep])))

    return {
        "estimated_drv_m_s": est,
        "estimated_drv_err_m_s": err,
        "residual_m_s": (est - drv) if np.isfinite(est) else float("nan"),
        "n_matched": n,
        "n_kept": n_kept,
        "drv_m_s": drv,
        "keep": keep,
        "matched": matched,
        "agg_a_minus_agg_b_diagnostic_m_s": float("nan"),  # filled by caller if desired
    }
