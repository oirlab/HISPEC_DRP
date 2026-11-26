"""RV recovery of one deep, isolated line from the BT-Settl lte040 template.

The line is the ccfopt entry at 1.049898 um on BSPEC order 139. It is about
70 percent deep, 61 km/s from the next mask line, and more than 1000 km/s
from either edge of the order, so a +/-20 km/s inject grid stays on the
detector. The spectrum is painted with ``n_fine=512``. ``StellarRVPrimitive``
sees a one-line delta mask and the ideal-spectrum line-by-line settings.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..primitives.stellar_rv import StellarRVPrimitive
from ..utils.line_mask_utils import MaskLines, package_data_masks_dir
from ..utils.physical_constants import C_M_S
from .spectra import (
    build_l2,
    quiet_pipeline_logging,
    stellar_rv_kwargs_for_ideal_spectrum,
)

ORDER = 139
FIBER = 1
SPECTROGRAPH = "BSPEC"
TEMPLATE = "lte040.0-4.5-0.0a+0.0.BT-Settl.spec.7.h5"
N_FINE = 512
PACKAGED_MASK_NAME = "lte040.0-4.5-0.0a+0.0.BT-Settl.spec.7.ccfopt.delta.mas"
# Deep, unblended ccfopt line, well inside order 139.
TARGET_LAM_UM = 1.049898019887
# Same inject grid as the single-Gaussian example.
DEFAULT_INJECTED_RV_M_S = np.linspace(-20_000.0, 20_000.0, 41)
# Disable chunk outlier rejection, as in the original single-line notebook.
OUTLIER_SIGMA = 1.0e30


@dataclass
class TemplateLineRVResult:
    """Recovered RVs for one template line."""

    injected_m_s: np.ndarray
    recovered_m_s: np.ndarray
    rv_err_m_s: np.ndarray
    lam_rest_um: float
    line_index: int
    order: int = ORDER
    n_fine: int = N_FINE
    template: str = TEMPLATE

    @property
    def bias_m_s(self) -> np.ndarray:
        """Recovered minus injected, in m/s."""
        return self.recovered_m_s - self.injected_m_s

    @property
    def delta_m_s(self) -> np.ndarray:
        """Bias relative to the bias at injected RV = 0."""
        bias = self.bias_m_s
        at0 = np.where(np.isclose(self.injected_m_s, 0.0))[0]
        if at0.size != 1:
            raise ValueError("delta requires an injected RV of exactly 0")
        return bias - bias[int(at0[0])]


def packaged_mask_path() -> Path:
    """Packaged lte040 ccfopt mask."""
    path = package_data_masks_dir() / PACKAGED_MASK_NAME
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def selected_mask_line(mask_path: str | Path | None = None) -> tuple[float, int]:
    """Rest wavelength (um) and index of the chosen order-139 line."""
    path = packaged_mask_path() if mask_path is None else Path(mask_path)
    lams = np.asarray(MaskLines.load(str(path)).line_lambda_um, dtype=np.float64)
    index = int(np.argmin(np.abs(lams - TARGET_LAM_UM)))
    if abs(float(lams[index]) - TARGET_LAM_UM) > 1.0e-6:
        raise RuntimeError(f"mask has no line at {TARGET_LAM_UM} um")
    return float(lams[index]), index


def _one_line_mask_path(directory: Path, lam_rest_um: float, line_index: int) -> Path:
    path = directory / "single_line.delta.mas"
    MaskLines.from_delta_centers_um(
        np.array([lam_rest_um]),
        np.array([1.0]),
        metadata={"n_lines": 1, "order": ORDER, "line_idx": int(line_index)},
    ).save(str(path))
    return path


def recover_template_line(
    rv_m_s: float,
    *,
    mask_path: str | Path,
    lam_rest_um: float | None = None,
) -> tuple[float, float]:
    """Recover one injected RV. Returns ``(recovered_m_s, rv_err_m_s)`` from the chunk."""
    l2 = build_l2(
        float(rv_m_s) / 1000.0,
        spectrograph=SPECTROGRAPH,
        template=TEMPLATE,
        fiber=FIBER,
        order_start=ORDER,
        order_end=ORDER,
        n_fine=N_FINE,
    )
    kw = stellar_rv_kwargs_for_ideal_spectrum(
        float(rv_m_s) / 1000.0,
        template=TEMPLATE,
        mask_path=str(mask_path),
        spectrograph=SPECTROGRAPH,
        fiber=FIBER,
        order_start=ORDER,
        order_end=ORDER,
        outlier_sigma=OUTLIER_SIGMA,
    )
    with quiet_pipeline_logging():
        out = StellarRVPrimitive.call(l2, **kw)
    rvc = getattr(out, f"rvccf_fib{FIBER}")
    if len(rvc) != 1:
        raise RuntimeError(f"expected one mask line, got {len(rvc)}")
    return float(rvc["RV"][0]), float(rvc["RV_ERR"][0])


def run_template_line_rv_grid(
    injected_rv_m_s: np.ndarray | None = None,
) -> TemplateLineRVResult:
    """Recover every injected RV for the chosen order-139 template line."""
    if injected_rv_m_s is None:
        injected = np.asarray(DEFAULT_INJECTED_RV_M_S, dtype=np.float64).copy()
    else:
        injected = np.asarray(injected_rv_m_s, dtype=np.float64).ravel()
    lam_rest, line_index = selected_mask_line()
    recovered = np.empty(injected.shape, dtype=np.float64)
    rv_err = np.empty(injected.shape, dtype=np.float64)
    with tempfile.TemporaryDirectory(prefix="template_line_rv_") as directory:
        mask_path = _one_line_mask_path(Path(directory), lam_rest, line_index)
        for i, rv in enumerate(injected):
            recovered[i], rv_err[i] = recover_template_line(float(rv), mask_path=mask_path)
    return TemplateLineRVResult(
        injected_m_s=injected,
        recovered_m_s=recovered,
        rv_err_m_s=rv_err,
        lam_rest_um=lam_rest,
        line_index=line_index,
    )


def plot_template_line_profile(
    out_path: str | Path,
    *,
    lam_rest_um: float,
    zoom_km_s: float = 20.0,
) -> None:
    """Painted template line at injected RV = 0, versus velocity from the mask line."""
    import matplotlib.pyplot as plt

    l2 = build_l2(
        0.0,
        spectrograph=SPECTROGRAPH,
        template=TEMPLATE,
        fiber=FIBER,
        order_start=ORDER,
        order_end=ORDER,
        n_fine=N_FINE,
    )
    wave = np.asarray(getattr(l2, f"wave_fib{FIBER}")[0], dtype=np.float64)
    flux = np.asarray(getattr(l2, f"spec_fib{FIBER}")[0], dtype=np.float64)
    v_km = (C_M_S * (wave / float(lam_rest_um) - 1.0)) / 1000.0
    keep = np.abs(v_km) <= float(zoom_km_s)
    wings = keep & (np.abs(v_km) >= 8.0)
    continuum = float(np.median(flux[wings])) if np.any(wings) else float(np.median(flux[keep]))
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(v_km[keep], flux[keep] / continuum, color="C0", lw=1.2, label="simulated pixels")
    ax.axvline(0.0, color="0.4", ls=":", lw=1.0, label="mask line")
    ax.set_xlabel("Velocity from mask line (km/s)")
    ax.set_ylabel("Flux / continuum")
    ax.set_title(f"BT-Settl lte040 line on order {ORDER}, RV = 0")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_template_line_rv_recovery(
    result: TemplateLineRVResult,
    out_path: str | Path,
) -> None:
    """Recovered RV versus injected RV, and the bias relative to RV = 0."""
    import matplotlib.pyplot as plt

    inj_km = result.injected_m_s / 1000.0
    rec_km = result.recovered_m_s / 1000.0
    delta = result.delta_m_s
    # Trials whose line window is still on the order stay within a few m/s.
    # The rest run off the blue edge and would otherwise set the scale.
    on_order = np.abs(delta) < 50.0
    rms = float(np.sqrt(np.mean(delta[on_order] ** 2)))
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 7.0), sharex=True)
    ax = axes[0]
    ax.plot(inj_km, inj_km, color="k", ls="--", lw=1, label="recovered = injected")
    ax.plot(inj_km, rec_km, "o", ms=4, color="C0", label="StellarRV")
    ax.set_ylabel("Recovered RV (km/s)")
    ax.legend(frameon=False)
    ax.set_title("Single BT-Settl line")
    ax = axes[1]
    ax.axhline(0.0, color="k", ls="--", lw=1)
    ax.plot(inj_km, delta, "o", ms=4, color="C1")
    half = max(float(np.max(np.abs(delta[on_order]))) * 1.35, 0.5)
    ax.set_ylim(-half, half)
    ax.set_xlabel("Injected RV (km/s)")
    ax.set_ylabel("bias(v) - bias(0) (m/s)")
    n_off = int(np.count_nonzero(~on_order))
    title = f"RMS of Delta = {rms:.3f} m/s"
    if n_off:
        title += " where the line stays on the order"
    ax.set_title(title)
    fig.tight_layout()
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_template_line_summary(result: TemplateLineRVResult, path: str | Path) -> None:
    """Machine-readable bias summary. Paths are not recorded."""
    bias = result.bias_m_s
    delta = result.delta_m_s
    at0 = int(np.where(np.isclose(result.injected_m_s, 0.0))[0][0])
    payload = {
        "order": int(result.order),
        "template": result.template,
        "n_fine": int(result.n_fine),
        "lam_rest_um": float(result.lam_rest_um),
        "line_index": int(result.line_index),
        "n_injected": int(result.injected_m_s.size),
        "injected_rv_min_m_s": float(np.min(result.injected_m_s)),
        "injected_rv_max_m_s": float(np.max(result.injected_m_s)),
        "injected_rv_step_m_s": float(np.median(np.diff(result.injected_m_s))),
        "bias_at_0_m_s": float(bias[at0]),
        "max_abs_bias_m_s": float(np.max(np.abs(bias))),
        "max_abs_delta_m_s": float(np.max(np.abs(delta))),
        "rms_delta_m_s": float(np.sqrt(np.mean(delta**2))),
        "ptp_delta_m_s": float(np.ptp(delta)),
        "max_rv_err_m_s": float(np.max(result.rv_err_m_s)),
        "n_on_order": int(np.count_nonzero(np.abs(delta) < 50.0)),
        "rms_delta_on_order_m_s": float(np.sqrt(np.mean(delta[np.abs(delta) < 50.0] ** 2))),
        "max_abs_delta_on_order_m_s": float(np.max(np.abs(delta[np.abs(delta) < 50.0]))),
        "ptp_delta_on_order_m_s": float(np.ptp(delta[np.abs(delta) < 50.0])),
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
