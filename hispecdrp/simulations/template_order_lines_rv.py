"""Per-line RV delta for a handful of lines on one BSPEC order.

Five ccfopt lines on order 139, spread across the order and clear of its
edges. One mask, one ``StellarRVPrimitive`` call per injected RV. Delta for
each line is its bias minus the bias at RV = 0.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..primitives.stellar_rv import StellarRVPrimitive
from ..utils.line_mask_utils import MaskLines, package_data_masks_dir
from .spectra import (
    build_l2,
    quiet_pipeline_logging,
    stellar_rv_kwargs_for_ideal_spectrum,
)
from .template_line_rv import (
    DEFAULT_INJECTED_RV_M_S,
    FIBER,
    N_FINE,
    ORDER,
    OUTLIER_SIGMA,
    PACKAGED_MASK_NAME,
    SPECTROGRAPH,
    TEMPLATE,
)

# Spread across order 139. Each is isolated and stays on the detector for a
# +/-20 km/s inject grid.
TARGET_LAM_UM = np.array(
    [1.04725164, 1.04890714, 1.04989802, 1.05128918, 1.05351125],
    dtype=np.float64,
)


@dataclass
class OrderLinesRVResult:
    """Per-line recovered RVs. ``recovered_m_s`` has shape (n_rv, n_lines)."""

    injected_m_s: np.ndarray
    recovered_m_s: np.ndarray
    lam_rest_um: np.ndarray
    order: int = ORDER

    @property
    def bias_m_s(self) -> np.ndarray:
        return self.recovered_m_s - self.injected_m_s[:, None]

    @property
    def delta_m_s(self) -> np.ndarray:
        bias = self.bias_m_s
        at0 = np.where(np.isclose(self.injected_m_s, 0.0))[0]
        if at0.size != 1:
            raise ValueError("delta requires an injected RV of exactly 0")
        return bias - bias[int(at0[0])]

    def pairwise_correlation(self) -> np.ndarray:
        """Pearson correlation of Delta between lines."""
        return np.corrcoef(self.delta_m_s, rowvar=False)


def selected_order_lines() -> np.ndarray:
    """Exact mask wavelengths (um) nearest the chosen order-139 targets."""
    path = package_data_masks_dir() / PACKAGED_MASK_NAME
    lams = np.asarray(MaskLines.load(str(path)).line_lambda_um, dtype=np.float64)
    chosen = np.empty(TARGET_LAM_UM.shape, dtype=np.float64)
    for i, target in enumerate(TARGET_LAM_UM):
        index = int(np.argmin(np.abs(lams - target)))
        if abs(float(lams[index]) - float(target)) > 1.0e-5:
            raise RuntimeError(f"mask has no line near {target} um")
        chosen[i] = float(lams[index])
    return chosen


def _mask_path(directory: Path, lam_rest_um: np.ndarray) -> Path:
    path = directory / "order_lines.delta.mas"
    MaskLines.from_delta_centers_um(
        np.asarray(lam_rest_um, dtype=np.float64),
        np.ones(lam_rest_um.shape, dtype=np.float64),
        metadata={"n_lines": int(lam_rest_um.size), "order": ORDER},
    ).save(str(path))
    return path


def _recovered_by_line(rvc, lam_rest_um: np.ndarray) -> np.ndarray:
    """Pair each rest wavelength with the nearest RVCCF row.

    The stored chunk wavelength is the observed window center, so it moves
    with the injected RV. The chosen lines are separated by ~0.001 um, much
    more than a 20 km/s shift.
    """
    got = np.asarray(rvc["WAVELENGTH"], dtype=np.float64)
    rv = np.asarray(rvc["RV"], dtype=np.float64)
    out = np.empty(lam_rest_um.shape, dtype=np.float64)
    used = np.zeros(got.size, dtype=bool)
    for i, lam in enumerate(lam_rest_um):
        dist = np.abs(got - lam)
        dist[used] = np.inf
        j = int(np.argmin(dist))
        if not np.isfinite(dist[j]) or dist[j] > 5.0e-4:
            raise RuntimeError(f"no RVCCF row for {lam} um (nearest {got})")
        used[j] = True
        out[i] = float(rv[j])
    return out


def run_order_lines_rv_grid(
    injected_rv_m_s: np.ndarray | None = None,
) -> OrderLinesRVResult:
    """Recover every injected RV for each chosen line on order 139."""
    if injected_rv_m_s is None:
        injected = np.asarray(DEFAULT_INJECTED_RV_M_S, dtype=np.float64).copy()
    else:
        injected = np.asarray(injected_rv_m_s, dtype=np.float64).ravel()
    lam_rest = selected_order_lines()
    recovered = np.empty((injected.size, lam_rest.size), dtype=np.float64)
    with tempfile.TemporaryDirectory(prefix="order_lines_rv_") as directory:
        mask_path = _mask_path(Path(directory), lam_rest)
        for i, rv in enumerate(injected):
            l2 = build_l2(
                float(rv) / 1000.0,
                spectrograph=SPECTROGRAPH,
                template=TEMPLATE,
                fiber=FIBER,
                order_start=ORDER,
                order_end=ORDER,
                n_fine=N_FINE,
            )
            kw = stellar_rv_kwargs_for_ideal_spectrum(
                float(rv) / 1000.0,
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
            if len(rvc) != lam_rest.size:
                raise RuntimeError(f"expected {lam_rest.size} lines, got {len(rvc)}")
            recovered[i] = _recovered_by_line(rvc, lam_rest)
    return OrderLinesRVResult(
        injected_m_s=injected,
        recovered_m_s=recovered,
        lam_rest_um=lam_rest,
    )


def plot_order_lines_delta(result: OrderLinesRVResult, out_path: str | Path) -> None:
    """Delta versus injected RV, one series per line."""
    import matplotlib.pyplot as plt

    inj_km = result.injected_m_s / 1000.0
    delta = result.delta_m_s
    corr = result.pairwise_correlation()
    iu = np.triu_indices(corr.shape[0], k=1)
    off = np.abs(corr[iu])
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.axhline(0.0, color="k", ls="--", lw=1)
    for j, lam in enumerate(result.lam_rest_um):
        ax.plot(inj_km, delta[:, j], "o-", ms=3.5, lw=1.0, label=f"{lam:.4f} um")
    ax.set_xlabel("Injected RV (km/s)")
    ax.set_ylabel("bias(v) - bias(0) (m/s)")
    ax.set_title(
        f"Order {ORDER}, {result.lam_rest_um.size} lines, "
        f"median absolute correlation {float(np.median(off)):.2f}"
    )
    ax.legend(
        frameon=False,
        fontsize=8,
        title="line",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
    )
    fig.tight_layout()
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_order_lines_summary(result: OrderLinesRVResult, path: str | Path) -> None:
    """Machine-readable Delta summary. Paths are not recorded."""
    delta = result.delta_m_s
    corr = result.pairwise_correlation()
    iu = np.triu_indices(corr.shape[0], k=1)
    off = corr[iu]
    payload = {
        "order": int(result.order),
        "n_lines": int(result.lam_rest_um.size),
        "lam_rest_um": [float(x) for x in result.lam_rest_um],
        "n_injected": int(result.injected_m_s.size),
        "injected_rv_min_m_s": float(np.min(result.injected_m_s)),
        "injected_rv_max_m_s": float(np.max(result.injected_m_s)),
        "injected_rv_step_m_s": float(np.median(np.diff(result.injected_m_s))),
        "injected_rv_m_s": [float(x) for x in result.injected_m_s],
        "delta_m_s": [[float(x) for x in row] for row in delta],
        "rms_delta_m_s": [float(x) for x in np.sqrt(np.mean(delta**2, axis=0))],
        "pairwise_correlation": [[float(x) for x in row] for row in corr],
        "median_abs_correlation": float(np.median(np.abs(off))),
        "max_abs_correlation": float(np.max(np.abs(off))),
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
