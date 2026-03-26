#!/usr/bin/env python3
"""
Regenerate ``mask_width_vs_wavelength.png``: SPIROU ``.mas`` segment width vs wavelength.

Each row is (λ_start, λ_end, weight) in vacuum nm (see ``load_mask_spirou_mas``).
Run from repo root: ``python hispecdrp/data/masks/plot_mask_widths.py``
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_mas(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    start, end = data[:, 0].astype(np.float64), data[:, 1].astype(np.float64)
    lo, hi = np.minimum(start, end), np.maximum(start, end)
    mean_nm = 0.5 * (lo + hi)
    width_nm = hi - lo
    return mean_nm, width_nm


def main() -> None:
    here = Path(__file__).resolve().parent
    files = sorted(here.glob("*.mas"))
    if not files:
        raise SystemExit("No .mas files next to this script.")

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    ax_scatter, ax_frac, ax_hist, ax_log = axes.ravel()

    colors = plt.cm.tab10(np.linspace(0, 0.35, len(files)))

    for f, c in zip(files, colors):
        mean_nm, w_nm = load_mas(f)
        label = f.stem
        ax_scatter.scatter(mean_nm, w_nm * 1000, s=4, alpha=0.35, c=[c], label=f"{label} (N={len(mean_nm)})")
        ax_frac.scatter(mean_nm, w_nm / mean_nm, s=4, alpha=0.35, c=[c], label=label)
        ax_log.scatter(np.log(mean_nm), w_nm / mean_nm, s=4, alpha=0.35, c=[c], label=label)
        ax_hist.hist(w_nm * 1000, bins=40, alpha=0.45, color=c, label=label)

    ax_scatter.set_xlabel("Mean wavelength (nm)")
    ax_scatter.set_ylabel("Segment width (pm)")
    ax_scatter.set_title(".mas segments: width vs mean λ (file units: vacuum nm)")
    ax_scatter.legend(loc="upper right", fontsize=8)

    ax_frac.set_xlabel("Mean wavelength (nm)")
    ax_frac.set_ylabel("Width / mean λ")
    ax_frac.set_title("Fractional width (~constant ⇒ fixed Δln λ per line)")

    ax_hist.set_xlabel("Width (pm)")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Width distribution")
    ax_hist.legend(fontsize=8)

    ax_log.set_xlabel("ln(mean λ / nm)")
    ax_log.set_ylabel("Width / mean λ")
    ax_log.set_title("Fractional width vs ln λ")

    plt.tight_layout()
    out = here / "mask_width_vs_wavelength.png"
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")

    for f in files:
        m, w = load_mas(f)
        r = w / m
        print(
            f"{f.name}: width_nm median={np.median(w):.6g}, "
            f"(w/λ) median={np.median(r):.6g}, (w/λ) range relative={r.ptp()/np.median(r):.3e}"
        )


if __name__ == "__main__":
    main()
