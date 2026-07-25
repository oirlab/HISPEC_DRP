#!/usr/bin/env python3
"""Plot segment half-width vs line center for bundled finite-width masks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hispecdrp.utils.line_mask_utils import MaskLines, fit_half_width_linear_from_mas


def main() -> None:
    here = Path(__file__).resolve().parent
    files = sorted(here.glob("*.tophat.mas"))
    if not files:
        files = sorted(here.glob("*.kpf.mas"))
    if not files:
        raise SystemExit(
            "No *.tophat.mas files next to this script (need three-column segment masks)."
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    for path in files:
        ml = MaskLines.load(str(path))
        centers_nm = ml.line_center.to_value("nm")
        half_nm = 0.5 * ml.line_width.to_value("nm")
        ax.scatter(centers_nm, half_nm, s=1, alpha=0.25, label=path.name)
        try:
            slope, intercept = fit_half_width_linear_from_mas(str(path))
            x = np.linspace(float(np.nanmin(centers_nm)), float(np.nanmax(centers_nm)), 200)
            ax.plot(x, slope * x + intercept, lw=1.5, label=f"{path.stem} fit")
        except ValueError:
            pass

    ax.set_xlabel("Line center (vacuum nm)")
    ax.set_ylabel("Segment half-width (nm)")
    ax.set_title("Finite-width (tophat) mask segment widths")
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, alpha=0.3)
    out = here / "mask_width_vs_wavelength.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
