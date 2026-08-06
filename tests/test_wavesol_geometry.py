"""
Slow geometry inject/recover test for deposit-inverse WAVE.

**Not run on GitHub.** ``pyproject.toml`` ``addopts`` includes
``--ignore=tests/test_wavesol_geometry.py``, so bare ``pytest`` (CI) never collects
this module. Run locally only::

    pytest tests/test_wavesol_geometry.py -v -s -o addopts=

Regenerates ``comb_wavesol_fib*_order*_dpix.png`` under ``test_data/wavesol_injected_comb/``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Belt-and-suspenders with --ignore in pyproject.toml addopts.
pytestmark = pytest.mark.slow

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "test_data" / "wavesol_injected_comb"
TARGET_CM_S = 10.0
FIBERS = (1, 2, 3)


def test_deposit_inverse_wavesol_geometry_generates_comb_plots():
    """
    Build deposit-inverse WAVE from the HDF spectrograph model for fibers 1–3,
    evaluate at exact deposit centers ``row(λ)``, assert residuals ≪10 cm/s, and
    write per-fiber / per-order PNGs for visual comparison.

    Products (under ``test_data/wavesol_injected_comb/``, gitignored):

    - ``comb_wavesol_fib{F}_order{N}_dpix.png`` — residual vs deposit pixel
    - ``comb_wavesol_fib{F}_per_order.png`` — MAD/RMS summary per fiber
    - ``comb_wavesol_per_fiber.png`` — across-fiber summary
    - ``comb_wavesol_report.json`` — numeric report
    - ``sim_hdf_wavesol_injected_comb.fits`` — WAVE model used
    """
    from hispecdrp.simulations.e2e_pipeline import validate_deposit_inverse_wavesol_geometry

    report = validate_deposit_inverse_wavesol_geometry(
        OUT_DIR,
        spectrograph="BSPEC",
        fibers=FIBERS,
        n_lines_per_order=1000,
        target_cm_s=TARGET_CM_S,
        write_plots=True,
        write_wavesol=True,
    )

    assert report["fibers"] == list(FIBERS)
    assert report["passed"], (
        f"Geometry WAVE recovery failed: max MAD={report['max_mad_cm_s']:.4f} cm/s "
        f"(target < {TARGET_CM_S:g} cm/s); see {OUT_DIR}"
    )
    assert report["max_mad_cm_s"] < TARGET_CM_S
    assert report["median_mad_cm_s"] < TARGET_CM_S

    for fib in FIBERS:
        fr = report["per_fiber"][fib]
        assert fr["n_orders"] > 0, f"Fiber {fib}: no orders evaluated"
        assert fr["passed"], (
            f"Fiber {fib} failed: max MAD={fr['max_mad_cm_s']:.4f} cm/s"
        )
        summary = OUT_DIR / f"comb_wavesol_fib{fib}_per_order.png"
        assert summary.is_file(), f"Missing summary plot: {summary}"
        order_plots = sorted(OUT_DIR.glob(f"comb_wavesol_fib{fib}_order*_dpix.png"))
        assert len(order_plots) == fr["n_orders"], (
            f"Fiber {fib}: expected {fr['n_orders']} order plots, "
            f"found {len(order_plots)} in {OUT_DIR}"
        )

    assert (OUT_DIR / "comb_wavesol_per_fiber.png").is_file()
    assert (OUT_DIR / "comb_wavesol_report.json").is_file()
    assert (OUT_DIR / "sim_hdf_wavesol_injected_comb.fits").is_file()

    for fib in FIBERS:
        fr = report["per_fiber"][fib]
        print(
            f"STELLAR_WAVESOL_GEOMETRY|BSPEC|fib{fib}|PASS|"
            f"median_mad_cm_s={fr['median_mad_cm_s']:.4f}|"
            f"max_mad_cm_s={fr['max_mad_cm_s']:.4f}|"
            f"n_orders={fr['n_orders']}"
        )
    print(
        f"STELLAR_WAVESOL_GEOMETRY|BSPEC|ALL|PASS|"
        f"median_mad_cm_s={report['median_mad_cm_s']:.4f}|"
        f"max_mad_cm_s={report['max_mad_cm_s']:.4f}|out={OUT_DIR}"
    )
