"""Direct rate-map → 1D extract bypass (skip UTR / Stage0 rate_err)."""

from __future__ import annotations

import numpy as np
import pytest

from hispecdrp.simulations.e2e_pipeline import (
    _order_list,
    extract_spectrum_from_rate_map,
    get_sim_hdf_trace_grids,
)


def test_extract_spectrum_from_rate_map_shapes():
    grids = get_sim_hdf_trace_grids("BSPEC")
    fiber = 2
    orders = _order_list("BSPEC")
    # Minimal synthetic rate: illuminate near fiber-2 order traces
    ny, nx = 4096, 4160
    rate = np.zeros((ny, nx), dtype=np.float64)
    for od in orders[::8]:  # sparse set of orders for speed
        y = grids[fiber][od]["y_pos"]
        for i in range(0, ny, 64):
            yc = y[i]
            if not np.isfinite(yc):
                continue
            c = int(round(yc))
            if 0 <= c < nx:
                rate[i, max(0, c - 1) : min(nx, c + 2)] = 10.0

    out = extract_spectrum_from_rate_map(
        rate,
        spectrograph="BSPEC",
        fiber=fiber,
        extraction_aperture=5.0,
        orders=list(orders[::8]),
    )
    n = len(out["orders"])
    assert n >= 1
    assert out["wavelength"].shape == (n, 4096)
    assert out["flux"].shape == (n, 4096)
    assert out["flux_err"].shape == (n, 4096)
    assert out["fiber"] == fiber
    assert np.any(np.isfinite(out["flux"]) & (out["flux"] > 0))


def test_extract_spectrum_from_rate_map_rejects_bad_fiber():
    with pytest.raises(ValueError, match="fiber"):
        extract_spectrum_from_rate_map(np.zeros((8, 8)), fiber=9)
