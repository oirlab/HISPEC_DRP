"""``find_emission_lines`` must only drop first/last peaks at the array edge."""

from __future__ import annotations

import numpy as np

from hispecdrp.primitives.line_fitting.find_lines import find_emission_lines


def _spec_with_lines(centers, n_pix=512):
    pix = np.arange(n_pix, dtype=float)
    spec = np.zeros(n_pix, dtype=np.float32)
    for mu in centers:
        spec += (200.0 * np.exp(-0.5 * ((pix - mu) / 1.1) ** 2)).astype(np.float32)
    dq = np.zeros(n_pix, dtype=np.uint32)
    return spec, dq


def test_interior_first_and_last_peaks_are_kept():
    centers = (60.0, 160.0, 260.0, 360.0, 460.0)
    spec, dq = _spec_with_lines(centers)
    lines, _ = find_emission_lines(spec, dq, min_line_spacing=10)
    assert len(lines) == len(centers)
    assert np.allclose(np.sort(lines), centers, atol=1.0)


def test_peaks_truncated_by_array_edge_are_dropped():
    centers = (6.0, 160.0, 260.0, 506.0)
    spec, dq = _spec_with_lines(centers)
    lines, _ = find_emission_lines(spec, dq, min_line_spacing=10)
    assert len(lines) == 2
    assert np.allclose(np.sort(lines), (160.0, 260.0), atol=1.0)
