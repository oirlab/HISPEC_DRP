"""Cross-talk extraction recovers blended fiber fluxes, with and without input errors."""

from __future__ import annotations

import numpy as np

from hispecdrp.primitives.spectral_extraction.extract_cross_talk import extract_spectra_cross_talk


def _make_frame(Nrows=8, Ncols=40, centers=(14.0, 19.0, 24.0), sigma=1.5, half_width=3):
    """Vertical orders: rows are the spectral axis, columns cross-dispersion.

    Profiles are zero beyond ``half_width`` pixels, inside the 7 px aperture, since the
    design matrix models each fiber only within its own aperture.
    """
    cols = np.arange(Ncols, dtype=float)
    Nfibers = len(centers)
    profile = np.zeros((Nfibers, Nrows, Ncols), dtype=np.float32)
    for i, c in enumerate(centers):
        p = np.exp(-0.5 * ((cols - c) / sigma) ** 2)
        p[np.abs(cols - c) > half_width] = 0
        profile[i] = p[None, :]
    trace_positions = np.tile(np.asarray(centers, dtype=float)[:, None], (1, Nrows))
    return profile, trace_positions


def _aperture_flux(profile, trace_positions, aperture):
    """Flux in the aperture for unit-amplitude profiles, as the extraction normalizes it."""
    from hispecdrp.primitives.spectral_extraction.extraction_utils import get_extraction_window

    Nfibers = profile.shape[0]
    out = np.zeros(Nfibers)
    for i in range(Nfibers):
        idx, W = get_extraction_window(trace_positions[i, 0], aperture, profile.shape[2])
        out[i] = np.sum(profile[i, 0, idx] * W)
    return out


def _run(err_value):
    profile, trace_positions = _make_frame()
    amps = np.array([100.0, 250.0, 60.0])
    data = np.einsum("i,irc->rc", amps, profile).astype(np.float32)
    Nrows, Ncols = data.shape
    aperture = 7.0
    err = np.full((Nrows, Ncols), err_value, dtype=np.float32)
    out = extract_spectra_cross_talk(
        input_data=data,
        input_err=err,
        input_dq=np.zeros((Nrows, Ncols), dtype=np.uint32),
        input_var_rnoise=np.full((Nrows, Ncols), 1.0, dtype=np.float32),
        input_var_poisson=np.zeros((Nrows, Ncols), dtype=np.float32),
        input_mid_time=None,
        trace_positions=trace_positions,
        profile=profile,
        extraction_aperture=aperture,
        orders_are_vertical=True,
    )
    expected = amps * _aperture_flux(profile, trace_positions, aperture)
    return out, expected


def test_cross_talk_weighted_recovers_blended_fluxes():
    out, expected = _run(err_value=2.0)
    np.testing.assert_allclose(out["spec"], np.tile(expected[:, None], (1, out["spec"].shape[1])), rtol=1e-4)
    assert np.all(np.isfinite(out["err"])) and np.all(out["err"] > 0)
    assert np.all(out["var_rnoise"] > 0)
    assert np.all(out["dq"] == 0)


def test_cross_talk_zero_errors_solves_unweighted():
    out, expected = _run(err_value=0.0)
    np.testing.assert_allclose(out["spec"], np.tile(expected[:, None], (1, out["spec"].shape[1])), rtol=1e-4)
    assert np.all(out["err"] == 0)
    # Read noise variance is still propagated through the unweighted solve
    assert np.all(out["var_rnoise"] > 0)
