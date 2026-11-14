from pathlib import Path

import numpy as np
import yaml
from hispecdrp import datamodels
from hispecdrp.simulations.utr import create_ramp
from hispecdrp.simulations.simulate_rate_map import SimulateRateMap

# Module-level generator: reproducible across a test session, but each call draws new noise,
# without reseeding the global np.random state other tests use.
RNG = np.random.default_rng(1)


def repo_root() -> Path:
    """Repository root (parent of ``tests/``)."""
    return Path(__file__).resolve().parents[1]


# Fixed epoch for cal-db / sim metadata in tests: inside IERS leap-second tables
# so Astropy/ERFA does not emit "dubious year" warnings (unlike Time.now() in 2026+
# or packaged wavesol placeholder MJD 0 / 1858-11-17).
SAFE_TEST_MJD = 60000.0  # 2023-02-25 UTC


def save_config_yaml(config_dict : dict, config_path : str):
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f)


def make_ramp_model(
    source : np.array,
    meta : dict | None = None,
    **kwargs
) -> datamodels.RampModel:
    ramp_data = create_ramp(
        source=source,
        **kwargs,
    )
    ramp_model = datamodels.RampModel(
        data=ramp_data['data'],
        dq=np.zeros(ramp_data['data'].shape[1:], dtype=np.uint32),
        dq_raw=np.zeros(ramp_data['data'].shape, dtype=np.uint8),
        meta={**(meta or {}), **ramp_data['meta']},
    )
    return ramp_model


def fast_ccf_orders(spectrograph: str) -> list[int]:
    """Three mid-band echelle orders for fast RV-recovery unit tests."""
    from hispecdrp.datamodels import hispec_config

    orders = list(hispec_config.HISPECConfig.get_order_list(spectrograph))
    if len(orders) <= 3:
        return orders
    mid = len(orders) // 2
    return [orders[mid - 1], orders[mid], orders[mid + 1]]


def add_noise_L0(
    model : datamodels.DataModel,
    max_snr : float,
    exposure_time : float | None = None,
    rn : float = 1.0,
    copy : bool = True,
    solve_for : str = 'rate',
):
    """
    Bring a noiseless L0 rate map to a peak SNR, then add Poisson and read noise.

    The brightest pixel is given SNR ``max_snr`` for the read noise ``rn``, either by
    rescaling the rate at a fixed exposure time (``solve_for='rate'``) or by keeping the
    rate and solving for the exposure time (``solve_for='exposure_time'``). The variances
    are computed from the noiseless rate, and ``rate_err = sqrt(var_poisson + var_rnoise)``,
    the same convention as ramp fitting. A Gaussian draw with that error is then added
    to the rate.

    Parameters
    ----------
    model : datamodels.DataModel
        L0 model with a noiseless ``rate`` in e-/s.
    max_snr : float
        SNR of the brightest pixel.
    exposure_time : float | None
        Exposure time in s for ``solve_for='rate'``. Defaults to ``model.meta.exposure_time``,
        or 60 s if unset. Must be None for ``solve_for='exposure_time'``.
    rn : float
        Effective read noise of the exposure in e-.
    copy : bool
        If True, modify a copy of the input model; otherwise modify it in place.
    solve_for : str
        What to change to reach ``max_snr``:

        - ``'rate'``: rescale the rate at the given exposure time.
        - ``'exposure_time'``: keep the rate, solve for the exposure time and write it to
          ``model.meta.exposure_time``.

    Returns
    -------
    datamodels.DataModel
        Model with noisy ``rate``, ``rate_err`` in e-/s, and ``rate_var_poisson`` and
        ``rate_var_rnoise`` in (e-/s)^2.
    """
    if solve_for not in ('rate', 'exposure_time'):
        raise ValueError(f"solve_for must be 'rate' or 'exposure_time', got {solve_for!r}")
    if solve_for == 'exposure_time' and exposure_time is not None:
        raise ValueError("exposure_time is solved for with solve_for='exposure_time'; don't pass it")
    if copy:
        model = model.copy()
    rate = np.asarray(model.rate, dtype=np.float64)
    peak_rate = np.nanmax(rate)

    # Peak counts N with N / sqrt(N + rn^2) = max_snr
    snr2 = float(max_snr)**2
    peak_counts = (snr2 + np.sqrt(snr2**2 + 4 * snr2 * rn**2)) / 2

    # Reach the peak counts by scaling the rate, or by choosing the exposure time
    if solve_for == 'rate':
        if exposure_time is None:
            exposure_time = model.meta.exposure_time or 60.0
        t = float(exposure_time)
        rate = rate * peak_counts / (peak_rate * t)
    else:
        t = peak_counts / peak_rate
        model.meta.exposure_time = t

    # Variances of the rate (e-/s)^2: Poisson N/t^2 = rate/t, read noise rn^2/t^2
    var_poisson = np.clip(rate, 0, None) / t
    var_rnoise = np.full(rate.shape, rn**2 / t**2)
    rate_err = np.sqrt(var_poisson + var_rnoise)

    model.rate = (rate + RNG.normal(0.0, rate_err)).astype(np.float32)
    model.rate_err = rate_err.astype(np.float32)
    model.rate_var_poisson = var_poisson.astype(np.float32)
    model.rate_var_rnoise = var_rnoise.astype(np.float32)
    return model