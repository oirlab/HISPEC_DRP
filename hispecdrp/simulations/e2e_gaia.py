"""
Gaia target selection and barycentric corrections for end-to-end simulations.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

import numpy as np
from astroquery.gaia import Gaia

from ..primitives.barycenter_correction.barycenter_corr import (
    BarycorrpyStarData,
    calc_barycenter_corrections,
)
from ..primitives.utils import get_stellar_data_gaia_from_source_id

logger = logging.getLogger(__name__)

DEFAULT_TELESCOPE = "Keck-II"


@dataclass(frozen=True)
class E2EGaiaTarget:
    """Gaia star used to drive E2E Doppler injection and barycentric metadata."""

    source_id: str
    star_data: dict
    target_meta: dict
    gaia_rv_km_s: float
    berv_km_s: float
    injected_rv_km_s: float
    mjd_mid: float


def query_random_bright_gaia_source_id(
    *,
    max_g_mag: float = 8.0,
    seed: int | None = None,
) -> str:
    """
    Return a random Gaia DR3 source_id for a bright star with a measured RV.

    Uses ``random_index`` on the Gaia archive (see Gaia ADQL documentation).
    """
    rng = random.Random(seed)
    threshold = rng.random()
    query = f"""
        SELECT TOP 1 source_id
        FROM gaiadr3.gaia_source
        WHERE phot_g_mean_mag < {max_g_mag}
          AND radial_velocity IS NOT NULL
          AND parallax IS NOT NULL
          AND parallax > 0
          AND random_index >= {threshold}
        ORDER BY random_index ASC
    """
    job = Gaia.launch_job(query)
    results = job.get_results()
    if len(results) == 0:
        raise RuntimeError(
            f"No Gaia DR3 match for bright-star query (G < {max_g_mag}, RV required)."
        )
    return str(results["source_id"][0])


def _finite_float(value, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def calc_berv_km_s(
    star_data: dict,
    mjd_utc: float,
    *,
    telescope: str = DEFAULT_TELESCOPE,
) -> float:
    """Barycentric Earth RV correction (km/s) at ``mjd_utc`` for the given Gaia star."""
    star = BarycorrpyStarData(
        ra=float(star_data["ra"]),
        dec=float(star_data["dec"]),
        epoch=_finite_float(star_data.get("ref_epoch"), 2016.0),
        pm_ra=_finite_float(star_data.get("pmra")),
        pm_dec=_finite_float(star_data.get("pmdec")),
        px=_finite_float(star_data.get("parallax")),
        rvsys=_finite_float(star_data.get("radial_velocity")),
    )
    _, vel_bc = calc_barycenter_corrections(
        mjd_utc,
        star,
        observatory=telescope,
    )
    # barycorrpy returns m/s; E2E Doppler injection uses km/s.
    return float(np.asarray(vel_bc).ravel()[0]) / 1000.0


def build_target_meta_from_gaia(
    star_data: dict,
    *,
    mjd_mid: float,
    target_name: str | None = None,
    telescope: str = DEFAULT_TELESCOPE,
) -> dict:
    """Build target metadata dict (same fields as ``tests/test_barycorr.py``)."""
    sid = str(star_data["source_id"])
    name = target_name or f"Gaia_{sid}"
    return {
        "target_name": name,
        "target_gaia_id": sid,
        "target_gaia_ra": float(star_data["ra"]),
        "target_gaia_dec": float(star_data["dec"]),
        "target_gaia_epoch": _finite_float(star_data.get("ref_epoch"), 2016.0),
        "target_gaia_rapm": _finite_float(star_data.get("pmra")),
        "target_gaia_decpm": _finite_float(star_data.get("pmdec")),
        "target_gaia_px": _finite_float(star_data.get("parallax")),
        "target_gaia_rvsys": _finite_float(star_data.get("radial_velocity")),
        "mjd_mid": float(mjd_mid),
        "mjd_start": float(mjd_mid),
        "telescope": telescope,
    }


def resolve_e2e_gaia_target(
    *,
    source_id: str | None,
    mjd_mid: float,
    max_g_mag: float = 8.0,
    random_seed: int | None = None,
    extra_rv_km_s: float = 0.0,
    telescope: str = DEFAULT_TELESCOPE,
    target_name: str | None = None,
) -> E2EGaiaTarget:
    """
    Resolve a Gaia target and total injected heliocentric RV (Gaia RV + BERV + extra).

    Stellar properties are loaded with
    :func:`~hispecdrp.primitives.utils.get_stellar_data_gaia_from_source_id`
    (same as ``tests/test_barycorr.py``).
    """
    if source_id is None:
        source_id = query_random_bright_gaia_source_id(
            max_g_mag=max_g_mag,
            seed=random_seed,
        )
        logger.info("E2E: selected random bright Gaia source_id=%s.", source_id)
    else:
        logger.info("E2E: using configured Gaia source_id=%s.", source_id)

    star_data = get_stellar_data_gaia_from_source_id(source_id)
    if star_data is None:
        raise RuntimeError(f"Gaia query returned no data for source_id={source_id!r}.")

    gaia_rv_km_s = _finite_float(star_data.get("radial_velocity"))
    berv_km_s = calc_berv_km_s(star_data, mjd_mid, telescope=telescope)
    injected_rv_km_s = gaia_rv_km_s + berv_km_s + float(extra_rv_km_s)

    meta = build_target_meta_from_gaia(
        star_data,
        mjd_mid=mjd_mid,
        target_name=target_name,
        telescope=telescope,
    )

    logger.info(
        "E2E Gaia target %s: G=%.2f, Gaia RV=%.3f km/s, BERV=%.3f km/s, "
        "extra RV=%.3f km/s, injected RV=%.3f km/s, mjd_mid=%.5f",
        meta["target_name"],
        _finite_float(star_data.get("phot_g_mean_mag"), float("nan")),
        gaia_rv_km_s,
        berv_km_s,
        extra_rv_km_s,
        injected_rv_km_s,
        mjd_mid,
    )

    return E2EGaiaTarget(
        source_id=str(star_data["source_id"]),
        star_data=star_data,
        target_meta=meta,
        gaia_rv_km_s=gaia_rv_km_s,
        berv_km_s=berv_km_s,
        injected_rv_km_s=injected_rv_km_s,
        mjd_mid=float(mjd_mid),
    )


def apply_target_meta_to_model(model, meta: dict) -> None:
    """Set target / exposure metadata fields on a data model."""
    for key, value in meta.items():
        if value is None:
            continue
        if hasattr(model.meta, key):
            setattr(model.meta, key, value)
