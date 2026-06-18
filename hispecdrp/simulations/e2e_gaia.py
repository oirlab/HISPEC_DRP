"""
Gaia target selection and barycentric corrections for end-to-end simulations.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

import numpy as np

from ..primitives.barycenter_correction.barycenter_corr import (
    BarycorrpyStarData,
    calc_barycenter_corrections,
)
from ..primitives.utils import (
    _convert_epoch_gaia,
    get_2mass_data_vizier,
    get_stellar_data_gaia_from_source_id,
)
from ..primitives.utils.gaia_query import GAIA_DR3_VIZIER_ID, VIZIER_MIRRORS

logger = logging.getLogger(__name__)

DEFAULT_TELESCOPE = "Keck-II"

# Fixed first-read MJD for reproducible E2E runs (UTC mid-2026). Exposure mid-time and
# BERV are derived deterministically from this value plus ``n_reads`` / ``readtime``.
DEFAULT_E2E_MJD_START = 61210.0

# Approximate mean epoch of 2MASS observations (J2000); Gaia positions are propagated
# back to this epoch before the 2MASS cone search to handle high-proper-motion stars.
_TWOMASS_EPOCH_JYEAR = 2000.0

# 2MASS magnitudes are reported in the Vega system; the template normalizer
# (``load_btsettl_spectrum``) works in AB. These are the per-band Vega->AB offsets
# (m_AB = m_Vega + offset) for the 2MASS JHKs bands (Blanton & Roweis 2007).
_TWOMASS_VEGA_TO_AB = {"J": 0.910, "H": 1.390, "K": 1.850, "KS": 1.850}

# Map a 2MASS band label to the catalog magnitude key returned by
# :func:`~hispecdrp.primitives.utils.get_2mass_data_vizier`.
_TWOMASS_BAND_TO_KEY = {"J": "j_m", "H": "h_m", "K": "ks_m", "KS": "ks_m"}


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
    extra_rv_km_s: float = 0.0
    twomass_data: dict | None = None

    @property
    def stellar_rv_km_s(self) -> float:
        """Barycentric stellar RV for E2E comparison (Gaia ``radial_velocity`` only)."""
        return float(self.injected_rv_km_s)

    @property
    def topocentric_doppler_rv_km_s(self) -> float:
        """Topocentric observed RV for template Doppler (``v_obs = v_bary - BERV``)."""
        return (
            float(self.gaia_rv_km_s)
            - float(self.berv_km_s)
            + float(self.extra_rv_km_s)
        )


def twomass_mag_ab(twomass_data: dict | None, band: str) -> float | None:
    """Return the 2MASS ``band`` magnitude (from VizieR) converted to the AB system.

    2MASS catalog magnitudes are in the Vega system; this applies the per-band
    Vega->AB offset so the value can be passed to the AB template normalizer.
    Returns ``None`` if the data or band magnitude is unavailable.
    """
    if not twomass_data:
        return None
    b = str(band).upper()
    key = _TWOMASS_BAND_TO_KEY.get(b)
    offset = _TWOMASS_VEGA_TO_AB.get(b)
    if key is None or offset is None:
        return None
    mag_vega = _finite_float(twomass_data.get(key), float("nan"))
    if not np.isfinite(mag_vega):
        return None
    return float(mag_vega) + float(offset)


def query_random_bright_gaia_source_id(
    *,
    max_g_mag: float = 8.0,
    seed: int | None = None,
    max_rounds: int = 25,
) -> str:
    """
    Return a random Gaia DR3 source_id for a bright star with a measured RV, via VizieR.

    Picks random sphere-uniform sky positions and cone-searches the VizieR Gaia DR3
    mirror (``I/355/gaiadr3``) for bright stars (G < ``max_g_mag``) with a defined RV and
    positive parallax, returning a random match. Mirrors are tried in order on failure.
    """
    from astroquery.vizier import Vizier
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    rng = random.Random(seed)
    column_filters = {
        "Gmag": f"<{max_g_mag}",
        "RV": ">-9000",   # range constraint also excludes null RVs
        "Plx": ">0",
    }
    # Keep the cone small: a large radius forces VizieR to scan millions of faint
    # sources before applying the G<max_g_mag filter. Bright RV stars are common enough
    # that a ~1.5 deg cone usually yields several; empty cones are handled by retrying.
    search_radius_deg = 1.5
    for _ in range(max_rounds):
        ra_deg = rng.uniform(0.0, 360.0)
        dec_deg = float(np.degrees(np.arcsin(rng.uniform(-1.0, 1.0))))
        center = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
        for server in VIZIER_MIRRORS:
            try:
                vizier = Vizier(
                    columns=["Source", "Gmag", "RV", "Plx"],
                    column_filters=column_filters,
                    vizier_server=server,
                )
                vizier.ROW_LIMIT = 500
                result = vizier.query_region(
                    center, radius=search_radius_deg * u.deg, catalog=GAIA_DR3_VIZIER_ID
                )
                if result is not None and len(result) > 0 and len(result[0]) > 0:
                    tbl = result[0]
                    idx = rng.randrange(len(tbl))
                    return str(tbl["Source"][idx])
            except Exception as e:  # noqa: BLE001 - mirror fallback is intentional
                logger.warning(
                    "VizieR mirror %s failed for random bright-star query: %s", server, e
                )
                continue
    raise RuntimeError(
        f"No Gaia DR3 match for bright-star query (G < {max_g_mag}, RV required) "
        f"after {max_rounds} random VizieR cone searches."
    )


def _finite_float(value, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def resolve_e2e_mjd_start(mjd_start: float | None) -> float:
    """Return the MJD of the first ramp read (fixed default when ``mjd_start`` is unset)."""
    if mjd_start is not None:
        return float(mjd_start)
    return float(DEFAULT_E2E_MJD_START)


def exposure_mid_mjd(mjd_start: float, *, n_reads: int, readtime: float) -> float:
    """Exposure mid-time (MJD) for a UTR ramp with uniform read spacing."""
    return float(mjd_start) + 0.5 * max(int(n_reads) - 1, 0) * float(readtime) / 86400.0


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
    Resolve a Gaia target for E2E simulation and RV validation.

    **Doppler injection:** template wavelengths are shifted by the **topocentric**
    observed RV ``Gaia RV - BERV + extra_rv_km_s`` (``barycorrpy``: ``v_bary =
    v_obs + BERV``) so extracted ``WAVE_FIB*`` matches real observations.
    **Comparison metric:** ``injected_rv_km_s`` is the Gaia catalog
    ``radial_velocity`` only; BERV is applied in Stage 2 CCF via ``ccf_zb``, not in
    the reported injected value. ``extra_rv_km_s`` shifts the simulated template
    Doppler only (not the injected comparison metric).

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
    injected_rv_km_s = gaia_rv_km_s

    # Look up the 2MASS counterpart (Vega JHKs) so the template can be normalized to this
    # star's real brightness. Propagate the Gaia position back to the 2MASS epoch so
    # high-proper-motion stars still fall inside the cone.
    twomass_data: dict | None = None
    try:
        ra_2mass, dec_2mass = _convert_epoch_gaia(
            ra=float(star_data["ra"]),
            dec=float(star_data["dec"]),
            epoch=_TWOMASS_EPOCH_JYEAR,
            pmra=_finite_float(star_data.get("pmra")),
            pmdec=_finite_float(star_data.get("pmdec")),
            px=_finite_float(star_data.get("parallax")),
            rvsys=_finite_float(star_data.get("radial_velocity")),
            to_gaia=False,
            release="DR3",
        )
    except Exception:
        logger.exception("E2E: epoch propagation to 2MASS failed; using Gaia position.")
        ra_2mass, dec_2mass = float(star_data["ra"]), float(star_data["dec"])
    try:
        twomass_data = get_2mass_data_vizier(ra_2mass, dec_2mass, radius_arcsec=5.0)
    except Exception:
        logger.exception("E2E: 2MASS VizieR lookup failed (continuing without it).")
    if twomass_data is not None:
        logger.info(
            "E2E 2MASS match for %s: id=%s, J=%.3f, H=%.3f, Ks=%.3f (Vega)",
            source_id,
            twomass_data.get("twomass_id"),
            _finite_float(twomass_data.get("j_m"), float("nan")),
            _finite_float(twomass_data.get("h_m"), float("nan")),
            _finite_float(twomass_data.get("ks_m"), float("nan")),
        )
    else:
        logger.warning("E2E: no 2MASS counterpart found for source_id=%s.", source_id)

    meta = build_target_meta_from_gaia(
        star_data,
        mjd_mid=mjd_mid,
        target_name=target_name,
        telescope=telescope,
    )

    logger.info(
        "E2E Gaia target %s: G=%.2f, Gaia RV=%.3f km/s, BERV=%.3f km/s (CCF ccf_zb only), "
        "extra RV=%.3f km/s, injected stellar RV=%.3f km/s, topo Doppler=%.3f km/s, mjd_mid=%.5f",
        meta["target_name"],
        _finite_float(star_data.get("phot_g_mean_mag"), float("nan")),
        gaia_rv_km_s,
        berv_km_s,
        extra_rv_km_s,
        injected_rv_km_s,
        gaia_rv_km_s - berv_km_s + float(extra_rv_km_s),
        mjd_mid,
    )

    return E2EGaiaTarget(
        source_id=str(star_data["source_id"]),
        star_data=star_data,
        target_meta=meta,
        gaia_rv_km_s=gaia_rv_km_s,
        berv_km_s=berv_km_s,
        injected_rv_km_s=injected_rv_km_s,
        extra_rv_km_s=float(extra_rv_km_s),
        mjd_mid=float(mjd_mid),
        twomass_data=twomass_data,
    )


def apply_target_meta_to_model(model, meta: dict) -> None:
    """Set target / exposure metadata fields on a data model."""
    for key, value in meta.items():
        if value is None:
            continue
        if hasattr(model.meta, key):
            setattr(model.meta, key, value)
