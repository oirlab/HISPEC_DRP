"""Tests for barycenter correction helpers and BERV plumbing.
"""
from hispecdrp import datamodels
from hispecdrp.primitives import BarycenterCorrectionPrimitive
import numpy as np

from hispecdrp.primitives.utils import (
    get_stellar_data_gaia_from_source_id,
)

# HD 189733
source_params = {
    "source_id": '1827242816201846144',
    "ra": 300.1821218062407,  # degrees
    "dec": 22.709741105168273,  # degrees
    "pm_ra": -3.208,  # mas/yr
    "pm_dec": -250.323,  # mas/yr
    "epoch": 2000.0,
}

def test_calc_barycorr(tmp_path, datadir):

    mjd_utc = 49254.5
    star_data = get_stellar_data_gaia_from_source_id(source_id=source_params["source_id"])
    meta = {
        'target_name': 'HD189733',
        'target_gaia_id' : '1827242816201846144',
        'target_gaia_ra': star_data['ra'],
        'target_gaia_dec': star_data['dec'],
        'target_gaia_epoch': star_data["ref_epoch"],
        'target_gaia_rapm': star_data["pmra"],
        'target_gaia_decpm': star_data["pmdec"],
        'target_gaia_px': star_data["parallax"],
        'target_gaia_rvsys': star_data["radial_velocity"],
        'mjd_mid' : mjd_utc,
    }
    input_model = datamodels.L2Model(meta=meta)
    num_orders = len(input_model.make_order_vec())
    npix = 4096
    input_model.spec_fib1 = np.zeros((num_orders, npix), dtype=np.float64)
    input_model.spec_err_fib1 = np.ones((num_orders, npix), dtype=np.float64)
    input_model.spec_dq_fib1 = np.zeros((num_orders, npix), dtype=np.uint32)
    input_model.mid_time_fib1 = np.full((num_orders, npix), mjd_utc, dtype=np.float64)

    output_model = BarycenterCorrectionPrimitive.call(input_model)

    bc_table = output_model.barycorr_fib1
    assert bc_table is not None
    assert len(bc_table) == 1
    mjdutc_out = bc_table['MJDUTC'][0]
    mbjd_out = bc_table['MBJD'][0]
    berv_out = bc_table['BERV'][0]

    assert mjdutc_out == mjd_utc
    assert 0.001 < abs(mbjd_out - mjd_utc) < 0.01
    assert -30_000 < berv_out < 30_000


def test_force_berv_m_s_skips_barycorrpy():
    """Simulated exposures can stamp BERV exactly without calling barycorrpy."""
    mjd_utc = 60577.0
    meta = {
        "target_name": "sim_rv0",
        "force_berv_m_s": 0.0,
        "mjd_mid": mjd_utc,
    }
    input_model = datamodels.L2Model(meta=meta)
    num_orders = len(input_model.make_order_vec())
    npix = 64
    input_model.spec_fib1 = np.zeros((num_orders, npix), dtype=np.float64)
    input_model.spec_err_fib1 = np.ones((num_orders, npix), dtype=np.float64)
    input_model.spec_dq_fib1 = np.zeros((num_orders, npix), dtype=np.uint32)
    input_model.mid_time_fib1 = np.full((num_orders, npix), mjd_utc, dtype=np.float64)

    output_model = BarycenterCorrectionPrimitive.call(input_model)
    bc_table = output_model.barycorr_fib1
    assert bc_table is not None
    assert float(bc_table["BERV"][0]) == 0.0
    assert float(bc_table["MJDUTC"][0]) == mjd_utc
    assert float(bc_table["MBJD"][0]) == mjd_utc


def test_berv_m_s_to_ccf_zb_units():
    from hispecdrp.primitives.barycenter_correction.barycenter_corr import (
        berv_m_s_to_ccf_zb,
        resolve_ccf_zb,
    )
    from hispecdrp.utils.physical_constants import C_M_S

    berv_m_s = 1244.0
    zb = berv_m_s_to_ccf_zb(berv_m_s)
    assert zb == berv_m_s / C_M_S
    assert abs(zb * C_M_S - berv_m_s) < 1e-12


def test_resolve_ccf_zb_from_barycorr_table():
    from astropy.table import Table

    from hispecdrp.primitives.barycenter_correction.barycenter_corr import (
        berv_m_s_to_ccf_zb,
        resolve_ccf_zb,
    )

    berv_m_s = -2399.0
    model = datamodels.L2Model(meta={"target_name": "test"})
    model.barycorr_fib1 = Table(
        {"MJDUTC": [61209.7], "MBJD": [61209.701], "BERV": [berv_m_s]}
    )
    assert resolve_ccf_zb(model, 1) == berv_m_s_to_ccf_zb(berv_m_s)
    assert resolve_ccf_zb(model, 1, ccf_zb=0.5, from_barycorr=False) == 0.5
    assert resolve_ccf_zb(model, 2, ccf_zb=0.25) == 0.25


def test_stellar_rv_uses_barycorr_ccf_zb_per_fiber():
    from hispecdrp.primitives import BarycenterCorrectionPrimitive, StellarRVPrimitive

    mjd_utc = 49254.5
    star_data = get_stellar_data_gaia_from_source_id(source_id=source_params["source_id"])
    meta = {
        "target_name": "HD189733",
        "target_gaia_id": "1827242816201846144",
        "target_gaia_ra": star_data["ra"],
        "target_gaia_dec": star_data["dec"],
        "target_gaia_epoch": star_data["ref_epoch"],
        "target_gaia_rapm": star_data["pmra"],
        "target_gaia_decpm": star_data["pmdec"],
        "target_gaia_px": star_data["parallax"],
        "target_gaia_rvsys": star_data["radial_velocity"],
        "mjd_mid": mjd_utc,
    }
    input_model = datamodels.L2Model(meta=meta)
    num_orders = len(input_model.make_order_vec())
    npix = 4096
    input_model.spec_fib1 = np.zeros((num_orders, npix), dtype=np.float64)
    input_model.spec_err_fib1 = np.ones((num_orders, npix), dtype=np.float64)
    input_model.spec_dq_fib1 = np.zeros((num_orders, npix), dtype=np.uint32)
    input_model.mid_time_fib1 = np.full((num_orders, npix), mjd_utc, dtype=np.float64)

    bary_model = BarycenterCorrectionPrimitive.call(input_model)
    berv_m_s = float(bary_model.barycorr_fib1["BERV"][0])

    prim = StellarRVPrimitive(bary_model, do_fibers=['1'], do_orders=[139])
    from hispecdrp.primitives.barycenter_correction.barycenter_corr import berv_m_s_to_ccf_zb
    from hispecdrp.utils.physical_constants import C_M_S

    zb = StellarRVPrimitive.resolve_ccf_zb_for_fiber(prim, bary_model, "1")
    assert abs(zb - berv_m_s_to_ccf_zb(berv_m_s)) < 1e-15
    assert abs(zb * C_M_S - berv_m_s) < 1e-9


def test_calc_solar_barycenter_corrections(tmp_path, datadir):
    mjd_utc = 49254.5
    star_data = get_stellar_data_gaia_from_source_id(source_id=source_params["source_id"])
    meta = {
        'target_name': 'SOLAR',
        'target_gaia_id' : '1827242816201846144',
        'target_gaia_ra': star_data['ra'],
        'target_gaia_dec': star_data['dec'],
        'target_gaia_epoch': star_data["ref_epoch"],
        'target_gaia_rapm': star_data["pmra"],
        'target_gaia_decpm': star_data["pmdec"],
        'target_gaia_px': star_data["parallax"],
        'target_gaia_rvsys': star_data["radial_velocity"],
        'mjd_mid' : mjd_utc,
    }
    input_model = datamodels.L2Model(meta=meta)
    num_orders = len(input_model.make_order_vec())
    npix = 4096
    input_model.spec_fib1 = np.zeros((num_orders, npix), dtype=np.float64)
    input_model.spec_err_fib1 = np.ones((num_orders, npix), dtype=np.float64)
    input_model.spec_dq_fib1 = np.zeros((num_orders, npix), dtype=np.uint32)
    input_model.mid_time_fib1 = np.full((num_orders, npix), mjd_utc, dtype=np.float64)

    output_model = BarycenterCorrectionPrimitive.call(input_model)

    bc_table = output_model.barycorr_fib1
    assert bc_table is not None
    assert len(bc_table) == 1
    mjdutc_out = bc_table['MJDUTC'][0]
    mbjd_out = bc_table['MBJD'][0]
    berv_out = bc_table['BERV'][0]

    assert mjdutc_out == mjd_utc
    assert 0.001 < abs(mbjd_out - mjd_utc) < 0.01
    assert -1000 < berv_out < 1000


def test_calc_barycorr_with_simulated_spec_dq(tmp_path, datadir):
    """Regression: spec_dq==0 marks good pixels (see dqflags.GOOD)."""
    mjd_utc = 61211.5
    star_data = get_stellar_data_gaia_from_source_id(source_id=source_params["source_id"])
    meta = {
        'target_name': 'HD189733',
        'target_gaia_id': '1827242816201846144',
        'target_gaia_ra': star_data['ra'],
        'target_gaia_dec': star_data['dec'],
        'target_gaia_epoch': star_data["ref_epoch"],
        'target_gaia_rapm': star_data["pmra"],
        'target_gaia_decpm': star_data["pmdec"],
        'target_gaia_px': star_data["parallax"],
        'target_gaia_rvsys': star_data["radial_velocity"],
        'mjd_mid': mjd_utc,
        'telescope': 'Keck-II',
    }
    input_model = datamodels.L2Model(meta=meta)
    num_orders = len(input_model.make_order_vec())
    npix = 4096
    rng = np.random.default_rng(0)
    input_model.spec_fib1 = rng.normal(size=(num_orders, npix))
    input_model.spec_err_fib1 = rng.uniform(0.5, 2.0, size=(num_orders, npix))
    input_model.spec_err_fib1[0, :10] = np.nan
    input_model.spec_dq_fib1 = np.zeros((num_orders, npix), dtype=np.uint32)
    input_model.spec_dq_fib1[0, 100:200] = datamodels.DQ_FLAGS["DO_NOT_USE"]
    input_model.mid_time_fib1 = np.linspace(
        mjd_utc - 0.001, mjd_utc + 0.001, num=num_orders * npix
    ).reshape(num_orders, npix)

    output_model = BarycenterCorrectionPrimitive.call(input_model)
    bc_table = output_model.barycorr_fib1
    assert bc_table is not None
    assert np.isfinite(bc_table['MJDUTC'][0])
    assert abs(bc_table['MJDUTC'][0] - mjd_utc) < 0.01