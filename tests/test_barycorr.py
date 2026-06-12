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
    input_model.mid_time_fib1 = np.full((num_orders, npix), mjd_utc, dtype=np.float64)

    prim = BarycenterCorrectionPrimitive()
    output_model = prim.apply(input_model)

    bc_table = output_model.barycorr_fib1
    assert bc_table is not None
    assert len(bc_table) == 1
    mjdutc_out = bc_table['MJDUTC'][0]
    mbjd_out = bc_table['MBJD'][0]
    berv_out = bc_table['BERV'][0]

    assert mjdutc_out == mjd_utc
    assert 0.001 < abs(mbjd_out - mjd_utc) < 0.01
    assert -30_000 < berv_out < 30_000

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
    input_model.mid_time_fib1 = np.full((num_orders, npix), mjd_utc, dtype=np.float64)

    prim = BarycenterCorrectionPrimitive()
    output_model = prim.apply(input_model)

    bc_table = output_model.barycorr_fib1
    assert bc_table is not None
    assert len(bc_table) == 1
    mjdutc_out = bc_table['MJDUTC'][0]
    mbjd_out = bc_table['MBJD'][0]
    berv_out = bc_table['BERV'][0]

    assert mjdutc_out == mjd_utc
    assert 0.001 < abs(mbjd_out - mjd_utc) < 0.01
    assert -1000 < berv_out < 1000