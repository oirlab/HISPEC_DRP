from hispecdrp import datamodels
from hispecdrp import FitLFCLinesPrimitive, ProcessLFCPipeline
from hispecdrp.utils import download_gdrive_file

import numpy as np
import os

def test_save_result_primitive(tmp_path, datadir):

    # Setup
    lfc_path = download_gdrive_file('PARVI/L2/PR.20230725.83766.15_spec-LFC.fits', use_cached=True, output_dir=datadir)
    do_orders = [90, 91, 92, 93, 94, 96, 97, 98, 99]
    do_fibers = ['3']

    # Test passing output_dir
    prim = FitLFCLinesPrimitive()
    suffix = 'my-test-file'
    line_fits_model = prim.apply(
        lfc_path,
        do_orders=do_orders,
        do_fibers=do_fibers,
        save_result=True,
        output_dir=tmp_path,
        suffix=suffix
    )
    assert os.path.isfile(line_fits_model._filepath)

    # Test that the suffix was applied
    assert line_fits_model._filepath.endswith(f'{suffix}.fits')
    assert line_fits_model.meta.filename.endswith(f'{suffix}.fits')

    # Test passing output_path
    output_path = str(tmp_path / 'my-test-file.fits')
    prim = FitLFCLinesPrimitive()
    line_fits_model = prim.apply(
        lfc_path,
        do_orders=do_orders,
        do_fibers=do_fibers,
        save_result=True,
        output_path=output_path,
    )
    assert line_fits_model._filepath == output_path
    assert line_fits_model.meta.filename == os.path.basename(output_path)
    assert os.path.isfile(output_path)

    # Test output_dir with in memory data model (model._filepath = model.meta.filename = None)
    lfc_model = datamodels.open(lfc_path)
    lfc_model._filepath = None
    lfc_model.meta.filename = None
    prim = FitLFCLinesPrimitive()
    line_fits_model = prim.apply(
        lfc_model,
        do_orders=do_orders,
        do_fibers=do_fibers,
        save_result=True,
        output_dir=tmp_path,
    )
    output_path = str(tmp_path / line_fits_model.meta.filename)
    assert line_fits_model._filepath == output_path
    assert os.path.isfile(output_path)

def test_save_result_pipeline(tmp_path, datadir):
    do_orders = [90, 91, 92, 93, 94, 96, 97, 98, 99]
    do_fibers = ['3']
    lfc_path = download_gdrive_file('PARVI/L2/PR.20230725.83766.15_spec-LFC.fits', use_cached=True, output_dir=datadir)
    os.environ['KOA_CALIBRATION_CACHE'] = str(tmp_path) + '/KOA_CALIBRATION_CACHE/'
    prim_args = {
        'fit_lfc_lines' : {
            'do_orders': do_orders,
            'do_fibers': do_fibers,
        },
        'lfc_wavesol' : {
            'fit_orders': do_orders,
            'opt_weights': True,
            'do_fibers': do_fibers,
            'deg_inter_order': 6,
            'deg_intra_order': 9,
        },
        'lsf' : {
            'do_fibers': do_fibers,
            'deg_inter_order': 3,
            'deg_intra_order': 3,
            'fit_orders': do_orders,
        },
        'register_wave_cal' : {
            'run': True,
        },
        'register_lsf_cal' : {
            'run': True
        }
    }

    # Test saving a particular primitive's result.
    _prim_args = prim_args.copy()
    _prim_args['lfc_wavesol']['save_result'] = True
    _prim_args['lfc_wavesol']['output_dir'] = tmp_path
    pipe = ProcessLFCPipeline()
    result = pipe.apply(lfc_path, prim_args=_prim_args)
    wavesol_model = result['wavesol']
    assert os.path.isfile(wavesol_model._filepath)

    # Test pipeline level saving
    _prim_args = prim_args.copy()
    suffix = 'saved-from-pipeline'
    pipe = ProcessLFCPipeline()
    result = pipe.apply(lfc_path, prim_args=_prim_args, save_result=True, output_dir=tmp_path, suffix=suffix)
    wavesol_model = result['wavesol']
    lsf_model = result['lsf']
    assert wavesol_model._filepath == str(tmp_path / wavesol_model.generate_filename(suffix=suffix))
    assert lsf_model._filepath == str(tmp_path / lsf_model.generate_filename(suffix=suffix))
    assert os.path.isfile(wavesol_model._filepath)
    assert os.path.isfile(lsf_model._filepath)