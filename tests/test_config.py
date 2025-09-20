from hispecdrp import FitLFCLinesPrimitive, ProcessLFCPipeline
from hispecdrp.utils import initialize_from_config

import os
import textwrap


def save_config_string(config_str : str, path : str):
    with open(path, 'w') as f:
        f.write(textwrap.dedent(config_str))

def test_primitive_config(tmp_path):

    # Test with no config file or args
    prim = FitLFCLinesPrimitive()
    assert prim.do_orders is None
    assert prim.fwhm_bounds == prim.ARGUMENTS['fwhm_bounds']

    # Initial args
    do_orders = [90, 91, 92, 93, 94, 96, 97, 98, 99]
    fwhm_bounds = [0.1, 2.5]
    do_fibers = [3]
    lfc_nu0 = 100.0
    lfc_delta_nu = 0.1

    # Specify config file
    config_string = f"""
    %YAML 1.2
    ---
    class: FitLFCLinesPrimitive
    arguments:
      do_orders: {do_orders}
      do_fibers: {do_fibers}
      fwhm_bounds: {fwhm_bounds}
      lfc_nu0: {lfc_nu0}
      lfc_delta_nu: {lfc_delta_nu}
    """
    config_path = os.path.join(tmp_path, 'test_config.yaml')
    save_config_string(config_string, config_path)

    # Test with config file
    prim = FitLFCLinesPrimitive(config=config_path)
    assert prim.do_orders == do_orders
    assert prim.fwhm_bounds == fwhm_bounds

    # Override config file argument with a different value
    do_orders = [90, 91, 92, 93, 94, 96]
    prim = FitLFCLinesPrimitive(config=config_path, do_orders=do_orders)
    assert prim.do_orders == do_orders

    # Get primitive from class attr
    new_fwhm_bounds = [0.5, 1.123]
    prim = initialize_from_config(config_path, fwhm_bounds=new_fwhm_bounds)
    assert isinstance(prim, FitLFCLinesPrimitive)
    assert prim.lfc_nu0 == lfc_nu0
    assert prim.lfc_delta_nu == lfc_delta_nu
    assert prim.fwhm_bounds == new_fwhm_bounds


def test_pipeline_config(tmp_path):

    # Initial args
    do_orders = [90, 91, 92, 93, 94, 96, 97, 98, 99]
    fwhm_bounds = [0.1, 2.5]
    do_fibers = [3]
    lfc_nu0 = 100.0
    lfc_delta_nu = 0.1
    prim_args = {
        'fit_lfc_lines' : {
            'do_orders': do_orders,
            'do_fibers': do_fibers,
            'fwhm_bounds': fwhm_bounds,
            'mu_bounds': (-1, 1),
            'lfc_nu0': lfc_nu0,
            'lfc_delta_nu': lfc_delta_nu,
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
    }

    # Test with no config file or args
    pipe = ProcessLFCPipeline(prim_args=prim_args)
    assert pipe.fit_lfc_lines.do_orders == do_orders
    assert pipe.fit_lfc_lines.fwhm_bounds == fwhm_bounds

    # Pipeline config file
    config_string = f"""
    %YAML 1.2
    ---
    class: ProcessLFCPipeline
    primitives:
      fit_lfc_lines:
        arguments:
          do_orders: {do_orders}
          do_fibers: {do_fibers}
          fwhm_bounds: {fwhm_bounds}
          lfc_nu0: {lfc_nu0}
          lfc_delta_nu: {lfc_delta_nu}
    """
    config_path = os.path.join(tmp_path, 'test_pipe_config.yaml')
    save_config_string(config_string, config_path)

    # Test with config file
    pipe = ProcessLFCPipeline(config=config_path)
    assert pipe.fit_lfc_lines.do_orders == do_orders
    assert pipe.fit_lfc_lines.fwhm_bounds == fwhm_bounds

    # Override config file argument with a different value
    do_orders = [90, 91]
    prim = FitLFCLinesPrimitive(config=config_path, do_orders=do_orders)
    assert prim.do_orders == do_orders

    pipe_config_path = os.path.join(tmp_path, 'test_pipe_config.yaml')
    prim_config_path = os.path.join(tmp_path, 'test_prim_config.yaml')

    # Pipeline config file with primitive config file
    prim_config_string = f"""
    %YAML 1.2
    ---
    arguments:
      do_orders: {do_orders}
      do_fibers: {do_fibers}
      fwhm_bounds: {list(fwhm_bounds)}
      lfc_nu0: {lfc_nu0}
      lfc_delta_nu: {lfc_delta_nu}
    """
    new_fwhm_bounds = [0.5, 1.123]
    pipe_config_string = f"""
    %YAML 1.2
    ---
    class: ProcessLFCPipeline
    primitives:
      fit_lfc_lines:
        config_file: {prim_config_path}
        arguments:
          fwhm_bounds: {new_fwhm_bounds}
    """
    save_config_string(prim_config_string, prim_config_path)
    save_config_string(pipe_config_string, pipe_config_path)

    # Test with config file
    pipe = ProcessLFCPipeline(config=pipe_config_path)
    assert pipe.fit_lfc_lines.do_orders == do_orders
    assert pipe.fit_lfc_lines.fwhm_bounds == new_fwhm_bounds

    # Override config file argument with a different value
    do_orders = [90, 91, 99]
    prim = FitLFCLinesPrimitive(config=pipe_config_path, do_orders=do_orders)
    assert prim.do_orders == do_orders

    # Get primitive from class attr
    pipe = initialize_from_config(pipe_config_path)
    assert isinstance(pipe, ProcessLFCPipeline)
    assert pipe.fit_lfc_lines.lfc_nu0 == lfc_nu0
    assert pipe.fit_lfc_lines.lfc_delta_nu == lfc_delta_nu
    assert pipe.fit_lfc_lines.fwhm_bounds == new_fwhm_bounds