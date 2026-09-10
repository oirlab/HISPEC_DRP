"""Tests for noiseless template-on-WAVE L2 builder."""

from pathlib import Path

from hispecdrp.simulations.spectra import (
    build_ccfopt_mask,
    build_l2,
    ensure_lsf_ccfopt_mask,
    paint_order_list,
)

LTE040 = "lte040.0-4.5-0.0a+0.0.BT-Settl.spec.7.h5"


def test_paint_order_list_bspec_defaults_no_hdf():
    orders = paint_order_list("BSPEC", 1, None, None)
    assert orders[0] == 110
    assert orders[-1] == 148
    assert len(orders) == 39


def test_apply_ccfopt_defaults_for_line_by_line():
    from hispecdrp.utils.line_mask_utils import (
        CCFOPT_SINGLE_LINE_STELLAR_RV_KW,
        STELLAR_RV_WIDE_WINDOW_UM,
        apply_ccfopt_defaults_for_line_by_line,
    )

    ws, vh, vs = apply_ccfopt_defaults_for_line_by_line(
        window_size_um=STELLAR_RV_WIDE_WINDOW_UM,
        v_half_window=15000.0,
        v_step_coarse=25.0,
    )
    kw = CCFOPT_SINGLE_LINE_STELLAR_RV_KW
    assert ws == kw["window_size_um"]
    assert vh == kw["v_half_window_m_s"]
    assert vs == kw["v_step_m_s"]

    ws2, vh2, vs2 = apply_ccfopt_defaults_for_line_by_line(
        window_size_um=0.3,
        v_half_window=12000.0,
        v_step_coarse=5.0,
    )
    assert ws2 == 0.3
    assert vh2 == 12000.0
    assert vs2 == 5.0


def test_build_l2_single_order_shape():
    orders = paint_order_list("BSPEC", 1, 130, 130)
    assert orders == (130,)
    l2 = build_l2(
        10.0,
        spectrograph="BSPEC",
        template=LTE040,
        fiber=1,
        order_start=130,
        order_end=130,
    )
    assert l2.meta.order_start == 130
    assert l2.meta.order_end == 130
    assert l2.spec_fib1.shape == (1, 4096)
    assert l2.wave_fib1.shape == (1, 4096)
    from hispecdrp.simulations.e2e_gaia import DEFAULT_E2E_MJD_START

    assert l2.meta.mjd_start == DEFAULT_E2E_MJD_START
    assert l2.meta.datetime_obs


def test_ensure_lsf_ccfopt_mask_reuses_full_band(tmp_path):
    dest = tmp_path / f"{Path(LTE040).stem}.lsf.ccfopt.delta.mas"
    dest.write_text(
        '# HISPEC_DRP_MASK_META {"n_lines": 867, "n_seed_lines": 1066}\n'
        "1000.0 1.0\n"
    )
    path = ensure_lsf_ccfopt_mask(LTE040, tmp_path, fiber=1)
    assert path == str(dest)


def test_build_ccfopt_mask_narrow_orders(tmp_path):
    seed_path, ccfopt_path, l2 = build_ccfopt_mask(
        LTE040,
        tmp_path,
        order_start=130,
        order_end=131,
    )
    assert Path(seed_path).is_file()
    assert Path(ccfopt_path).is_file()
    assert l2.meta.order_start == 130
    assert l2.meta.order_end == 131
    assert l2.spec_fib1.shape[0] == 2
