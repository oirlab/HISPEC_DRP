import numpy as np
import matplotlib.pyplot as plt
from hispecdrp import datamodels, TracePositionPrimitive, TraceProfilePrimitive, CoaddFramesPrimitive, SpectralExtractionPrimitive
from hispecdrp.utils import download_gdrive_file

def test_stage1_hispec(tmp_path, datadir):
    
    fiberflat_filepaths = [
        download_gdrive_file('HISPEC/L0/HB.20240924.0000.00_rate-FIBERFLAT-FIB1.fits', use_cached=True, output_dir=datadir),
        download_gdrive_file('HISPEC/L0/HB.20240924.0000.00_rate-FIBERFLAT-FIB2.fits', use_cached=True, output_dir=datadir),
        download_gdrive_file('HISPEC/L0/HB.20240924.0000.00_rate-FIBERFLAT-FIB3.fits', use_cached=True, output_dir=datadir)
    ]
    fibers = ['1', '2', '3']
    ff_models = [datamodels.open(fp) for fp in fiberflat_filepaths]

    # Coadd Fiberflats per fiber
    prim = CoaddFramesPrimitive()
    ff_models_coadded = prim.apply(ff_models)

    # Trace Positions per fiber
    yrange = (100, 4050)
    xrange = (500, 4000)
    deg_pos = 4
    deg_width = 2
    prim = TracePositionPrimitive()
    trace_pos_model = prim.apply(ff_models_coadded, yrange=yrange, deg_pos=deg_pos, deg_width=deg_width, xrange=xrange)

    # Trace Profiles per fiber
    prim = TraceProfilePrimitive()
    trace_prof_model = prim.apply(ff_models_coadded, tracepos=trace_pos_model)

    # Create combined fiberflat model for cross-talk extraction
    fiberflat_all_fibers_model = datamodels.HISPECL0Model()
    fiberflat_all_fibers_model.meta = ff_models_coadded[0].meta
    fiberflat_all_fibers_model.rate = getattr(ff_models_coadded[0], 'rate') + getattr(ff_models_coadded[1], 'rate') + getattr(ff_models_coadded[2], 'rate')
    fiberflat_all_fibers_model.rate_err = np.sqrt(getattr(ff_models_coadded[0], 'rate_err')**2 + getattr(ff_models_coadded[1], 'rate_err')**2 + getattr(ff_models_coadded[2], 'rate_err')**2)
    fiberflat_all_fibers_model.rate_var_rnoise = getattr(ff_models_coadded[0], 'rate_var_rnoise') + getattr(ff_models_coadded[1], 'rate_var_rnoise') + getattr(ff_models_coadded[2], 'rate_var_rnoise')
    fiberflat_all_fibers_model.rate_var_poisson = getattr(ff_models_coadded[0], 'rate_var_poisson') + getattr(ff_models_coadded[1], 'rate_var_poisson') + getattr(ff_models_coadded[2], 'rate_var_poisson')
    fiberflat_all_fibers_model.dq = getattr(ff_models_coadded[0], 'dq') | getattr(ff_models_coadded[1], 'dq') | getattr(ff_models_coadded[2], 'dq')

    order_vec = trace_pos_model.make_order_vec()

    # Extract Spectra per fiber/order
    extraction_aperture = 10
    do_orders = order_vec
    
    prim = SpectralExtractionPrimitive()
    spec_box = prim.apply(
        fiberflat_all_fibers_model,
        tracepos=trace_pos_model,
        do_fibers=fibers,
        do_orders=do_orders,
        traceprof=trace_prof_model,
        xrange=xrange,
        extraction_aperture=extraction_aperture,
        method='box',
    )

    spec_opt = prim.apply(
        fiberflat_all_fibers_model,
        tracepos=trace_pos_model,
        do_fibers=fibers,
        do_orders=do_orders,
        traceprof=trace_prof_model,
        xrange=xrange,
        extraction_aperture=extraction_aperture,
        method='optimal',
    )

    # NOTE: This works well for HISPEC but ignore tests for now.
    prim = SpectralExtractionPrimitive()
    spec_ct = prim.apply(
        fiberflat_all_fibers_model,
        tracepos=trace_pos_model,
        do_fibers=fibers,
        do_orders=do_orders,
        traceprof=trace_prof_model,
        xrange=xrange,
        extraction_aperture=extraction_aperture,
        method='cross_talk',
    )

    # Check average SNR for each order
    for fiber in fibers:
    	for order in do_orders:

            oi = np.abs(order - trace_pos_model.meta.order_start)
            print(f"Order index: {oi}")

            spectrum_box = getattr(spec_box, f'spec_uncal_fib{fiber}')[oi, :]
            error_box = getattr(spec_box, f'spec_uncal_err_fib{fiber}')[oi, :]
            snr_box = np.nanmean(spectrum_box / error_box)
            
            print(f"Fiber {fiber} Order {order}: Box SNR = {snr_box:.2f}")
            assert snr_box > 10, f"Fiber {fiber} Order {order}: Box SNR < 10"

            spectrum_opt = getattr(spec_opt, f'spec_uncal_fib{fiber}')[oi, :]
            error_opt = getattr(spec_opt, f'spec_uncal_err_fib{fiber}')[oi, :]
            snr_opt = np.nanmean(spectrum_opt / error_opt)
            
            print(f"Fiber {fiber} Order {order}: OPT SNR = {snr_opt:.2f}")
            assert snr_opt > 10, f"Fiber {fiber} Order {order}: OPT SNR < 10"

            spectrum_ct = getattr(spec_ct, f'spec_uncal_fib{fiber}')[oi, :]
            error_ct = getattr(spec_ct, f'spec_uncal_err_fib{fiber}')[oi, :]
            snr_ct = np.nanmean(spectrum_ct / error_ct)
            
            print(f"Fiber {fiber} Order {order}: CT SNR = {snr_ct:.2f}")
            assert snr_ct > 10, f"Fiber {fiber} Order {order}: CT SNR < 10"

            # plt.plot(spectrum_box, label='BOX')
            # plt.plot(spectrum_opt, label='OPT')
            # plt.plot(spectrum_ct, label='CT')
            # plt.legend()
            # plt.show()