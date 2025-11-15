import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from hispecdrp import datamodels, TracePositionPrimitive, TraceProfilePrimitive, CoaddFramesPrimitive, SpectralExtractionPrimitive
from hispecdrp.utils import download_gdrive_file
from hispecdrp.primitives.spectral_extraction.extraction_utils import get_extraction_window

def test_stage1(tmp_path, datadir):
    
    fiberflat_filepaths = {
        '1' : [
            download_gdrive_file('PARVI/L0/PR.20230901.81619.65_rate-FIBERFLAT-FIB1.fits', output_dir=datadir),
            download_gdrive_file('PARVI/L0/PR.20230901.81656.55_rate-FIBERFLAT-FIB1.fits', output_dir=datadir),
            download_gdrive_file('PARVI/L0/PR.20230901.81693.45_rate-FIBERFLAT-FIB1.fits', output_dir=datadir),
        ],
        '2' : [
            download_gdrive_file('PARVI/L0/PR.20230901.82173.17_rate-FIBERFLAT-FIB2.fits', output_dir=datadir),
            download_gdrive_file('PARVI/L0/PR.20230901.82210.07_rate-FIBERFLAT-FIB2.fits', output_dir=datadir),
            download_gdrive_file('PARVI/L0/PR.20230901.82246.97_rate-FIBERFLAT-FIB2.fits', output_dir=datadir),
        ],
    }
    fibers = list(fiberflat_filepaths.keys())
    ff_models = {}
    for fiber, filepaths in fiberflat_filepaths.items():
        ff_models[fiber] = [datamodels.open(fp) for fp in filepaths]
    
    # Coadd Fiberflats per fiber
    prim = CoaddFramesPrimitive()
    ff_models_coadded = prim.apply(ff_models)

    yrange_parvi = {
        '1' : (155, 1915),
        '2' : (167, 1930),
    }
    xrange = (200, 2048-200)
    deg_pos = 4
    deg_width = 2
    prim = TracePositionPrimitive()
    trace_pos_model = prim.apply(ff_models_coadded, yrange=yrange_parvi, deg_pos=deg_pos, deg_width=deg_width, xrange=xrange)

    # Trace Profiles per fiber
    prim = TraceProfilePrimitive()
    trace_prof_model = prim.apply(ff_models_coadded, tracepos=trace_pos_model)
    for fiber in fibers:
        P = getattr(trace_prof_model, f'prof_fib{fiber}')
        P[P > 0] = 1

    order_vec = trace_pos_model.make_order_vec()

    # Extraction parameters
    extraction_aperture = 5
    do_orders = [94]

    filepath_solar = download_gdrive_file('PARVI/L0/PR.20230802.82176.93_rate-SOLAR.fits', output_dir=datadir)
    temp_model = datamodels.open(filepath_solar)

    for fiber in fibers:
        for x in range(temp_model.rate.shape[1]):
            oi = np.abs(do_orders[0] - trace_pos_model.meta.order_start)
            trace_positions = getattr(trace_pos_model, f'POS_FIB{fiber}')[oi, x]
            window, W = get_extraction_window(trace_positions, extraction_aperture+4)
            temp_model.rate[window, x] = 100
            temp_model.rate_err[window, x] = 1
            if x % 10 == 0:
                temp_model.dq[window[3], x] = 1

    # Box extraction
    spec_box = {}
    for fiber in fibers:
        prim = SpectralExtractionPrimitive()
        spec_box[fiber] = prim.apply(
            temp_model,
            tracepos=trace_pos_model,
            do_fibers=[fiber],
            do_orders=do_orders,
            traceprof=trace_prof_model,
            xrange=xrange,
            extraction_aperture=extraction_aperture,
            method='box',
        )

    # Optimal Extraction
    spec_opt = {}
    for fiber in fibers:
        prim = SpectralExtractionPrimitive()
        spec_opt[fiber] = prim.apply(
            temp_model,
            tracepos=trace_pos_model,
            do_fibers=[fiber],
            do_orders=do_orders,
            traceprof=trace_prof_model,
            xrange=xrange,
            extraction_aperture=extraction_aperture,
            method='optimal',
        )

    # Cross-talk
    prim = SpectralExtractionPrimitive()
    spec_ct = prim.apply(
        temp_model,
        tracepos=trace_pos_model,
        do_fibers=['1', '2'],
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

            spectrum_box = getattr(spec_box[fiber], f'spec_uncal_fib{fiber}')[oi, :]
            good = np.where(np.isfinite(spectrum_box))[0]
            assert len(good) > 0, f"Fiber {fiber} Order {order}: No good pixels in box spectrum"
            np.testing.assert_allclose(spectrum_box[good], 500, atol=1E-3)
            
            spectrum_opt = getattr(spec_opt[fiber], f'spec_uncal_fib{fiber}')[oi, :]
            good = np.where(np.isfinite(spectrum_opt))[0]
            assert len(good) > 0, f"Fiber {fiber} Order {order}: No good pixels in optimal spectrum"
            np.testing.assert_allclose(spectrum_opt[good], 500, atol=1E-3)

            spectrum_ct = getattr(spec_ct, f'spec_uncal_fib{fiber}')[oi, :]
            good = np.where(np.isfinite(spectrum_ct))[0]
            assert len(good) > 0, f"Fiber {fiber} Order {order}: No good pixels in cross-talk spectrum"
            np.testing.assert_allclose(spectrum_ct[good], 500, atol=1E-3)

            # plt.plot(spectrum_box, label='Box')
            # plt.plot(spectrum_opt, label='Optimal')
            # plt.plot(spectrum_ct, label='CT')
            # plt.legend()
            # plt.show()