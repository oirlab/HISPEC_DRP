import numpy as np
from hispecdrp import datamodels, TracePositionPrimitive, TraceProfilePrimitive, CoaddFramesPrimitive, SpectralExtractionPrimitive, Stage1Pipeline, ProcessFiberflatsPipeline
import matplotlib.pyplot as plt
from hispecdrp.utils import download_gdrive_file
import os

# TODO: Update this test to mirror test_stage1_hispec.py but for PARVI data.
def _test_stage1_parvi(tmp_path, datadir):

    os.environ['KOA_CALIBRATION_CACHE'] = str(tmp_path) + '/KOA_CALIBRATION_CACHE/'

    fiberflat_filepaths = [
        download_gdrive_file('PARVI/L0/PR.20230901.81619.65_rate-FIBERFLAT-FIB1.fits', output_dir=datadir),
        download_gdrive_file('PARVI/L0/PR.20230901.81656.55_rate-FIBERFLAT-FIB1.fits', output_dir=datadir),
        download_gdrive_file('PARVI/L0/PR.20230901.81693.45_rate-FIBERFLAT-FIB1.fits', output_dir=datadir),
        download_gdrive_file('PARVI/L0/PR.20230901.82173.17_rate-FIBERFLAT-FIB2.fits', output_dir=datadir),
        download_gdrive_file('PARVI/L0/PR.20230901.82210.07_rate-FIBERFLAT-FIB2.fits', output_dir=datadir),
        download_gdrive_file('PARVI/L0/PR.20230901.82246.97_rate-FIBERFLAT-FIB2.fits', output_dir=datadir),
    ]
    fibers = ['1', '2']
    ff_models = [datamodels.open(fp) for fp in fiberflat_filepaths]

    lfc_wavesol = download_gdrive_file('PARVI/Cals/PR.20230725.83774.17.master-lfc_wavesol.fits', output_dir=datadir)

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

    order_vec = trace_pos_model.make_order_vec()

    # Extraction parameters
    extraction_aperture = 9
    do_orders = order_vec
    #do_orders = [94]

    filepath_solar = download_gdrive_file('PARVI/L0/PR.20230802.82176.93_rate-SOLAR.fits', output_dir=datadir)
    solar_model = datamodels.open(filepath_solar)
    #plt.imshow(solar_model.rate, vmin=0, vmax=np.nanquantile(solar_model.rate, 0.995)); plt.plot(trace_pos_model.pos_fib1[34, :]); plt.ylim(0, 2048); plt.show()

    # Box extraction
    prim = SpectralExtractionPrimitive()
    spec_box = prim.apply(
        solar_model,
        tracepos=trace_pos_model,
        do_fibers=fibers,
        do_orders=do_orders,
        traceprof=trace_prof_model,
        xrange=xrange,
        extraction_aperture=extraction_aperture,
        method='box',
    )

    # Optimal Extraction
    prim = SpectralExtractionPrimitive()
    spec_opt = prim.apply(
        solar_model,
        tracepos=trace_pos_model,
        do_fibers=fibers,
        do_orders=do_orders,
        traceprof=trace_prof_model,
        xrange=xrange,
        extraction_aperture=extraction_aperture,
        method='optimal',
    )

    # Cross-talk
    prim = SpectralExtractionPrimitive()
    spec_ct = prim.apply(
        solar_model,
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

            spectrum_box = getattr(spec_box, f'spec_uncal_fib{fiber}')[oi, :]
            error_box = getattr(spec_box, f'spec_uncal_err_fib{fiber}')[oi, :]
            snr_box = np.nanmean(spectrum_box / error_box)
            
            print(f"Fiber {fiber} Order {order}: Box SNR = {snr_box:.2f}")
            assert snr_box > 0, f"Fiber {fiber} Order {order}: Box SNR <= 10"

            spectrum_opt = getattr(spec_opt, f'spec_uncal_fib{fiber}')[oi, :]
            error_opt = getattr(spec_opt, f'spec_uncal_err_fib{fiber}')[oi, :]
            snr_opt = np.nanmean(spectrum_opt / error_opt)
            
            print(f"Fiber {fiber} Order {order}: OPT SNR = {snr_opt:.2f}")
            assert snr_opt > 0, f"Fiber {fiber} Order {order}: OPT SNR <= 10"

            spectrum_ct = getattr(spec_ct, f'spec_uncal_fib{fiber}')[oi, :]
            error_ct = getattr(spec_ct, f'spec_uncal_err_fib{fiber}')[oi, :]
            snr_ct = np.nanmean(spectrum_ct / error_ct)
            assert snr_ct > 0, f"Fiber {fiber} Order {order}: CT SNR <= 0"
            print(f"Fiber {fiber} Order {order}: CT SNR = {snr_ct:.2f}")

            # plt.plot(spectrum_box, label='Box')
            # plt.plot(spectrum_opt, label='Optimal')
            # plt.plot(spectrum_ct, label='CT')
            # plt.legend()
            # plt.show()

    # Test Fiberflat Pipeline
    pipe = ProcessFiberflatsPipeline()
    pipe.trace_pos.deg_pos = 4
    pipe.trace_pos.deg_width = 2
    pipe.trace_pos.yrange = yrange_parvi
    pipe.trace_pos.xrange = xrange
    pipe.spec_extract.extraction_aperture = extraction_aperture
    pipe.set_wavesol.wavesol = lfc_wavesol
    trace_results = pipe.apply(ff_models)

    # Test Stage1 Pipeline
    pipe = Stage1Pipeline()
    pipe.spec_extract.tracepos = trace_pos_model
    pipe.spec_extract.traceprof = trace_prof_model
    pipe.spec_extract.do_fibers = fibers
    pipe.spec_extract.do_orders = do_orders
    pipe.spec_extract.xrange = xrange
    pipe.spec_extract.extraction_aperture = extraction_aperture
    pipe.spec_extract.method = 'optimal'
    pipe.set_wavesol.wavesol = lfc_wavesol

    spec_opt = pipe.apply(solar_model)

    for fiber in fibers:
    	for order in do_orders:

            oi = np.abs(order - trace_pos_model.meta.order_start)
            
            spectrum_opt = getattr(spec_opt, f'spec_uncal_fib{fiber}')[oi, :]
            error_opt = getattr(spec_opt, f'spec_uncal_err_fib{fiber}')[oi, :]
            snr_opt = np.nanmean(spectrum_opt / error_opt)
            
            print(f"Fiber {fiber} Order {order}: OPT SNR = {snr_opt:.2f}")
            assert snr_opt > 0, f"Fiber {fiber} Order {order}: OPT SNR <= 10"
