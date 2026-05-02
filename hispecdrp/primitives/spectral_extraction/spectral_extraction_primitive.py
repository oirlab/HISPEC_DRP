from ..base_primitive import BasePrimitive
from ... import datamodels
import numpy as np
from typing import Sequence
import logging

from .extract_box import extract_spectrum_box
from .extract_optimal import extract_spectrum_optimal
from .extract_cross_talk import extract_spectra_cross_talk
from ...calibrations import TracePosSelector, TraceProfSelector

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

__all__ = ['SpectralExtractionPrimitive']


class SpectralExtractionPrimitive(BasePrimitive):

    """
    Extracts the 2D echelleogram into 1D spectra for each fiber and order. Pre-determined trace positions and profiles (for optimal extraction) must be provided.

    TODO: Fix normalization to extend to effective infinity regardless of aperture size. Aperture size only determines fit window and we can round here.

    Parameters
    ----------
    suffix : str
        Suffix to add to the output filename. Default is 'specextract'.
    method : str
        Extraction method to use. Options are:
            - **box**: Sum all pixels within a fixed aperture around the trace position.
            - **optimal**: Fit a 1D profile to the data.
            - **cross_talk**: Simultaneously fit 1D profiles for all fibers to account for cross-talk between fibers. Only applicable when multiple fibers are present.
    extraction_aperture : tuple[int, int]
        Size of the extraction aperture in pixels. Tuple of (lower, upper) bounds relative to the trace position.
    do_fibers : list[int] or None
        List of fiber numbers to extract. If None, all fibers will be extracted.
    do_orders : list[int] or None
        List of order numbers to extract. If None, all orders will be extracted.
    pix_range : tuple[int, int] or None
        Range of x pixels to extract. If None, the full range will be extracted.

    Returns
    -------
    output : datamodels.L1Model or list[datamodels.L1Model]
        Extracted spectra for each fiber and order.
    """

    ARGUMENTS = {
        'suffix' : 'specextract',
        'method' : 'box',
        'extraction_aperture' : (-2, 2),
        'do_fibers' : None,
        'do_orders' : None,
        'pix_range' : None,
        'output_model' : 'L1Model',
    }

    CALIBRATIONS = {
        'tracepos': {
            'selector': TracePosSelector,
            'selector_kwargs': {}
        },
        'traceprof': {
            'selector': TraceProfSelector,
            'selector_kwargs': {}
        },
    }

    def _perform(self):

        if isinstance(self.input, (datamodels.DataModel, str)):
            output = None
            input = [self.input]
        elif isinstance(self.input, Sequence):
            output = []
            input = self.input

        # Loop over inputs (now a list)
        for _input in input:
        
            # Open the input model
            input_model = datamodels.open(_input)
            if input_model.meta.orders_are_vertical:
                npix = input_model.rate.shape[0]
            else:
                npix = input_model.rate.shape[1]

            # Get calibrations
            tracepos_model, _ = self.get_calibration(input_model, 'tracepos')
            traceprof_model, _ = self.get_calibration(input_model, 'traceprof')

            # Init output model
            output_model = self.get_output_model(input_model)

            # Which fibers and orders to process
            do_fibers, do_orders = self.get_do_traces(input_model, self.do_fibers, self.do_orders)
            order_vec = input_model.make_order_vec()
            n_orders = len(order_vec)

            # Initialize output arrays if not already set
            for fiber in do_fibers:
                if getattr(output_model, f'SPEC_UNCAL_FIB{fiber}') is None:
                    setattr(
                        output_model, f'SPEC_UNCAL_FIB{fiber}',
                        np.full((n_orders, npix), np.nan, dtype=np.float32)
                    )
                    setattr(
                        output_model, f'SPEC_UNCAL_ERR_FIB{fiber}',
                        np.full((n_orders, npix), np.nan, dtype=np.float32))
                    setattr(
                        output_model, f'SPEC_UNCAL_VAR_POISSON_FIB{fiber}', np.full((n_orders, npix), np.nan, dtype=np.float32)
                    )
                    setattr(
                        output_model, f'SPEC_UNCAL_VAR_RNOISE_FIB{fiber}',
                        np.full((n_orders, npix), np.nan, dtype=np.float32)
                    )
                    setattr(
                        output_model, f'SPEC_DQ_FIB{fiber}',
                        np.zeros((n_orders, npix), dtype=input_model.dq.dtype)
                    )
                    setattr(
                        output_model, f'MID_TIME_FIB{fiber}',
                        np.full((n_orders, npix), np.nan, dtype=input_model.mid_time.dtype)
                    )

            # Dispatch to extraction method
            if self.method.lower() == 'box':
                self.extract_box(
                    input_model=input_model,
                    output_model=output_model,
                    tracepos_model=tracepos_model,
                    do_fibers=do_fibers,
                    do_orders=do_orders
                )
            elif self.method.lower() == 'optimal':
                self.extract_optimal(
                    input_model=input_model,
                    output_model=output_model,
                    tracepos_model=tracepos_model,
                    traceprof_model=traceprof_model,
                    do_fibers=do_fibers,
                    do_orders=do_orders
                )
            elif self.method.lower() == 'cross_talk':
                self.extract_cross_talk(
                    input_model=input_model,
                    output_model=output_model,
                    tracepos_model=tracepos_model,
                    traceprof_model=traceprof_model,
                    do_fibers=do_fibers,
                    do_orders=do_orders
                )
            else:
                raise ValueError(f"Unknown extraction method: {self.method}")
                
            # Append to output list
            if isinstance(output, list):
                output.append(output_model)
            else:
                output = output_model

        # Return
        return dict(output=output)

    def extract_box(
        self,
        input_model : datamodels.L0Model,
        output_model : datamodels.L1Model,
        tracepos_model : datamodels.TracePosModel,
        do_fibers : np.ndarray,
        do_orders : np.ndarray,
    ):
        order_vec = input_model.make_order_vec()
        for fiber in do_fibers:

            # Get the trace position coefficients for the fiber
            trace_positions = getattr(tracepos_model, f'POS_FIB{fiber}')
            
            # Sanity check
            assert trace_positions.shape[0] == len(order_vec), "Trace position coefficients do not match number of orders"

            # Alias outputs
            spec_out = getattr(output_model, f'SPEC_UNCAL_FIB{fiber}')
            err_out = getattr(output_model, f'SPEC_UNCAL_ERR_FIB{fiber}')
            var_rnoise_out = getattr(output_model, f'SPEC_UNCAL_VAR_RNOISE_FIB{fiber}')
            var_poisson_out = getattr(output_model, f'SPEC_UNCAL_VAR_POISSON_FIB{fiber}')
            dq_out = getattr(output_model, f'SPEC_DQ_FIB{fiber}')
            mid_time_out = getattr(output_model, f'MID_TIME_FIB{fiber}')

            for j, order in enumerate(order_vec):

                if order not in do_orders:
                    continue

                logger.info(f"Extracting {input_model} with method {self.method} [Fiber {fiber}][Order {order}]")

                # Perform box extraction
                result = extract_spectrum_box(
                    input_data=input_model.rate,
                    input_err=input_model.rate_err,
                    input_dq=input_model.dq,
                    input_var_rnoise=input_model.rate_var_rnoise,
                    input_var_poisson=input_model.rate_var_poisson,
                    input_mid_time=input_model.mid_time,
                    trace_positions=trace_positions[j, :],
                    extraction_aperture=self.extraction_aperture,
                    pix_range=self.pix_range,
                    orders_are_vertical=input_model.meta.orders_are_vertical,
                )

                spec_out[j, :] = result['spec']
                err_out[j, :] = result['err']
                var_rnoise_out[j, :] = result['var_rnoise']
                var_poisson_out[j, :] = result['var_poisson']
                dq_out[j, :] = result['dq']
                mid_time_out[j, :] = result['mid_time']
                # NOTE: Keep for debugging
                # breakpoint()
                # import matplotlib.pyplot as plt; plt.plot(spec_out[j, :]); plt.show()

    def extract_optimal(
        self,
        input_model : datamodels.L0Model,
        output_model : datamodels.L1Model,
        tracepos_model : datamodels.TracePosModel,
        traceprof_model : datamodels.TraceProfModel,
        do_fibers : np.ndarray,
        do_orders : np.ndarray,
    ):
        order_vec = input_model.make_order_vec()
        for fiber in do_fibers:

            # Get the trace position coefficients for the fiber
            trace_positions = getattr(tracepos_model, f'POS_FIB{fiber}')
            # Sanity check
            #assert trace_pos_coeffs.shape[0] == len(order_vec), "Trace position coefficients do not match number of orders"
            fiberflat_profile = getattr(traceprof_model, f'PROF_FIB{fiber}')

            # Alias outputs
            spec_out = getattr(output_model, f'SPEC_UNCAL_FIB{fiber}')
            err_out = getattr(output_model, f'SPEC_UNCAL_ERR_FIB{fiber}')
            var_rnoise_out = getattr(output_model, f'SPEC_UNCAL_VAR_RNOISE_FIB{fiber}')
            var_poisson_out = getattr(output_model, f'SPEC_UNCAL_VAR_POISSON_FIB{fiber}')
            dq_out = getattr(output_model, f'SPEC_DQ_FIB{fiber}')
            mid_time_out = getattr(output_model, f'MID_TIME_FIB{fiber}')

            for j, order in enumerate(order_vec):

                if order not in do_orders:
                    continue
                logger.info(f"Extracting {input_model} with method {self.method} [Fiber {fiber}][Order {order}]")

                # # NOTE: Keep for debugging
                # if order == 94:
                #     breakpoint()
                #     pixarr = np.arange(input_model.rate.shape[0])
                #     plt.imshow(input_model.rate); plt.plot(trace_positions[j+1], pixarr, c='red'); plt.plot(trace_positions[j], pixarr, c='red'); plt.show()

                # Perform optimal extraction
                result = extract_spectrum_optimal(
                    input_data=input_model.rate,
                    input_err=input_model.rate_err,
                    input_dq=input_model.dq,
                    input_var_rnoise=input_model.rate_var_rnoise,
                    input_var_poisson=input_model.rate_var_poisson,
                    input_mid_time=input_model.mid_time,
                    trace_positions=trace_positions[j, :],
                    profile=fiberflat_profile,
                    extraction_aperture=self.extraction_aperture,
                    pix_range=self.pix_range,
                    orders_are_vertical=input_model.meta.orders_are_vertical,
                )

                spec_out[j, :] = result['spec']
                err_out[j, :] = result['err']
                var_rnoise_out[j, :] = result['var_rnoise']
                var_poisson_out[j, :] = result['var_poisson']
                dq_out[j, :] = result['dq']
                mid_time_out[j, :] = result['mid_time']

                # NOTE: Keep for debugging
                # import matplotlib.pyplot as plt; plt.imshow(input_model.rate, vmin=0, vmax=np.nanquantile(input_model.rate, 0.995)); plt.plot(trace_positions,color='r'); plt.show()
                # plt.plot(spec_out[j, :]); plt.show()
    
    def extract_cross_talk(
        self,
        input_model : datamodels.L0Model,
        output_model : datamodels.L1Model,
        tracepos_model : datamodels.TracePosModel,
        traceprof_model : datamodels.TraceProfModel,
        do_fibers : np.ndarray,
        do_orders : np.ndarray,
    ):
        
        n_fibers = len(do_fibers)
        ny, nx = input_model.rate.shape
        order_vec = input_model.make_order_vec()
        fiberflat_profiles = np.zeros((n_fibers, ny, nx), dtype=input_model.rate.dtype)
        
        for j, fiber in enumerate(do_fibers):
            fiberflat_profiles[j, :, :] = getattr(traceprof_model, f'PROF_FIB{fiber}')

        for i, order in enumerate(order_vec):
            if order not in do_orders:
                continue
            trace_positions = np.zeros((n_fibers, nx), dtype=input_model.rate.dtype)
            for j, fiber in enumerate(do_fibers):
                trace_positions[j, :] = getattr(tracepos_model, f'POS_FIB{fiber}')[i, :]
                
            logger.info(f"Extracting {input_model} with method {self.method} [Fibers {do_fibers}][Order {order}]")

            # NOTE: Keep for debugging
            #breakpoint()
            # plt.imshow(input_model.rate); plt.plot(trace_positions[0], c='red'); plt.plot(trace_positions[1], c='red'); plt.show()

            # Extract all pixels, all fibers for this order
            result = extract_spectra_cross_talk(
                input_data=input_model.rate,
                input_err=input_model.rate_err,
                input_dq=input_model.dq,
                input_var_rnoise=input_model.rate_var_rnoise,
                input_var_poisson=input_model.rate_var_poisson,
                input_mid_time=input_model.mid_time,
                trace_positions=trace_positions,
                profile=fiberflat_profiles,
                extraction_aperture=self.extraction_aperture,
                pix_range=self.pix_range,
                orders_are_vertical=input_model.meta.orders_are_vertical,
            )
            for j, fiber in enumerate(do_fibers):
                
                spec_uncal = getattr(output_model, f'SPEC_UNCAL_FIB{fiber}')
                spec_uncal[i, :] = result['spec'][j, :]

                err_output_fiber = getattr(output_model, f'SPEC_UNCAL_ERR_FIB{fiber}')
                err_output_fiber[i, :] = result['err'][j, :]

                var_rnoise_output_fiber = getattr(output_model, f'SPEC_UNCAL_VAR_RNOISE_FIB{fiber}')
                var_rnoise_output_fiber[i, :] = result['var_rnoise'][j, :]

                var_poisson_output_fiber = getattr(output_model, f'SPEC_UNCAL_VAR_POISSON_FIB{fiber}')
                var_poisson_output_fiber[i, :] = result['var_poisson'][j, :]

                dq_output_fiber = getattr(output_model, f'SPEC_DQ_FIB{fiber}')
                dq_output_fiber[i, :] = result['dq'][j, :]

                mid_time_output_fiber = getattr(output_model, f'MID_TIME_FIB{fiber}')
                mid_time_output_fiber[i, :] = result['mid_time'][j, :]

            # NOTE: Keep for debugging
            # breakpoint()
            # import matplotlib.pyplot as plt; plt.imshow(input_model.rate, vmin=0, vmax=np.nanquantile(input_model.rate, 0.995)); plt.plot(trace_positions,color='r'); plt.show()
            # plt.plot(spec_uncal[j, :]); plt.show()



    # elif method == 'optimal':
    #     for i in range(1,4):
    #         ### Get the trace positions for the fiber
    #         pos = tracepos_model[f'POS_FIB{i}']
    #         prof = traceprof_model[f'POF_FIB{i}']
    #         ### Set up the output arrays
    #         flux = np.empty(pos.shape)
    #         err, rnoise, poisson = np.empty(pos.shape), np.empty(pos.shape), np.empty(pos.shape)
    #         dq = np.empty(pos.shape)
    #         ### Loop over orders and wavlength channels for the specified fiber
    #         ### We can definitely find ways to speed up the nested loop
    #         for xpix in range(pos.shape[1]):
    #             ### Get data slice around position of each order
    #             cutdata = np.zeros((prof.shape[0],prof.shape[-1]))
    #             cutprof = np.zeros((prof.shape[0],prof.shape[-1]))
    #             for i in range(prof.shape[0]):
    #                 ### Cut data. Careful that this is done exactly how it was done when the 
    #                 ### trace profile was calculated
    #                 center = int(pos[i,xpix])
    #                 width = (prof.shape[-1]-1) // 2
    #                 dslice = data[center-width:center+width+1,xpix]
    #                 pslice = np.atleast_2d(prof[i,xpix])
    #                 ### Now we do the linear algebra to get the flux, solving
    #                 ### data = flux x profile for the OLS flux value
    #                 ### Did the linalg explicitly so that we could maybe
    #                 ### jit compile this, and because scipy is super slow
    #                 inv = np.linalg.inv(np.dot(pslice,pslice.T))
    #                 pfactor = np.dot(pslice.T, inv)
    #                 flux[i,xpix] = np.sum(dslice*pfactor[:,0])
    #                 ### FIXME how does error propegation work with this
                    

    #         ### Save the results for each fiber to a separate extension
    #         out_dict[f'SPEC_UNCAL_FIB{i}'] = flux
    #         out_dict[f'SPEC_UNCAL_ERR_FIB{i}'] = err
    #         out_dict[f'SPEC_UNCAL_VAR_POISSON_FIB{i}'] = poisson
    #         out_dict[f'SPEC_UNCAL_VAR_RNOISE{i}'] = rnoise
    #         out_dict[f'SPEC_DQ_FIB{i}'] = dq


    # ### Write the output model
    # model_result = HISPECL1Model(data=out_dict,meta=input_model.meta)

    # return model_result
