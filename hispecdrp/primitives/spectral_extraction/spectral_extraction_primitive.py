from ..base_primitive import BasePrimitive
from ... import datamodels
import numpy as np
from typing import Sequence
import logging

from .extract_box import extract_spectrum_box
from .extract_optimal import extract_spectrum_optimal
from .extract_cross_talk import extract_spectra_cross_talk
from .extract_sheared_sp import extract_sheared_banded_ls
from .extract_spectroperfectionism import extract_order_spectroperfectionism_from_hdf
from .extract_spectroperfection import extract_spectroperfection
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
            - **sheared_sp**: Banded LS inverse of the separable sheared scatter operator.
            - **spectroperfection**: Banded LS with fiberflat spatial profile + LFC
              iso-λ shear. Default bilinear kernel (pixel-grid 1-D basis); set
              ``spectral_kernel='gaussian'`` to invert the measured LSF.
            - **spectroperfectionism**: Banded LS with HDF PSF deposit operator (sim).
    extraction_aperture : float
        Size of the extraction aperture in pixels, centered on the trace position.
    do_fibers : list[int] or None
        List of fiber numbers to extract. If None, all fibers will be extracted.
    do_orders : list[int] or None
        List of order numbers to extract. If None, all orders will be extracted.
    pix_range : tuple[int, int] or None
        Range of x pixels to extract. If None, the full range will be extracted.
    shear : array-like or None
        Optional iso-λ shear ``(n_orders, n_pix)`` for ``optimal`` / ``sheared_sp``.
        When None and ``method`` is ``sheared_sp``, shear is loaded from the sim HDF
        for the input spectrograph (if available).
    continuum_smooth_pix : int
        Spectral smooth length for building continuum ``P`` in ``sheared_sp``.
    spectral_kernel : str
        For ``spectroperfection``: ``bilinear`` (default, pixel-grid iso-λ) or
        ``gaussian`` (invert LFC-measured LSF).

    Returns
    -------
    output : datamodels.L1Model or list[datamodels.L1Model]
        Extracted spectra for each fiber and order.
    """

    PARAMETERS = {
        'suffix' : 'specextract',
        'method' : 'box',
        'extraction_aperture' : 10,
        'do_fibers' : None,
        'do_orders' : None,
        'pix_range' : None,
        'output_model' : 'L1Model',
        'shear' : None,
        'continuum_smooth_pix' : 81,
        'spectral_kernel': 'bilinear',
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

        if isinstance(self.dataset, (datamodels.DataModel, str)):
            output = None
            input = [self.dataset]
        elif isinstance(self.dataset, Sequence):
            output = []
            input = self.dataset
        elif isinstance(self.dataset, dict):
            output = {}
            input_keys = list(self.dataset.keys())
            input_vals = list(self.dataset.values())
            input = input_vals
        else:
            raise ValueError(f"Unsupported input type: {type(self.dataset)}")

        # Loop over inputs
        for _input_i, _input in enumerate(input):
        
            # Open the input model
            input_model = datamodels.open(_input)
            npix = input_model.shape[0] if input_model.meta.orders_are_vertical else input_model.shape[1]

            # Get calibrations (get_calibration returns a filepath or model)
            tracepos_model, _ = self.get_calibration(input_model, 'tracepos')
            tracepos_model = datamodels.open(tracepos_model)
            traceprof_model, _ = self.get_calibration(input_model, 'traceprof')
            traceprof_model = datamodels.open(traceprof_model)

            # Init output model
            output_model = self.get_output_model(input_model)

            # Which fibers and orders to process
            do_fibers, do_orders = self.get_do_traces(input_model, self.do_fibers, self.do_orders)
            order_vec = input_model.make_order_vec()
            n_orders = len(order_vec)

            # Initialize output arrays for all fibers if not already set.
            # Mock / sim L0s may omit mid_time (or dq); fall back to safe dtypes.
            dq_dtype = (
                input_model.dq.dtype
                if getattr(input_model, "dq", None) is not None
                else np.uint32
            )
            mid_time = getattr(input_model, "mid_time", None)
            mid_time_dtype = (
                mid_time.dtype if mid_time is not None else np.float64
            )
            for fiber in do_fibers:
                if not output_model.has_extension_value(f"spec_uncal_fib{fiber}"):
                    setattr(
                        output_model, f'SPEC_UNCAL_FIB{fiber}',
                        np.full((n_orders, npix), np.nan, dtype=np.float32)
                    )
                    setattr(
                        output_model, f'SPEC_UNCAL_ERR_FIB{fiber}',
                        np.full((n_orders, npix), np.nan, dtype=np.float32)
                    )
                    setattr(
                        output_model, f'SPEC_UNCAL_VAR_POISSON_FIB{fiber}', np.full((n_orders, npix), np.nan, dtype=np.float32)
                    )
                    setattr(
                        output_model, f'SPEC_UNCAL_VAR_RNOISE_FIB{fiber}',
                        np.full((n_orders, npix), np.nan, dtype=np.float32)
                    )
                    setattr(
                        output_model, f'SPEC_DQ_FIB{fiber}',
                        np.zeros((n_orders, npix), dtype=dq_dtype)
                    )
                    setattr(
                        output_model, f'MID_TIME_FIB{fiber}',
                        np.full((n_orders, npix), np.nan, dtype=mid_time_dtype)
                    )

            # Dispatch to extraction method
            method = str(self.method).lower()
            if method == 'box':
                self.extract_box(
                    input_model=input_model,
                    output_model=output_model,
                    tracepos_model=tracepos_model,
                    do_fibers=do_fibers,
                    do_orders=do_orders
                )
            elif method == 'optimal':
                self.extract_optimal(
                    input_model=input_model,
                    output_model=output_model,
                    tracepos_model=tracepos_model,
                    traceprof_model=traceprof_model,
                    do_fibers=do_fibers,
                    do_orders=do_orders
                )
            elif method == 'cross_talk':
                self.extract_cross_talk(
                    input_model=input_model,
                    output_model=output_model,
                    tracepos_model=tracepos_model,
                    traceprof_model=traceprof_model,
                    do_fibers=do_fibers,
                    do_orders=do_orders
                )
            elif method == 'sheared_sp':
                self.extract_sheared_sp(
                    input_model=input_model,
                    output_model=output_model,
                    tracepos_model=tracepos_model,
                    do_fibers=do_fibers,
                    do_orders=do_orders,
                )
            elif method in ('spectroperfection', 'spectroperfectionist'):
                self.extract_spectroperfection(
                    input_model=input_model,
                    output_model=output_model,
                    tracepos_model=tracepos_model,
                    traceprof_model=traceprof_model,
                    do_fibers=do_fibers,
                    do_orders=do_orders,
                )
            elif method in ('spectroperfectionism', 'sp'):
                self.extract_spectroperfectionism(
                    input_model=input_model,
                    output_model=output_model,
                    do_fibers=do_fibers,
                    do_orders=do_orders,
                )
            else:
                raise ValueError(f"Unknown extraction method: {self.method}")
                
            # Add to output
            if isinstance(self.dataset, (datamodels.DataModel, str)):
                output = output_model
            elif isinstance(self.dataset, Sequence):
                output.append(output_model)
            elif isinstance(self.dataset, dict):
                output[input_keys[_input_i]] = output_model

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
        # All orders
        order_vec = input_model.make_order_vec()

        for fiber in do_fibers:

            trace_positions = getattr(tracepos_model, f'pos_fib{fiber}')

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

                # Get the data attributes based on the primary ext name
                pext_name = input_model.primary_extension_name
                input_data = getattr(input_model, pext_name)
                input_err = getattr(input_model, pext_name + '_err', None)
                input_dq = getattr(input_model, 'dq', None)
                input_var_rnoise = getattr(input_model, pext_name + '_var_rnoise', None)
                input_var_poisson = getattr(input_model, pext_name + '_var_poisson', None)
                input_mid_time = getattr(input_model, 'mid_time', None)

                # Sanity check
                assert trace_positions.shape[1] == input_data.shape[0], f"Trace positions shape[1]={trace_positions.shape[1]} does not match number of pixels={input_data.shape[0]}"

                # Perform box extraction
                result = extract_spectrum_box(
                    input_data=input_data,
                    input_err=input_err,
                    input_dq=input_dq,
                    input_var_rnoise=input_var_rnoise,
                    input_var_poisson=input_var_poisson,
                    input_mid_time=input_mid_time,
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
        
        # All orders
        order_vec = input_model.make_order_vec()

        # Sanity check
        #assert trace_positions.shape[0] == len(order_vec), f"Trace positions shape[0]={trace_positions.shape[0]} does not match number of orders={len(order_vec)}"

        for fiber in do_fibers:

            # Cals
            trace_positions = getattr(tracepos_model, f'pos_fib{fiber}')
            trace_profile = getattr(traceprof_model, f'prof_fib{fiber}')

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

                # Get the data attributes based on the primary ext name
                pext_name = input_model.primary_extension_name
                input_data = getattr(input_model, pext_name)
                input_err = getattr(input_model, pext_name + '_err', None)
                input_dq = getattr(input_model, 'dq', None)
                input_var_rnoise = getattr(input_model, pext_name + '_var_rnoise', None)
                input_var_poisson = getattr(input_model, pext_name + '_var_poisson', None)
                input_mid_time = getattr(input_model, 'mid_time', None)

                # Sanity check
                assert trace_positions.shape[1] == input_data.shape[0], f"Trace positions shape[1]={trace_positions.shape[1]} does not match number of pixels={input_data.shape[0]}"
                assert trace_profile.shape == input_data.shape, f"Trace profile shape: {trace_profile.shape} does not match input data shape: {input_data.shape}"

                # # NOTE: Keep for debugging
                # if order == 94:
                #     breakpoint()
                #     pixarr = np.arange(input_model.rate.shape[0])
                #     plt.imshow(input_data, vmin=0, vmax=np.nanquantile(input_data, 0.995)); plt.plot(trace_positions[j+1], pixarr, c='red'); plt.plot(trace_positions[j], pixarr, c='red'); plt.show()

                # Perform optimal extraction
                shear_row = self._shear_for_order(input_model, fiber, j, order_vec.size)
                result = extract_spectrum_optimal(
                    input_data=input_data,
                    input_err=input_err,
                    input_dq=input_dq,
                    input_var_rnoise=input_var_rnoise,
                    input_var_poisson=input_var_poisson,
                    input_mid_time=input_mid_time,
                    trace_positions=trace_positions[j, :],
                    profile=trace_profile,
                    extraction_aperture=self.extraction_aperture,
                    pix_range=self.pix_range,
                    orders_are_vertical=input_model.meta.orders_are_vertical,
                    shear=shear_row,
                )

                spec_out[j, :] = result['spec']
                err_out[j, :] = result['err']
                var_rnoise_out[j, :] = result['var_rnoise']
                var_poisson_out[j, :] = result['var_poisson']
                dq_out[j, :] = result['dq']
                mid_time_out[j, :] = result['mid_time']

                # NOTE: Keep for debugging
                # import matplotlib.pyplot as plt; plt.imshow(input_data, vmin=0, vmax=np.nanquantile(input_data, 0.995)); plt.plot(trace_positions,color='r'); plt.show()
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
        input_data = getattr(input_model, pext_name)
        pext_name = input_model.primary_extension_name
        input_data = getattr(input_model, pext_name)
        ny, nx = input_data.shape
        order_vec = input_model.make_order_vec()
        npix = ny if input_model.meta.orders_are_vertical else nx

        # Trace profiles for all fibers for a single order (resuse for all orders)
        trace_profiles = np.zeros((n_fibers, ny, nx), dtype=input_data.dtype)
        for j, fiber in enumerate(do_fibers):
            trace_profiles[j, :, :] = getattr(traceprof_model, f'prof_fib{fiber}')

        # Loop over orders, extract all fibers simultaneously
        for i, order in enumerate(order_vec):

            if order not in do_orders:
                continue

            # Trace positions for all fibers but only this order
            trace_positions = np.zeros((n_fibers, npix), dtype=np.float64)
            for j, fiber in enumerate(do_fibers):
                trace_positions[j, :] = getattr(tracepos_model, f'pos-fib{fiber}')[i, :]
                
            logger.info(f"Extracting {input_model} with method {self.method} [Fibers {do_fibers}][Order {order}]")

            # NOTE: Keep for debugging
            #breakpoint()
            # plt.imshow(input_model.rate); plt.plot(trace_positions[0], c='red'); plt.plot(trace_positions[1], c='red'); plt.show()

            # Get the data attributes based on the primary ext name
            input_err = getattr(input_model, pext_name + '_err', None)
            input_dq = getattr(input_model, 'dq', None)
            input_var_rnoise = getattr(input_model, pext_name + '_var_rnoise', None)
            input_var_poisson = getattr(input_model, pext_name + '_var_poisson', None)
            input_mid_time = getattr(input_model, 'mid_time', None)

            # Sanity check
            assert trace_positions.shape[1] == input_data.shape[0], f"Trace positions shape[1]={trace_positions.shape[1]} does not match number of pixels={input_data.shape[0]}"
            #assert trace_profile.shape == input_data.shape, f"Trace profile shape: {trace_profile.shape} does not match input data shape: {input_data.shape}"

            # Extract all pixels, all fibers for this order
            result = extract_spectra_cross_talk(
                input_data=input_data,
                input_err=input_err,
                input_dq=input_dq,
                input_var_rnoise=input_var_rnoise,
                input_var_poisson=input_var_poisson,
                input_mid_time=input_mid_time,
                trace_positions=trace_positions,
                profile=trace_profiles,
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

    def _resolve_shear_array(self, input_model, n_orders: int, n_pix: int) -> np.ndarray | None:
        """Return ``(n_orders, n_pix)`` shear or ``None``."""
        shear = getattr(self, "shear", None)
        if shear is None:
            return None
        arr = np.asarray(shear, dtype=np.float64)
        if arr.ndim == 1:
            if arr.size != n_pix:
                raise ValueError(
                    f"1-D shear length {arr.size} does not match n_pix={n_pix}"
                )
            return np.broadcast_to(arr, (n_orders, n_pix)).copy()
        if arr.shape != (n_orders, n_pix):
            raise ValueError(
                f"shear shape {arr.shape} does not match {(n_orders, n_pix)}"
            )
        return arr

    def _shear_for_order(
        self,
        input_model,
        fiber: int,
        order_index: int,
        n_orders: int,
    ) -> np.ndarray | None:
        pext_name = input_model.primary_extension_name
        input_data = getattr(input_model, pext_name)
        n_pix = (
            input_data.shape[0]
            if input_model.meta.orders_are_vertical
            else input_data.shape[1]
        )
        shear_2d = self._resolve_shear_array(input_model, n_orders, n_pix)
        if shear_2d is None:
            return None
        return shear_2d[int(order_index)]

    def _hdf_shear_or_none(self, input_model, fiber: int) -> np.ndarray | None:
        shear = getattr(self, "shear", None)
        if shear is not None:
            return np.asarray(shear, dtype=np.float64)
        spectrograph = getattr(input_model.meta, "spectrograph", None)
        if spectrograph is None:
            return None
        try:
            from ...simulations.e2e_pipeline import create_trace_shear_from_sim_hdf

            return create_trace_shear_from_sim_hdf(
                str(spectrograph), fiber=int(fiber)
            )
        except Exception as exc:
            logger.warning("Could not load HDF shear for sheared_sp: %s", exc)
            return None

    def extract_sheared_sp(
        self,
        input_model: datamodels.L0Model,
        output_model: datamodels.L1Model,
        tracepos_model: datamodels.TracePosModel,
        do_fibers: np.ndarray,
        do_orders: np.ndarray,
    ):
        from ...simulations.plotting.stage1_extraction_noise import (
            build_profile_image,
            continuum_rate_from_spectral_smooth,
        )

        order_vec = input_model.make_order_vec()
        pext_name = input_model.primary_extension_name
        input_data = getattr(input_model, pext_name)
        input_err = getattr(input_model, pext_name + "_err", None)
        input_dq = getattr(input_model, "dq", None)
        input_mid_time = getattr(input_model, "mid_time", None)

        for fiber in do_fibers:
            trace_positions = getattr(tracepos_model, f"pos_fib{fiber}")
            shear_2d = self._hdf_shear_or_none(input_model, int(fiber))
            spec_out = getattr(output_model, f"SPEC_UNCAL_FIB{fiber}")
            err_out = getattr(output_model, f"SPEC_UNCAL_ERR_FIB{fiber}")
            var_rnoise_out = getattr(output_model, f"SPEC_UNCAL_VAR_RNOISE_FIB{fiber}")
            var_poisson_out = getattr(output_model, f"SPEC_UNCAL_VAR_POISSON_FIB{fiber}")
            dq_out = getattr(output_model, f"SPEC_DQ_FIB{fiber}")
            mid_time_out = getattr(output_model, f"MID_TIME_FIB{fiber}")

            for j, order in enumerate(order_vec):
                if order not in do_orders:
                    continue
                logger.info(
                    "Extracting %s with method sheared_sp [Fiber %s][Order %s]",
                    input_model,
                    fiber,
                    order,
                )
                ty = np.asarray(trace_positions[j, :], dtype=np.float64)
                cont = continuum_rate_from_spectral_smooth(
                    np.asarray(input_data, dtype=np.float64),
                    ty,
                    extraction_aperture=float(self.extraction_aperture),
                    smooth_pix=int(self.continuum_smooth_pix),
                )
                profile = build_profile_image(
                    cont, ty, extraction_aperture=float(self.extraction_aperture)
                )
                if input_err is not None:
                    rate_var = np.maximum(
                        np.asarray(input_err, dtype=np.float64), 1.0e-30
                    ) ** 2
                else:
                    rate_var = np.ones_like(input_data, dtype=np.float64)
                shear_row = None if shear_2d is None else shear_2d[j]
                result = extract_sheared_banded_ls(
                    np.asarray(input_data, dtype=np.float64),
                    rate_var,
                    profile,
                    ty,
                    extraction_aperture=float(self.extraction_aperture),
                    shear=shear_row,
                    rate_dq=None if input_dq is None else np.asarray(input_dq),
                )
                spec_out[j, :] = result["spec"]
                err_out[j, :] = result["err"]
                var_rnoise_out[j, :] = np.nan
                var_poisson_out[j, :] = np.nan
                dq_out[j, :] = 0
                if input_mid_time is not None:
                    mid_time_out[j, :] = np.nanmean(
                        np.asarray(input_mid_time, dtype=np.float64), axis=1
                    )
                else:
                    mid_time_out[j, :] = np.nan

    def _traceprof_moment_row(
        self,
        traceprof_model,
        fiber: int,
        order_index: int,
        n_pix: int,
        name: str,
    ) -> np.ndarray | None:
        attr = f"{name}_fib{fiber}"
        try:
            arr = getattr(traceprof_model, attr)
        except Exception:
            return None
        if arr is None:
            return None
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim == 1:
            if arr.size != n_pix:
                return None
            return arr
        if arr.ndim == 2 and order_index < arr.shape[0] and arr.shape[1] == n_pix:
            return arr[int(order_index)]
        return None

    def extract_spectroperfection(
        self,
        input_model: datamodels.L0Model,
        output_model: datamodels.L1Model,
        tracepos_model: datamodels.TracePosModel,
        traceprof_model: datamodels.TraceProfModel,
        do_fibers: np.ndarray,
        do_orders: np.ndarray,
    ):
        from ...simulations.plotting.stage1_extraction_noise import (
            build_profile_image,
            continuum_rate_from_spectral_smooth,
        )

        order_vec = input_model.make_order_vec()
        pext_name = input_model.primary_extension_name
        input_data = getattr(input_model, pext_name)
        input_err = getattr(input_model, pext_name + "_err", None)
        input_dq = getattr(input_model, "dq", None)
        input_mid_time = getattr(input_model, "mid_time", None)
        n_pix = (
            input_data.shape[0]
            if input_model.meta.orders_are_vertical
            else input_data.shape[1]
        )

        for fiber in do_fibers:
            trace_positions = getattr(tracepos_model, f"pos_fib{fiber}")
            trace_profile = getattr(traceprof_model, f"prof_fib{fiber}")
            spec_out = getattr(output_model, f"SPEC_UNCAL_FIB{fiber}")
            err_out = getattr(output_model, f"SPEC_UNCAL_ERR_FIB{fiber}")
            var_rnoise_out = getattr(output_model, f"SPEC_UNCAL_VAR_RNOISE_FIB{fiber}")
            var_poisson_out = getattr(output_model, f"SPEC_UNCAL_VAR_POISSON_FIB{fiber}")
            dq_out = getattr(output_model, f"SPEC_DQ_FIB{fiber}")
            mid_time_out = getattr(output_model, f"MID_TIME_FIB{fiber}")

            for j, order in enumerate(order_vec):
                if order not in do_orders:
                    continue
                logger.info(
                    "Extracting %s with method spectroperfection [Fiber %s][Order %s]",
                    input_model,
                    fiber,
                    order,
                )
                ty = np.asarray(trace_positions[j, :], dtype=np.float64)
                prof = np.asarray(trace_profile, dtype=np.float64)
                # Prefer the fiberflat image as P; fall back to spectrally
                # smoothed science if the profile is a placeholder (unity).
                if np.allclose(prof, 1.0) or not np.any(np.isfinite(prof) & (prof > 0)):
                    cont = continuum_rate_from_spectral_smooth(
                        np.asarray(input_data, dtype=np.float64),
                        ty,
                        extraction_aperture=float(self.extraction_aperture),
                        smooth_pix=int(self.continuum_smooth_pix),
                    )
                    prof = build_profile_image(
                        cont, ty, extraction_aperture=float(self.extraction_aperture)
                    )
                shear_row = self._traceprof_moment_row(
                    traceprof_model, int(fiber), j, n_pix, "shear"
                )
                if shear_row is None or not np.any(np.isfinite(shear_row)):
                    hdf_shear = self._hdf_shear_or_none(input_model, int(fiber))
                    shear_row = None if hdf_shear is None else hdf_shear[j]
                sig_row = self._traceprof_moment_row(
                    traceprof_model, int(fiber), j, n_pix, "sigma_spec"
                )
                if input_err is not None:
                    rate_var = np.maximum(
                        np.asarray(input_err, dtype=np.float64), 1.0e-30
                    ) ** 2
                else:
                    rate_var = np.ones_like(input_data, dtype=np.float64)
                result = extract_spectroperfection(
                    np.asarray(input_data, dtype=np.float64),
                    rate_var,
                    prof,
                    ty,
                    extraction_aperture=float(self.extraction_aperture),
                    shear=shear_row,
                    sigma_spec=sig_row,
                    rate_dq=None if input_dq is None else np.asarray(input_dq),
                    pix_range=self.pix_range,
                    spectral_kernel=getattr(self, "spectral_kernel", "bilinear"),
                )
                spec_out[j, :] = result["spec"]
                err_out[j, :] = result["err"]
                var_rnoise_out[j, :] = np.nan
                var_poisson_out[j, :] = np.nan
                dq_out[j, :] = 0
                if input_mid_time is not None:
                    mid_time_out[j, :] = np.nanmean(
                        np.asarray(input_mid_time, dtype=np.float64), axis=1
                    )
                else:
                    mid_time_out[j, :] = np.nan

    def extract_spectroperfectionism(
        self,
        input_model: datamodels.L0Model,
        output_model: datamodels.L1Model,
        do_fibers: np.ndarray,
        do_orders: np.ndarray,
    ):
        order_vec = input_model.make_order_vec()
        spectrograph = getattr(input_model.meta, "spectrograph", None)
        if spectrograph is None:
            raise ValueError(
                "spectroperfectionism extract requires input_model.meta.spectrograph"
            )
        pext_name = input_model.primary_extension_name
        input_data = getattr(input_model, pext_name)
        input_err = getattr(input_model, pext_name + "_err", None)
        input_dq = getattr(input_model, "dq", None)
        input_mid_time = getattr(input_model, "mid_time", None)
        rate = np.asarray(input_data, dtype=np.float64)
        if input_err is not None:
            rate_var = np.maximum(np.asarray(input_err, dtype=np.float64), 1.0e-30) ** 2
        else:
            rate_var = np.ones_like(rate, dtype=np.float64)

        for fiber in do_fibers:
            spec_out = getattr(output_model, f"SPEC_UNCAL_FIB{fiber}")
            err_out = getattr(output_model, f"SPEC_UNCAL_ERR_FIB{fiber}")
            var_rnoise_out = getattr(output_model, f"SPEC_UNCAL_VAR_RNOISE_FIB{fiber}")
            var_poisson_out = getattr(output_model, f"SPEC_UNCAL_VAR_POISSON_FIB{fiber}")
            dq_out = getattr(output_model, f"SPEC_DQ_FIB{fiber}")
            mid_time_out = getattr(output_model, f"MID_TIME_FIB{fiber}")

            for j, order in enumerate(order_vec):
                if order not in do_orders:
                    continue
                logger.info(
                    "Extracting %s with method spectroperfectionism [Fiber %s][Order %s]",
                    input_model,
                    fiber,
                    order,
                )
                result = extract_order_spectroperfectionism_from_hdf(
                    rate,
                    rate_var,
                    spectrograph=str(spectrograph),
                    fiber=int(fiber),
                    order=int(order),
                    rate_dq=None if input_dq is None else np.asarray(input_dq),
                )
                n = min(spec_out.shape[1], result["spec"].size)
                spec_out[j, :n] = result["spec"][:n]
                err_out[j, :n] = result["err"][:n]
                var_rnoise_out[j, :] = np.nan
                var_poisson_out[j, :] = np.nan
                dq_out[j, :] = 0
                mid_time_out[j, :] = np.nan
                if input_mid_time is not None:
                    mid_time_out[j, :] = np.nanmean(input_mid_time, axis=1)
