from ..base_primitive import BasePrimitive
from ... import datamodels
import numpy as np
from ...datamodels.dqflags import DQ_FLAGS
from typing import Sequence
from copy import copy
import logging

from numba import njit, prange
from ...utils.parallelization_utils import numba_thread_scope
from ..coadd.coadd_frames_primitive import coadd_group

logger = logging.getLogger(__name__)
from .make_dark_primitive import _detect_hot_pixels


__all__ = ['MakeMasterDarkPrimitive']

HOT_FLAG =  DQ_FLAGS['HOT']
DO_NOT_USE_FLAG = DQ_FLAGS["DO_NOT_USE"]

class MakeMasterDarkPrimitive(BasePrimitive):
    """
    Create a master dark calibration model from a list of dark calibrations (DarkModel).

    A master calibration is identified by setting the `meta.master_cal` attribute to True.

    TODO: implement automatic selection of darks to be combined for making the master dark

    Parameters
    ----------
    input : list of str or list of DarkModel
        List of input file paths or dark models to process.
    mean_calc : str, optional
        Method for calculating the mean when combining darks. Default is 'wmean' (weighted mean).
    err_calc : str, optional
        Method for calculating the error when combining darks. Default is 'wse' (weighted standard error).
    hot_pixel_threshold : float, optional
        Threshold dark rate above which pixels are flagged as hot in the DQ array. Default is 5 e-/s.
    max_cores : int, optional
        Maximum number of CPU cores to use for parallel processing. Default is 1.

    Returns
    -------
    output : DarkModel
        The derived master dark calibration model containing the combined dark rate map,
        combined error map, and updated DQ array.
    """

    ARGUMENTS = {
        'output_model' : 'DarkModel',
        'mean_calc' : 'wmean',
        'err_calc' : 'wse',
        'hot_pixel_threshold' : 5,  # e-/s,  Threshold above which pixels are bad
        'max_cores' : 1,
    }


    def _perform(self):

        if self.input is not None: # How do you call a primitive without input?
            _inputs = self.input
        else:
            # todo implement automatic query of caldb to get all the darks between some dates with xyz filters
            raise NotImplementedError("Automatic calibration retrieval not yet implemented. List of darks must be provided as input.")


        input_type = type(_inputs)

        # Make sure input is a list of models
        if input_type in (str, datamodels.DataModel):
            logger.warning("Input is a single model or path, returning without coadding.")
            output_dark = _inputs.copy()
        if input_type in (list, tuple):
            output_dark = coadd_group(_inputs, self, self.mean_calc, self.err_calc)

        # Flag this calibration as a master calibration
        output_dark.meta.master_cal = True

        # # Median dark rate for instrument monitoring
        # median_dark_level = np.nanmedian(output_dark.rate)
        # output_dark.meta["MEDDARK"] = median_dark_level

        with numba_thread_scope(self.max_cores):
            _detect_hot_pixels(output_dark.rate, output_dark.dq,self.hot_pixel_threshold)

        return dict(output=output_dark)

