from ..base_primitive import BasePrimitive
from ... import datamodels
import numpy as np
from .barycenter_corr import calc_barycenter_corrections, calc_solar_barycenter_corrections
from astropy.table import Table

from .barycenter_corr import get_barycorrpy_star_data

import logging
logger = logging.getLogger(__name__)

__all__ = [
    'BarycenterCorrectionPrimitive',
]


class BarycenterCorrectionPrimitive(BasePrimitive):
    """
    Primitive to calculate the barycenter corrections for the input data.

    Parameters
    ----------
    input : str or L0Model
        Input file path or data model to process. Currently, this is expected
        to be a level 0 (rate map) data product.
    suffix : str or None, optional
        Suffix to append to the output filename. Default is ``'barycorr'``.
    max_cores : int, optional
        Maximum number of CPU cores to use for Numba parallel processing. Default is 1.

    Calibrations
    ------------
    dark : DarkModel
        - selector: :class:`~hispecdrp.calibrations.selectors.DarkSelector`

    Returns
    -------
    output
        The barycenter-corrected data model with updated error and data quality arrays.
    """
    ARGUMENTS = {
        'suffix' : 'barycorr',
        'max_cores' : 1,
        'solar' : None,
    }

    def _perform(self):

        # Load the input model
        input_model = datamodels.open(self.input)
        output_model = self.get_output_model(input_model)

        # Observatory
        telescope = input_model.meta.telescope

        # Solar RVs?
        if self.solar is None:
            solar = any(k in input_model.meta.target_name.lower() for k in ("sun", "solar"))
            
        if solar:
            logger.info("Calculating SOLAR barycenter corrections...")
        else:
            logger.info("Calculating barycenter corrections...")
            
            # Collect star data
            star_data = get_barycorrpy_star_data(input_model)

        # Orders
        order_vec = input_model.make_order_vec()

        # Observatory
        for fiber in range(1, input_model.num_fibers + 1):

            # Init bc table for this fiber
            bc_table = getattr(output_model, 'barycorr_fib{fiber}', None)
            if bc_table is None:
                bc_table = Table(
                    names=['MJDUTC', 'MBJD', 'BERV'],
                    dtype=[np.float64, np.float64, np.float64],
                )

            # Mid points in UTC for this fiber
            mjd_utcs = getattr(input_model, f'mid_time_fib{fiber}', None)
            if mjd_utcs is None:
                continue
            
            def is_valid_mjd(mjd: float) -> bool:
                return 30_000 <= mjd <= 80_000

            spec_err = getattr(input_model, f'spec_err_fib{fiber}', None)
            dq_mask = getattr(input_model, f'spec_dq_fib{fiber}', None)
            weights = 1 / spec_err**2
            weights[dq_mask == 0] = 0.0
            
            # Exposure average MJD UTC for this fiber
            mjd_mean = np.sum(weights * mjd_utcs) / np.sum(weights)

            # Calculate corrections
            if solar:
                mbjds, vel_bcs = calc_solar_barycenter_corrections(
                    mjd_mean,
                    observatory=telescope,
                )
            else:
                mbjds, vel_bcs = calc_barycenter_corrections(
                    mjd_mean,
                    star_data,
                    observatory=telescope,
                )

            # NOTE: For now, only single value until utility in
            # per-order corrections is needed and validated
            mbjd, vel_bc = mbjds[0], vel_bcs[0]
            bc_table = Table(
                {
                    'MJDUTC' : [mjd_mean],
                    'MBJD' : [mbjd],
                    'BERV' : [vel_bc],
                }
            )

            setattr(output_model, f'barycorr_fib{fiber}', bc_table)

        # Output
        return dict(output=output_model)