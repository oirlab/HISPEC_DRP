from ..base_primitive import BasePrimitive
from ... import datamodels
from ...datamodels.dqflags import DQ_FLAGS
import numpy as np
from .barycenter_corr import calc_barycenter_corrections, calc_solar_barycenter_corrections
from astropy.table import Table

from .barycenter_corr import get_barycorrpy_star_data

import logging
logger = logging.getLogger(__name__)

DO_NOT_USE = DQ_FLAGS["DO_NOT_USE"]

__all__ = [
    'BarycenterCorrectionPrimitive',
]


class BarycenterCorrectionPrimitive(BasePrimitive):
    """
    Primitive to calculate the barycenter corrections for the input data.

    When ``meta.force_berv_m_s`` is a finite value (simulations), that BERV is
    written for every processed fiber and ``barycorrpy`` is skipped.

    Parameters
    ----------
    input : str or L0Model
        Input file path or data model to process.
    suffix : str or None, optional
        Suffix to append to the output filename. Default is ``'barycorr'``.
    max_cores : int, optional
        Maximum number of CPU cores to use for Numba parallel processing. Default is 1.

    Returns
    -------
    output
        The data model with per-fiber ``barycorr_fib*`` tables.
    """
    PARAMETERS = {
        'suffix' : 'barycorr',
        'max_cores' : 1,
        'solar' : None,
    }

    def _perform(self):

        # Load the input model
        input_model = datamodels.open(self.dataset)
        output_model = self.get_output_model(input_model)

        # Observatory
        telescope = input_model.meta.telescope

        force_berv = getattr(input_model.meta, "force_berv_m_s", None)
        try:
            force_berv = float(force_berv) if force_berv is not None else None
        except (TypeError, ValueError):
            force_berv = None
        if force_berv is not None and not np.isfinite(force_berv):
            force_berv = None

        # Solar RVs?
        if self.solar is None:
            target_name = getattr(input_model.meta, "target_name", None) or ""
            solar = any(k in str(target_name).lower() for k in ("sun", "solar"))
        else:
            solar = bool(self.solar)

        if force_berv is not None:
            logger.info(
                "Using forced BERV = %.6f m/s (meta.force_berv_m_s); skipping barycorrpy.",
                force_berv,
            )
        elif solar:
            logger.info("Calculating SOLAR barycenter corrections...")
        else:
            logger.info("Calculating barycenter corrections...")
            
            # Collect star data
            star_data = get_barycorrpy_star_data(input_model)

        # Orders
        order_vec = input_model.make_order_vec()

        # Observatory
        for fiber in range(1, input_model.num_fibers + 1):

            # Mid points in UTC for this fiber
            mjd_utcs = getattr(input_model, f'mid_time_fib{fiber}', None)
            if mjd_utcs is None:
                continue

            spec_err = getattr(input_model, f'spec_err_fib{fiber}', None)
            if spec_err is None:
                spec_err = getattr(input_model, f'spec_uncal_err_fib{fiber}', None)
            dq_mask = getattr(input_model, f'spec_dq_fib{fiber}', None)

            if spec_err is not None:
                weights = np.zeros_like(spec_err, dtype=np.float64)
                valid_err = np.isfinite(spec_err) & (spec_err > 0)
                weights[valid_err] = 1.0 / spec_err[valid_err] ** 2
                if dq_mask is not None:
                    weights[(dq_mask & DO_NOT_USE) != 0] = 0.0
                wsum = float(np.sum(weights))
                if wsum > 0 and np.isfinite(wsum):
                    mjd_mean = float(np.sum(weights * mjd_utcs) / wsum)
                else:
                    mjd_mean = float(np.nanmean(mjd_utcs))
            else:
                mjd_mean = float(np.nanmean(mjd_utcs))

            if not np.isfinite(mjd_mean):
                meta_mjd = getattr(input_model.meta, "mjd_mid", None)
                if meta_mjd is None:
                    meta_mjd = getattr(input_model.meta, "mjd_start", None)
                if meta_mjd is not None and np.isfinite(meta_mjd):
                    mjd_mean = float(meta_mjd)
                else:
                    logger.warning(
                        "Skipping barycenter correction for fiber %s: "
                        "no finite exposure mid-time.",
                        fiber,
                    )
                    continue

            # Calculate corrections
            if force_berv is not None:
                mbjd, vel_bc = mjd_mean, float(force_berv)
            elif solar:
                mbjds, vel_bcs = calc_solar_barycenter_corrections(
                    mjd_mean,
                    observatory=telescope,
                )
                mbjd, vel_bc = mbjds[0], vel_bcs[0]
            else:
                mbjds, vel_bcs = calc_barycenter_corrections(
                    mjd_mean,
                    star_data,
                    observatory=telescope,
                )
                mbjd, vel_bc = mbjds[0], vel_bcs[0]

            # NOTE: For now, only single value until utility in
            # per-order corrections is needed and validated
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
