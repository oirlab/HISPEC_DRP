from ... import datamodels
from ...datamodels.dqflags import DQ_FLAGS
from ..base_primitive import BasePrimitive

from ...calibrations import WavelengthSolutionSelector
from ...datamodels.dqflags import DQ_FLAGS

import numpy as np


__all__ = ['SetInitialWLSPrimitive']


class SetInitialWLSPrimitive(BasePrimitive):

    """
    Set the initial wavelength solution for each fiber/order.
    This should be accurate to ~ < 10% of a resolution element.

    Parameters
    ----------
    input : str or L1Model
        Input L1 model or file name.
    suffix : str
        Suffix to add to the output file name. Default is 'wavesolinit'.
    do_fibers : list of int, optional
        List of fibers to process. Default is all fibers.
    do_orders : list of int, optional
        List of orders to process. Default is all orders.

    Calibrations
    ------------
    wavesol : str or CalibrationModel
        - selector: :class:`~hispecdrp.calibrations.selectors.WavelengthSolutionSelector`

    Notes
    -----
    *This depends on the type of wavelength solution and is under development*

    Returns
    -------
    output : L2Model
        The output model with the initial wavelength solution set.
    
    """

    ARGUMENTS = {
        'suffix' : 'wavesolinit',
        'do_fibers' : None,
        'do_orders': None,
    }

    CALIBRATIONS = {
        'wavesol': {
            'selector': WavelengthSolutionSelector,
            'selector_kwargs': {}
        }
    }

    def _perform(self):

        outputs = {}

        for key, input in self._iter_inputs():

            # Open the input model
            input_model = datamodels.open(input)
            
            wavesol, _ = self.get_calibration(input_model, 'wavesol')
            wavesol_model = datamodels.open(wavesol)

            output_model = self.get_output_model(input_model)

            # Get fibers and orders
            do_fibers, do_orders = self.get_do_traces(
                input_model, self.do_fibers, self.do_orders
            )

            for fiber in do_fibers:

                if output_model.has_extension_value(f"spec_fib{fiber}"):
                    n_orders, n_pix = getattr(output_model, f'spec_fib{fiber}').shape
                elif output_model.has_extension_value(f"spec_uncal_fib{fiber}"):
                    n_orders, n_pix = getattr(output_model, f'spec_uncal_fib{fiber}').shape

                # Initialize arrays if not set yet
                if getattr(output_model, f'wave_fib{fiber}', None) is None:
                    setattr(
                        output_model,
                        f'wave_fib{fiber}',
                        np.full((n_orders, n_pix), np.nan, dtype=np.float64)
                    )
                if getattr(output_model, f'spec_dq_fib{fiber}', None) is None:
                    setattr(
                        output_model,
                        f'spec_dq_fib{fiber}',
                        np.zeros((n_orders, n_pix), dtype=np.uint32)
                    )

                # Get the wavelength solution for the fiber
                wavesol_cal_arr = getattr(wavesol_model, f'wave_fib{fiber}')
                dq_cal_arr = getattr(wavesol_model, f'spec_dq_fib{fiber}')

                wavesol_output_arr = getattr(output_model, f'wave_fib{fiber}')
                dq_output_arr = getattr(output_model, f'spec_dq_fib{fiber}')

                orders = list(output_model.make_order_vec())
                
                for i, order in enumerate(orders):
                    if order in do_orders:
                        wavesol_output_arr[i, :] = wavesol_cal_arr[i, :]

                    if dq_cal_arr is not None:
                        flag_value = DQ_FLAGS['UNRELIABLE_WS']
                        mask = (dq_cal_arr & flag_value) != 0
                        dq_output_arr[mask] |= flag_value

            outputs[key] = output_model

        return dict(
            output=self._wrap_outputs(outputs),
            wavesol=wavesol
        )