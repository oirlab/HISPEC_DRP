from ... import datamodels
from ...datamodels.dqflags import DQ_FLAGS
from ..base_primitive import BasePrimitive

from ...calibrations import WavelengthSolutionSelector
from ...datamodels.dqflags import DQ_FLAGS


__all__ = ['SetInitialWLSPrimitive']


class SetInitialWLSPrimitive(BasePrimitive):

    """
    Set the initial wavelength solution for each fiber/order. This should be accurate to ~ < 10% of a resolution element.

    Parameters
    ----------
    input : str or L1Model
        Input L1 model or file name.
    suffix : str
        Suffix to add to the output file name. Default is 'wavesolinit'.
    do_fibers : list of int, optional
        List of fibers to process. If None, all fibers will be processed. Default is None.

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
        'run': True,
        'suffix' : 'wavesolinit',
        'do_fibers' : None,
        'do_orders': None,
        'wavesol' : None,  # If set (e.g. LFCWavesolModel), use instead of calibration lookup
    }

    CALIBRATIONS = {
        'wavesol': {
            'selector': WavelengthSolutionSelector,
            'selector_kwargs': {}
        }
    }

    def _perform(self):

        # Open the input model
        input_model = datamodels.open(self.input)
        if not getattr(self, "run", True):
            return dict(output=input_model, wavesol=None)

        wavesol_model = getattr(self, 'wavesol', None)
        if wavesol_model is None:
            wavesol_path, _ = self.get_calibration(input_model, 'wavesol')
            wavesol_model = datamodels.open(wavesol_path)
        elif not hasattr(wavesol_model, 'wave_fib1'):
            wavesol_model = datamodels.open(wavesol_model)

        output_model = self.get_output_model(input_model)

        # Get fibers and orders
        do_fibers, _ = self.get_do_traces(input_model, self.do_fibers, None)
        do_orders = getattr(self, "do_orders", None)
        if do_orders is not None:
            try:
                do_orders = list(do_orders)
            except TypeError:
                do_orders = None

        # Best-effort: infer wavesol order vector for mapping if metadata exists
        wavesol_orders = None
        if do_orders is not None and hasattr(wavesol_model, "make_order_vec"):
            try:
                wavesol_orders = list(wavesol_model.make_order_vec())
            except Exception:
                wavesol_orders = None

        for fiber in do_fibers:
            
            # Get the wavelength solution for the fiber
            wave_full = getattr(wavesol_model, f'wave_fib{fiber}')
            dq_full = getattr(wavesol_model, f'spec_dq_fib{fiber}')
            wave = wave_full
            wavesol_dq = dq_full
            if do_orders is not None:
                # Map order numbers -> row indices if possible; otherwise fall back to leading rows
                if wavesol_orders is not None and wave_full is not None and getattr(wave_full, "shape", None) and len(wavesol_orders) == wave_full.shape[0]:
                    try:
                        idx = [wavesol_orders.index(int(o)) for o in do_orders]
                        wave = wave_full[idx, :]
                        wavesol_dq = dq_full[idx, :]
                    except Exception:
                        wave = wave_full[: len(do_orders), :]
                        wavesol_dq = dq_full[: len(do_orders), :]
                elif wave_full is not None and getattr(wave_full, "shape", None) and wave_full.shape[0] >= len(do_orders):
                    wave = wave_full[: len(do_orders), :]
                    wavesol_dq = dq_full[: len(do_orders), :]
            setattr(output_model, f'wave_fib{fiber}', wave)

            # Update unreliable DQ flags
            input_dq = getattr(output_model, f'spec_dq_fib{fiber}')
            if input_dq is not None and wavesol_dq is not None:
                # Some models (e.g. L1Model) do not expose DQ_FLAGS; use global bit definitions.
                flag_table = getattr(output_model, "DQ_FLAGS", DQ_FLAGS)
                flag_value = flag_table['UNRELIABLE_WS']
                mask = (wavesol_dq & flag_value) != 0
                input_dq[mask] |= flag_value


        return dict(output=output_model, wavesol=wavesol_model)