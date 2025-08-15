from ... import datamodels
from ..base_primitive import BasePrimitive

from ...calibrations import WavelengthSolutionSelector


__all__ = ['SetInitialWLSPrimitive']


class SetInitialWLSPrimitive(BasePrimitive):

    ARGUMENTS = {
        'suffix' : 'wavesolinit',
        'do_fibers' : None,
        'do_orders' : None,
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
        wavesol_path, _ = self.get_calibration(input_model, 'wavesol')
        wavesol_model = datamodels.open(wavesol_path)

        for fiber in self.do_fibers:
            
            # Get the wavelength solution for the fiber
            wave = getattr(wavesol_model, f'wave_fib{fiber}')
            setattr(input_model, f'wave_fib{fiber}', wave)

            # Update unreliable DQ flags
            input_dq = getattr(input_model, f'spec_dq_fib{fiber}')
            wave_dq = getattr(wavesol_model, f'spec_dq_fib{fiber}')
            flag_value = input_model.DQ_FLAGS['UNRELIABLE_WS']
            mask = (wave_dq & flag_value) != 0
            input_dq[mask] |= flag_value

        return dict(output=input_model, wavesol=wavesol_model)