from ... import datamodels
from ..base_primitive import BasePrimitive

from ...calibrations import WavelengthSolutionSelector


__all__ = ['SetInitialWLSPrimitive']


class SetInitialWLSPrimitive(BasePrimitive):

    ARGUMENTS = {
        'suffix' : 'wavesolinit',
        'do_fibers' : None,
        'output_model' : 'input',
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

        output_model = self.get_output_model(input_model)

        # Get fibers and orders
        do_fibers, _ = self.get_do_traces(input_model, self.do_fibers, None)

        for fiber in do_fibers:
            
            # Get the wavelength solution for the fiber
            wave = getattr(wavesol_model, f'wave_fib{fiber}')
            setattr(output_model, f'wave_fib{fiber}', wave)

            # Update unreliable DQ flags
            input_dq = getattr(output_model, f'spec_dq_fib{fiber}')
            wavesol_dq = getattr(wavesol_model, f'spec_dq_fib{fiber}')
            flag_value = output_model.DQ_FLAGS['UNRELIABLE_WS']
            mask = (wavesol_dq & flag_value) != 0
            input_dq[mask] |= flag_value


        return dict(output=output_model, wavesol=wavesol_model)