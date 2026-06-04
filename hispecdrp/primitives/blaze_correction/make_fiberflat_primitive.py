from ..base_primitive import BasePrimitive
from ... import datamodels

__all__ = ['MakeFiberflatPrimitive']

class MakeFiberflatPrimitive(BasePrimitive):
    """
    Create a fiber flat calibration model from a combined rate map.

    Parameters
    ----------
    input : dict[str, L0Model]
        Input L1 models of fiber flats.

    Returns
    -------
    output : FiberflatModel
        The derived fiber flat calibration model.
    """

    ARGUMENTS = {
        'output_model' : 'FiberFlatPrimitive',
    }

    def _perform(self):

        ff_coadded_models_L1 = self.input

        # Prepare output fiberflat model
        ff_output_model = datamodels.FiberflatModel.from_datamodels(ff_coadded_models_L1, primitive=self)
        
        for fiber, ff_model in ff_coadded_models_L1.items():

            # Set spectrum data attributes
            setattr(
                ff_output_model,
                f'WAVE_FIB{fiber}',
                getattr(ff_model, f'WAVE_FIB{fiber}')
            )
            
            # Set spectrum data attributes
            setattr(
                ff_output_model,
                f'SPEC_FIB{fiber}',
                getattr(ff_model, f'SPEC_UNCAL_FIB{fiber}')
            )

            setattr(
                ff_output_model,
                f'SPEC_ERR_FIB{fiber}',
                getattr(ff_model, f'SPEC_UNCAL_ERR_FIB{fiber}')
            )

            setattr(
                ff_output_model,
                f'SPEC_VAR_POISSON_FIB{fiber}',
                getattr(ff_model, f'SPEC_UNCAL_VAR_POISSON_FIB{fiber}')
            )

            setattr(
                ff_output_model,
                f'SPEC_VAR_RNOISE_FIB{fiber}',
                getattr(ff_model, f'SPEC_UNCAL_VAR_RNOISE_FIB{fiber}')
            )

            setattr(
                ff_output_model,
                f'SPEC_DQ_FIB{fiber}',
                getattr(ff_model, f'SPEC_DQ_FIB{fiber}')
            )

            setattr(
                ff_output_model,
                f'MID_TIME_FIB{fiber}',
                getattr(ff_model, f'MID_TIME_FIB{fiber}')
            )

        return dict(output=ff_output_model)