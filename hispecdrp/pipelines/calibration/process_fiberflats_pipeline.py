from hispecdrp import datamodels
from ..calibration_pipeline_base import CalibrationPipeline
from ...primitives import CoaddFramesPrimitive, TracePositionPrimitive, TraceProfilePrimitive, RegisterCalibrationPrimitive, SpectralExtractionPrimitive, SetInitialWLSPrimitive
from typing import Sequence

__all__ = ['ProcessFiberflatsPipeline']


class ProcessFiberflatsPipeline(CalibrationPipeline):

    PRIMITIVES = {
        'coadd_frames': CoaddFramesPrimitive,
        'trace_pos': TracePositionPrimitive,
        'trace_prof': TraceProfilePrimitive,
        'register_tracepos_cal': RegisterCalibrationPrimitive,
        'register_traceprof_cal': RegisterCalibrationPrimitive,
        'spec_extract': SpectralExtractionPrimitive,
        'set_wavesol' : SetInitialWLSPrimitive,
        'register_fiberflat_cal': RegisterCalibrationPrimitive,
    }

    def _perform(self):

        # Coadd individual fiberflat frames for each fiber individually
        ff_coadded_models = self.coadd_frames.apply(self.input)

        # Create trace pos and prof cals
        trace_position_model = self.trace_pos.apply(ff_coadded_models)
        trace_profile_model = self.trace_prof.apply(ff_coadded_models)

        # Register the trace pos and prof cals
        trace_pos_reg = self.register_tracepos_cal.apply(trace_position_model)
        trace_prof_reg = self.register_traceprof_cal.apply(trace_profile_model)
        
        # Set the trace pos and prof cals in the spectral extraction primitive
        if self.spec_extract.tracepos is not None:
            self.log.warning(f"Overriding spec_extract.tracepos with {trace_pos_reg}.")
        self.spec_extract.tracepos = trace_pos_reg
        if self.spec_extract.traceprof is not None:
            self.log.warning(f"Overriding spec_extract.traceprof with {trace_prof_reg}.")
        self.spec_extract.traceprof = trace_prof_reg

        # Extract the coadded fiberflats
        ff_coadded_models_extracted = self.spec_extract.apply(ff_coadded_models)

        # Associate L1 by fiber
        ff_coadded_models_extracted = datamodels.associate_fibers(ff_coadded_models_extracted)

        # Set wavelength solutions
        #for fiber in ff_coadded_models_extracted.keys():
        #    self.set_wavesol.apply(ff_coadded_models_extracted[fiber], do_fibers=[fiber])

        # Prepare output fiberflat model
        ff_output_model = self.get_output_model(ff_coadded_models_extracted, output_model='FiberflatModel', include_data=False)

        for fiber, ff_model in ff_coadded_models_extracted.items():
            #output_model FiberFlat data keys = dict_keys(['PRIMARY', 'SPEC_FIB1', 'SPEC_ERR_FIB1', 'SPEC_VAR_POISSON_FIB1', 'SPEC_VAR_RNOISE_FIB1', 'SPEC_DQ_FIB1', 'WAVE_FIB1', 'MID_TIME_FIB1', 'TRANS_FIB1', 'TRANS_ERR_FIB1', 'SPEC_FIB2', 'SPEC_ERR_FIB2', 'SPEC_VAR_POISSON_FIB2', 'SPEC_VAR_RNOISE_FIB2', 'SPEC_DQ_FIB2', 'WAVE_FIB2', 'MID_TIME_FIB2', 'TRANS_FIB2', 'TRANS_ERR_FIB2', 'CALIBRATIONS', 'DATA_ANCESTORS', 'RECEIPT'])
            # ff coadded model keys = dict_keys(['PRIMARY', 'SPEC_UNCAL_FIB1', 'SPEC_UNCAL_ERR_FIB1', 'SPEC_UNCAL_VAR_POISSON_FIB1', 'SPEC_UNCAL_VAR_RNOISE_FIB1', 'SPEC_DQ_FIB1', 'WAVE_FIB1', 'MID_TIME_FIB1', 'BARYCORR_FIB1', 'SPEC_UNCAL_FIB2', 'SPEC_UNCAL_ERR_FIB2', 'SPEC_UNCAL_VAR_POISSON_FIB2', 'SPEC_UNCAL_VAR_RNOISE_FIB2', 'SPEC_DQ_FIB2', 'WAVE_FIB2', 'MID_TIME_FIB2', 'BARYCORR_FIB2', 'CALIBRATIONS', 'DATA_ANCESTORS', 'RECEIPT'])
            # Set spectrum data attributes
            setattr(ff_output_model, f'SPEC_FIB{fiber}', getattr(ff_coadded_models_extracted[fiber], f'SPEC_UNCAL_FIB{fiber}'))
            setattr(ff_output_model, f'SPEC_ERR_FIB{fiber}', getattr(ff_coadded_models_extracted[fiber], f'SPEC_UNCAL_ERR_FIB{fiber}'))
            setattr(ff_output_model, f'SPEC_VAR_POISSON_FIB{fiber}', getattr(ff_coadded_models_extracted[fiber], f'SPEC_UNCAL_VAR_POISSON_FIB{fiber}'))
            setattr(ff_output_model, f'SPEC_VAR_RNOISE_FIB{fiber}', getattr(ff_coadded_models_extracted[fiber], f'SPEC_UNCAL_VAR_RNOISE_FIB{fiber}'))
            setattr(ff_output_model, f'SPEC_DQ_FIB{fiber}', getattr(ff_coadded_models_extracted[fiber], f'SPEC_DQ_FIB{fiber}'))

            # Set mid times
            setattr(ff_output_model, f'MID_TIME_FIB{fiber}', getattr(ff_coadded_models_extracted[fiber], f'MID_TIME_FIB{fiber}'))

            # Add ancestors
            self.add_data_ancestors(ff_output_model, ff_model)

        # Get the wavelength solution
        # TODO: NEED TO INPUT WAVELENGTH SOLUTION HERE
        # DO WE GET THE WAVESOLS SEPARATELY OR FROM THE MERGED FRAME?
        ff_output_model = self.set_wavesol.apply(ff_output_model)

        # Register the fiberflat cal
        ff_reg = self.register_fiberflat_cal.apply(ff_output_model)

        return dict(
            output=dict(
                trace_pos=trace_pos_reg, trace_prof=trace_prof_reg,
                fiberflat=ff_reg
            )
        )
