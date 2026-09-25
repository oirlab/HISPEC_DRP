import logging
from typing import Sequence

from hispecdrp import datamodels
from ..calibration_pipeline_base import CalibrationPipeline
from ...primitives import CoaddFramesPrimitive, TracePositionPrimitive, TraceProfilePrimitive, RegisterCalibrationPrimitive, SpectralExtractionPrimitive, SetInitialWLSPrimitive, MakeFiberflatPrimitive


logger = logging.getLogger(__name__)

__all__ = ['ProcessFiberflatsPipeline']


class ProcessFiberflatsPipeline(CalibrationPipeline):

    PARAMETERS = {
        'use_derived_trace_cals' : True # True for boostrapping until we have good trace cals
    }

    PRIMITIVES = {
        'coadd_frames': CoaddFramesPrimitive,
        'trace_pos': TracePositionPrimitive,
        'trace_prof': TraceProfilePrimitive,
        'register_tracepos_cal': RegisterCalibrationPrimitive,
        'register_traceprof_cal': RegisterCalibrationPrimitive,
        'spec_extract': SpectralExtractionPrimitive,
        'set_wavesol' : SetInitialWLSPrimitive,
        'make_fiberflat': MakeFiberflatPrimitive,
        'register_fiberflat_cal': RegisterCalibrationPrimitive,
    }

    def _perform(self):

        # Coadd individual fiberflat frames for each fiber individually
        # Models split by fiber
        ff_coadded_models_L0 = self.coadd_frames.apply(self.dataset)

        ff_coadded_models_L0 = datamodels.group_by_fiber_combination(ff_coadded_models_L0)
        
        first_key = next(iter(ff_coadded_models_L0))

        # Current keys: ('WHITELIGHT', 'BLOCKED', 'BLOCKED') : file, etc.
        # Want: ('1' : file, '2' : file, '3' : file)
        if not str(first_key).isdigit():
            ff_coadded_models_L0 = {
                str(t.index(next(x for x in t if x.upper() != "BLOCKED")) + 1) : v
                for t, v in ff_coadded_models_L0.items()
            }

        # Make sure each value is a single datamodel
        for key, value in ff_coadded_models_L0.items():
            if isinstance(value, Sequence):
                assert len(value) == 1, f"Expected only one model for fiber {key}, but got {len(value)}."
                ff_coadded_models_L0[key] = value[0]

        # Create trace pos and prof cals
        # All fibers merged into one model
        trace_position_model = self.trace_pos.apply(ff_coadded_models_L0)
        trace_profile_model = self.trace_prof.apply(ff_coadded_models_L0)

        # Register the trace pos and prof cals
        trace_pos_reg = self.register_tracepos_cal.apply(trace_position_model)
        trace_prof_reg = self.register_traceprof_cal.apply(trace_profile_model)
        
        # Set the trace pos and prof cals in the spectral extraction primitive
        # NOTE: Final behavior TBD - sometimes we want to use the these trace cals, sometimes we want older trace cals.
        if self.use_derived_trace_cals:
            if self.spec_extract.tracepos is not None:
                logger.warning(f"Overriding spec_extract.tracepos with {trace_pos_reg}.")
            self.spec_extract.tracepos = trace_pos_reg
            if self.spec_extract.traceprof is not None:
                logger.warning(f"Overriding spec_extract.traceprof with {trace_prof_reg}.")
            self.spec_extract.traceprof = trace_prof_reg

        # Extract the coadded fiberflats (models split by fiber)
        ff_coadded_models_L1 = {}
        for fiber, model in ff_coadded_models_L0.items():
            ff_coadded_models_L1[fiber] = self.spec_extract.apply(model)

        # Set wavelength solution for each model
        for fiber, model in ff_coadded_models_L1.items():
            ff_coadded_models_L1[fiber] = self.set_wavesol.apply(model)

        # Create the fiberflat cal
        fiberflat_model = self.make_fiberflat.apply(ff_coadded_models_L1)

        # Register the fiberflat cal
        ff_reg = self.register_fiberflat_cal.apply(fiberflat_model)

        return dict(
            output=dict(
                trace_pos=trace_pos_reg, trace_prof=trace_prof_reg,
                fiberflat=ff_reg
            )
        )
