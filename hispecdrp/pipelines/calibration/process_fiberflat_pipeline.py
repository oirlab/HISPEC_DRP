from hispecdrp import datamodels
from ..calibration_pipeline_base import CalibrationPipeline
from ...primitives import CoaddFramesPrimitive, TracePositionPrimitive, TraceProfilePrimitive, RegisterCalibrationPrimitive

__all__ = ['ProcessFiberflatPipeline']


class ProcessFiberflatPipeline(CalibrationPipeline):

    PRIMITIVES = {
        'coadd_frames': CoaddFramesPrimitive,
        'trace_pos': TracePositionPrimitive,
        'trace_prof': TraceProfilePrimitive,
        'register_tracepos_cal': RegisterCalibrationPrimitive,
        'register_traceprof_cal': RegisterCalibrationPrimitive,
    }

    def _perform(self):
        ff_coadded_models = self.coadd_frames.apply(self.input)
        trace_position_model = self.trace_pos.apply(ff_coadded_models)
        trace_profile_model = self.trace_prof.apply(ff_coadded_models)
        trace_pos_reg = self.register_tracepos_cal.apply(trace_position_model)
        trace_prof_reg = self.register_traceprof_cal.apply(trace_profile_model)
        return dict(output=dict(trace_pos=trace_pos_reg, trace_prof=trace_prof_reg))
