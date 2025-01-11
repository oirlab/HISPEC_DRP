from keckdrpframework.primitives.base_primitive import BasePrimitive
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.action import Action
from keckdrpframework.models.event import Event
from keckdrpframework.pipelines.base_pipeline import BasePipeline
from keckdrpframework.models.processing_context import ProcessingContext
from keckdrpframework.config.framework_config import ConfigClass

from typing import Type
from ..primitives import HISPECPrimitive

class KDRPFHISPECPipeline(BasePipeline):

    event_table = {

        # File management
        'ingest_file': ('IngestHISPECFile', 'ingest_file_started', 'classify_file'),
        'classify_file': ('ClassifyHISPECFile', 'classify_file_started', 'action_planner'),
        'action_planner': ('action_planner', 'action_planner_started', None),

        # Main Pipeline
        'stage0': ('HISPECStage0Pipeline', 'stage0_started', 'stage1'),
        'stage1': ('HISPECStage1Pipeline', 'stage1_started', 'stage2'),
        'stage2': ('HISPECStage2Pipeline', 'stage2_started', None),

        # QL Pipeline
        'stage0ql': ('HISPECStage0QLPipeline', 'stage0ql_started', 'stage1ql'),
        'stage1ql': ('HISPECStage1QLPipeline', 'stage1ql_started', 'stage2ql'),
        'stage2ql': ('HISPECStage2QLPipeline', 'stage2ql_started', None),

        # Calibration Pipelines
        'autocal_dark': ('AutocalDarkPipeline', 'autocaldark_started', None),
        'autocal_flat': ('AutocalFlatPipeline', 'autocalflat_started', None),
        'wavecal': ('WavelengthCalibration', 'autocalflat_started', None),

    }

    def action_planner(self, action, context):
        # Determine where to enter event table
        # action and context tells us the kind of file
        # and if the DRP is in QL or normal mode
        # ...
        pass

    def _get_action_apply_method(self, _class : Type[HISPECPrimitive]):
        # Instantiating a primitive with action/context is done through a classmethod
        def func(action, context):
            primitive = _class.from_action_context(action, context)
            return primitive.apply()
        return func

    