from ..primitives.base_primitive import BasePrimitive
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.action import Action
from keckdrpframework.models.event import Event
from keckdrpframework.pipelines.base_pipeline import BasePipeline as BaseKeckPipeline
from keckdrpframework.models.processing_context import ProcessingContext
from keckdrpframework.config.framework_config import ConfigClass

from typing import Type
from ..primitives import BasePrimitive


__all__ = ['HISPECRTIPipeline']

class HISPECRTIPipeline(BaseKeckPipeline):

    event_table = {}

    def action_planner(self, action, context):
        # Determine where to enter event table
        # action and context tells us the kind of file
        # and if the DRP is in QL or normal mode
        # ...
        pass

    def _get_action_apply_method(self, _class : Type[BasePrimitive]):
        # Instantiating a primitive with action/context is done through a classmethod
        def func(action, context):
            primitive = _class.from_action_context(action, context)
            return primitive.apply()
        return func