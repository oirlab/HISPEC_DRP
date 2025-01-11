from keckdrpframework.models.arguments import Arguments
from keckdrpframework.config.framework_config import ConfigClass

from ..primitives.base_primitive import HISPECPrimitive


class HISPECPipeline(HISPECPrimitive):

    primitive_table : dict[str, type[HISPECPrimitive]]

    def __init__(
            self,
            config : ConfigClass | str | None = None,
            **kwargs
        ):
        """
        Parameters:
            config_file (str): Filename for pipeline-level configuration file containing configuration for all primitives.
        """
        super().__init__(config=config)
        self.init_primitives()
        
        # Get cals?

    def init_primitives(self):
        """
        Initialize each primitive in the primitive table.
        """
        # for attr, prim in self.primitive_table.items():
        pass


    def _perform(self):
        """
        Invokes all pipeline steps manually
        """
        pass