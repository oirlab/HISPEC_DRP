from keckdrpframework.models.arguments import Arguments
from ..utils.config import Config
from .. import datamodels
import os

from ..primitives.base_primitive import BasePrimitive


class BasePipeline(BasePrimitive):

    ARGUMENTS = {
        'save_results': 'final', # Overrides individual primitive save_results. Set to None to manually control each primitive.
    }

    PRIMITIVES = {}

    def __init__(
            self,
            config : Config | str | None = None,
            **kwargs
        ):
        """
        Args:
            config (str | Config | None): The configuration file or Config to further specify parameters.
            **kwargs: Any additional keyword arguments to set or override in `self.args` for this primitive only.
        """
        super().__init__(config=config, **kwargs)
        self.init_primitives()

    def init_primitives(self):
        """
        Initialize each primitive in the primitive table.
        """
        # TODO: Clean this up?
        if self.config_file is not None:
            config_dir = os.path.dirname(self.config_file)
        else:
            config_dir = ""
        for prim_alias, prim_class in self.PRIMITIVES.items():
            if prim_alias not in self.config["primitives"]:
                self.config["primitives"][prim_alias] = Config({'class': prim_class.__name__, 'arguments': Config(), 'logger': Config()})
            prim_config_from_pipeline = self.config["primitives"][prim_alias]
            if "config_file" in prim_config_from_pipeline:
                prim_config_file = os.path.join(config_dir, prim_config_from_pipeline["config_file"])
            else:
                prim_config_file = None
            prim_args = {}
            if prim_config_file is not None:
                prim_config_from_file = prim_class._load_config(config_file=prim_config_file)
                prim_args.update(prim_config_from_file.arguments)
            for key, val in prim_config_from_pipeline['arguments'].items():
                if key != 'config_file':
                    prim_args[key] = val
            primitive = prim_class(
                config_file=prim_config_file,
                **prim_args
            )
            if primitive.args.output_dir is None:
                primitive.args.output_dir = self.args.output_dir
            if self.args.save_results == 'final':
                primitive.args.save_results = False
            if self.args.save_results == 'all':
                primitive.args.save_results = True
            setattr(self, prim_alias, primitive)