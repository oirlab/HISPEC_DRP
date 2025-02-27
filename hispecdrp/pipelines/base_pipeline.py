import os

from ..utils.config import Config
from ..primitives.base_primitive import BasePrimitive


class BasePipeline(BasePrimitive):

    ARGUMENTS = {
        'save_results': 'final', # Overrides individual primitive save_results. Set to None to manually control each primitive.
    }

    PRIMITIVES = {}

    def __init__(
            self,
            config : Config | None = None,
            config_file : Config | str | None = None,
            **kwargs
        ):
        """
        Args:
            config (str | Config | None): The configuration file or Config to further specify parameters.
            **kwargs: Any additional keyword arguments to set or override in `self.args` for this primitive only.
        """
        super().__init__(config=config, config_file=config_file, **kwargs)
        self.init_primitives()


    def init_primitives(self):
        """
        Initialize each primitive in the primitive table.
        """
        # TODO: Sort out config dirs
        if self.config_file is not None:
            config_dir = os.path.dirname(self.config_file)
        else:
            config_dir = ""
        for prim_alias, prim_class in self.PRIMITIVES.items():
            if prim_alias in self.config["primitives"]:
                prim_config_from_pipeline = self.config["primitives"][prim_alias]
                if "config_file" in prim_config_from_pipeline:
                    prim_config_file = os.path.join(config_dir, prim_config_from_pipeline["config_file"])
                else:
                    prim_config_file = None
            else:
                prim_config_from_pipeline = None
                prim_config_file = None

            # !!! NOTE !!! Keep an eye on how we propogate the pipeline's logger to the primitive.
            prim_config = prim_class._get_config(config_file=prim_config_file, config=prim_config_from_pipeline)
            prim_config['logging'].update(**self.config['logging'])
            primitive = prim_class(
                config_file=prim_config_file,
                config=prim_config,
            )

            # !!! NOTE !!! Keep an eye on how we propogate the pipeline's logger to the primitive.
            if primitive.config.get('output_dir') is None and primitive.args.output_dir is None:
                primitive.args.output_dir = self.args.output_dir
            if self.args.save_results == 'final':
                #primitive.config['save_results'] = False
                primitive.args.save_results = False
            if self.args.save_results == 'all':
                #primitive.config['save_results'] = True
                primitive.args.save_results = True

            # Set the primitive
            setattr(self, prim_alias, primitive)

    @classmethod
    def _get_config(cls, config : Config | None = None, config_file : str | None = None) -> Config:
        """
        Load a config file for this primitive or pipeline, or construct one from scratch.
        """
        _config = super()._get_config(config_file=config_file, config=config)
        if 'primitives' not in _config:
            _config['primitives'] = {key : Config() for key in cls.PRIMITIVES.keys()}

        return _config