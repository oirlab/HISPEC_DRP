import os
import traceback

from ..utils.config import Config
from ..primitives.base_primitive import BasePrimitive
from .. import datamodels


class BasePipeline(BasePrimitive):

    ARGUMENTS = {
        'save_result': 'final', # Overrides individual primitive save_result. Set to None to manually control each primitive.
        'full_output': False, # If True, all primitives will return their full output, and the pipeline will collect these extra results in a dict.
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
            if self.args.save_result == 'final':
                #primitive.config['save_result'] = False
                primitive.args.save_result = False
            if self.args.save_result == 'all':
                #primitive.config['save_result'] = True
                primitive.args.save_result = True

            # Set the primitive
            setattr(self, prim_alias, primitive)

    def apply(self, name, **kwargs):
        self.init_logger()
        self.output = None
        self.args.full_output = any(
            getattr(getattr(self, prim_alias).args, 'full_output', False) for prim_alias in self.PRIMITIVES
        )
        try:
            self.set_args(name=name, **kwargs)
            if not self._pre_condition():
                self.log.info(f"Skipping {self.__class__.__name__} with arguments:\n{self.prepr_args(self.args)}")
                self.on_skip()
            else:
                self.log.info(f"Applying {self.__class__.__name__} with arguments:\n{self.prepr_args(self.args)}")
                output = self._perform()
                if self._post_condition():
                    self.output = output
                    if isinstance(self.output.name, datamodels.DataModel):
                        #self.finalize_model(self.output.name)
                        if self.args.save_result:
                            self.save_model(self.output.name)
                    elif isinstance(self.output.name, list):
                        for result in self.output.name:
                            if isinstance(result, datamodels.DataModel):
                                self.finalize_model(result)
                                if self.args.save_result:
                                    self.save_model(result)
                    elif isinstance(self.output, dict):
                        for key in self.output:
                            if isinstance(self.output[key], datamodels.DataModel):
                                self.finalize_model(self.output[key])
                                if self.args.save_result:
                                    self.save_model(self.output[key])
        except Exception as e:
            self.log.error(f"Failed executing primitive {self.__class__.__name__}: {e}\n{traceback.format_exc()}")
            if os.environ.get('HISPECDRP_DEBUG', '0') == '1':
                raise e
            return name
        
        if self.args.full_output:
            return self.output
        else:
            return self.output.name
        
    def set_args(self, **kwargs) -> dict:
        """
        Set the arguments for this primitive or pipeline.
        Args:
            name: The primary input.
            **kwargs: Additional keyword arguments to set in `self.args`. A special key `prim_args` can be used to set arguments for individual primitives in the pipeline, where each subkey is the alias of a primtiive.
        """
        kwargs_sup = {k : v for k, v in kwargs.items() if k.lower() != 'prim_args'}
        super().set_args(**kwargs_sup)
        if 'prim_args' in kwargs:
            prim_args = kwargs['prim_args']
            for prim_alias in self.PRIMITIVES:
                if prim_alias in prim_args:
                    if not hasattr(self, prim_alias):
                        raise ValueError(f"Bad key in prim_args: {prim_alias} for {self.__class__.__name__}. Available primitives: {list(self.PRIMITIVES.keys())}")
                    prim = getattr(self, prim_alias)
                    prim.set_args(**prim_args[prim_alias.lower()])

    @classmethod
    def _get_config(cls, config : Config | None = None, config_file : str | None = None) -> Config:
        """
        Load a config file for this primitive or pipeline, or construct one from scratch.
        """
        _config = super()._get_config(config_file=config_file, config=config)
        if 'primitives' not in _config:
            _config['primitives'] = {key : Config() for key in cls.PRIMITIVES.keys()}

        return _config