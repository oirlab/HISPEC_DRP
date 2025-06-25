import os
from .. import utils
from .. import datamodels

from ..primitives.base_primitive import ProcessingComponent

class BasePipeline(ProcessingComponent):
    """
    Base pipeline class for managing a sequence of primitives.
    
    ARGUMENTS:
        save_result (str | None):
            * `None` : Let primitives control saving results.
            * `'all'` : Save the result from each primitive. Passes output_dir to each primitive if not already set.
            * `'final'` : Save only the final result of the pipeline. This sets save_result to False for all primitives, and the final is saved after returning from the last primitive.
        
        output_dir (str | None): Directory to save results. Can be passed to primitives if save_result == 'all'
        suffix (str | None): Suffix to append to the output filename for the pipeline only.
        full_output (bool): If True, return the full output of all primitives.
    """

    PRIMITIVES = {}

    def __init__(
        self,
        config : dict | str | None = None,
        prim_args : dict | None = None,
        **kwargs
    ):
        """
        Args:
            config (str | Config | None): The configuration file or Config to further specify parameters.
            **kwargs: Any additional keyword arguments to set or override in `self.args` for this primitive only.
        """
        self.init_config(config)
        self.set_args(**kwargs)
        self.init_primitives(prim_args)

    def set_args(self, **kwargs) -> dict:
        """
        If `self.args` already exists, this updates `self.args` with the contents of `**kwargs`.
        `self.args.name` should be the name of the primary datamodel or filename to be processed.
        """
        if not hasattr(self, 'args'):
            self.init_args()
        for key in kwargs:
            if key in self.ARGUMENTS or key == 'name':
                self.args[key] = kwargs[key]
            else:
                raise ValueError(f"Invalid argument '{key}' for primitive {self.__class__.__name__}. Valid arguments are: {list(self.ARGUMENTS.keys())}")
        return self.args
    
    def set_prim_args(self, prim_args : dict | None = None):
        """
        Set the primitive arguments for this pipeline.
        
        Args:
            prim_args (dict | None): A dictionary of primitive arguments to set.
        """
        if prim_args is None:
            return
        for prim_alias in prim_args:
            if prim_alias in self.PRIMITIVES:
                prim_instance = getattr(self, prim_alias)
                prim_instance_args = getattr(prim_instance, 'args')
                for arg_name in prim_args[prim_alias]:
                    if arg_name in prim_instance.ARGUMENTS:
                        setattr(prim_instance_args, arg_name, prim_args[prim_alias][arg_name])
                    else:
                        raise ValueError(f"Invalid argument '{arg_name}' for primitive {prim_alias}. Valid arguments are: {list(prim_instance.ARGUMENTS.keys())}")
            else:
                raise ValueError(f"Invalid primitive argument '{prim_alias}' for {self.__class__.__name__}. Valid arguments are: {list(self.args.keys())}")
    
    def init_primitives(self, prim_args : dict | None = None):
        """
        Initialize each primitive in the primitive table.
        """
        for prim_alias, prim_class in self.PRIMITIVES.items():
            primitive = prim_class(config=self.config.get('primitives', {}).get(prim_alias, {}))
            setattr(self, prim_alias, primitive)
        self.set_prim_args(prim_args)

    def apply(self, name, prim_args : dict | None = None, **kwargs):
        self.init_logger()
        self.output = None

        try:
            self.set_args(name=name, **kwargs)
            self.set_prim_args(prim_args)
            self.args.full_output = any(
                getattr(getattr(self, prim_alias).args, 'full_output', False) for prim_alias in self.PRIMITIVES
            ) or self.args.full_output
            self.log.info(f"Applying {self.__class__.__name__} with arguments:\n{self.prepr_args(self.args)}")
            self.output = self._perform()
            if isinstance(self.output.name, datamodels.DataModel):
                if self.args.save_result is True:
                    self.save_model(self.output.name)
            elif isinstance(self.output.name, list):
                for result in self.output.name:
                    if isinstance(result, datamodels.DataModel):
                        if self.args.save_result:
                            self.save_model(result)
            elif isinstance(self.output, dict):
                for key in self.output:
                    if isinstance(self.output[key], datamodels.DataModel):
                        if self.args.save_result:
                            self.save_model(self.output[key])
        except Exception as e:
            self.log.error(f"Failed executing pipeline {self.__class__.__name__}: {e}", exc_info=True)
            if os.environ.get('HISPECDRP_DEBUG', '1') != '0':
                raise e
            return name
        
        if self.args.full_output:
            return self.output
        else:
            return self.output.name
    
    def init_config(self, config : dict | str | None = None) -> dict:
        """
        Initialize the configuration for this primitive.

        Args:
            config (dict | str | None): The configuration object or path to the configuration file.
        """
        if config is None:
            self.config = {
                'class': self.__class__.__module__ + '.' + self.__class__.__name__,
                'arguments': {},
                'logging': self.LOGGING.copy(),
                'primitives': {key: {} for key in self.PRIMITIVES},
            }
        elif isinstance(config, dict):
            self.config = config
        elif isinstance(config, str):
            self.config = utils.read_config_file(config)
        else:
            raise ValueError(f"Invalid config type: {type(config)}. Expected dict or str.")
        
        if 'logging' not in self.config:
            self.config['logging'] = self.LOGGING.copy()
        else:
            self.config['logging'] = self.LOGGING | self.config['logging']
        if 'arguments' not in self.config:
            self.config['arguments'] = {}
        for prim_alias in self.PRIMITIVES:
            if prim_alias not in self.config['primitives']:
                self.config['primitives'][prim_alias] = {}
            self.config['primitives'][prim_alias]['class'] = self.PRIMITIVES[prim_alias].__module__ + '.' + self.PRIMITIVES[prim_alias].__name__
            self.config['primitives'][prim_alias]['logging'] = self.LOGGING.copy()
            if 'arguments' not in self.config['primitives'][prim_alias]:
                self.config['primitives'][prim_alias]['arguments'] = {}
            if 'config_file' in self.config['primitives'][prim_alias]:
                prim_config_file = utils.find_config_path(self.config['primitives'][prim_alias]['config_file'])
                prim_config_from_file = utils.read_config_file(prim_config_file)
                prim_config_from_file = prim_config_from_file.get('arguments', {})
                for key in prim_config_from_file:
                    if key not in self.config['primitives'][prim_alias]['arguments']:
                        self.config['primitives'][prim_alias]['arguments'][key] = prim_config_from_file[key]
        return self.config
    
    def __repr__(self):
        s = f"{self.__class__.__name__}:\n"
        s += "  Primitives:\n"
        for prim_alias, prim_class in self.PRIMITIVES.items():
            s += f"    {prim_alias}: {prim_class.__name__}\n"
        return s