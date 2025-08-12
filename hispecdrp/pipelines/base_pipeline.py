import os
from .. import utils
from .. import datamodels

from ..primitives.base_primitive import ProcessingComponent

class BasePipeline(ProcessingComponent):
    """
    Base pipeline class for managing a sequence of primitives.
    
    ARGUMENTS:
        save_result (str | None):
            - `None` : Let primitives control saving results.
            - `'all'` : Save the result from each primitive. Passes output_dir to each primitive if not already set.
            - `'final'` : Save only the final result of the pipeline. This sets save_result to False for all primitives, and the final is saved after returning from the last primitive.
        
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
            prim_args (dict | None): A dictionary of primitive arguments to set for this pipeline.
            **kwargs: Any additional keyword arguments to set as direct attributes for this pipeline.
        """
        self.init_config(config)
        self.set_args(**kwargs)
        self.init_primitives(prim_args)

    def set_args(self, **kwargs) -> dict:
        """
        Set arguments directly as top-level attributes on the instance.
        """
        if not hasattr(self, '_args_initialized'):
            self.init_args()
            self._args_initialized = True
        
        for key, value in kwargs.items():
            if key in self.ARGUMENTS:
                setattr(self, key, value)
            # NOTE: Ignore invalid keys?

        return kwargs
    
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
                for arg_name in prim_args[prim_alias]:
                    if arg_name in prim_instance.ARGUMENTS:
                        prim_instance.set_args(**{arg_name: prim_args[prim_alias][arg_name]})
                    else:
                        raise ValueError(f"Invalid argument '{arg_name}' for primitive {prim_alias}. Valid arguments are: {list(prim_instance.ARGUMENTS.keys())}")
            else:
                raise ValueError(f"Invalid primitive argument '{prim_alias}' for {self.__class__.__name__}. Valid arguments are: {list(self.PRIMITIVES.keys())}")
    
    def init_primitives(self, prim_args : dict | None = None):
        """
        Initialize each primitive in the primitive table.
        """
        for prim_alias, prim_class in self.PRIMITIVES.items():
            primitive = prim_class(config=self.config.get('primitives', {}).get(prim_alias, {}))
            setattr(self, prim_alias, primitive)
        self.set_prim_args(prim_args)

    def apply(self, input, prim_args : dict | None = None, **kwargs):
        self.init_logger()
        self._output = None
        try:
            self.set_args(input=input, **kwargs)
            self.set_prim_args(prim_args)
            self.full_output = any(
                getattr(getattr(self, prim_alias), 'full_output', False) for prim_alias in self.PRIMITIVES
            ) or getattr(self, 'full_output', False)
            self.log.info(f"Applying {self.__class__.__name__} with arguments:\n{self.prepr_args()}")
            self._output = self._perform()
            if isinstance(self._output['output'], datamodels.DataModel):
                if getattr(self, 'save_result', False) is True:
                    self.save_model(self._output['output'])
            elif isinstance(self._output['output'], list):
                for result in self._output['output']:
                    if isinstance(result, datamodels.DataModel):
                        if getattr(self, 'save_result', False):
                            self.save_model(result)
            elif isinstance(self._output['output'], dict):
                for key in self._output['output']:
                    if isinstance(self._output['output'][key], datamodels.DataModel):
                        if getattr(self, 'save_result', False):
                            self.save_model(self._output['output'][key])
        except Exception as e:
            self.log.error(f"Failed executing pipeline {self.__class__.__name__}: {e}", exc_info=True)
            if os.environ.get('HISPECDRP_DEBUG', '1') != '0':
                raise e
            return input
        
        if getattr(self, 'full_output', False):
            return self._output
        else:
            return self._output['output']
    
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