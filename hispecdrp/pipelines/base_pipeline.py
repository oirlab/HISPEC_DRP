import os
from .. import utils

from ..primitives.base_primitive import ProcessingComponent

class BasePipeline(ProcessingComponent):
    """
    Base pipeline class for managing a sequence of primitives.
    
    ARGUMENTS:
        output_dir (str | None): Directory to save results.
        suffix (str | None): Suffix to append to the output filename for the pipeline only.
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

                if self.output_dir is not None and prim_instance.output_dir is None and prim_instance.save_result:
                    prim_instance.output_dir = self.output_dir
            else:
                raise ValueError(f"Invalid primitive argument '{prim_alias}' for {self.__class__.__name__}. Valid arguments are: {list(self.PRIMITIVES.keys())}")
    
    def init_primitives(self, prim_args : dict | None = None):
        """
        Initialize each primitive object in the primitive table passing the appropriate configuration.
        """
        for prim_alias, prim_class in self.PRIMITIVES.items():
            primitive = prim_class(config=self.config.get('primitives', {}).get(prim_alias, {}))
            setattr(self, prim_alias, primitive)
        self.set_prim_args(prim_args)

    def apply(self, input, prim_args : dict | None = None, **kwargs):
        """
        The main method a user calls to run a pipeline.

        Args:
            input: The input data to process and is up to the subclass to decide what this represents, but is typically a datamodel, or simple containers of datamodels (or filepaths).
            prim_args (dict | None): A dictionary of primitive-specific arguments. Keys must match the PRIMITIVES dictionary keys.
            **kwargs: Additional keyword arguments to set or override for this pipeline. These will override any arguments set during initialization.
        
        Returns:
            dict: The main output of the pipeline (self._output['output']), which is typically a datamodel or a container of datamodels. This is typically the last primitive's output, but is up to the pipeline implementation.
        """
        self.init_logger()
        self._output = None
        try:
            self.set_args(input=input, **kwargs)
            self.set_prim_args(prim_args)
            self.log.info(f"Applying {self.__class__.__name__} with arguments:\n{self.prepr_args()}")
            self._output = self._perform()
            if self.save_result:
                self.save_output() # Already finalized in the primitive's perform.
        except Exception as e:
            self.log.error(f"Failed executing pipeline {self.__class__.__name__}: {e}", exc_info=True)
            if os.environ.get('HISPECDRP_DEBUG', '1') != '0':
                raise e
            return input

        # Get main output
        output = self._output.get('output', None) if self._output is not None else None

        # Reset input
        self.input = None

        # Return main output
        return output
    
    def init_config(self, config : dict | str | None = None) -> dict:
        """
        Initialize the configuration for the pipeline and all primitives.
        Any sub-config files are also loaded.

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