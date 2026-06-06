import os
import logging

from ..utils.config import read_config_file, get_config_path
from ..utils.logging import logging_context
from ..processing_component import ProcessingComponent

logger = logging.getLogger(__name__)

class BasePipeline(ProcessingComponent):
    """
    Base pipeline class for managing a sequence of primitives.
    
    ARGUMENTS
    ---------
    output_dir (str | None): Directory to save results.
    suffix (str | None): Suffix to append to the output filename for the pipeline only.
    """

    ARGUMENTS = {
        'save_result': True,
    }

    PRIMITIVES = {}

    def __init__(
        self,
        config : dict | str | None = None,
        log_config : dict | str | None = None,
        prim_args : dict | None = None,
        **kwargs
    ):
        """
        Parameters
        ----------
        config : str | dict | None
            The configuration file or Config to further specify parameters.
        log_config : dict | str | None
            Logging configuration override applied to this pipeline and all its
            primitives. Can be a dict of logging keys, a path to a logging YAML
            file, or None to use package defaults.
        prim_args : dict | None
            A dictionary of primitive arguments to set for this pipeline.
        **kwargs
            Any additional keyword arguments to set as direct attributes for this pipeline.
        """
        self.init_config(config)
        self._init_log_config(log_config)
        self.init_args(**kwargs)
        self.init_primitives(prim_args)
    
    def set_prim_args(self, prim_args : dict | None = None):
        """
        Set the primitive arguments for this pipeline.
        
        Parameters
        ----------
        prim_args : dict | None, optional
            A dictionary of primitive-specific arguments. Keys must match the PRIMITIVES dictionary keys.
        """

        if prim_args is not None:
            for prim_alias in prim_args:
                if prim_alias in self.PRIMITIVES:
                    prim_instance = getattr(self, prim_alias)
                else:
                    raise ValueError(f"Invalid primitive alias '{prim_alias}'. Valid primitive aliases are: {list(self.PRIMITIVES.keys())}")
                
                valids_args = prim_instance.ARGUMENTS | prim_instance.CALIBRATIONS
                for arg_name in prim_args[prim_alias]:
                    if arg_name in valids_args:
                        prim_instance.set_args(**{arg_name: prim_args[prim_alias][arg_name]})
                    else:
                        raise ValueError(f"Invalid argument '{arg_name}' for primitive {prim_alias}. Valid arguments are: {valids_args}")
            
        # Propagate output directory to primitives
        input_dir = self.get_input_dir()
        for prim_alias in self.PRIMITIVES:
            prim_instance = getattr(self, prim_alias)
            if (prim_instance.save_result or prim_instance.save_preview) and prim_instance.output_dir is None:
                if self.output_dir is not None:
                    prim_instance.output_dir = self.output_dir
                elif input_dir is not None:
                    prim_instance.output_dir = input_dir

    def init_primitives(self, prim_args : dict | None = None):
        """
        Initialize each primitive object in the primitive table passing the appropriate configuration.
        """
        for prim_alias, prim_class in self.PRIMITIVES.items():
            primitive = prim_class(config=self.config['primitives'][prim_alias], parent=self)
            primitive.log_config = self.log_config
            setattr(self, prim_alias, primitive)
        self.set_prim_args(prim_args)

    def apply(self, input, prim_args : dict | None = None, **kwargs):
        """
        The main method a user calls to run a pipeline.

        Parameters
        ----------
        input
            The input data to process and is up to the subclass to decide what this represents, but is typically a datamodel, or simple containers of datamodels (or filepaths).
        prim_args : dict | None, optional
            A dictionary of primitive-specific arguments. Keys must match the PRIMITIVES dictionary keys.
        **kwargs
            Additional keyword arguments to set or override for this pipeline. These will override any arguments set during initialization.

        Returns
        -------
        dict
            The main output of the pipeline (self._output['output']), which is typically a datamodel or a container of datamodels. This is typically the last primitive's output, but is up to the pipeline implementation.
        """
        self._output = None
        with logging_context(self.log_config):
            try:

                self.set_args(input=input, **kwargs)

                input_dir = self.get_input_dir()

                #if self.save_result in (None, False) and self.output_dir is not None:
                #if self.save_result is None and self.output_dir is not None:
                #    self.save_result = True
                #if self.save_result and self.output_dir is None:
                #    self.output_dir = input_dir

                self.set_prim_args(prim_args)

                logger.info(f"Applying {self.__class__.__name__} with arguments:\n{self._prepr_args()}")

                self._output = self._perform()

                if self.save_result:
                    self.save_output() # Already finalized in the primitive's perform.
            #except Exception as e:
                #logger.error(f"Failed executing pipeline {self.__class__.__name__}: {e}", exc_info=True)
                #raise
            except Exception as e:
                self._output = dict(output=input)
                self._status = "FAILED"
                self.on_exception(e)
                self.finalize_output()
                logger.exception(
                    f"Failed executing pipeline {self.__class__.__name__}"
                )
                raise

        # Get main output
        output = self._output.get('output', None) if self._output is not None else None

        # Reset input
        self.input = None

        # Return main output
        return output
    
    def init_config(
        self,
        config : dict | str | None = None
    ) -> dict:
        """
        Initialize the configuration for the pipeline and all primitives.
        Any sub-config files are also loaded.

        Parameters
        ----------
        config : dict | str | None, optional
            The configuration dictionary or path to configuration file. If None, default configuration is used.

        Returns
        -------
        dict
            The full configuration dictionary for this pipeline at self.config.
        """

        # Determine class name
        class_name = self.__class__.__module__ + '.' + self.__class__.__name__

        # No user config provided, use defaults
        if config is None:
            self.config = {
                'class': class_name,
                'arguments': self.ARGUMENTS.copy(),
                'primitives': {key: {} for key in self.PRIMITIVES},
            }
        elif isinstance(config, str):
            config_filepath = get_config_path(config)
            self.config = read_config_file(config_filepath)
        elif isinstance(config, dict):
            self.config = config
        else:
            raise ValueError(f"Invalid config type: {type(config)}. Expected dict or str.")
        
        # Set class name
        self.config['class'] = class_name

        # NOTE: Fully populate config with all defaults

        # Default arguments
        if 'arguments' not in self.config:
            self.config['arguments'] = self.ARGUMENTS.copy()
        else:
            self.config['arguments'] = self.ARGUMENTS | self.config['arguments']

        # Ensure primitives dict exists
        if 'primitives' not in self.config:
            self.config['primitives'] = {}

        def _deep_merge(d1: dict, d2: dict) -> dict:
            result = d1.copy()
            for k, v in d2.items():
                if (
                    k in result
                    and isinstance(result[k], dict)
                    and isinstance(v, dict)
                ):
                    result[k] = _deep_merge(result[k], v)
                else:
                    result[k] = v
            return result

        # Populate primitive configs
        for prim_alias in self.PRIMITIVES:

            # Set default primitive config dict
            if prim_alias in self.config['primitives']:
                prim_config = self.config['primitives'][prim_alias]
            else:
                self.config['primitives'][prim_alias] = {}
                prim_config = self.config['primitives'][prim_alias]

            # Primitive config file
            if 'config_file' in prim_config:
                prim_config_file = get_config_path(prim_config['config_file'])
                prim_config_from_file = read_config_file(prim_config_file)
                del prim_config['config_file']
            else:
                prim_config_from_file = {}

            # Merge file config with inline config
            prim_config = _deep_merge(prim_config_from_file, prim_config)

            # Set primitive class name
            prim_class_name = self.PRIMITIVES[prim_alias].__module__ + '.' + self.PRIMITIVES[prim_alias].__name__
            prim_config['class'] = prim_class_name

            # Ensure arguments dict exists
            if 'arguments' not in prim_config:
                prim_config['arguments'] = {}

            # Set primitive config
            self.config['primitives'][prim_alias] = prim_config

        return self.config
    
    def init_args(self, **kwargs):
        """
        Initialize arguments including calibrations as top-level attributes with the following priorities:
            2. `self.config['arguments']` (from config)
            1. Direct attribute setting (highest priority)
        
        This must be called after `self.config` is initialized.

        Parameters
        ----------
        **kwargs
            Direct attribute settings to override config and defaults.
            To override a calibration, provide the calibration name as a keyword argument.
            Selector logic is not handled here.
        """

        # Config already populated with default args
        args = self.config["arguments"]
        args.update(**kwargs)

        for k, v in args.items():
            if k not in self.ARGUMENTS:
                logger.error(
                    f"Invalid argument name '{k}' for {self.__class__.__name__}. "
                    f"Valid arguments are: {list(self.ARGUMENTS.keys())}"
                )
            setattr(self, k, v)
    
    def __repr__(self):
        s = f"{self.__class__.__name__}:\n"
        s += "  Primitives:\n"
        for prim_alias, prim_class in self.PRIMITIVES.items():
            s += f"    {prim_alias}: {prim_class.__name__}\n"
        return s