import os
import logging
from copy import deepcopy
from typing import Any

from ..driver.logging import logging_context
from ..driver import ConfigFile, ConfigStore, ConfigSelector, get_config_filepath
from ..processing_component import ProcessingComponent

logger = logging.getLogger(__name__)

class BasePipeline(ProcessingComponent):
    """
    Base pipeline class for managing a sequence of primitives.
    
    PARAMETERS
    ---------
    output_dir (str | None): Directory to save results.
    suffix (str | None): Suffix to append to the output filename for the pipeline only.
    """

    PARAMETERS = {
        'save_result': True,
    }

    PRIMITIVES = {}

    ########################
    #### INITIALIZATION ####
    ########################

    def __init__(
        self,
        dataset : Any = None,
        config : dict | ConfigFile | str | None = None,
        context : dict | None = None,
        log_config : dict | str | None = None,
        primitives : dict | None = None,
        **kwargs
    ):
        """
        Parameters
        ----------
        config : dict | ConfigFile | str | None
            The configuration file or Config to further specify parameters.
        log_config : dict | str | None
            Logging configuration override applied to this pipeline and all its
            primitives. Can be a dict of logging keys, a path to a logging YAML
            file, or None to use package defaults.
        primitives : dict | None
            A dictionary of primitive instances to use for this pipeline.
        **kwargs
            Any additional keyword arguments to set as direct attributes for this pipeline.
        """

        self.dataset = dataset
        
        # Load the configuration file or Config object
        self.init_config(dataset, config, context)

        # Resolve logging config (merges defaults -> config file -> kwarg)
        self._init_log_config(log_config)

        # Set any kwargs passed to init
        self.init_parameters(**kwargs)

        # Initialize the primitives
        self.init_primitives(context, primitives)

        # Init the output, status, and calibrations used
        self._output = None
        self._status = None
        self._calibrations_used = {}

    def init_primitives(
        self,
        context : dict | None = None,
        primitives : dict | None = None,
    ):
        """
        Initialize all primitives in the pipeline.

        Parameters
        ----------
        context : dict, optional
            A dictionary of context information to use for selecting the configuration file for each primitive.
        primitives : dict, optional
            A dictionary of primitive aliases and user-specified parameters.
        """

        # Default steps dict from user input
        primitives = primitives or {}
        context = context or {}

        # Alias pipeline context
        pipe_context = self.config.context

        valid_primitives = set(self.PRIMITIVES.keys())
        for key in primitives.keys():
            if key not in valid_primitives:
                raise ValueError(f"Invalid primitive alias '{key}' in pipeline {self.__class__.__name__}. Valid primitive aliases are: {valid_primitives}")

        # Loop over and instantiate all primitives in the pipeline
        for prim_alias, prim_class in self.PRIMITIVES.items():

            # TODO: Consider resolving step config file here to more robustly handle step parameters if needed. Currently, step config files are not resolved until the step is instantiated. If we want to override step parameters like output_dir, etc, we may want to refactor this loop.

            # Config for this step from pipeline config file (if any)
            prim_config_from_pipe_config = self.config.primitives.get(prim_alias, {})
            prim_config_from_steps_dict = primitives.get(prim_alias, {})

            # Context for this step
            prim_context_from_pipe_config = prim_config_from_pipe_config.get('context', {})
            prim_context_from_steps_dict = prim_config_from_steps_dict.get('context', {})
            prim_context = pipe_context | context | prim_context_from_pipe_config | prim_context_from_steps_dict

            # Parameters for this step
            if 'parameters' in prim_config_from_pipe_config:
                prim_pars_from_pipe_config = prim_config_from_pipe_config['parameters']
            else:
                prim_pars_from_pipe_config = prim_config_from_pipe_config

            if 'parameters' in prim_config_from_steps_dict:
                prim_pars_from_steps_dict = prim_config_from_steps_dict['parameters']
            else:
                prim_pars_from_steps_dict = prim_config_from_steps_dict

            #prim_pars_from_pipe_config = prim_config_from_pipe_config
            #prim_pars_from_steps_dict = prim_config_from_steps_dict

            prim_pars = prim_pars_from_pipe_config | prim_pars_from_steps_dict
            prim_pars.pop('context', None)

            # Initialize the step
            primitive = prim_class(
                dataset=self.dataset,
                parent=self,
                context=prim_context,
                config=None,
                log_config=self.log_config,
                **prim_pars
            )

            # Set as attribute of the pipeline
            setattr(self, prim_alias, primitive)

    def init_config(
        self,
        dataset : Any = None,
        config : dict | None = None,
        context : dict | None = None
    ):
        if config is not None:
            if isinstance(config, str):
                self.config = ConfigFile.from_file(config)
            elif isinstance(config, dict):
                self.config = ConfigFile.from_dict(config)
            elif isinstance(config, ConfigFile):
                self.config = config
            else:
                raise TypeError("config must be a str (filename or filepath), dict, or ConfigFile instance.")
        else:
            if dataset is None:
                raise ValueError(
                    "A dataset must be passed to the Pipeline constructor if no config file is provided."
                )
            self.config = self.get_config(
                dataset, context=context
            )

    def init_parameters(self, **kwargs):
        """
        Initialize parameter values as top-level attributes.

        Parameters
        ----------
        **kwargs
            Any additional parameter values to override.
        """
        # Get all step parameter values from config file and input kwargs
        pars = self.PARAMETERS | self.config.parameters | kwargs

        # Set attributes
        allowed = self.allowed_parameters
        for k, v in pars.items():
            if k not in allowed:
                msg = f"Invalid parameter name '{k}' for {self.__class__.__name__}. Valid parameters are: {allowed}"
                logger.error(msg)
                raise NameError(msg)
            setattr(self, k, v)

    @classmethod
    def from_config_file(
        cls,
        config : ConfigFile | str,
        dataset : Any = None,
        log_config : dict | str | None = None,
        **kwargs
    ):
        """
        Initialize a pipeline from a configuration file.

        Parameters
        ----------
        dataset : Any, optional
            The dataset to process.
        config : str, dict, or ConfigFile, optional.
            The configuration for the pipeline. Can be one of:
                - str: A full path to a configuration file or a filename in the ConfigStore.
                - dict: A dictionary of configuration parameters.
                - ConfigFile: A ConfigFile instance.

            If not provided, the pipeline will attempt to auto-configure based on the input dataset and context.
            If the dataset is required for auto-configuration, it must be provided.
        log_config : dict, str, or None, optional
            The logging configuration to use for the pipeline.
        kwargs : dict
            Additional keyword arguments to pass to the pipeline constructor.

        Returns
        -------
        BasePipeline
            An instance of the pipeline with the configuration loaded.
        """

        # If ConfigFile object is passed, use it directly.
        # Otherwise, load the config file.
        if isinstance(config, str):
            config_filepath = get_config_filepath(config)
            config_file = ConfigFile.from_file(config_filepath)
        elif isinstance(config, dict):
            config_file = ConfigFile.from_dict(config)
        elif isinstance(config, ConfigFile):
            config_file = config
        else:
            raise ValueError("Input config must be a filepath (str) or a ConfigFile object")

        # If class is not provided, then use the current cls context
        pipe_class = config_file.class_
        if pipe_class is None:
            pipe_class = cls

        # Check the class is a valid Step or Pipeline class
        if not issubclass(pipe_class, cls):
            raise ValueError(f"Config file {config_file} specifies class {pipe_class}, which is not a subclass of {cls}")
        from ..primitives.base_primitive import BasePrimitive
        if pipe_class in (BasePrimitive, BasePipeline):
            raise ValueError(f"Config file {config_file} does not specify a class, and the current context is {cls}, which is not a valid class to initialize.")

        if issubclass(pipe_class, BasePrimitive):
            raise ValueError(f"Config file {config_file} specifies class {pipe_class}, which is a Primitive. Use BasePrimitive.from_config_file to initialize a Primitive.")

        return pipe_class(
            dataset=dataset,
            config=config_file,
            log_config=log_config,
            **kwargs
        )

    ##################
    #### MAIN API ####
    ##################

    def apply(
        self,
        dataset : Any | None = None,
        primitives: dict | None = None,
        **kwargs
    ):
        """
        The main method a user calls to run a pipeline.

        Parameters
        ----------
        dataset
            The dataset to process and is up to the subclass to decide what this represents, but is typically a datamodel, or simple containers of datamodels (or filepaths).
        primitives : dict | None
            A dictionary of primitive aliases and user-specified parameters to override the default primitive parameters for this pipeline.
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

                if dataset is not None:
                    self.dataset = dataset

                self.set_pars(**kwargs)

                # Set additional parameters from the primitives dict.
                if primitives is not None:
                    for prim_alias, prim_config in primitives.items():
                        if prim_alias not in self.PRIMITIVES:
                            raise ValueError(f"Invalid primitive alias '{prim_alias}' for pipeline {self.__class__.__name__}. Valid primitive aliases are: {list(self.PRIMITIVES.keys())}")
                        if 'parameters' in prim_config:
                            prim_pars = prim_config['parameters']
                        else:
                            prim_pars = prim_config
                        prim_pars.pop('context', None)
                        prim_instance = getattr(self, prim_alias)
                        prim_instance.set_pars(**prim_pars)

                for prim_alias in self.PRIMITIVES:
                    prim_instance = getattr(self, prim_alias)

                    # Forward output_dir to primitives.
                    if (
                        prim_instance.output_dir is None
                        and (prim_instance.save_result or prim_instance.output_file is not None)
                    ):
                        prim_instance.output_dir = self.output_dir

                    # Forward max_cores to primitives
                    if prim_instance.max_cores is None:
                        prim_instance.max_cores = self.max_cores
                    


                #input_dir = self.get_input_dir()
                #if self.save_result in (None, False) and self.output_dir is not None:
                #if self.save_result is None and self.output_dir is not None:
                #    self.save_result = True
                #if self.save_result and self.output_dir is None:
                #    self.output_dir = input_dir

                logger.info(f"Applying {self.__class__.__name__} with parameters:\n{self.pars_str}")

                self._output = self._perform()

                if self.save_result:
                    self.save_output() # Already finalized in the primitive's perform.
            except Exception as e:
                self._output = dict(output=input)
                self._status = "FAILED"
                logger.exception(
                    f"Failed executing pipeline {self.__class__.__name__}"
                )
                raise

        # Get main output
        output = self._output.get('output', None) if self._output is not None else None

        # Reset input
        self.dataset = None

        # Return main output
        return output

    @classmethod
    def call(
        cls,
        dataset : Any,
        config : dict | str | ConfigFile | None = None,
        context : dict | None = None,
        primitives : dict | None = None,
        return_instance : bool = False,
        **kwargs
    ):
        """
        Top level method to instantiate a Step or Pipeline and run it in one call.

        Parameters
        ----------
        dataset : object
            The dataset to process.
        context : object, optional
            The context in which the step is being called.
        config : object, optional
            The configuration to use for the step. Can be a dict, a config file path, or a ConfigFile object.
        parent : LigerIRISStep, optional
            The parent step or pipeline of this step. Used to determine a fully-qualified name for this step, and to determine the mode in which to run this step.
        return_instance : bool, optional
            If True, return a tuple of (result, step_instance).
            If False, return only the result.
        kwargs : dict
            Override step parameters.

        Returns
        -------
        results: Any
            The result of processing the input with the step.
        """
        pipe = cls(
            dataset,
            config=config,
            context=context,
            primitives=primitives,
            **kwargs
        )
        result = pipe.apply()
        if return_instance:
            return result, pipe
        else:
            return result

    ###################
    #### UTILITIES ####
    ###################

    def get_effective_config(self):
        config_file = deepcopy(self.config)
        config_file.parameters |= self.get_pars()
        for prim_alias, prim_class in self.PRIMITIVES.items():
            prim = getattr(self, prim_alias)
            prim_config = prim.get_effective_config()
            config_file.primitives[prim_alias] = {
                'class': prim_class.__name__,
                'name' : prim_config.name,
                'context': prim_config.context,
                'parameters': prim_config.parameters,
                'description': prim_config.description,
            }
        return config_file
    
    def __repr__(self):
        s = f"{self.__class__.__name__}:\n"
        s += "  Primitives:\n"
        for prim_alias, prim_class in self.PRIMITIVES.items():
            s += f"    {prim_alias}: {prim_class.__name__}\n"
        return s

    @property
    def default_suffix(self):
        """
        Return the default suffix for this pipeline, which is the lowercase class name.
        """
        return None