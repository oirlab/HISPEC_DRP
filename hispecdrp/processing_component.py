from __future__ import annotations
import os
from keckdrpframework.primitives.base_primitive import BasePrimitive as _BasePrimitive

from .datamodels import DataModel
from . import datamodels
from .driver import load_config_file, get_config_dir

from typing import Any, Sequence

import logging
logger = logging.getLogger(__name__)

__all__ = ['ProcessingComponent']

class ProcessingComponent(_BasePrimitive):

    PARAMETERS = {
        'run' : True,
        'save_result': None,
        'output_model': 'input',
        'output_file' : None,
        'output_dir': None,
        'suffix': None,
        'save_preview' : False,
        'show_preview' : False,
    }

    LOG_CONFIG_FILE = "logging.yaml"

    ########################
    #### INITIALIZATION ####
    ########################

    def __init_subclass__(cls, **kwargs):
        # Ensure proper cooperative multiple inheritance behavior
        super().__init_subclass__(**kwargs)

        # Merge ARGUMENTS from the full (reversed) MRO so base defaults apply first
        merged_args = {}
        for _cls in cls.mro()[::-1]:
            if issubclass(_cls, ProcessingComponent):
                merged_args.update(getattr(_cls, 'PARAMETERS', {}))
        cls.PARAMETERS = merged_args

    def _init_log_config(self, log_config : dict | str | None = None) -> None:
        """
        Resolve and store ``self.log_config`` by merging (lowest → highest priority):

        1. Package defaults from ``hispecdrp/configs/logging.yaml``
        2. ``logging`` attribute embedded in ``self.config``.
        3. ``log_config`` kwarg passed at construction time (dict or file path/name)
        """

        logging_file = os.path.join(get_config_dir(), self.LOG_CONFIG_FILE)

        # 1. Package defaults
        try:
            defaults = load_config_file(logging_file)
        except FileNotFoundError:
            logger.warning(f"Logging config file not found: {logging_file}")
            defaults = {}

        # 2. Embedded logging block inside the run config
        config_log = self.config.logging

        # 3. Runtime kwarg
        if isinstance(log_config, str):
            #kwarg_log = read_config_file(get_config_path(log_config))
            kwarg_log = load_config_file(log_config)
        elif isinstance(log_config, dict):
            kwarg_log = log_config
        else:
            kwarg_log = {}

        self.log_config = defaults | config_log | kwarg_log

    ##################
    #### MAIN API ####
    ##################

    def apply(self, dataset : Any = None, **kwargs):
        """
        One of the two primary methods a user calls to run a primitive. The other is ``Primitive.call()``.
        Users should **not** need to override this method, and care should be taken if doing so.

        Both BasePrimitive and BasePipeline implement apply().
        """
        raise NotImplementedError(f"instance method apply not implemented for type {self.__class__.__name__}")

    def _perform(self):
        """
        **Required for all subclasses.**
        This method must be overridden to implement the main logic of the primitive.
        It should return a dictionary where the primary output is the first element.
        The return value of ``_perform`` must be one of the following:
        
            - ``dict`` where the key 'output' is the primary return value. The primary output can be a:
            - Single DataModel
            - List of DataModels
            - Dictionary of DataModels
        
        When returned to apply(), the output will always get stored at ``self._output``, and the main output will be at the key 'output'.
        If a ``dict``, additional keys can be included for other outputs. These are accessible at ``self._output['key_name']``.
        """
        raise NotImplementedError(f"instance method _perform not implemented for type {self.__class__.__name__}")

    def _create_preview(self):
        """
        Create a preview image for the output data model.
        This method can be overridden by subclasses to implement specific preview generation logic.
        By default, this method does nothing.
        """
        pass
    
    def _pre_condition(self):
        raise NotImplementedError(f"instance method _pre_condition not implemented for type {self.__class__.__name__}")
    
    def _post_condition(self):
        # NOTE: This is to block anyone from calling this method since we don't support it.
        raise NotImplementedError(f"instance method _post_condition not implemented for type {self.__class__.__name__}")

    #############################
    #### PARAMETER UTILITIES ####
    #############################

    @property
    def allowed_parameters(self):
        return list(self.PARAMETERS.keys()) + list(getattr(self, "CALIBRATIONS", {}).keys())

    def set_pars(self, **kwargs):
        """
        Set parameter values including calibrations as top-level attributes.

        Parameters
        ----------
        **kwargs
            Parameters to set as attributes.
            For primitives, one can also override calibration values.
        
        Notes
        -----
        Selector logic is not run here.
        """
        allowed = self.allowed_parameters
        for k, v in kwargs.items():
            try:
                if k not in allowed:
                    raise ValueError(
                        f"Invalid parameter name '{k}' for {self.__class__.__name__}. "
                        f"Valid arguments are: {list(allowed)}"
                    )
                setattr(self, k, v)
            except ValueError:
                logger.exception(f"Failed to set parameter '{k}'")

    def get_pars(self):
        """
        Get the current parameter values.

        Returns
        -------
        dict
            A dictionary of parameter names and their current values.
            Primitive parameters include both parameters and calibrations.
        """
        allowed = self.allowed_parameters
        pars = {}
        for k in allowed:
            pars[k] = getattr(self, k, None)
        return pars

    def pars_str(self) -> str:
        """
        Pretty print the current parameters and calibrations for this primitive or pipeline.
        
        Returns
        -------
        str
            A formatted string showing all current parameter values.
        """
        return "\n".join(f"{k} = {v}" for k, v in self.get_pars().items())

    #######################
    #### INPUT/OUTPUTS ####
    #######################

    def save_output(self) -> str | None:
        """
        Saves all datamodels stored at `self._output['output']`.
        """
        if self._output is None or 'output' not in self._output:
            logger.info('No output to save.')
            return None

        def _save(item):
            if isinstance(item, datamodels.DataModel):
                self.save_model(item)
            elif isinstance(item, dict):
                for val in item.values():
                    _save(val)
            elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                for val in item:
                    _save(val)
            else:
                logger.warning(f"Skipping unsupported output type: {type(item)}")

        _save(self._output['output'])

    def save_model(
        self,
        model : datamodels.DataModel,
        output_path : str | None = None,
    ) -> str:
        """
        Save a DataModel to a file in the context of this processing component.
        The file suffix will be added to the existing filename (``model.meta.filename``) if present and filepath is not set.

        Parameters
        ----------
        model : DataModel
            The model to save.
        output_path : str | None, optional
            The full filepath where to save the model.
            Supersedes ``self.output_dir`` and ``self.suffix``.

        Returns
        -------
        str
            The file path where the model was saved.
        """
        if output_path is None:
            output_path = self.generate_output_path(model)
        model.save(filepath=output_path)
        logger.info(f"Saved {model}")
        return output_path

    def generate_output_path(
        self,
        model : str | DataModel,
        output_dir : str | None = None,
    ):
        """
        Generate an output filepath for a model.

        Parameter
        ---------
        model : str | DataModel
            The model to generate the output path for.
        output_dir : str | None, optional
            The output directory to use. If None, the output directory is determined by ``self.get_output_dir(model)``.
        """
        
        # Get output directory
        if output_dir is None:
            output_dir = self.get_output_dir(model)

        # Get the filename
        if self.output_file is not None:
            filename = self.output_file
        else:
            filename = model.generate_filename(suffix=self.suffix)

        # Construct the full output path
        output_path = os.path.abspath(os.path.join(output_dir, filename))

        # Return the output path
        return output_path

    def get_output_dir(self, model : str | DataModel) -> str:
        """
        Get the output directory for saving any results in the context of a model.
        The priority for determining the output directory is as follows:

        1. ``self.output_dir`` if explicitly set (input model not used).
        2. ``model._filepath`` directory if available If an input model is provided, the fallback is the directory of self.dataset.
        3. Current working directory as a final fallback.

        Parameters
        ----------
        model : str | DataModel
            The input model to determine the output directory from if ``self.output_dir`` is not set.

        Returns
        -------
        str
            The output directory for saving results.
        """
        
        # If output_dir is explicitly set, use it regardless of the input model
        if self.output_dir is not None:
            return self.output_dir

        # Get the input directory for this model
        input_dir = self.get_input_dir(model)
        if input_dir is not None:
            return input_dir

        # Fallback is the current working directory
        return os.getcwd()

    def get_input_dir(self, model : str | DataModel | None = None):
        """
        Get the input directory for a given model.

        Parameters
        ----------
        model : str | DataModel | None
            The input model to determine the input directory from. If None, uses ``self.dataset``.
        """
        
        # Model contenxt
        if model is None:
            model = self.dataset

        # Helper function for below.
        def _get_model_filepath(model):
            _filepath = getattr(model, '_filepath', None)
            if _filepath is not None:
                return os.path.dirname(os.path.abspath(_filepath))
            else:
                return None
        
        # Case 1: Filepath string
        if isinstance(model, str) and os.path.exists(model):
            return os.path.dirname(os.path.abspath(model))
        
        # Case 2: DataModel instance with _filepath attr
        elif isinstance(model, datamodels.DataModel):
            return _get_model_filepath(model)
        
        # Case 3: Sequence
        elif isinstance(model, (list, tuple)):
            first_input = model[0]
            return _get_model_filepath(first_input)
        
        # Case 4: Dict
        elif isinstance(model, dict):
            first_input = next(iter(model.values()))
            return _get_model_filepath(first_input)
        
        return None

    #####################
    #### OTHER UTILS ####
    #####################

    def __repr__(self):
        s = self.__class__.__name__
        for key in self.allowed_parameters:
            s += f"\n  {key}: {getattr(self, key)}"
        return s
    
    @staticmethod
    def get_do_traces(
        input_model: datamodels.DataModel,
        do_fibers: list[str] | None = None,
        do_orders: list[int] | None = None,
    ) -> dict[str, list[int]]:
        """
        Determine which fibers and orders to process based on the input model and optional user-specified lists.

        Parameters
        ----------
        input_model : datamodels.DataModel
            The input data model.
        do_fibers : list[str] | None, optional
            A list of fiber numbers to process. If None, all fibers are processed.
        do_orders : list[int] | None, optional
            A list of order numbers to process. If None, all orders are processed.

        Returns
        -------
        do_fibers : list[str | int]
            The list of fiber numbers to process.
        do_orders : list[int]
            The list of order numbers to process.
        """

        if do_fibers is None:
            do_fibers = [str(f + 1) for f in range(input_model.num_fibers)]

        if do_orders is None:
            do_orders = input_model.make_order_vec()

        return do_fibers, do_orders

    def _iter_inputs(self, input=None):
        """
        Yields (key, item) pairs from self.input, normalized
        to a single item, list, or dict depending on the input.
        """
        if input is None:
            input = self.input
        if isinstance(input, (DataModel, str)):
            yield None, input
        elif isinstance(input, dict):
            yield from input.items()
        elif isinstance(input, Sequence):
            yield from enumerate(input)
        else:
            raise ValueError(f"Unsupported input type: {type(input)}")

    def _wrap_outputs(self, keyed_outputs: dict, input=None):
        """
        Reconstruct a keyed_outputs dict (built via _iter_inputs) into the original container shape.
        """
        if input is None:
            input = self.input
        if isinstance(input, (DataModel, str)):
            return keyed_outputs[None]
        elif isinstance(input, dict):
            return keyed_outputs
        else:  # Sequence
            return [keyed_outputs[i] for i in range(len(keyed_outputs))]