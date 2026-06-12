from __future__ import annotations
import os
from keckdrpframework.primitives.base_primitive import BasePrimitive as _BasePrimitive

from .datamodels import DataModel
from . import datamodels
from .utils.config import get_config_path, read_config_file

from typing import Sequence

import logging
logger = logging.getLogger(__name__)

__all__ = ['ProcessingComponent']

class ProcessingComponent(_BasePrimitive):

    ARGUMENTS = {
        'input': None,
        'run' : True,
        'save_result': None,
        'output_model': 'input',
        'output_file' : None,
        'output_dir': None,
        'suffix': None,
        'save_preview' : False,
        'show_preview' : False,
    }

    def _init_log_config(self, log_config=None) -> None:
        """
        Resolve and store ``self.log_config`` by merging (lowest → highest priority):

        1. Package defaults from ``hispecdrp/configs/logging.yaml``
        2. ``logging`` key embedded in ``self.config`` (removed from ``self.config`` after reading)
        3. ``log_config`` kwarg passed at construction time (dict or file path/name)
        """

        from .pipelines.base_pipeline import BasePipeline
        if isinstance(self, BasePipeline):
            logging_file = "logging_pipeline.yaml"
        else:
            logging_file = "logging_primitive.yaml"

        # 1. Package defaults
        try:
            defaults = read_config_file(get_config_path(logging_file))
        except FileNotFoundError:
            defaults = {}

        # 2. Embedded logging block inside the run config (mutates self.config)
        config_log = self.config.pop("logging", {}) or {}

        # 3. Runtime kwarg
        if isinstance(log_config, str):
            kwarg_log = read_config_file(get_config_path(log_config))
        elif isinstance(log_config, dict):
            kwarg_log = log_config
        else:
            kwarg_log = {}

        self.log_config = defaults | config_log | kwarg_log

    def apply(self, *args, **kwargs):
        """
        The main method a user calls to run a primitive.
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
    
    def _pre_condition(self):
        raise NotImplementedError(f"instance method _pre_condition not implemented for type {self.__class__.__name__}")
    
    def _post_condition(self):
        # NOTE: This is to block anyone from calling this method since we don't support it.
        raise NotImplementedError(f"instance method _post_condition not implemented for type {self.__class__.__name__}")

    def __init_subclass__(cls, **kwargs):
        # Ensure proper cooperative multiple inheritance behavior
        super().__init_subclass__(**kwargs)

        # Merge ARGUMENTS from the full (reversed) MRO so base defaults apply first
        merged_args = {}
        for _cls in cls.mro()[::-1]:
            if issubclass(_cls, ProcessingComponent):
                merged_args.update(getattr(_cls, 'ARGUMENTS'))
        cls.ARGUMENTS = merged_args

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
        model : 'DataModel',
        filepath : str | None = None,
    ) -> str:
        """
        Save a DataModel to a file in the context of this processing component.
        The file suffix will be added to the existing filename (``model.meta.filename``) if present and filepath is not set.

        Parameters
        ----------
        model : DataModel
            The model to save.
        filepath : str | None, optional
            The full filepath where to save the model.
            Supersedes ``self.output_dir`` and ``self.suffix``.

        Returns
        -------
        str
            The file path where the model was saved.
        """
        output_path = self.generate_output_path(model)
        model.save(filepath=output_path)
        logger.info(f"Saved {model}")
        return filepath

    def get_args(self) -> dict:
        """
        Get the current arguments as a dictionary.

        Returns
        -------
        dict
            A dictionary of the current argument values.
        """
        args = {}
        for k in self.ARGUMENTS:
            args[k] = getattr(self, k, self.ARGUMENTS[k])
        for k in getattr(self, "CALIBRATIONS", {}):
            args[k] = getattr(self, k, self.CALIBRATIONS[k])
        return args

    def set_args(self, **kwargs):
        """
        Set arguments including calibrations as top-level attributes.

        Parameters
        ----------
        **kwargs
            Arguments to set as attributes.
            To override a calibration, provide the calibration name as a keyword argument.
            Selector logic is not handled here.
        """

        allowed = set(self.ARGUMENTS.keys()) | set(getattr(self, "CALIBRATIONS", {}).keys())

        for k, v in kwargs.items():
            if k not in allowed:
                logger.error(
                    f"Invalid argument name '{k}' for {self.__class__.__name__}. "
                    f"Valid arguments are: {list(allowed)}"
                )
            setattr(self, k, v)

    def _prepr_args(self) -> str:
        """
        Pretty print the current arguments and calibrations for this object only.
        
        Returns
        -------
        str
            A formatted string showing all current argument values.
        """
        args = self.get_args()
        keys = list(self.ARGUMENTS) + list(getattr(self, "CALIBRATIONS", {}))
        return "\n".join(f"{k} = {args[k]}" for k in keys)

    @staticmethod
    def get_do_traces(
        input_model: datamodels.DataModel,
        do_fibers: list[str] | None = None,
        do_orders: list[int] | None = None,
    ) -> dict[str, list[int]]:

        if do_fibers is None:
            do_fibers = [str(f + 1) for f in range(input_model.num_fibers)]

        if do_orders is None:
            do_orders = input_model.make_order_vec()

        return do_fibers, do_orders

    def _create_preview(self):
        """
        Create a preview image for the output data model.
        This method can be overridden by subclasses to implement specific preview generation logic.
        By default, this method does nothing.
        """
        pass

    def generate_output_path(
        self,
        model : str | DataModel,
        output_dir : str | None = None,
    ):
        """
        Generate an output filepath for a model.
        """

        # Open model
        #model = datamodels.open(model, meta_only=True)

        # If the model has a filepath, use it
        #if model._filepath is not None:
        #    return model._filepath
        
        # Get output directory
        output_dir = self.get_output_dir(model)

        # Get the filename
        if self.output_file is not None:
            filename = self.output_file
        else:
            filename = model.generate_filename(suffix=self.suffix)
        output_path = os.path.abspath(os.path.join(output_dir, filename))
        return output_path

    def get_output_dir(
        self,
        model : str | DataModel
    ) -> str:
        """
        Get the output directory for saving any results in the context of a model.
        The priority for determining the output directory is as follows:
            1. ``self.output_dir`` if explicitly set (input model not used).
            2. ``model._filepath`` directory if available.
            3. Current working directory as a fallback.



        - If ``self.output_dir`` is set, it is returned and the input model is not used.
        - Otherwise, the output dir is determined from the input model's filepath if available, and if not, the current working directory is used.

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

    def get_input_dir(self, model=None):
        """
        Get the input directory for a given model.
        """
        
        # Model contenxt
        if model is None:
            model = self.input

        # Helper function for below.
        def _get_filepath(model):
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
            return _get_filepath(model)
        
        # Case 3: Sequence
        elif isinstance(model, (list, tuple)):
            first_input = model[0]
            return _get_filepath(first_input)
        
        # Case 4: Dict
        elif isinstance(model, dict):
            first_input = next(iter(model.values()))
            return _get_filepath(first_input)
        
        return None
    
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