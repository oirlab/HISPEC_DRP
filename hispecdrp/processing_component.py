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
        'output_path': None,
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

        # 1. Package defaults
        try:
            defaults = read_config_file(get_config_path("logging_pipeline.yaml"))
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
        output_dir : str | None = None,
        suffix : str | None = None
    ) -> str:
        """
        Save a DataModel to a file.
        The file suffix will be added to the existing filename (``model.meta.filename``) if present and filepath is not set.

        Parameters
        ----------
        model : DataModel
            The model to save.
        filepath : str | None, optional
            The full filepath where to save the model. Supersedes output_dir and suffix.
        output_dir : str | None, optional
            The output directory.
        suffix : str | None, optional
            The file suffix.

        Returns
        -------
        str
            The file path where the model was saved.
        """
        if filepath is None:
            filepath = self.output_path
        if output_dir is None:
            output_dir = self.output_dir
        if suffix is None:
            suffix = self.suffix
        model.save(filepath=filepath, output_dir=output_dir, suffix=suffix)
        logger.info(f"Saved {model.__class__.__name__} to {model._filepath}")
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

    def get_primary_input_dir(self, input=None):
        
        if input is None:
            input = self.input
        
        # Case 1: String path
        if isinstance(input, str) and os.path.exists(input):
            return os.path.dirname(os.path.abspath(input))
        
        # Case 2: DataModel (or similar object with _filepath)
        elif hasattr(input, '_filepath') and input._filepath:
            return os.path.dirname(os.path.abspath(input._filepath))
        
        # Case 3: Dict container
        elif isinstance(input, dict):
            for val in input.values():
                res = self.get_primary_input_dir(val)
                if res:
                    return res
        
        # Case 4: List/Tuple container
        elif isinstance(input, (list, tuple)):
            for val in input:
                res = self.get_primary_input_dir(val)
                if res:
                    return res
        
        return None