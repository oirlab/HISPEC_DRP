from keckdrpframework.primitives.base_primitive import BasePrimitive as _BasePrimitive

from . import datamodels
from .utils.logging import get_logger

from typing import Sequence

__all__ = ['ProcessingComponent']

class ProcessingComponent(_BasePrimitive):

    ARGUMENTS = {
        'input': None,
        'save_result': False,
        'output_path': None,
        'output_dir': None,
        'suffix': None,
        'full_output': False,
    }
    
    LOGGING = {
        'level' : 'info',
        'log_path' : None,
    }

    def apply(self, *args, **kwargs):
        """
        The main method a user calls to run a primitive.
        Users should **not** need to override this method, and care should be taken if doing so.
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
    
    def init_logger(self):
        """
        Initialize the logger for this primitive and store at `self._logger`.

        Returns:
            logging.Logger: The initialized logger.
        """
        self._logger = get_logger(self.__class__.__name__, self.config)
        return self._logger

    @property
    def log(self):
        return self._logger
    
    def __init_subclass__(cls, **kwargs):
        # Ensure proper cooperative multiple inheritance behavior
        super().__init_subclass__(**kwargs)

        # Merge ARGUMENTS from the full (reversed) MRO so base defaults apply first
        merged_args = {}
        for _cls in cls.mro()[::-1]:
            merged_args.update(getattr(_cls, 'ARGUMENTS', {}) or {})
        cls.ARGUMENTS = merged_args

        # Merge CALIBRATIONS similarly; not all classes will define this.
        merged_cals = {}
        for _cls in cls.mro()[::-1]:
            merged_cals.update(getattr(_cls, 'CALIBRATIONS', {}) or {})
        cls.CALIBRATIONS = merged_cals

    def save_output(self) -> str:
        """
        Saves all datamodels stored at `self._output['output']`.
        """
        if self._output is None or 'output' not in self._output:
            self.log.info('No output to save.')
            return None

        # Only save main output
        output = self._output['output']

        # Single model
        if isinstance(output, datamodels.DataModel): # Assume it can't be a string
            self.save_model(output)
        # Dictionary of models
        elif isinstance(output, dict):
            for model in output.values():
                self.save_model(model)
        # List of models
        elif isinstance(output, Sequence):
            for model in output:
                self.save_model(model)
        else:
            self.log.error("Unsupported output type.")
            raise TypeError("Unsupported output type.")

    def save_model(
        self,
        model : datamodels.DataModel,
        filepath : str | None = None,
        output_dir : str | None = None,
        suffix : str | None = None
    ) -> str:
        """
        Save a DataModel to a file.
        The file suffix will be added to the existing filename (``model.meta.filename``) if present and filepath is not set.

        Args:
            model (DataModel): The model to save.
            filepath (str, optional): The full filepath where to save the model. Supersedes output_dir and suffix.
            output_dir (str, optional): The output directory.
            suffix (str, optional): The file suffix.

        Returns:
            str: The file path where the model was saved.
        """
        if filepath is None:
            filepath = self.output_path
        if output_dir is None:
            output_dir = self.output_dir
        if suffix is None:
            suffix = self.suffix
        model.save(filepath=filepath, output_dir=output_dir, suffix=suffix)
        self.log.info(f"Saved {model.__class__.__name__} to {model._filepath}")
        return filepath

    def init_args(self):
        """
        Initialize arguments and calibrations as top-level attributes with the following priorities:
            3. `self.ARGUMENTS` (defaults)
            2. `self.config['arguments']` (from config)
            1. Direct attribute setting (highest priority)
        """
        # Set defaults from ARGUMENTS (including input)
        for key, default_value in self.ARGUMENTS.items():
            setattr(self, key, default_value)
        
        # Override with config values
        for key, config_value in self.config['arguments'].items():
            if key in self.ARGUMENTS:
                setattr(self, key, config_value)
        
        # Initialize calibration attributes to None (will be set by user or remain None for defaults)
        if hasattr(self, 'CALIBRATIONS'):
            for cal_key in self.CALIBRATIONS.keys():
                if not hasattr(self, cal_key):
                    setattr(self, cal_key, None)

    def reset_args(self):
        """
        Reset all arguments to their default values by clearing the initialization flag 
        and calling init_args again.
        """
        if hasattr(self, '_args_initialized'):
            delattr(self, '_args_initialized')
        self.init_args()
        self._args_initialized = True

    def set_args(self, **kwargs) -> dict:
        """
        Set arguments and calibrations directly as top-level attributes on the instance.
        """
        # Initialize defaults from ARGUMENTS and CALIBRATIONS
        if not hasattr(self, '_args_initialized'):
            self.init_args()
            self._args_initialized = True
        
        # Set provided kwargs as top-level attributes
        for key, value in kwargs.items():
            if key in self.ARGUMENTS:
                setattr(self, key, value)
            elif hasattr(self, 'CALIBRATIONS') and key in self.CALIBRATIONS:
                setattr(self, key, value)

    def prepr_args(self) -> str:
        """
        Pretty print the current arguments and calibrations.
        
        Returns:
            str: A formatted string showing all current argument values.
        """
        s = ''
        for k in self.ARGUMENTS:
            if hasattr(self, k):
                s += f'{k} = {getattr(self, k)}\n'
        if hasattr(self, 'CALIBRATIONS'):
            for k in self.CALIBRATIONS:
                if hasattr(self, k):
                    s += f'{k} = {getattr(self, k)}\n'
        s = s[:-1] if s.endswith('\n') else s
        return s
    
    @property
    def arguments(self):
        """
        Pretty print the current arguments and calibrations.
        
        Returns:
            str: A formatted string showing all current argument values.
        """
        return print(self.prepr_args())