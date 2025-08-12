from keckdrpframework.primitives.base_primitive import BasePrimitive as _BasePrimitive

from . import datamodels
from .utils.logging import get_logger

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
        raise NotImplementedError(f"instance method apply not implemented for type {self.__class__.__name__}")

    def _perform(self):
        raise NotImplementedError(f"instance method _perform not implemented for type {self.__class__.__name__}")
    
    def _pre_condition(self):
        raise NotImplementedError(f"instance method _pre_condition not implemented for type {self.__class__.__name__}")
    
    def _post_condition(self):
        # NOTE: NOT IMPLEMENTED ANYWHERE. Raise error if user calls _post_condition.
        raise NotImplementedError(f"instance method _post_condition not implemented for type {self.__class__.__name__}")
    
    def init_logger(self):
        self._logger = get_logger(self.__class__.__name__, self.config)
        return self._logger

    @property
    def log(self):
        return self._logger
    
    @property
    def arguments(self) -> str:
        """
        Pretty print the current arguments and calibrations.
        
        Returns:
            str: A formatted string showing all current argument values.
        """
        return print(self.prepr_args())
    
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

    def prepr_args(self) -> str:
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
    
    def init_args(self) -> None:
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
    
    def save_output(self) -> str:
        """
        Save the result from _perform after processing and finalize_output are called.

        Args:
            model (output): The output to save.
        """
        assert self._output is not None and 'output' in self._output, 'No output to save.'
        output = self._output['output']
        instance_output_path = getattr(self, 'output_path', None)
        instance_output_dir = getattr(self, 'output_dir', None)
        instance_suffix = getattr(self, 'suffix', None)
        if isinstance(output, datamodels.DataModel): # Assume it can't be a string
            output.save(
                filepath=instance_output_path,
                output_dir=instance_output_dir,
                suffix=instance_suffix   
            )
        elif isinstance(output, dict):
            for model in output.values():
                model.save(
                    filepath=instance_output_path,
                    output_dir=instance_output_dir,
                    suffix=instance_suffix
                )
        elif isinstance(output, list):
            for model in output:
                model.save(
                    filepath=instance_output_path,
                    output_dir=instance_output_dir,
                    suffix=instance_suffix
                )
        else:
            self.log.error("Unsupported output type.")
            raise TypeError("Unsupported output type.")

    def reset_args(self):
        """
        Reset all arguments to their default values by clearing the initialization flag 
        and calling init_args again.
        """
        if hasattr(self, '_args_initialized'):
            delattr(self, '_args_initialized')
        self.init_args()
        self._args_initialized = True