from keckdrpframework.primitives.base_primitive import BasePrimitive as _BasePrimitive
from keckdrpframework.models.arguments import Arguments

from . import datamodels
from .calibrations import DATAMODEL_ORM_MAP
from .utils.logging import get_logger

__all__ = ['ProcessingComponent']

class ProcessingComponent(_BasePrimitive):

    ARGUMENTS = {
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
    
    def __init_subclass__(cls):
        ARGUMENTS = {}
        for _cls in cls.mro()[::-1]:
            if issubclass(_cls, ProcessingComponent):
                ARGUMENTS.update(_cls.ARGUMENTS)
        cls.ARGUMENTS = ARGUMENTS

    def prepr_args(self, args : Arguments | None = None) -> str:
        if args is None:
            args = self.args
        s = f'name = {args.name}\n'
        for k in args.iter_kw():
            if k != 'name':
                s += f'{k} = {args[k]}\n'
        s = s[:-1]
        return s
    
    def init_args(self) -> dict:
        """
        Coalesces the user input into a single `Arguments` object from scratch with the following priorities:
            3. `self.ARGUMENTS`
            2. `self.config`
            1. `self.args`

        And is stored in `self.args`.
        """
        self.args = Arguments(name=None)
        self.args.update(**self.ARGUMENTS.copy())
        self.args.update(**self.config['arguments'])
        return self.args
    
    def save_model(self, model: datamodels.DataModel) -> str:
        """
        Save a datamodel to a file.

        Args:
            model (DataModel): The model to save.
        """
        instance_output_path = getattr(self.args, 'output_path', None)
        instance_output_dir = getattr(self.args, 'output_dir', None)
        instance_suffix = getattr(self.args, 'suffix', None)
        return model.save(
            filepath=instance_output_path,
            output_dir=instance_output_dir,
            suffix=instance_suffix   
        )
    
    @classmethod
    def get_orm_class(cls, input_model : str | datamodels.DataModel) -> type:
        """
        Get the ORM class associated with the input model type from the provided ORM register.
        """
        # NOTE: Consider alternative approaches to link datamodel with ORM class.
        model_cls = type(input_model)
        for base_cls, orm_cls in DATAMODEL_ORM_MAP.items():
            if issubclass(model_cls, base_cls):
                return orm_cls
        raise ValueError(f"No ORM class registered for model type {model_cls}")