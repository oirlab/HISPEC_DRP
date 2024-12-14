
import traceback
import logging

from keckdrpframework.primitives.base_primitive import BasePrimitive
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.action import Action
from keckdrpframework.models.event import Event
from keckdrpframework.models.processing_context import ProcessingContext
from keckdrpframework.config.framework_config import ConfigClass

from ..utils.logging import get_logger


class HISPECPrimitive(BasePrimitive):
    
    default_args = {
        "skip" : False,
    }

    def __init__(
            self,
            config_file : str | None = None,
            **kwargs
        ):
        """
        Arguments:
            config_file (str | None): The configuration file to further specify default parameters.
            **kwargs: Any additional keyword arguments to set or override in `self.args`.
        """

        # Default config

        # Get a default Context and action
        super().__init__(action=None, context=None)

        # Logger
        self.logger = get_logger(self)

        # Set the default args
        self.set_args(**kwargs)

        # Initialize output to None
        self.output = None

    @classmethod
    def from_action_context(
            cls,
            action : Action | None = None,
            context : ProcessingContext | None = None,
        ):
        # This will interface with the Keck framework specific pipeline
        # context = ProcessingContext(event_queue, event_queue_hi, logger, config)
        raise NotImplementedError(f"class method from_action_context not implemented for {cls.__name__}")
    
    def _perform(self):
        raise NotImplementedError(f"instance method _perform not implemented for type {self.__class__.__name__}")


    def apply(self, **kwargs):
        try:
            self.set_args(**kwargs)
            self.logger = self.get_logger(self.__class__.__name__, self.args)
            if not self._pre_condition():
                self.logger.info(f"Skipping {self.__class__.__name__} with arguments:\n{self.args}:")
            else:
                self.logger.info(f"Applying {self.__class__.__name__} with arguments:\n{self.args}:")
                output = self._perform()
                if self._post_condition():
                    self.output = output
                    return self.output
        except Exception as e:
            self.logger.error(f"Failed executing primitive {self.__class__.__name__}: {e}\n{traceback.format_exc()}")
            raise e
        return None
    
    def default_context(self, logger : logging.Logger | None = None):
        self.context = ProcessingContext(None, None, logger)
        return self.context


    def set_args(self, **kwargs) -> Arguments:
        """
        Coalesces the user input into a single `Arguments` object with the following priorities:
            4. `self.default_args`
            3. `self.context.config`
            2. `self.action.args`
            1. `kwargs`
        And is stored in `self.args`.
        If `self.args` already exists, this updates `self.args` with the contents of `**kwargs`.
        """
        if not hasattr(self, "args") or not isinstance(self.args, Arguments):
            self.reset_args(**kwargs)
        else:
            self.args.update(**kwargs)
        return self.args

    def reset_args(self, **kwargs) -> Arguments:
        """
        Coalesces the user input into a single `Arguments` object from scratch with the following priorities:
            4. `self.default_args`
            3. `self.context.config`
            2. `self.action.args`
            1. `kwargs`
        And is stored in `self.args`.
        """
        self.args = Arguments(name=kwargs.get('name', None))
        self.args.update(**self.default_args)
        if self.context.config.has_section("ARGUMENT"):
            self.args.update(**self.context.config["ARGUMENT"])
        self.args.update(**kwargs)
        return self.args
    

    def __call__(self, **kwargs):
        """
        Wrapper for `self.apply`. Any valid arguments can be passed through `kwargs`.
        """
        return self.apply(**kwargs)
    

    def _pre_condition(self):
        return not super()._pre_condition() or not self.args.skip
    

    def __repr__(self):
        if hasattr(self, "args"):
            return f"{self.__class__.__name__} with Arguments: {self.args}"
        else:
            return self.__class__.__name__


    def hasarg(self, key: str):
        return hasattr(self.args, key)


    def get_calibration_file(self, kind : str, **kwargs):
        """
        This will implement the function 
        """
        raise NotImplementedError(f"instance method get_calibration_file not implemented for class {self.__class__.__name__}")
    

    def __init_subclass__(cls):
        subclasses = cls.mro()[::-1]
        for _cls in subclasses:
            if hasattr(_cls, "default_args"):
                cls.default_args.update(_cls.default_args)