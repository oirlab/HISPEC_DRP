"""
base_primitive.py
Defines the base primitive class for HISPEC.
"""

import traceback

from keckdrpframework.primitives.base_primitive import BasePrimitive
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.action import Action
from keckdrpframework.models.event import Event
from keckdrpframework.models.processing_context import ProcessingContext
from keckdrpframework.config.framework_config import ConfigClass

import logging
logger = logging.getLogger("HISPEC")


class HISPECPrimitive(BasePrimitive):
    """
    Base primitive for HISPEC.

    Class Attributes:
        default_args (dict) : A python dictionary.

    Attributes:
        action (Action): The action.
        context (ProcessingContext): The processing context.
        args (Arguments): The current Arguments object that will be used in _perform.
    """

    default_args = {}

    def __init__(
            self,
            action : Action | None = None,
            context : ProcessingContext | None = None,
            **kwargs
        ):
        """
        Arguments:
            action (Action | None): A default action will be created if not provided.
            context (ProcessingContext | None): A default context will be created if not provided.
            **kwargs: Any keyword arguments to set in `self.args`. See documentation on [primitive parameters](url).
        """

        # Default action
        if action is None:
            name = self.__class__.__name__
            args = Arguments()
            event = Event(name, args)
            action = Action(event, (name, None, None), args)

        # Default context
        if context is None:
            config = ConfigClass()
            event_queue, event_queue_hi = None, None
            context = ProcessingContext(event_queue, event_queue_hi, logger, config)

        # Super init
        super().__init__(action, context)

        # Store logger at top level
        self.logger = context.logger

        # Set the default args
        self.set_args(**kwargs)

        # Default output
        self.output = None

    
    def _perform(self):
        raise NotImplementedError(f"instance method _perform not implemented for class {self.__class__.__name__}")


    def apply(self, **kwargs):
        try:
            self.set_args(**kwargs)
            if self._pre_condition():
                self.logger.info(f"Applying {self.__class__.__name__} with arguments:\n{self.args}:")
                output = self._perform()
                if self._post_condition():
                    self.output = output
                    return self.output
        except Exception as e:
            self.logger.error(f"Failed executing primitive {self.__class__.__name__}: {e}\n{traceback.format_exc()}")
            raise e
        return None


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
        self.args = Arguments()
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
    
    
    def __repr__(self):
        return f"{self.__class__.__name__}\n{self.args}"
    