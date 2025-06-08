
import traceback
import datetime
import os
import functools

from keckdrpframework.primitives.base_primitive import BasePrimitive as _BasePrimitive
from keckdrpframework.models.arguments import Arguments

import hispecdrp
from ..calibrations import get_calibration_store, CalibrationSelector
from .. import datamodels
from .. import utils as utils
from ..utils.config import Config
from ..utils.logging import get_logger


__all__ = ['BasePrimitive']


class BasePrimitive(_BasePrimitive):

    class_alias = None
    
    ARGUMENTS = {
        'run' : True,
        'save_result' : False,
        'output_dir' : None,
        'suffix' : '',
        'full_output' : False,
    }

    LOGGING = {
        'enable' : True,
        'level' : 'info',
        'log_path' : None,
        'verbose' : True, # NOTE: not used yet, look at logger.py
    }

    def __init__(
            self,
            config_file : str | None = None,
            config : Config | None = None,
            **kwargs
        ):
        """
        Args:
            config (str | Config | None): The configuration file or Config to further specify parameters.
            **kwargs: Any additional keyword arguments to set or override in `self.args` for this primitive only.
        """

        # Load the configuration file or Config object
        self.init_config(config=config, config_file=config_file)

        # Set any kwargs passed to init
        self.set_args(**kwargs)

        # Init the output and status
        self.output = None
        self.status = None

    @classmethod
    def _get_config(cls, config : Config | None = None, config_file : str | None = None) -> Config:
        """
        Load a config file for this primitive or pipeline, or construct one from scratch.
        """
        if config_file is None:
            _config = Config(data={'class': cls.__module__ + '.' + cls.__name__, 'arguments': Config(), 'logging': cls.LOGGING.copy()})
        else:
            _config = Config(config_file=config_file)
            if 'logging' not in _config:
                _config['logging'] = cls.LOGGING.copy()
            else:
                _config['logging'] = cls.LOGGING | _config['logging']
            if 'arguments' not in _config:
                _config['arguments'] = {}
        if config is not None:
            _config.update(**config)

        return _config


    def init_config(self, config : Config | None = None, config_file : str | None = None) -> Config:
        """
        Load the config_file for this primitive or pipeline and any sub config_files.
        """
        self.config_file = config_file
        self.config = self._get_config(config=config, config_file=config_file)
        return self.config

    def _perform(self):
        raise NotImplementedError(f"instance method _perform not implemented for type {self.__class__.__name__}")

    def apply(self, name, **kwargs):
        self.init_logger()
        self.output = None
        try:
            # This allows pipelines to work whether full_output is True or False.
            # Pipelines can pass in a datamodel or an Arguments object, in which case the first arg is fetched.
            # NOTE: Consider alternative approach. A primitive should not have full_output=True when invoked by a pipeline.
            if isinstance(name, Arguments):
                name = name.name
            self.set_args(name=name, **kwargs)
            if not self._pre_condition():
                self.log.info(f"Skipping {self.__class__.__name__} with arguments:\n{self.prepr_args(self.args)}")
                self.on_skip()
            else:
                self.log.info(f"Applying {self.__class__.__name__} with arguments:\n{self.prepr_args(self.args)}")
                output = self._perform()
                if self._post_condition():
                    self.output = output
                    if isinstance(self.output.name, datamodels.DataModel):
                        self.finalize_model(self.output.name)
                        if self.args.save_result:
                            self.save_model(self.output.name)
                    elif isinstance(self.output.name, list):
                        for result in self.output.name:
                            if isinstance(result, datamodels.DataModel):
                                self.finalize_model(result)
                                if self.args.save_result:
                                    self.save_model(result)
                    elif isinstance(self.output, dict):
                        for key in self.output:
                            if isinstance(self.output[key], datamodels.DataModel):
                                self.finalize_model(self.output[key])
                                if self.args.save_result:
                                    self.save_model(self.output[key])
        except Exception as e:
            self.log.error(f"Failed executing primitive {self.__class__.__name__}: {e}\n{traceback.format_exc()}")
            if os.environ.get('HISPECDRP_DEBUG', '0') == '0':
                raise e
            return name
        if self.args.full_output:
            return self.output
        else:
            return self.output.name
    
    def init_logger(self):
        """
        Initialize the logger for this primitive. If logging is disabled, `self.logger` will be None.
        """
        self.logger = get_logger(self.__class__.__name__, self.config)
        return self.logger

    @property
    def log(self):
        if (hasattr(self, 'logger') and self.logger is None) \
            or not self.config.get('logging', self.LOGGING).get('enable', True):
            return self._fallback_log
        return self.logger

    @property
    def _fallback_log(self):
        # Use a lambda to dynamically handle log methods
        return type(
            "_FallbackLog",  # Anonymous type name
            (),  # No base class
            {
                method: functools.partial(lambda message: print(f"[{method.upper()} LOGGING DISABLED] {message}"))
                for method in ["warn", "info", "error", "debug"] # Log methods to handle
            },
        )()
                
    def set_args(self, **kwargs) -> dict:
        """
        Coalesces the user input into a single `Arguments` obejct with the following priorities:
            4. `self.ARGUMENTS`
            3. `self.config['arguments']`
            2. `self.args`
            1. `kwargs`
        
        And is stored in `self.args`.

        If `self.args` already exists, this updates `self.args` with the contents of `**kwargs`.
        `self.args.name` should be the name of the primary datamodel or filename to be processed.
        """
        if not hasattr(self, "args"):
            self.init_args()
        if 'name' in 'kwargs':
            self.args.name = kwargs.pop('name')
        for key in kwargs:
            if key in self.ARGUMENTS or key == 'name':
                self.args[key] = kwargs[key]
            else:
                raise ValueError(f"Invalid argument '{key}' for primitive {self.__class__.__name__}. Valid arguments are: {list(self.ARGUMENTS.keys())}")
        return self.args

    def init_args(self) -> dict:
        """
        Coalesces the user input into a single `Arguments` object from scratch with the following priorities:
            3. `self.ARGUMENTS`
            2. `self.config`
            1. `self.args`

        And is stored in `self.args`.

        Keyword Args:
            Any valid arguments for the primitive.
        """
        self.args = Arguments(name=None)
        self.args.update(**self.ARGUMENTS)
        if 'arguments' in self.config:
            self.args.update(**self.config['arguments'])
        return self.args
        
    def _pre_condition(self):
        return super()._pre_condition() and self.args.run
    
    def _post_condition(self):
        if super()._post_condition():
            self.status = "COMPLETE"
            return True
        else:
            self.status = "FAILED"
            return False
    
    def on_skip(self):
        self.status = "SKIPPED"

    def __repr__(self):
        if hasattr(self, "args"):
            s = self.__class__.__name__
            for key in self.args.iter_kw():
                s += f"\n  {key}: {self.args[key]}"
            return s
        else:
            return self.__class__.__name__
    
    def __init_subclass__(cls):
        ARGUMENTS = {}
        for _cls in cls.mro()[::-1]:
            if issubclass(_cls, BasePrimitive):
                ARGUMENTS.update(_cls.ARGUMENTS)
        cls.ARGUMENTS = ARGUMENTS

    def add_receipt_entry(self, model : datamodels.DataModel):
        """
        Add an entry to the receipt table of the model.
        """
        # add the row to the bottom of the table
        from ..pipelines import BasePipeline
        if not isinstance(self, BasePipeline):
            row = {
                'Time' : datetime.datetime.now().isoformat(),
                'HISPECDRP_Version': hispecdrp.__version__,
                'Class': f"{self.__class__.__module__}.{self.__class__.__name__}",
                'Filename' : model.meta.filename or 'None',
                'Arguments': repr(self.args),
                'Status' : self.status
            }
            model.receipt.add_row(row)
    
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

    def finalize_model(self, model_result : datamodels.DataModel):
        """
        Hook called at the end of `primitive.apply` to finalize the model.

        Args:
            model_result (DataModel): The model to finalize.
        """
        # NOTE: Dropping model.meta.drp_step.prim_alias = STATUS
        #   for now in favor of receipt until we decide the utility of tracking processing.
        self.add_receipt_entry(model_result)
        model_result.meta.drp_version = hispecdrp.__version__ # Always override the current DRP version.
        return model_result
    
    def get_calibration(self, name, selector : CalibrationSelector, use_cached : bool = True, **kwargs):
        cal_store = get_calibration_store()
        return cal_store.get_calibration(name, selector, use_cached=use_cached, **kwargs)
    
    def prepr_args(self, args : Arguments | None = None) -> str:
        if args is None:
            args = self.args
        s = f'name = {args.name}\n'
        for k in args.iter_kw():
            if k != 'name':
                s += f'{k} = {args[k]}\n'
        s = s[:-1]
        return s