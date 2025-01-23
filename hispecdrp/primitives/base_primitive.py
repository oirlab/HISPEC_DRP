
import traceback
from typing import Self
from collections.abc import Sequence
import datetime
import git
import os
import functools

from keckdrpframework.primitives.base_primitive import BasePrimitive as _BasePrimitive
from keckdrpframework.models.arguments import Arguments

#from keckdrpframework.models.action import Action as _Action
#from keckdrpframework.models.event import Event as _Event
#from keckdrpframework.models.processing_context import ProcessingContext as _ProcessingContext
#from keckdrpframework.config.framework_config import ConfigClass as _ConfigClass

from .. import datamodels
from ..datamodels import DataModel
from .. import utils as utils
from ..utils.config import Config
from ..utils.logging import get_logger
from ..utils.utils import prepr_args


__all__ = ['BasePrimitive']


class BasePrimitive(_BasePrimitive):

    class_alias = None
    
    ARGUMENTS = {
        'skip' : False,
        'save_results' : True,
        'output_dir' : None,
        'suffix' : ''
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
        # NOTE HERE: Finalize how args are set
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
            if 'class' not in _config:
                raise ValueError(f"'class' entry not found in config {config_file}")
            _class = utils.resolve_primitive_class(_config['class'])
            if _class is not cls:
                raise ValueError(f"Config 'class' '{_class}' from {config_file} not consistent with instantiated primitive {cls.__name__}")
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


    def apply(self, name : DataModel, **kwargs):
        self.init_logger()
        self.output = None
        try:
            self.set_args(name, **kwargs)
            if not self._pre_condition():
                self.log.info(f"Skipping {self.__class__.__name__} with arguments:\n{prepr_args(self.args)}")
                self.on_skip()
            else:
                self.log.info(f"Applying {self.__class__.__name__} with arguments:\n{prepr_args(self.args)}")
                output = self._perform()
                if self._post_condition():
                    self.finalize_model(output)
                    self.output = output
                    if self.args.save_results:
                        if isinstance(self.output, Sequence):
                            for out in self.output:
                                self.save_model(out)
                        else:
                            self.save_model(self.output)
        except Exception as e:
            self.log.error(f"Failed executing primitive {self.__class__.__name__}: {e}\n{traceback.format_exc()}")
            raise e
        return self.output
    

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
        
    def set_args(self, name : str | DataModel | None = None, **kwargs) -> dict:
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
        if name is not None:
            self.args.name = name
        #self.args.update(**self.config['arguments'])
        for key in kwargs: # Only want to pass valid kwargs to self.args, don't call dict update()
            if key in self.ARGUMENTS:
                self.args[key] = kwargs[key]
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
    

    def __call__(self, name : DataModel, **kwargs):
        """
        Wrapper for `self.apply`. Any valid arguments can be passed through `kwargs`.
        """
        return self.apply(name, **kwargs)
    

    def _pre_condition(self):
        return not super()._pre_condition() or not self.args.skip
    

    def _post_condition(self):
        if super()._post_condition():
            self.status = "COMPLETE"
        return True
    

    def on_skip(self):
        self.status = "SKIPPED"
    

    def __repr__(self):
        if hasattr(self, "args"):
            s = self.__class__.__name__
            # for k, arg in enumerate(self.args):
            #     s += f"\n  {k}: {arg}"
            for key in self.args.iter_kw():
                s += f"\n  {key}: {self.args[key]}"
            return s
        else:
            return self.__class__.__name__


    def hasarg(self, key: str):
        return hasattr(self.args, key)
    

    def __init_subclass__(cls):
        ARGUMENTS = {}
        for _cls in cls.mro()[::-1]:
            if issubclass(_cls, BasePrimitive):
                ARGUMENTS.update(_cls.ARGUMENTS)
        cls.ARGUMENTS = ARGUMENTS


    def add_receipt_entry(self, model : DataModel):
        """
        Taken from KPF DRP and modified:
            https://github.com/Keck-DataReductionPipelines/KPF-Pipeline/blob/master/kpfpipe/models/base_model.py
        """

        # TODO: Get accurate GitHub info
        #repo = git.Repo(search_parent_directories=True)
        #git_commit_hash = repo.head.object.hexsha
        #git_branch = repo.active_branch.name
        #git_tag = str(repo.tags[-1])
        repo = 'None'
        git_commit_hash = 'None'
        git_branch = 'None'
        git_tag = 'None'

        # add the row to the bottom of the table
        row = {
            'Time' : datetime.datetime.now().isoformat(),
            'Code_Release': git_tag,
            'Branch_Name': git_branch,
            'Commit_Hash': git_commit_hash,
            'Class': f"{self.__class__.__module__}.{self.__class__.__name__}",
            'Filename' : model.filename,
            'Arguments': repr(self.args),
            'Status' : self.status
        }
        model.receipt.loc[len(model.receipt)] = row

    
    def make_output_path(self, model : DataModel, suffix : str = None):
        """
        Args:
            model (DataModel): The model to generate the output path for.
            suffix (str): The suffix to append to the output file. Defaults to `self.args.suffix`.
        """
        if suffix is None:
            suffix = self.args.suffix
        return os.path.join(self.args.output_dir, model.generate_filename(suffix=suffix))
    

    def save_model(
            self,
            model: DataModel,
            filename: str | None = None,
            output_path: str | None = None,
            suffix: str = None,
            output_dir: str | None = None,
            update_receipt: bool = True
        ):
        """
        Save a model to a file. The output_path is determined as follows:
        1. If output_path is input to this method, use that.
        2. If self.output_path is provided, use that.
        3. If output_dir is provided, use that for the directory.
        4. If self.output_dir is provided, use that for the directory.
        5. If filename is provided, use that for the filename.
        6. Generate filename from model.filename and either self.suffix or suffix passed in.

        Args:
            model (DataModel): The model to save.
            filename (str, optional): The filename to save the model to.
            output_path (str, optional): The full path where the model should be saved.
            suffix (str, optional): The suffix to append to the output file. Defaults to self.args.suffix.
            output_dir (str, optional): The directory to save the file in.
            update_receipt (bool, optional): Whether to update the receipt. Defaults to True.
        """
        # Check output_path priority
        if output_path is not None:
            final_path = output_path
        else:
            # Check instance output_path
            instance_output_path = getattr(self.args, 'output_path', None)
            if instance_output_path is not None:
                final_path = instance_output_path
            else:
                # Determine the directory
                if output_dir is None:
                    output_dir = getattr(self, 'output_dir', None)
                
                # Determine the filename
                if filename is not None:
                    final_filename = filename
                else:
                    # Use provided suffix or fall back to self.args.suffix
                    current_suffix = suffix or getattr(self.args, 'suffix', None)
                    final_filename = self.make_output_path(model, suffix=current_suffix)
                
                # Combine directory and filename if directory exists
                if output_dir is not None:
                    final_path = os.path.join(output_dir, os.path.basename(final_filename))
                else:
                    final_path = final_filename

        # Update the receipt
        if update_receipt:
            self.add_receipt_entry(model)
        
        # Save the model
        model.save(filename=final_path)

    def open_model(self, name : DataModel, _copy=True):
        """
        Open a model from a file or copy a datamodel.
        """
        if isinstance(name, DataModel):
            if _copy:
                return name.copy()
            else:
                return name
        if isinstance(name, str):
            return datamodels.open(name)
        else:
            raise ValueError(f"Cannot open model from {name}")
        
    
    def finalize_model(self, model_result : DataModel):
        """
        Hook called at the end of `pipeline.apply` to finalize the model.

        Args:
            model_result (DataModel): The model to finalize.
        """
        try:
            setattr(model_result.meta.drp_step, self.class_alias, str(self.status))
        except:
            pass
        return model_result


    # @classmethod
    # def from_keck_pipeline(cls, keck_pipeline):
    #     """
    #     Create a new instance of this primitive from an action and context.
    #     """
    #     name = action.args.name
    #     self = cls(config=context.config, )
    #     self._action = action
    #     self._context = context
    #     return self