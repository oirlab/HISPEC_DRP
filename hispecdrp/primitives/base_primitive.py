
import datetime
import os
import yaml

from keckdrpframework.models.arguments import Arguments

import hispecdrp
from koa_middleware import get_calibration_store, CalibrationSelector
from .. import datamodels
from .. import utils

from ..processing_component import ProcessingComponent


__all__ = ['BasePrimitive']


class BasePrimitive(ProcessingComponent):

    ARGUMENTS = {
        'run': True,
    }

    def __init__(
        self,
        config : dict | str | None = None,
        **kwargs
    ):
        """
        Args:
            config (dict | str | None): A configuration dictionary, or the path to to a configuration file.
            **kwargs: Any additional keyword arguments to set or override in `self.args` for this primitive only.
        """

        # Load the configuration file or Config object
        self.init_config(config)

        # Set any kwargs passed to init
        self.set_args(**kwargs)

        # Init the output and status
        self.output = None
        self.status = None

    def init_config(self, config : dict | str | None = None, validate_class : bool = False) -> dict:
        """
        Initialize the configuration for this primitive.

        Args:
            config (dict | str | None): The configuration object or path to the configuration file.
            validate_class (bool): If True, validates that the class name in the configuration matches the class name of this primitive.
        """
        class_name = self.__class__.__module__ + '.' + self.__class__.__name__
        if config is None:
            self.config = {
                'class': self.__class__.__module__ + '.' + self.__class__.__name__,
                'arguments': {},
                'logging': self.LOGGING.copy()
            }
        elif isinstance(config, dict):
            self.config = config
            if validate_class:
                if 'class' not in self.config:
                    raise KeyError(f"'class' not found in config.")
                else:
                    assert class_name == config['class'], f"Config has class {self.config['class']} but expected {class_name}."
        elif isinstance(config, str):
            config_filepath = utils.find_config_path(config)
            self.config = utils.read_config_file(config_filepath)
        else:
            raise TypeError(f"Invalid config type: {type(config)}. Expected dict or str.")
        if 'arguments' not in self.config:
            self.config['arguments'] = {}
        return self.config
    
    def _pre_condition(self):
        return self.args.run

    def apply(self, name, **kwargs):
        self.init_logger()
        self.output = None

        try:
            # This allows pipelines to directly pass the result from a primitive to the next primitive if full_output is True or False.
            # Pipelines can pass in a datamodel or an Arguments object, in which case the first arg is fetched.
            # NOTE: Consider alternative approach.
            if isinstance(name, Arguments):
                name = name.name
            self.set_args(name=name, **kwargs)
            if not self._pre_condition():
                self.log.info(f"Skipping {self.__class__.__name__} with arguments:\n{self.prepr_args(self.args)}")
                self.on_skip()
            else:
                self.log.info(f"Applying {self.__class__.__name__} with arguments:\n{self.prepr_args(self.args)}")
                output = self._perform()
                if self.status is None:
                    self.status = "COMPLETE"
                    self.output = output
                self.finalize_model(self.output.name)
                if self.args.save_result:
                    self.save_model(self.output.name)
        except Exception as e:
            self.log.error(f"Failed executing primitive {self.__class__.__name__}: {e}", exc_info=True)
            self.on_exception(e)
            self.finalize_model(self.output.name)
            if os.environ.get('HISPECDRP_DEBUG', '1') != '0':
                raise e
            return name
        
        if self.args.full_output:
            return self.output
        else:
            return self.output.name

    def set_args(self, **kwargs) -> dict:
        """
        If `self.args` already exists, this updates `self.args` with the contents of `**kwargs`.
        `self.args.name` should be the name of the primary datamodel or filename to be processed.
        """
        if not hasattr(self, 'args'):
            self.init_args()
        for key in kwargs:
            if key in self.ARGUMENTS or key == 'name':
                self.args[key] = kwargs[key]
            else:
                raise ValueError(f"Invalid argument '{key}' for primitive {self.__class__.__name__}. Valid arguments are: {list(self.ARGUMENTS.keys())}")
        return self.args
    
    def on_skip(self):
        """
        Hook called when the component is skipped.
        """
        self.status = "SKIPPED"

    def on_exception(self, exc : Exception):
        """
        Hook called when an exception occurs during processing.
        """
        self.status = "FAILED"

    def __repr__(self):
        if hasattr(self, "args"):
            s = self.__class__.__name__
            for key in self.args.iter_kw():
                s += f"\n  {key}: {self.args[key]}"
            return s
        else:
            return self.__class__.__name__
    
    def add_receipt_entry(self, model : datamodels.DataModel):
        """
        Add an entry to the receipt table of the model.
        """
        row = {
            'Time' : datetime.datetime.now().isoformat(),
            'HISPECDRP_Version': hispecdrp.__version__,
            'Class': f"{self.__class__.__module__}.{self.__class__.__name__}",
            'Filename' : model.meta.filename or 'None',
            'Arguments': repr(self.args),
            'Status' : self.status
        }
        model.receipt.add_row(row)

    def finalize_model(self, model_result : datamodels.DataModel):
        """
        Hook called at the end of `primitive.apply` to finalize the model. Only called if _post_condition() is True.

        Args:
            model_result (DataModel): The model to finalize.
        """
        model_result.meta.drp_version = hispecdrp.__version__ # Always override the current DRP version.
        self.add_receipt_entry(model_result)
        return model_result
    
    def get_calibration(self, name, selector : CalibrationSelector | None = None, use_cached : bool = True, **kwargs):
        store = get_calibration_store(name)
        if selector is None:
            selector = self.get_calibration_selector(name)
        return store.get_calibration(name, selector=selector, use_cached=use_cached, **kwargs)
    
    def get_output_model(self, input_model : datamodels.DataModel | None = None, model_suffix : str | None = None) -> datamodels.DataModel:
        """
        Utility method to determine the output model class and instantiate it, optionally from an existing input model. This should be called within _perform() but i is left to each primitive to use how it likes.

        This method checks the `self.args.output_model` attribute to determine the type of output model to create. If `output_model` is set to 'input', it returns the input model as the output. If it is set to 'input_copy', it creates a copy of the input model without its filepath and filename. Otherwise if `output_model` is a string, it attempts to create a new model type based on that string. If it is a type, it instantiates that type from the input model by calling `DataModel.from_datamodel(input_model)`.

        Args:
            input_model (DataModel | None): An existing model to base the output model on.
            model_suffix (str | None): A suffix to determine the output model type.

        Returns:
            DataModel: An instance of the output model type.
        """
        if hasattr(self.args, 'output_model') and self.args.output_model is not None:
            if self.args.output_model == 'input':
                return input_model
            if self.args.output_model == 'input_copy':
                output_model = input_model.copy()
                output_model._filepath = None
                output_model.meta.filename = None
                return output_model
            if isinstance(self.args.output_model, str):
                output_model = datamodels.model_type_from_name(self.args.output_model)
                output_model = output_model.from_datamodel(input_model)
                output_model._filepath = None
                output_model.meta.filename = None
                return output_model
            elif isinstance(self.args.output_model, type):
                return self.args.output_model.from_datamodel(input_model)
            else:
                raise TypeError(f"Invalid output_model type: {type(self.args.output_model)}")
        elif model_suffix is not None:
            instrument = input_model.meta.instrument.name
            output_model = datamodels.model_type_from_name(instrument.upper() + model_suffix)
            return output_model.from_datamodel(input_model)
        else:
            raise ValueError("Cannot determine output model type.")