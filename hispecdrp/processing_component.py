from keckdrpframework.primitives.base_primitive import BasePrimitive as _BasePrimitive

from . import datamodels
from .utils.logging import get_logger

from typing import Sequence, Any

import os

__all__ = ['ProcessingComponent']

class ProcessingComponent(_BasePrimitive):

    ARGUMENTS = {
        'input': None,
        'save_result': False,
        'output_path': None,
        'output_dir': None,
        'suffix': None,
        'full_output': False,
        'output_model' : None,
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
    
    @staticmethod
    def get_do_traces(input_model : datamodels.DataModel, do_fibers : list[str] | None = None, do_orders : list[int] | None = None) -> tuple[list[str], list[int]]:
        """
        Helper to get the default fibers and orders to process. By default, all fibers and orders in the input model are processed if not provided.
        This method may get updated depending on where the DRP determines which fibers to process.

        Args:
            input_model (datamodels.DataModel): The input data model.
            do_fibers (list[str], optional): List of fiber numbers to process. If None, process all fibers.
            do_orders (list[int], optional): List of order numbers to process. If None, process all orders.

        Returns:
            tuple[list[str], list[int]]: A tuple of (do_fibers, do_orders).
        """
        if do_fibers is None:
            do_fibers = [str(f + 1) for f in range(input_model.num_fibers)]
        if do_orders is None:
            do_orders = input_model.make_order_vec()
        return do_fibers, do_orders
    
    def get_output_model(
        self,
        input : Any,
        output_model : str | datamodels.DataModel | type[datamodels.DataModel] = None,
        **kwargs
    ) -> datamodels.DataModel:
        """
        Get the output model based on the input model and the specified output model type. See :py:meth:`_get_output_model <hispecdrp.primitives.base_primitive.BasePrimitive._get_output_model>` for details.
        """
        if output_model is None:
            output_model = getattr(self, 'output_model', None)
        return self._get_output_model(input, output_model, **kwargs)
    
    @staticmethod
    def _get_output_model(
        input : Any,
        output_model : str | datamodels.DataModel | type[datamodels.DataModel] = None,
        **kwargs
        ) -> datamodels.DataModel:
        """
        Get the output model based on the input model and the specified output model type.

        Args:
            input_model (datamodels.DataModel, optional): The input data model to base the output on.
            output_model (str | datamodels.DataModel | type[datamodels.DataModel], optional): The desired output model type or specification. It can take on several values:
                - DataModel instance (datamodels.DataModel): Returns `output_model` as is.
                - 'input' (str): Returns `input_model` as is.
                - 'input_copy' (str): Returns a copy of `input_model`.
                - Valid filepath (str): Opens and returns the model from the specified file.
                - DataModel class (type[DataModel]): Instantiates and returns a new instance of the specified DataModel class using `input_model` as a template with ``output_model.from_datamodels(input_model)``. If input_model.meta.instrument_name is set, it will attempt to use an instrument-specific subclass if available.
                - DataModel class name for specific instrument or instrument agnostic class (str): Instantiates and returns a new instance of the specified DataModel class with ``output_model.from_datamodels(input_model)``. If input_model.meta.instrument_name is set, it will attempt to use an instrument-specific subclass if available.
                - None (default value): Raises an error.
            **kwargs: Additional keyword arguments passed to `DataModel.from_datamodels()`.

        Returns:
            datamodels.DataModel: The instantiated output DataModel class.
        """

        ###################################
        #### Return output model as is ####
        ###################################
        if isinstance(output_model, datamodels.DataModel):
            return output_model

        ##################################
        #### Return input model as is ####
        ##################################
        if isinstance(output_model, str) and output_model.lower() == 'input':
            assert isinstance(
                input, (datamodels.DataModel, str)
            ), "Input DataModel or filepath must be provided if output_model is 'input'."
            return datamodels.open(input) # Assume only one input

        ####################################
        #### Return copy of input model ####
        ####################################
        if isinstance(output_model, str) and output_model.lower() == 'input_copy':
            assert isinstance(
                input, (datamodels.DataModel, str)
            ), "Input DataModel or filepath must be provided if output_model is 'input_copy'."
            input_model = datamodels.open(input) # Assume only one input
            return input_model.from_datamodels(input_model, **kwargs)

        ##################
        #### Filepath ####
        ##################
        if isinstance(output_model, str) and os.path.isfile(output_model):
            out_model = datamodels.open(output_model)
            return out_model

        #####################################
        #### DataModel class name (str) #####
        #####################################
        if isinstance(output_model, str):
            model_class = datamodels.get_model_type_from_name(output_model)
            return model_class.from_datamodels(input, **kwargs)

        #################################
        #### DataModel class (type) #####
        #################################
        if isinstance(output_model, type) and issubclass(output_model, datamodels.DataModel):
            return model_class.from_datamodels(input, **kwargs)

        #####################################
        #### Everything else is an error ####
        #####################################
        raise ValueError(f"Invalid output_model argument {output_model=}.")
    
    def add_data_ancestors(self, input_model : datamodels.DataModel, sources : datamodels.DataModel | str | list[datamodels.DataModel | str]):
        if not isinstance(sources, list):
            sources = [sources]
        for f in sources:
            source_model = datamodels.open(f, meta_only=True)
            input_model.data_ancestors.add_row({
                'FILENAME' : str(source_model.meta.filename),
                'DATETIME_OBS' : str(source_model.meta.datetime_obs),
                'PRIMITIVE' : self.__class__.__name__
            })