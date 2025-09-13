
import datetime
import os
import uuid

import hispecdrp
from koa_middleware import CalibrationStore
from .. import datamodels
from .. import utils

from koa_middleware import CalibrationORM

from ..processing_component import ProcessingComponent


__all__ = ['BasePrimitive']


class BasePrimitive(ProcessingComponent):

    ARGUMENTS = {
        'run': True,
        'output_model' : 'input',
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

        # Init the output, status, and calibrations used
        self._output = None
        self._status = None
        self._calibrations_used = {}

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
        return self.run

    def apply(self, input, **kwargs):
        
        # CURRENT PROCESSING TIME, USED FOR CONSISTENT LOGGING OF PROCESSING AT DIFFERENT PLACES.
        # NOTE: TBD IF THIS IS THE RIGHT APPROACH
        self._date_time = datetime.datetime.now().isoformat()

        # Init logger for this primitive
        self.init_logger()

        # Reset output, status, and calibrations used
        self._output = None
        self._status = None
        self._calibrations_used = {}

        try:
            # This allows pipelines to directly pass the result from a primitive to the next primitive if full_output is True or False.
            # Pipelines can pass in a datamodel or a dict object, in which case the first arg is fetched.
            #if isinstance(input, dict) and 'output' in input:
            #    input = input['output']
            self.set_args(input=input, **kwargs)
            if not self._pre_condition():
                self.log.info(f"Skipping {self.__class__.__name__} with arguments:\n{self.prepr_args()}")
                self.on_skip()
                self._output = dict(output=input)
            else:
                self.log.info(f"Applying {self.__class__.__name__} with arguments:\n{self.prepr_args()}")
                self._output = self._perform()
                if self._status is None:
                    self._status = "COMPLETE"
                self.finalize_output()
                if self.save_result:
                    self.save_output()
        except Exception as e:
            self.log.error(f"Failed executing primitive {self.__class__.__name__}: {e}", exc_info=True)
            self._output = dict(output=input)
            self.on_exception(e)
            self.finalize_output()
            if os.environ.get('HISPECDRP_DEBUG', '1') != '0':
                raise e
            return input
        
        # Get main output and store extra outputs
        if self._output is not None:
            output = self._output.get('output', None)
            #self._extra_output.update({k: v for k, v in self._output.items() if k != 'output'})
        else:
            output = None

        # Return main output
        return output

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
    
    def on_skip(self):
        """
        Hook called when the component is skipped.
        """
        self._status = "SKIPPED"

    def on_exception(self, exc : Exception):
        """
        Hook called when an exception occurs during processing.
        """
        self._status = "FAILED"

    def __repr__(self):
        s = self.__class__.__name__
        for key in self.ARGUMENTS:
            if hasattr(self, key):
                s += f"\n  {key}: {getattr(self, key)}"
        if hasattr(self, 'CALIBRATIONS'):
            for key in self.CALIBRATIONS:
                if hasattr(self, key):
                    s += f"\n  {key}: {getattr(self, key)}"
        return s
    
    def add_receipt_entry(self, model : datamodels.DataModel, date_time : str | None = None):
        """
        Add an entry to the receipt table of the model.
        """
        date_time = self._date_time or date_time
        row = {
            'PROC_TIME' : date_time,
            'DRP_VERSION': hispecdrp.__version__,
            'CLASS': f"{self.__class__.__module__}.{self.__class__.__name__}",
            'FILENAME' : model.meta.filename or 'None',
            'ARGUMENTS': self._get_args_repr(),
            'STATUS' : self._status
        }
        model.receipt.add_row(row)
    
    def _get_args_repr(self):
        """Helper to get a string representation of current arguments."""
        args_dict = {'input': getattr(self, 'input', None)}
        for key in self.ARGUMENTS:
            if hasattr(self, key):
                args_dict[key] = getattr(self, key)
        if hasattr(self, 'CALIBRATIONS'):
            for key in self.CALIBRATIONS:
                if hasattr(self, key):
                    args_dict[key] = getattr(self, key)
        return str(args_dict)

    def finalize_output(self):
        """
        Hook called at the end of `primitive.apply` to finalize the output. Only called if _post_condition() is True.
        """
        assert self._output is not None and 'output' in self._output, 'No output to finalize.'
        output = self._output['output']
        if isinstance(output, (str, datamodels.DataModel)):
            self.finalize_model(output)
        elif isinstance(output, dict):
            for model in output.values():
                self.finalize_model(model)
        elif isinstance(output, list):
            for model in output:
                self.finalize_model(model)
        else:
            self.log.error("Unsupported output type.")
            raise TypeError("Unsupported output type.")
        
    def finalize_model(self, model : datamodels.DataModel):
        model = datamodels.open(model)
        model.meta.drp_version = hispecdrp.__version__ # Always override the current DRP version.
        if self.run:
            model.meta.last_updated = self._date_time
        self.add_receipt_entry(model, date_time=self._date_time)
        for cal_role, calibration in self._calibrations_used.items():
            model.add_calibration(calibration, cal_role, date_time=self._date_time)

    def get_calibration(
        self,
        name : str | datamodels.DataModel,
        cal_role : str, # Match key in self.CALIBRATIONS. Typically 1-1 with ORM.cal_type but not always e.g. (etalon_before, etalon_after).
        use_cached : bool | None = None,
        **kwargs
    ) -> tuple[str | datamodels.DataModel, CalibrationORM | None]:
        """
        Primary method for a primitive to call in order to get a calibration file.

        Args:
            name (str): The filepath or datamodel to get a calibration for.
            cal_role (str): The type of calibration to get, which should match a key in `self.CALIBRATIONS`.
            use_cached (bool | None): If True, use cached calibrations if available. If None, use the default setting from the configuration.
            **kwargs: Additional keyword arguments to pass to the selector.

        Returns:
            tuple[str | datamodels.DataModel, CalibrationORM | None]: A tuple of (calibration_file_or_model, calibration_orm).
                The ORM will be None if the calibration is not in the database.
        """

        # User provided value
        cal_value = getattr(self, cal_role, None)

        ####################
        #### Case 1: ID ####
        ####################
        if isinstance(cal_value, str):
            is_uuid = False
            try:
                _ = uuid.UUID(cal_value)  # Check if it's a valid UUID
                is_uuid = True
            except ValueError:
                pass
            if is_uuid:
                input_model = datamodels.open(name, meta_only=True)
                with CalibrationStore(orm_class=input_model.META_MODEL.get_orm_class(), use_cached=use_cached) as store:
                    local_filepath, calibration_orm = store.get_calibration_by_id(calibration_id=cal_id)
                    if local_filepath is not None:
                        self._calibrations_used[cal_role] = calibration_orm
                        return input_model, calibration_orm
                    else:
                        # Msg probably not necessary, but useful for debugging.
                        self.log.warning(f"No calibration found in the store for the given cal_id: {cal_id}.")
                        return None, None

        ###############################################################
        #### Case 2: filepath string or datamodel (uuid ruled out) ####
        ###############################################################
        if isinstance(cal_value, (str, datamodels.DataModel)):

            # Open to get cal id (open in full mode to return for processing)
            cal_model = datamodels.open(cal_value)
            assert isinstance(
                cal_model,
                (
                    datamodels.HISPECCalibrationModel,
                    datamodels.PARVICalibrationModel,
                )
            ), "Input must be a valid CalibrationDataModel."

            cal_id = cal_model.meta.cal_id

            # If no cal_id, return the model itself.
            #if cal_id is None:
            #self.log.warning(f"No calibration ID found for selected calibration: {cal_model}.")
            self._calibrations_used[cal_role] = cal_model
            return cal_model, None  # Return the cal model as is. Ignore ORM.

            # Check if the calibration is in the DB.
            # with CalibrationStore(orm_class=cal_model.META_MODEL.get_orm_class(), use_cached=True) as store:
            #     local_filepath, calibration_orm = store.get_calibration_by_id(calibration_id=cal_id)
            #     if local_filepath is not None:
            #         self._calibrations_used[cal_role] = calibration_orm
            #         return cal_model, calibration_orm
            #     else:
            #         self.log.warning(f"No calibration found in the store for the given cal_id: {cal_id}.")
            #         self._calibrations_used[cal_role] = cal_model
            #         return cal_model, None

        #######################################
        #### Case 3: CalibrationORM object ####
        #######################################
        if isinstance(cal_value, CalibrationORM):
            self._calibrations_used[cal_role] = cal_value
            local_filepath = None  # Get local filepath with store
            return local_filepath, cal_value

        ################################################################
        ##### Case 4: None - use default selector from CALIBRATIONS ####
        ################################################################
        if cal_value is None:
            if not hasattr(self, 'CALIBRATIONS') or cal_role not in self.CALIBRATIONS:
                raise ValueError(f"No calibration configuration found for '{cal_role}' in CALIBRATIONS")

            calibration_config = self.CALIBRATIONS[cal_role]
            selector_class = calibration_config['selector']
            selector_kwargs = calibration_config.get('selector_kwargs', {})
            selector = selector_class(**selector_kwargs)
            input_model = datamodels.open(name, meta_only=True)
            with CalibrationStore(orm_class=input_model.META_MODEL.get_orm_class()) as store:
                local_filepath, calibration_orm = store.get_calibration(input_model.meta.model_dump(), selector=selector)
                return local_filepath, calibration_orm

        # Else raise error
        msg = f"Invalid calibration value for '{cal_role}': {type(cal_value)}"
        self.log.error(msg)
        raise ValueError(msg)

    def get_output_model(
        self,
        input_model : datamodels.DataModel | None = None,
        output_model : str | datamodels.DataModel | type[datamodels.DataModel] = None
    ) -> datamodels.DataModel:
        """
        Get the output model based on the input model and the specified output model type.
        """
        if output_model is None:
            output_model = getattr(self, 'output_model', None)
        return self._get_output_model(input_model, output_model)
    
    @staticmethod
    def _get_output_model(
        input_model : datamodels.DataModel | None = None,
        output_model : str | datamodels.DataModel | type[datamodels.DataModel] = None
        ) -> datamodels.DataModel:
        if output_model is not None:
            if isinstance(output_model, str):
                if output_model.lower() == 'input': # Return input model
                    assert isinstance(input_model, datamodels.DataModel), "Input model must be provided if output_model is 'input'."
                    return input_model
                elif output_model.lower() == 'input_copy': # Return a copy of the input model
                    out_model = input_model.copy()
                    out_model._filepath = None # NOTE: Revisit if we want to keep the filepath.
                    out_model.meta.filename = None
                    return out_model
                elif os.path.isfile(output_model): # If it's a file path, open the model from that file
                    out_model = datamodels.open(output_model)
                    return out_model
                elif any(m.endswith(output_model) for m in datamodels.MODEL_REGISTER):
                    instrument = input_model.meta.instrument_name
                    model_class = datamodels.model_type_from_name(instrument.upper() + output_model)
                    out_model = model_class.from_datamodel(input_model)
                    return out_model
                else:
                    raise ValueError(f"Invalid output_model argument {output_model=}.")
            elif isinstance(output_model, type): # Type of datamodel (python class)
                return output_model.from_datamodel(input_model)
            elif isinstance(output_model, datamodels.DataModel): # If it's already a DataModel, return it
                return output_model
            else:
                raise ValueError(f"Invalid output_model argument {output_model=}.")
        else:
            # Default
            return input_model
        
    @staticmethod
    def get_do_traces(input_model : datamodels.DataModel, do_fibers : list[int] | None = None, do_orders : list[int] | None = None) -> tuple[list[str], list[int]]:
        """
        Helper to get the fibers and orders to process.
        NOTE: This will need updated once we dynamically determine which fibers contain what data.
        """
        if do_fibers is None:
            do_fibers = [str(f + 1) for f in range(input_model.NUM_FIBERS)]
        if do_orders is None:
            do_orders = input_model.make_order_vec()
        return do_fibers, do_orders
    
    def __getattr__(self, attr):
        if '_output' in self.__dict__ and attr in self._output:
            return self._output[attr]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'.")