
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
            **kwargs: Any additional keyword arguments to set or override (including ``config``) for this primitive.
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
            config (dict | str | None): A configuration dictionary, or the path to to a configuration file.
            validate_class (bool): If True, validates that the class name in the configuration matches the class name of this primitive. *This may get removed in the future.*
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
        """
        Checks if all preconditions are met in order to run this primitive.
        The base class checks if ``self.run`` is ``True``. If not met, the primitive will be skipped.
        This can be overridden to implement custom pre-condition checks.

        Returns:
            bool: The precondition status.
        """
        return self.run

    def apply(self, input, **kwargs):
        """
        The main method a user calls to run a primitive.

        Args:
            input: The input data to process and is up to the subclass to decide what this represents, but is typically a datamodel, or simple containers of datamodels (or filepaths).
            **kwargs: Additional keyword arguments to set or override for this primitive. These will override any arguments set during initialization.

        Returns:
            dict: The main output of the primitive (self._output['output']), which is typically a datamodel or a container of datamodels.
        """

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
                if isinstance(self._output, dict):
                    if 'output' not in self._output:
                        self._output = dict(output=self._output)
                else:
                    self._output = dict(output=self._output)
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
        
        # Get main output
        output = self._output.get('output', None) if self._output is not None else None

        # Reset input
        self.input = None

        # Return main output
        return output
    
    def on_skip(self):
        """
        Hook called when the component is skipped. By default, it sets the status (`self._status`) to "SKIPPED".
        """
        self._status = "SKIPPED"

    def on_exception(self, exc : Exception):
        """
        Hook called when an exception occurs during processing. By default, it sets the status (`self._status`) to "FAILED".
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
        Add an entry to the receipt table of the model summarizing the processing.

        Args:
            model (DataModel): The data model to add the receipt entry to.
            date_time (str, optional): The date and time of processing in ISO format.
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
        Hook called at the end of ``primitive.apply`` to finalize the output.
        See ``finalize_model`` for details.
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
        """
        Finalize a single datamodel by adding receipt entries, used calibrations, and updating metadata.
        """
        model = datamodels.open(model)
        model.meta.drp_version = hispecdrp.__version__ # Always override the current DRP version.
        if self.run:
            model.meta.last_updated = self._date_time
        self.add_receipt_entry(model, date_time=self._date_time)
        for cal_role, calibration in self._calibrations_used.items():
            model.add_calibration(calibration, cal_role, date_time=self._date_time)

    def get_calibration(
        self,
        input : str | datamodels.DataModel,
        cal_role : str,
        use_cached : bool | None = None,
        **kwargs
    ) -> tuple[str | datamodels.DataModel, CalibrationORM | None]:
        """
        Primary method for a primitive to call in order to get a calibration file.

        Args:
            input (str | datamodels.DataModel): The filepath or datamodel to get a calibration for.
            cal_role (str): The type of calibration to get, which should match a key in `self.CALIBRATIONS`. This should be the same string as ORM.cal_type but is not enforced if multiple calibrations of a single type are needed (e.g., using multiple calibrations and interpolating).
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
                input_model = datamodels.open(input, meta_only=True)
                with CalibrationStore(orm_class=input_model.get_calibration_orm_class(), use_cached=use_cached) as store:
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
                    datamodels.CalibrationModel,
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
            # with CalibrationStore(orm_class=cal_model.get_calibration_orm_class(), use_cached=True) as store:
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
            input_model = datamodels.open(input, meta_only=True)
            with CalibrationStore(orm_class=input_model.get_calibration_orm_class()) as store:
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
        Get the output model based on the input model and the specified output model type. See :py:meth:`_get_output_model <hispecdrp.primitives.base_primitive.BasePrimitive._get_output_model>` for details.
        """
        if output_model is None:
            output_model = getattr(self, 'output_model', None)
        return self._get_output_model(input_model, output_model)
    
    @staticmethod
    def _get_output_model(
        input_model : datamodels.DataModel | None = None,
        output_model : str | datamodels.DataModel | type[datamodels.DataModel] = None
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
                - DataModel class (type[DataModel]): Instantiates and returns a new instance of the specified DataModel class using `input_model` as a template with ``output_model.from_datamodel(input_model)``. If input_model.meta.instrument_name is set, it will attempt to use an instrument-specific subclass if available.
                - DataModel class name for specific instrument or instrument agnostic class (str): Instantiates and returns a new instance of the specified DataModel class with ``output_model.from_datamodel(input_model)``. If input_model.meta.instrument_name is set, it will attempt to use an instrument-specific subclass if available.
                - None (default value): Raises an error.

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
                input_model, datamodels.DataModel
            ), "Input model must be provided if output_model is 'input'."
            return input_model

        ####################################
        #### Return copy of input model ####
        ####################################
        if isinstance(output_model, str) and output_model.lower() == 'input':
            assert isinstance(
                input_model, datamodels.DataModel
            ), "Input model must be provided if output_model is 'input_copy'."
            out_model = input_model.copy()
            out_model._filepath = None # NOTE: Revisit if we want to keep the filepath.
            out_model.meta.filename = None
            return out_model

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
            if input_model.meta.instrument_name is not None:
                model_class = datamodels.model_type_from_name(input_model.meta.instrument_name.upper() + output_model)
            else:
                model_class = datamodels.model_type_from_name(output_model)
            return model_class.from_datamodel(input_model)

        #################################
        #### DataModel class (type) #####
        #################################
        if isinstance(output_model, type) and issubclass(output_model, datamodels.DataModel):
            assert isinstance(
                input_model, datamodels.DataModel
            ), "Input model must be provided if output_model is a DataModel class."
            if input_model.meta.instrument_name is not None:
                model_class = datamodels.model_type_from_name(input_model.meta.instrument_name.upper() + output_model.__class__.__name__)
            else:
                model_class = output_model
            return output_model.from_datamodel(input_model)

        #####################################
        #### Everything else is an error ####
        #####################################
        raise ValueError(f"Invalid output_model argument {output_model=}.")

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
            do_fibers = [str(f + 1) for f in range(input_model.NUM_FIBERS)]
        if do_orders is None:
            do_orders = input_model.make_order_vec()
        return do_fibers, do_orders

    def __getattr__(self, attr):
        if '_output' in self.__dict__ and attr in self._output:
            return self._output[attr]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'.")
    
    @staticmethod
    def associate_fibers(input: list[str | datamodels.DataModel]) -> dict[str, object]:
        fiber_map = {}
        input_models = [datamodels.open(f, meta_only=True) for f in input]

        for i, model in enumerate(input_models):
            for fiber in range(1, model.NUM_FIBERS + 1):
                val = getattr(model.meta, f'object_fiber{fiber}', None)
                if val is None or val == 'BLOCKED':
                    continue  # skip blocked fibers
                key = str(fiber)
                if key in fiber_map:
                    raise ValueError(f"Multiple models for fiber {key}")
                fiber_map[key] = input[i]

        return fiber_map
    
    @staticmethod
    def group_input_by_fiber_object(input: list[str | datamodels.DataModel]) -> dict[tuple, list]:
        inputs_grouped = {}
        input_models = [datamodels.open(f, meta_only=True) for f in input]

        ref_set = None

        for i, model in enumerate(input_models):
            objs = []
            for fiber in range(1, model.NUM_FIBERS + 1):
                val = getattr(model.meta, f'object_fiber{fiber}', None)
                objs.append('BLOCKED' if val is None or val == 'BLOCKED' else val)

            objs = tuple(objs)
            objs_set = frozenset(o for o in objs if o != 'BLOCKED')

            if ref_set is None:
                ref_set = objs_set
            elif objs_set != ref_set:
                raise ValueError(f"Inconsistent illuminated fiber types: {ref_set} vs {objs_set}")

            inputs_grouped.setdefault(objs, []).append(input[i])

        return inputs_grouped