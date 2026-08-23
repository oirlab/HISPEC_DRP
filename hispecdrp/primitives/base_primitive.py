from __future__ import annotations
import datetime
import numpy as np
from copy import deepcopy
from typing import Sequence
import os

from koa_middleware import CalibrationSelector
from koa_middleware.utils import is_valid_uuid

import hispecdrp
from .. import datamodels
from ..datamodels import DataModel
from ..driver.logging import logging_context
from ..calibrations.utils import get_calibration_selector_type
from ..calibrations import HISPECCalibrationStore
from ..driver import ConfigFile, ConfigStore, ConfigSelector, get_config_filepath
from ..processing_component import ProcessingComponent

import logging
logger = logging.getLogger(__name__)

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..pipelines import BasePipeline


__all__ = ['BasePrimitive']


class BasePrimitive(ProcessingComponent):
    
    CALIBRATIONS = {}

    ##########################################
    #### INITIALIZATION AND CONFIGURATION ####
    ##########################################

    def __init__(
        self,
        dataset : Any = None,
        config : dict | ConfigFile | str | None = None,
        context : dict | None = None,
        log_config : dict | str | None = None,
        parent : Any | None = None,
        **kwargs
    ):
        """
        Parameters
        ----------
        dataset : Any
            The dataset to process.
        config : dict | str | None
            A configuration dictionary, or the path to to a configuration file.
        log_config : dict | str | None
            Logging configuration override. Can be a dict of logging keys, a path
            to a logging YAML file, or None to use package defaults.
        parent : Any | None
            The parent object that instantiated this primitive.
        **kwargs
            Any additional keyword arguments to set or override (including ``config``) for this primitive.
        """

        # Parent pipeline instance
        self.parent = parent
        self.dataset = dataset

        # Load the configuration file or Config object
        self.init_config(dataset, config, context)

        # Resolve logging config (merges defaults -> config file -> kwarg)
        self._init_log_config(log_config)

        # Set any kwargs passed to init
        self.init_parameters(**kwargs)


        # Init the output, status, and calibrations used
        self._output = None
        self._status = None
        self._calibrations_used = {}

    def init_config(
        self,
        dataset : Any = None,
        config : dict | None = None,
        context : dict | None = None
    ):
        if config is not None:
            if isinstance(config, str):
                self.config = ConfigFile.from_file(config)
            elif isinstance(config, dict):
                self.config = ConfigFile.from_dict(config)
            elif isinstance(config, ConfigFile):
                self.config = config
            else:
                raise TypeError("config must be a str (filename or filepath), dict, or ConfigFile instance.")
        else:
            if dataset is None:
                raise ValueError(
                    "A dataset must be passed to the Primitive constructor if no config file is provided."
                )
            self.config = self.get_config(
                dataset, context=context
            )

    @classmethod
    def from_config_file(cls, config : ConfigFile | str, **kwargs):
        """
        Initialize a primitive from a configuration file.

        Parameters
        ----------
        config : str, dict, or ConfigFile, optional.
            The configuration for the primitive. Can be one of:
                - str: A full path to a configuration file or a filename in the ConfigStore.
                - dict: A dictionary of configuration parameters.
                - ConfigFile: A ConfigFile instance.

            If not provided, the primitive will attempt to auto-configure based on the input dataset and context.
            If the dataset is required for auto-configuration, it must be provided.
        kwargs : dict
            Additional keyword arguments to pass to the primitive constructor.

        Returns
        -------
        BasePrimitive
            An instance of the primitive with the configuration loaded.
        """

        # If ConfigFile object is passed, use it directly.
        # Otherwise, load the config file.
        if isinstance(config, str):
            config_filepath = get_config_filepath(config)
            config_file = ConfigFile.from_file(config_filepath)
        elif isinstance(config, dict):
            config_file = ConfigFile.from_dict(config)
        elif isinstance(config, ConfigFile):
            config_file = config
        else:
            raise ValueError("Input config must be a filepath (str) or a ConfigFile object")

        # If class is not provided, then use the current cls context
        prim_class = config_file.class_
        if prim_class is None:
            prim_class = cls

        # Check the class is a valid Step or Pipeline class
        if not issubclass(prim_class, cls):
            raise ValueError(f"Config file {config_file} specifies class {prim_class}, which is not a subclass of {cls}")
        from ..pipelines.base_pipeline import BasePipeline
        if prim_class in (BasePrimitive, BasePipeline):
            raise ValueError(f"Config file {config_file} does not specify a class, and the current context is {cls}, which is not a valid class to initialize.")

        if issubclass(prim_class, BasePipeline):
            raise ValueError(f"Config file {config_file} specifies class {prim_class}, which is a Pipeline. Use BasePipeline.from_config_file to initialize a Pipeline.")

        return prim_class(
            config=config_file,
            **kwargs
        )
    
    def init_calibrations(self):
        """
        Process calibration configurations applying selector and kwargs logic.
        
        Handles 4 cases for each calibration:
        1. No entry in config -> use default selector and kwargs
        2. Selector provided -> merge kwargs if same as default, else use user kwargs only
        3. Kwargs only -> use default selector and merge kwargs
        4. Both selector and kwargs -> validate and apply appropriately
        
        Parameters
        ----------
        cal_config_dict : dict
            The calibrations config dictionary to process
        """
        pass
        for cal in self.CALIBRATIONS:
            
            # Alias 
            cal_config = self.config.parameters.get(cal)

            # Case 1: No entry in config dict or entry is empty dict
            # -> use default selector and kwargs
            if not cal_config:
                #self.config['arguments'][cal] = self.CALIBRATIONS[cal].copy()
                continue
            
            # Case 2: User provided selector class (string or class)
            # -> If selector matches default selector, use default kwargs (merged with user kwargs)
            # -> If selector does not match, use only user kwargs (or empty dict)
            # if 'selector' in cal_config:

            #     user_selector = cal_config['selector']
                
            #     # Convert string to class if needed
            #     if isinstance(user_selector, str):
            #         user_selector = get_calibration_selector_type(user_selector)
            #         cal_config['selector'] = user_selector
                
            #     # Validate selector is a CalibrationSelector subclass
            #     if not issubclass(user_selector, CalibrationSelector):
            #         raise TypeError(f"Invalid selector for calibration '{cal}': {user_selector}. Must be a CalibrationSelector subclass.")
                
            #     # Check if same selector as default
            #     same_selector = user_selector is self.CALIBRATIONS[cal]['selector']
                
            #     if same_selector:
            #         # Merge default kwargs with user kwargs
            #         default_kwargs = self.CALIBRATIONS[cal].get('selector_kwargs', {})
            #         user_kwargs = cal_config.get('selector_kwargs', {})
            #         cal_config['selector_kwargs'] = default_kwargs | user_kwargs
            #     else:
            #         # Different selector, use only user kwargs (or empty dict)
            #         if 'selector_kwargs' not in cal_config:
            #             cal_config['selector_kwargs'] = {}
            
            # # Case 3: Filepath provided
            # elif isinstance(cal_config, str):
            #     # Assume it's a filepath to a calibration model, but do not validate here
            #     # User can provide filepath directly as argument value instead of using selector logic
            #     continue

            # # Case 4: No selector class specified, but user may have kwargs
            # # -> Use default selector and merge kwargs
            # else:
            #     cal_config['selector'] = self.CALIBRATIONS[cal]['selector']
            #     default_kwargs = self.CALIBRATIONS[cal].get('selector_kwargs', {})
            #     user_kwargs = cal_config.get('selector_kwargs', {})
            #     cal_config['selector_kwargs'] = default_kwargs | user_kwargs
    
    def init_parameters(self, **kwargs):
        """
        Initialize parameters including calibrations as top-level attributes with the following priorities:
            1. `self.PARAMETERS` (lowest priority)
            2. `self.config.parameters`
            3. Direct attribute setting (highest priority)

        Parameters
        ----------
        **kwargs
            Direct attribute settings to override config and defaults.
            To override a calibration, provide the calibration name as a keyword argument.
            Selector logic is not run here.
        """

        # Allowed set of parameter names
        allowed = list(set(self.PARAMETERS.keys()) | set(self.CALIBRATIONS.keys()))

        # Get all step parameter values from config file and input kwargs
        pars = self.PARAMETERS | self.config.parameters | kwargs

        # Set attributes
        for k, v in pars.items():
            if k not in allowed:
                msg = f"Invalid parameter name '{k}' for {self.__class__.__name__}. Valid parameters are: {allowed}"
                logger.error(msg)
                raise NameError(msg)
            setattr(self, k, v)

        # Set additional dynamic defaults
        if self.max_cores is None:
            if self.parent is not None:
                self.max_cores = self.parent.max_cores or 1

    ##################
    #### MAIN API ####
    ##################

    def _pre_condition(self):
        """
        Checks if all preconditions are met in order to run this primitive.
        The base class checks if ``self.run`` is ``True``.
        If not met, the primitive will be skipped.
        This can be overridden to implement custom pre-condition checks.

        Returns
        -------
        pre_condition : bool
            The precondition status.
        """
        return self.run in (True, None)

    def apply(self, dataset : Any = None, **kwargs):
        """
        The main method a user calls to run a primitive.

        Parameters
        ----------
        input
            The input data to process and is up to the subclass to decide what this represents, but is typically a datamodel, or simple containers of datamodels (or filepaths).
        **kwargs
            Additional keyword arguments to set or override for this primitive. These will override any arguments set during initialization.

        Returns
        -------
        dict
            The main output of the primitive (self._output['output']), which is typically a datamodel or a container of datamodels.
        """

        # Reset output, status, and calibrations used
        self._output = None
        self._status = None
        self._calibrations_used = {}

        with logging_context(self.log_config):

            # Start time of processing
            self._datetime_start = datetime.datetime.now().isoformat(timespec='microseconds')

            try:
                # Set new parameters
                self.set_pars(**kwargs)

                # Primary input
                if dataset is not None:
                    self.dataset = dataset

                if self.save_result is None and self.output_dir is not None:
                    self.save_result = True
                if self.save_result and self.output_dir is None:
                    self.output_dir = self.get_input_dir(self.dataset)

                if not self._pre_condition():
                    logger.info(f"Skipping {self.__class__.__name__}")
                    self._status = "SKIPPED"
                    self._on_skip()
                    self._output = dict(output=self.dataset)
                    self.finalize_output()
                    # NOTE: No saving or previews if skipped
                else:
                    logger.info(f"Applying {self.__class__.__name__} with parameters:\n{self.pars_str}")
                    self._output = self._perform()
                    # Check again if skipped since primitives can do this dynamically during processing.
                    if not self.run:
                        logger.info(f"Skipping {self.__class__.__name__} after processing due to run=False.")
                        self._status = "SKIPPED"
                        self._on_skip()
                    else:
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
                        if self.save_preview or self.show_preview:
                            self._create_preview()
            except Exception as e:
                self._output = dict(output=input)
                self._status = "FAILED"
                self._on_exception(e)
                self.finalize_output()
                logger.exception(
                    f"Failed executing primitive {self.__class__.__name__}"
                )
                raise
            
            # Get main output
            output = self._output.get('output', None) if self._output is not None else None

            # Reset input
            self.dataset = None

        # import threading
        # import multiprocessing as mp
        # print(mp.active_children())
        # print(threading.enumerate())

        # Return main output
        return output

    @classmethod
    def call(
        cls,
        dataset : Any,
        config : dict | str | ConfigFile | None = None,
        context : dict | None = None,
        parent : BasePipeline | None = None,
        log_config : dict | str | None = None,
        return_instance : bool = False,
        **kwargs
    ):
        """
        Top level method to instantiate a primitive or pipeline and run it in one call.

        Parameters
        ----------
        dataset : object
            The dataset to process.
        context : object, optional
            The context in which the primitive is being called.
        config : object, optional
            The configuration to use for the primitive. Can be a dict, a config file path, or a ConfigFile object.
        parent : BasePipeline, optional
            The parent pipeline of this primitive.
        log_config : dict, str, or None, optional
            The logging configuration to use for the primitive.
        return_instance : bool, optional
            If True, return a tuple of (result, primitive_instance).
            If False, return only the result.
        kwargs : dict
            Override primitive parameters.

        Returns
        -------
        results: Any
            The result of processing the input with the primitive.
        """
        prim = cls(
            dataset,
            config=config,
            parent=parent,
            context=context,
            log_config=log_config,
            **kwargs
        )
        result = prim.apply()
        if return_instance:
            return result, prim
        else:
            return result

    def get_calibration(
        self,
        input : str | DataModel,
        cal_role : str,
        selector : CalibrationSelector | None = None,
        use_cached : bool | None = None,
    ) -> tuple[str | DataModel, dict | None]:
        """
        Primary method for a primitive to call in order to get a calibration file in the context of a primitive.

        Parameters
        ----------
        input : str or DataModel
            The filepath or datamodel to get a calibration for.
        cal_role : str
            The role of calibration in this primitive, which should match a key in
            ``self.CALIBRATIONS``.
        selector : CalibrationSelector or None
            An optional selector to use instead of the default selector
            defined in ``self.CALIBRATIONS``.
        use_cached : bool or None
            If True, use cached calibrations if available. If None, use the
            default setting from the configuration.

        Returns
        -------
        tuple[str | DataModel, dict | None]:
            A tuple of (calibration_file_or_model, cal_record).
        """

        # User provided value
        cal_value = getattr(self, cal_role, None)

        # Input model
        input_model = datamodels.open(input, meta_only=True)

        ####################
        #### Case 1: ID ####
        ####################
        if isinstance(cal_value, str) and is_valid_uuid(cal_value):
            cal_id = cal_value
            with HISPECCalibrationStore(
                instrument_name=input_model.meta.instrument_name,
                use_cached=use_cached,
                # Pipelines use the local caldb; remote catalog sync belongs in
                # verify_koa_calibrations / run_caldb, not every primitive lookup.
                connect_remote=False,
                sync_on_init=False,
            ) as store:
                
                local_filepath, cal_record = store.get_calibration(calibration=cal_id)
                
                self._calibrations_used[cal_role] = cal_record

                logger.info(
                    f"Calibration role={cal_role}: Using file={cal_record['filename']} ID={cal_record['id']} for input={input_model}"
                )
                
                return local_filepath, cal_record

        ###############################################################
        #### Case 2: filepath string or datamodel (uuid ruled out) ####
        ###############################################################
        if isinstance(cal_value, (str, DataModel)):

            # Open to get cal id (open in full mode to return for processing)
            if isinstance(cal_value, str):

                if os.path.isfile(cal_value):
                    cal_model = datamodels.open(cal_value)
                elif os.path.isfile(os.path.join(os.environ['KOA_CALIBRATION_CACHE'], cal_value)):
                    cal_filepath = os.path.join(os.environ['KOA_CALIBRATION_CACHE'], cal_value)
                    cal_model = datamodels.open(cal_filepath)

            elif isinstance(cal_value, datamodels.CalibrationModel):
                cal_model = cal_value

            assert isinstance(
                cal_model, datamodels.CalibrationModel
            ), "Input must be a valid CalibrationDataModel."

            cal_record = cal_model.to_record()
            self._calibrations_used[cal_role] = cal_record

            logger.info(
                f"Calibration role={cal_role}: Using file={cal_record['filename']} ID={cal_record['id']} for input={input_model}"
            )

            return cal_model, cal_record

        ######################################
        #### Case 3: Selector config dict ####
        ######################################
        if isinstance(cal_value, dict) and 'selector' in cal_value and 'selector_kwargs' in cal_value:

            selector_class = cal_value.get(
                'selector',
                self.CALIBRATIONS[cal_role]['selector']
            )
            selector_kwargs = cal_value.get(
                'selector_kwargs',
                self.CALIBRATIONS[cal_role]['selector_kwargs']
            )
            selector = selector_class(**selector_kwargs)
            
            with HISPECCalibrationStore(
                instrument_name=input_model.meta.instrument_name,
                use_cached=use_cached,
                connect_remote=False,
                sync_on_init=False,
            ) as store:
                local_filepath, cal_record = store.select_and_get_calibration(
                    input_model.meta,
                    selector=selector
                )
            
            self._calibrations_used[cal_role] = cal_record
            logger.info(
                f"Calibration role={cal_role}: Using file={cal_record['filename']} ID={cal_record['id']} for input={input_model}"
            )

            return local_filepath, cal_record

        ######################################
        #### Case 4: Calibration Selector ####
        ######################################
        if isinstance(cal_value, CalibrationSelector):

            selector = cal_value
            input_model = datamodels.open(input, meta_only=True)

            with HISPECCalibrationStore(
                instrument_name=input_model.meta.instrument_name,
                use_cached=use_cached,
                connect_remote=False,
                sync_on_init=False,
            ) as store:
                
                local_filepath, cal_record = store.select_and_get_calibration(
                    input_model.meta,
                    selector=selector
                )
            
            self._calibrations_used[cal_role] = cal_record

            logger.info(
                f"Calibration role={cal_role}: Using file={cal_record['filename']} ID={cal_record['id']} for input={input_model}"
            )

            return local_filepath, cal_record

        ################################################################
        ##### Case 5: None - use default selector from CALIBRATIONS ####
        ################################################################
        if cal_value is None:
            if selector is None:
                if not hasattr(self, 'CALIBRATIONS') or cal_role not in self.CALIBRATIONS:
                    raise ValueError(f"No calibration configuration found for '{cal_role}' in CALIBRATIONS")

                #calibration_config = self.config['calibrations'][cal_role]
                #calibration_config = self.config['calibrations'][cal_role]
                calibration_config = self.CALIBRATIONS[cal_role]
                selector_class = calibration_config['selector']
                selector_kwargs = calibration_config.get('selector_kwargs', {})
                selector = selector_class(**selector_kwargs)
            
            with HISPECCalibrationStore(
                instrument_name=input_model.meta.instrument_name,
                use_cached=use_cached,
                connect_remote=False,
                sync_on_init=False,
            ) as store:
                
                local_filepath, cal_record = store.select_and_get_calibration(
                    input_model.meta,
                    selector=selector
                )

            self._calibrations_used[cal_role] = cal_record

            logger.info(
                f"Calibration role={cal_role}: Using file={cal_record['filename']} ID={cal_record['id']} for input={input_model}"
            )
            return local_filepath, cal_record

        # Else raise error
        msg = f"Invalid calibration value for '{cal_role}': {type(cal_value)}"
        logger.error(msg)
        raise ValueError(msg)

    #################
    #### OUTPUTS ####
    #################

    def finalize_output(self):
        """
        Hook called at the end of ``primitive.apply`` to finalize the output.
        See ``finalize_model`` for details.
        """
        assert self._output is not None and 'output' in self._output, 'No output to finalize.'
        output = self._output['output']
        self._datetime_end = datetime.datetime.now().isoformat(timespec='microseconds')
        
        def _finalize(item):
            if isinstance(item, (str, DataModel)):
                self.finalize_model(item)
            elif isinstance(item, dict):
                for val in item.values():
                    self.finalize_model(val)
            elif isinstance(item, list):
                for val in item:
                    self.finalize_model(val)

        _finalize(output)
        
    def finalize_model(self, model : DataModel | str):
        """
        Finalize a single datamodel by adding receipt entries, used calibrations, and updating metadata.
        """
        if isinstance(model, str):
            return
        
        if model.meta.filename is None and not isinstance(model, datamodels.CalibrationModel):
            model.generate_filename(suffix=self.suffix)

        model.meta.drp_version = hispecdrp._version.__version__ # Always override the current DRP version.
        if self.run:
            model.meta.last_processed = self._datetime_end
        self.add_receipt_entry(model)
        for cal_role, calibration in self._calibrations_used.items():
            self.add_calibration_entry(model, calibration, cal_role=cal_role)

    def add_receipt_entry(self, model : DataModel):
        """
        Add an entry to the receipt table of the model summarizing the processing.

        Parameters
        ----------
        model : DataModel
            The data model to add the receipt entry to.
        """
        config_filename = self.config.filename or self.config.name or "InMemoryConfigFile"

        row = {
            'PROC_START_TIME' : self._datetime_start,
            'PROC_END_TIME' : self._datetime_end,
            'CLASS': f"{self.__class__.__module__}.{self.__class__.__name__}",
            'FILENAME' : model.meta.filename or 'None',
            'DRP_VERSION': hispecdrp._version.__version__,
            'DRP_GIT_COMMIT_ID' : hispecdrp._version.__commit_id__,
            'PARAMETERS': self.get_pars_str_for_receipt(),
            'STATUS' : self._status,
            'CONFIG_FILE' : config_filename
        }
        model.receipt.add_row(row)
    
    def add_calibration_entry(
        self,
        model : DataModel,
        calibration : DataModel | str | dict,
        cal_role : str,
    ):
        """
        Add a calibration to the DataModel's calibrations table.
        
        Parameters
        ----------
        model : DataModel
            The data model to add the calibration to.
        calibration : DataModel | str | dict
            The calibration to add. Can be:

                - DataModel object
                - String filepath (will load metadata to create entry)
                - Dict with calibration column values

        cal_role : str
            The role of the calibration.
        """

        # Case 1: Either metadata dict or SQL record (dict)
        # Assume same keys for now
        if isinstance(calibration, datamodels.MetaModel):
            new_row = {
                'CAL_ROLE': cal_role,
                'FILENAME' : calibration['filename'] or 'None',
                'TIME_APPLIED' : self._datetime_start,
                'CAL_TYPE' : calibration['cal_type'],
                'DATETIME_OBS' : calibration['datetime_obs'],
                'LAST_PROCESSED' : calibration['last_processed'],
                'ID' : calibration['cal_id'],
                'CLASS' : self.__class__.__name__,
                'DRP_VERSION' : calibration['drp_version'],
            }

        if isinstance(calibration, dict):
            new_row = {
                'CAL_ROLE': cal_role,
                'FILENAME' : calibration['filename'] or 'None',
                'TIME_APPLIED' : self._datetime_start,
                'CAL_TYPE' : calibration['cal_type'],
                'DATETIME_OBS' : calibration['datetime_obs'],
                'LAST_PROCESSED' : calibration['last_processed'],
                'ID' : calibration['id'],
                'CLASS' : self.__class__.__name__,
                'DRP_VERSION' : calibration['drp_version'],
            }

        # Case 2: filepath or DataModel
        # NOTE: This path should be used with caution
        #   and limited to dev purposes or limited user cases, never at Keck.
        elif isinstance(calibration, (str, DataModel)):
            from ..datamodels import open as open_datamodel
            cal_model = open_datamodel(calibration, meta_only=True)
            new_row = {
                'CAL_ROLE': cal_role,
                'FILENAME' : cal_model.meta.filename or 'None',
                'TIME_APPLIED' : self._datetime_start,
                'CAL_TYPE' : cal_model.meta.cal_type,
                'DATETIME_OBS' : cal_model.meta.datetime_obs or 'None',
                'LAST_PROCESSED' : cal_model.meta.last_processed or None,
                'ID' : cal_model.meta.cal_id,
                'CLASS' : self.__class__.__name__,
                'DRP_VERSION' : cal_model.meta.drp_version,
            }
        else:
            raise ValueError("Invalid calibration input")

        # Add new row
        model.calibrations.add_row(new_row)

    def add_data_ancestors(
        self,
        model : DataModel,
        sources: DataModel | str | list[DataModel | str],
    ):
        if not isinstance(sources, list):
            sources = [sources]
        for f in sources:
            source_model = datamodels.open(f)
            
            row_data = {
                'FILENAME': str(source_model.meta.filename),
                'CLASS' : self.__class__.__name__,
                'TIME_APPLIED': self._datetime_start
            }

            # Add extra fields if include_ancestors is True and value is not None
            for prop, schema in source_model.meta._schema.get('properties', {}).items():
                if schema.get('include_ancestors', False):
                    val = getattr(source_model.meta, prop)
                    if val is not None:
                        row_data[prop.upper()] = val

            # Expand table if needed
            new_cols = [c for c in row_data.keys() if c not in model.data_ancestors.colnames]
            if new_cols:
                model._extend_data_ancestors_columns(new_cols)

            model.data_ancestors.add_row(row_data)

    def get_output_model(
        self,
        dataset : Any = None,
        output_model : str | 'DataModel' | type['DataModel'] | None = None,
        **kwargs
    ) -> 'DataModel':
        """
        Get the output model based on the input model and the specified output model type.

        Parameters
        ----------
        dataset : Any, optional
            The dataset to use for creating the output model.
        output_model : str | datamodels.DataModel | type[datamodels.DataModel], optional
            The desired output model type or specification. It can take on several values:
                
                - datamodels.DataModel instance: Returns `output_model` as is.
                - 'input' (str): Returns `input_model` as is.
                - 'input_copy' (str): Returns a copy of `input_model`.
                - Valid filepath (str): Opens and returns the model from the specified file.
                - DataModel class (type[DataModel]): Instantiates and returns a new instance of the specified DataModel class using `input_model` as a template with ``output_model.from_datamodels(input_model)``. If input_model.meta.instrument_name is set, it will attempt to use an instrument-specific subclass if available.
                - DataModel class name for specific instrument or instrument agnostic class (str): Instantiates and returns a new instance of the specified DataModel class with ``output_model.from_datamodels(input_model)``. If input_model.meta.instrument_name is set, it will attempt to use an instrument-specific subclass if available.
                - None (default value): Falls back to ``self.output_model``.

        **kwargs : Additional keyword arguments passed to `DataModel.from_datamodels()`.

        Returns
        -------
        datamodels.DataModel
            The instantiated output DataModel class.
        """
        if output_model is None:
            output_model = getattr(self, 'output_model', None)

        def _get_first_input(dataset):
            if isinstance(dataset, datamodels.DataModel):
                return dataset, None, 1
            if isinstance(dataset, str):
                return datamodels.open(dataset), None, 1
            if isinstance(dataset, Sequence):
                for item in dataset:
                    if isinstance(item, datamodels.DataModel):
                        return item, list, len(dataset)
                    if isinstance(item, str) and os.path.isfile(item):
                        return datamodels.open(item), list, len(dataset)
            if isinstance(dataset, dict):
                for val in dataset.values():
                    if isinstance(val, datamodels.DataModel):
                        return val, dict, len(dataset)
                    if isinstance(val, str) and os.path.isfile(val):
                        return datamodels.open(val), dict, len(dataset)
            raise ValueError("Input must be a datamodel, filepath, sequence, or dict.")

        # Get the first input model and its container type
        input_model, input_type, input_len = _get_first_input(dataset)

        ###################################
        #### Return output model as is ####
        ###################################
        if isinstance(output_model, datamodels.DataModel):
            return output_model

        # String ('input', 'input_copy', filepath or datamodel class name)
        if isinstance(output_model, str):

            ##################################
            #### Return input model as is ####
            ##################################
            if output_model.lower() == 'input':
                if not isinstance(dataset, (datamodels.DataModel, str)):
                    raise ValueError("Input DataModel or filepath must be provided if output_model is 'input'.")
                return datamodels.open(dataset)

            ####################################
            #### Return copy of input model ####
            ####################################
            if output_model.lower() == 'input_copy':
                if not isinstance(dataset, (datamodels.DataModel, str)):
                    raise ValueError("Input DataModel or filepath must be provided if output_model is 'input_copy'.")
                input_model = datamodels.open(dataset)
                kwargs.setdefault('primitive', self)
                return input_model.from_datamodels(input_model, **kwargs)

            ##################
            #### Filepath ####
            ##################
            if os.path.isfile(output_model):
                return datamodels.open(output_model)

            #####################################
            #### DataModel class name (str) #####
            #####################################
            output_model_class = datamodels.get_model_type(output_model)
            if (
                output_model_class is not None
                and input_len == 1
                and input_model.__class__ is output_model_class
            ):
                return input_model

            kwargs.setdefault('primitive', self)
            return output_model_class.from_datamodels(dataset, **kwargs)
            

        #################################
        #### DataModel class (type) #####
        #################################
        if isinstance(output_model, type) and issubclass(output_model, datamodels.DataModel):
            kwargs.setdefault('primitive', self)
            return output_model.from_datamodels(dataset, **kwargs)

        #####################################
        #### Everything else is an error ####
        #####################################
        raise ValueError(f"Invalid output_model argument {output_model=}.")

    ###############
    #### UTILS ####
    ###############

    def __getattr__(self, attr):
        if (
            '_output' in self.__dict__
            and isinstance(self.__dict__['_output'], dict)
            and attr in self.__dict__['_output']
        ):
            return self.__dict__['_output'][attr]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'.")

    def get_effective_config(self):
        config_file = deepcopy(self.config)
        config_file.parameters |= self.get_pars()
        return config_file

    @property
    def default_suffix(self):
        return f'proc-{self.class_alias}'

    def get_pars_str_for_receipt(self):
        pars = self.get_pars()
        pars['input'] = getattr(self, 'input', None)
        for key in pars:
            val = pars[key]
            if isinstance(val, np.ndarray) and val.size > 100: # MAX SIZE = 100 elements
                pars[key] = f"ndarray shape={val.shape} dtype={val.dtype}"
        return str(pars)

    def _on_exception(self, exc : Exception):
        """
        Hook called when an exception occurs during processing. By default, it does nothing.
        """
        pass

    def _on_skip(self):
        """
        Hook called when the primitive is skipped due to pre-condition check failing. By default, it does nothing.
        """
        pass