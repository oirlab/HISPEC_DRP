from __future__ import annotations # NOTE: Not needed in Python 3.13+
import os
import astropy.table
from astropy.io import fits
from copy import deepcopy

from hispecdrp import __version__
import numpy as np
import warnings
from astropy.io.fits.verify import VerifyWarning
from uuid import uuid4 as make_uuid4
from typing import Any, Sequence, Self, ClassVar

from hispecdrp.calibrations.hispec_store import HISPECCalibrationStore

from .utils import open as datamodel_open, get_instrument_config
from .meta_data import MetaModel, merge_meta, get_meta_schema
from ..utils import convert_endianness
from .extensions import RECEIPT, CALIBRATIONS, DATA_ANCESTORS
from .utils import _set_datetime_meta, _set_koa_meta

from koa_middleware import CalibrationStore
from koa_middleware.utils import get_koa_id_timestamp_from_datetime, get_env_var_bool

import logging
logger = logging.getLogger(__name__)


_VALID_HDU = fits.ImageHDU | fits.BinTableHDU | fits.PrimaryHDU

class DataModel:
    """
    Base class for data models.

    Parameters
    ----------
    EXTENSIONS : dict
        Extensions and their properties.
    _INHERIT_EXTENSIONS : bool
    Whether to inherit extensions from parent class. Defaults to True.
        Whether to inherit extensions from parent class. Defaults to True.
    _ALLOWED_ATTRIBUTES : list[str]
        List of attributes that can be set directly on the DataModel.
    _DEFAULT_INSTRUMENT : str
        Default instrument name for the DataModel.
    """

    EXTENSIONS = {
        'PRIMARY' : {'hdu_type' : fits.PrimaryHDU, 'dtype' : None},
        **RECEIPT,
        **CALIBRATIONS,
        **DATA_ANCESTORS,
    }

    _ALLOWED_ATTRIBUTES = [
        'meta', 'config', 'EXTENSIONS',
    ]

    _INHERIT_EXTENSIONS = True

    _DEFAULT_INSTRUMENT = 'HISPEC'

    def __init__(
        self,
        filepath : str | None = None,
        meta : MetaModel | dict | None = None,
        meta_only : bool = False,
        extensions : str | list[str] | None = None,
        correct_dtypes : bool = False,
        config : type | None = None,
        validate_meta : bool = True,
        instrument_name : str | None = None,
        **kwargs
    ):
        """
        Initialize a DataModel object.

        Parameters
        ----------
        filepath : str, optional
            Path to a FITS file to load.
        data : dict, optional
            Dictionary containing the data (numpy arrays or astropy tables). Keys should match the schema's extension names.
        meta : MetaModel | dict | None, optional
            Metadata as a MetaModel instance or dictionary. Keys should match the schema's property paths.
        meta_only : bool, optional
            If True, only metadata is loaded and no data is read from the FITS file. Defaults to False.
        extensions : str, list[str], optional
            List of extensions to load from the FITS file. Defaults to all extensions.
        correct_dtypes : bool, optional
            Whether to correct the data types of the data arrays upon reading them in. Defaults to False.
        config : type | None, optional
            The instrument config class. If None, it will be determined based on the filepath and instrument name.
        validate_meta : bool, optional
            Whether to validate the metadata against the schema. Defaults to True.
        kwargs
            Any valid data attributes.
        """

        # Resolve instrument config and name
        if config is not None:
            self.config = config
        else:
            self.config = self._get_instrument_config(
                meta=meta,
                filepath=filepath,
                instrument_name=instrument_name,
            )

        # Validate metadata or not
        self._validate_meta = validate_meta

        # Get the extension names now that we know the instrument
        self.EXTENSIONS = self._get_extensions()

        # Set the optional filepath
        if filepath is not None:
            if not os.path.isfile(filepath):
                raise FileNotFoundError(f"File not found: {filepath}")
            self._filepath = os.path.abspath(filepath)
            self.init_meta(meta=meta)
            self.meta.filename = os.path.basename(self._filepath)
            if not meta_only:
                self.read_data(extensions=extensions, correct_dtypes=correct_dtypes)
        else:
            self._filepath = None
            self.init_meta(meta=meta)
            if not meta_only:
                self.init_data(correct_dtypes=correct_dtypes, **kwargs)

        # Hook
        self.on_init()

    def _get_extensions(self):
        return self.EXTENSIONS

    @classmethod
    def _get_instrument_config(
        cls,
        meta : MetaModel | None,
        filepath : str | None,
        instrument_name : str | None = None,
    ):
        
        # Instrument name from filepath
        if instrument_name is not None:
            config = get_instrument_config(instrument_name=instrument_name)
            return config
        
        # Instrument name from provided metadata
        if isinstance(meta, (MetaModel, dict)):
            instrument_name = meta.get('instrument_name')
            if instrument_name is not None:
                config = get_instrument_config(instrument_name=instrument_name)
                return config
        
        # Instrument name from FITS header
        if instrument_name is None and isinstance(filepath, str):
            config = get_instrument_config(filepath)

        # Instrument name from environment variable
        if instrument_name is None:
            instrument_name = os.getenv('HISPECDRP_INSTRUMENT_NAME')
        
        # Log error
        if instrument_name is None:
            instrument_name = cls._DEFAULT_INSTRUMENT

        # Get config for this instrument
        config = get_instrument_config(instrument_name=instrument_name)

        return config
    
    def read_data(
        self,
        extensions : str | list[str] | None = None,
        do_not_scale_image_data : bool | None = None,
        correct_dtypes : bool = False
    ) -> dict[str, _VALID_HDU]:
        """
        Read the data from the FITS file.

        Parameters
        ----------
        extensions : str or list of str, optional
            List of extensions to read. Defaults to None.
        do_not_scale_image_data : bool or None, optional
            Whether to not scale data according to BZERO and BSCALE. See astropy behavior.
        correct_dtypes : bool, optional
            Whether to correct the data types of each data format to the specified data format.

        """
        extensions = self._resolve_extensions(extensions)
        with fits.open(
            self._filepath,
            memmap=False,
            do_not_scale_image_data=do_not_scale_image_data
        ) as hdulist:
            for extname in extensions:
                extl = extname.lower()
                if extname in hdulist:
                    data = self.read_extension_data(
                        hdulist[extname],
                        correct_dtype=correct_dtypes
                    )
                    setattr(self, extl, data)
                elif extname not in self.EXTENSIONS:
                    raise ValueError(f"Extension {extname} not found in FITS file: {self._filepath}")

        # Initialize remaining extensions not set by user or FITS file
        for extname in self.EXTENSIONS:
            extl = extname.lower()
            if getattr(self, extl, None) is None:
                if 'columns' in self.EXTENSIONS[extname]:
                    setattr(self, extl, astropy.table.Table({
                        name: np.array([], dtype=dtype)
                        for name, dtype in self.EXTENSIONS[extname]['columns'].items()
                    }))
                else:
                    setattr(self, extl, None) # Keep Image HDUs as None
    
    def read_extension_data(
        self, hdu : _VALID_HDU,
        correct_dtype : bool = False
    ) -> np.ndarray | astropy.table.Table:
        """
        Read data from a specific HDU.

        Parameters
        ----------
        hdu : _VALID_HDU
            The HDU to read data from.
        correct_dtype : bool, optional
            Whether to correct the data type of the data. Defaults to False.

        Returns
        -------
        data : np.ndarray or astropy.table.Table
            If an ImageHDU, returns a numpy ndarray.
            If a BinTableHDU, returns an astropy Table.
        """
        hdu_type = type(hdu)
        data = hdu.data
        if hdu_type is fits.ImageHDU:
            dtype = self.EXTENSIONS[hdu.name]['dtype']
            if data is not None:
                if not np.issubdtype(data.dtype, dtype):
                    if correct_dtype:
                        logger.warning(f"Converting data type of {hdu.name} from {hdu.data.dtype} to {dtype}")
                        data = data.astype(dtype, copy=False)
                data = convert_endianness(data, output_endianness='little')
        elif hdu_type is fits.BinTableHDU:
            data = astropy.table.Table(data)
        return data

    def save(
        self,
        filepath : str | None = None,
        filename : str | None = None,
        output_dir : str | None = None,
        suffix : str | None = None,
        include_missing_metadata : bool = False,
        include_empty_extensions : bool = False,
        correct_dtypes : bool = False,
        **kwargs
    ) -> str:
        """
        Save the DataModel to a FITS file.

        Parameters
        ----------
        filepath : str, optional
            The filepath to save to. Defaults to self._filepath.
        filename : str, optional
            The filename to save to. Defaults to None.
        output_dir : str, optional
            The directory to save to. Defaults to None.
        suffix : str, optional
            An optional suffix to append to the filename. Defaults to None.
        include_missing_metadata : bool, optional
            Whether to add missing metadata to the header. Defaults to False.
        include_empty_extensions : bool, optional
            Whether to include empty extensions in the output. Defaults to False.
        correct_dtypes : bool, optional
            Whether to correct the data types of the data arrays. Defaults to False.
        output_verify : str, optional
        **kwargs
            Additional keyword arguments to pass to the fits.HDUList.writeto method.

        """
        if filepath is not None:
            self._filepath = filepath
        else:
            self._filepath = self.get_output_path(
                filename=filename,
                output_dir=output_dir,
                suffix=suffix
            )
        self.on_save()
        hdulist = self.to_hdulist(
            include_empty_extensions=include_empty_extensions,
            include_missing_metadata=include_missing_metadata,
            correct_dtypes=correct_dtypes,
        )
        self._save_fits(hdulist, **kwargs)
        return self._filepath
    
    def get_output_path(
        self,
        filename : str | None = None,
        output_dir : str | None = None,
        suffix : str | None = None,
    ) -> str:
        if output_dir is not None and filename is not None:
            filepath = os.path.join(output_dir, filename)
        elif output_dir is not None and filename is None:
            filename = self.generate_filename(suffix=suffix)
            filepath = os.path.abspath(os.path.join(output_dir, filename))
        elif output_dir is None and filename is not None:
            raise ValueError("output_dir must be specified if filename is specified")
            #filepath = os.path.abspath(os.path.join(os.getcwd(), filename))
        else:
            raise ValueError("Either output_dir or filename must be specified")
            #filename = self.generate_filename(suffix=suffix)
            #filepath = os.path.abspath(os.path.join(os.getcwd(), filename))
        return filepath

    def to_hdulist(
        self,
        include_empty_extensions : bool = False,
        include_missing_metadata : bool = False,
        correct_dtypes : bool = False,
    ) -> str:
        """
        Write the DataModel to a FITS file.
        
        Parameters
        ----------
        include_empty_extensions : bool, optional
            Whether to include empty extensions in the output. Defaults to False.
        include_missing_metadata : bool, optional
            Whether to add missing metadata to the header. Defaults to False.
        correct_dtypes : bool, optional
            Whether to correct the data types of the data arrays. Defaults to False.

        Returns
        -------
        str
            The filepath the DataModel was saved to.
        """
        hdulist = fits.HDUList()
        extensions_list = list(self.EXTENSIONS.keys())
        extensions_list = ['PRIMARY'] \
            + [e for e in self.EXTENSIONS.keys() if e not in {'PRIMARY', 'RECEIPT', 'DATA_ANCESTORS', 'CALIBRATIONS'}] \
            + ['RECEIPT', 'DATA_ANCESTORS', 'CALIBRATIONS']
        for extname in extensions_list:
            if not include_empty_extensions \
                and getattr(self, extname.lower(), None) is None \
                and extname not in ('RECEIPT', 'PRIMARY'):
                continue
            hdu = self.make_hdu(
                extname,
                include_missing_metadata=include_missing_metadata,
                correct_dtypes=correct_dtypes
            )
            hdulist.append(hdu)
        
        return hdulist
    
    def _save_fits(
        self,
        hdulist : fits.HDUList,
        **kwargs
    ):
        if self._filepath is None:
            raise ValueError("No filepath specified. Set self._filepath to save the DataModel.")
        kwargs.setdefault('overwrite', True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', VerifyWarning)
            hdulist.writeto(
                self._filepath,
                **kwargs
            )

    def on_init(self):
        """
        Hook called after the DataModel is initialized.
        """
        self.meta.model_type = self.__class__.__name__
        if self._filepath is not None:
            self.meta.filename = os.path.basename(self._filepath)
        self.meta.drp_version = __version__

        _set_datetime_meta(self)
        _set_koa_meta(self)

        if self.meta.last_processed is None:
            self.meta.last_processed = self.meta.datetime_obs

        # if self.meta.filename is None:
        #     self.generate_filename()

    def on_save(self):
        """
        Hook called before a datamodel is saved. The default implementation:
            1. Updates `self.meta.filename` with base filename.
        """
        if self._filepath is None:
            raise ValueError("No filepath specified. Set self._filepath to save the DataModel.")
        self.meta.filename = os.path.basename(self._filepath)
        _set_koa_meta(self)
        self.meta.model_type = self.__class__.__name__

    def make_hdu(
        self,
        extname : str,
        include_missing_metadata : bool = False,
        correct_dtypes : bool = False
    ) -> _VALID_HDU:
        """
        Create an HDU for an extension name.
        
        Parameters
        ----------
        extname : str
            The extension name.
        include_missing_metadata : bool, optional
            Whether to add missing metadata to the header. Defaults to False.
        correct_dtypes : bool, optional
            Whether to correct the data types of the data arrays. Defaults to False.
        """
        extl = extname.lower()
        hdu_class = self.EXTENSIONS[extname]['hdu_type']
        header = self.meta.to_fits_header(
            extname=extname,
            include_missing_metadata=include_missing_metadata
        )

        # Primary HDU is empty, only header
        if hdu_class is fits.PrimaryHDU:
            hdu = hdu_class(header=header)
        
        # Convert Image HDU data to correct dtype if needed
        elif hdu_class is fits.ImageHDU:
            data = getattr(self, extl, None)
            if data is not None and correct_dtypes:
                dtype = self.EXTENSIONS[extname]['dtype']
                data = data.astype(dtype)
            hdu = hdu_class(data=data, name=extname)
        
        # Nothing special for BIN tables
        elif hdu_class is fits.BinTableHDU:
            data = getattr(self, extl, None)
            hdu = hdu_class(data=data, name=extname)

        return hdu

    def __repr__(self):
        if self.meta.filename is not None:
            return f"{self.__class__.__name__}('{self.meta.filename}')"
        else:
            return f"{self.__class__.__name__}()"
    
    def __init_subclass__(cls):
        subclasses = cls.mro()[::-1]
        for _cls in subclasses:
            if hasattr(_cls, "EXTENSIONS") and cls._INHERIT_EXTENSIONS:
                for extname in _cls.EXTENSIONS:
                    if extname not in cls.EXTENSIONS:
                        cls.EXTENSIONS[extname] = _cls.EXTENSIONS[extname]
            if 'PRIMARY' in cls.EXTENSIONS:
                cls.EXTENSIONS = {'PRIMARY': cls.EXTENSIONS['PRIMARY'], **{k: v for k, v in cls.EXTENSIONS.items() if k != 'PRIMARY'}}
            if 'RECEIPT' in cls.EXTENSIONS:
                cls.EXTENSIONS = {**{k: v for k, v in cls.EXTENSIONS.items() if k != 'RECEIPT'}, 'RECEIPT': cls.EXTENSIONS['RECEIPT']}

    def copy(self) -> Self:
        """
        Create a deep copy of the DataModel object.
        """
        model_out = deepcopy(self)
        model_out._filepath = None
        if model_out.meta.cal_type is not None:
            model_out.meta.cal_id = None
        return model_out
    
    def __deepcopy__(self, memo) -> Self:
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        for k, v in self.__dict__.items():
            # Skip things that shouldn't be deepcopied
            if isinstance(v, (type, type(deepcopy), type(None))):
                setattr(result, k, v)
            else:
                setattr(result, k, deepcopy(v, memo))

        return result

    def _resolve_extensions(self, extensions : str | list[str] | None = None):
        if extensions is None:
            return list(self.EXTENSIONS.keys())
        if isinstance(extensions, str):
            return [extensions.upper()]
        else:
            return [e.upper() for e in extensions]
        
    def __getattr__(self, attr):
        attrl = attr.lower()

        if attrl in self.__dict__:
            return self.__dict__[attrl]

        # Or try super's getattr if the parent class has it
        try:
            return super().__getattribute__(attrl)
        except AttributeError:
            pass

        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")

    def __setattr__(self, attr, value):
        if attr.startswith("_") or attr in self._ALLOWED_ATTRIBUTES:
            return super().__setattr__(attr, value)

        attru = attr.upper()
        if attru in self.EXTENSIONS:
            attrl = attru.lower()
            return super().__setattr__(attrl, value)

        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{attr}' "
            "or is not allowed to be set directly"
        )

    def init_data(
        self,
        correct_dtypes : bool = False,
        **kwargs
    ):
        """
        Initialize the data.

        Parameters
        ----------
        correct_dtypes : bool, optional
            Whether to correct the data types of the data arrays upon initializing them. Defaults to False.
        kwargs
            Any valid data attributes to set.
        """

        # User input data via kwargs
        for key in kwargs:
            keyu = key.upper()
            keyl = key.lower()
            val = kwargs[key]
            if keyu in self.EXTENSIONS:
                if isinstance(val, astropy.table.Table):
                    setattr(self, keyl, val)
                elif isinstance(val, np.ndarray):
                    if correct_dtypes:
                        setattr(self, keyl, val.astype(self.EXTENSIONS[keyu]['dtype']))
                    else:
                        setattr(self, keyl, val)
            else:
                raise ValueError(f"Unknown extension: {key.upper()}")

        # Initialize remaining extensions
        for extname in self.EXTENSIONS:
            extl = extname.lower()
            if getattr(self, extl, None) is None:
                if 'columns' in self.EXTENSIONS[extname]:
                    setattr(self, extl, astropy.table.Table({
                        name: np.array([], dtype=dtype)
                        for name, dtype in self.EXTENSIONS[extname]['columns'].items()
                    }))
                else:
                    setattr(self, extl, None)

    def init_meta(self, meta: dict | MetaModel | None = None):
        """
        Initialize the metadata and optionally merge in provided metadata.

        Parameters
        ----------
        meta : dict or MetaModel, optional
            Existing metadata to store or merge.
        """

        # Get meta schema
        meta_schema = get_meta_schema(config=self.config)

        # Meta from file headers
        if self._filepath is None:
            meta_from_file = {}
        else:
            meta_from_file = self.read_meta_from_fits_headers(schema=meta_schema)

        # Set initial meta from user input
        if isinstance(meta, MetaModel):
            self.meta = meta

        # Case 2: dict or None -> build MetaModel
        else:
            if meta is not None and not isinstance(meta, dict):
                raise ValueError(f"Unknown type for meta: {type(meta)}")

            merged = dict(meta_from_file)
            if meta:
                merged |= meta

            self.meta = MetaModel(meta_schema, merged)

        # Apply instrument metadata defaults
        self.config.set_meta_defaults(self)

    def read_meta_from_fits_headers(self, schema : dict) -> dict:
        """
        Read metadata from the FITS headers.

        Returns
        -------
        meta_dict : dict
            Dictionary of metadata read from the FITS headers.
            Keys are property names, values are the corresponding header values.
        """
        if schema is None:
            schema = self.meta._schema
        meta_dict = {}
        with fits.open(self._filepath, memmap=True) as hdulist:
            for prop_name in schema['properties']:
                sub_schema = schema['properties'][prop_name]
                fits_hdu = sub_schema['fits_hdu']
                fits_keyword = sub_schema['fits_keyword']
                header = hdulist[fits_hdu].header
                if fits_keyword in header:
                    meta_dict[prop_name] = header[fits_keyword]
        return meta_dict

    def _extend_data_ancestors_columns(self, new_columns : list[str]):
        def _json_schema_to_python_dtype(t : str):
            if t == 'string':
                return np.str_
            elif t == 'integer':
                return np.int64
            elif t == 'number':
                return np.float64
            elif t == 'boolean':
                return bool
            else:
                raise ValueError(f"Unsupported JSON schema type: {t}")

        for col in new_columns:
            col = col.upper()
            if col not in self.data_ancestors.colnames:
                # Add new column with object dtype to support None
                self.data_ancestors.add_column(astropy.table.Column(
                    name=col,
                    data=[None] * len(self.data_ancestors),
                    dtype=_json_schema_to_python_dtype(
                        self.meta._schema['properties'][col.lower()]['type']
                    )
                ))

    @property
    def extinfo(self):
        for extname, info in self.EXTENSIONS.items():
            extl = extname.lower()
            hdu_type = info['hdu_type']
            if hdu_type is fits.PrimaryHDU:
                continue
            if hdu_type is fits.ImageHDU:
                data = getattr(self, extl, None)
                if data is not None:
                    dtype = data.dtype
                    print(f"{extname}\n  dtype: {dtype}\n  shape: {data.shape}")
            elif hdu_type is fits.BinTableHDU:
                data = getattr(self, extl, None)
                if data is not None:
                    print(f"Extension: {extname}\n  columns: {list(data.columns)}\n  nrows: {len(data)})")

    @property
    def data_dict(self):
        data = {}
        for extname, info in self.EXTENSIONS.items():
            extl = extname.lower()
            hdu_type = info['hdu_type']
            if hdu_type is fits.PrimaryHDU:
                continue
            data[extl] = getattr(self, extl, None)
        return data

    @property
    def data_level_suffix(self):
        return None

    @classmethod
    def from_datamodels(
        cls,
        input : Any,
        append_receipts: bool = True,
        append_ancestors: bool = True,
        append_calibrations: bool = True,
        include_data : bool = False,
    ) -> 'DataModel':
        """
        Create a new DataModel by merging metadata from multiple input models.

        Parameters
        ----------
        input : DataModel | str | list[DataModel | str] | dict[str, DataModel | str]
            Input DataModel(s) or file path(s) to merge.
        append_receipts : bool, optional
            Whether to append receipts from input models. Defaults to True.
        append_ancestors : bool, optional
            Whether to append data ancestors from input models. Defaults to True.
        append_calibrations : bool, optional
            Whether to append calibrations from input models. Defaults to True.
        include_data : bool, optional
            Whether to include data from the input model (must only be one). Defaults to False.

        Returns
        -------
        DataModel
            A new DataModel with merged metadata and optional data.
        """

        if input is None:
            raise ValueError("No models provided")
        
        # Normalize to iterable
        if isinstance(input, (str, DataModel)):
            input = [input]
        elif isinstance(input, dict):
            input = [
                item
                for v in input.values()
                for item in (v if isinstance(v, list) else [v])
            ]
        elif not isinstance(input, Sequence):
            raise ValueError("Input must be a DataModel, filepath, list of DataModels/filepaths, or dict of DataModels/filepaths.")
        
        # Sort inputs by datetime_obs
        input = sorted(input, key=lambda x: datamodel_open(x, meta_only=True).meta.datetime_obs)

        # Merge metadata
        meta_out = merge_meta(input)

        # Get first model
        first_model = datamodel_open(input[0])

        # Create new model
        model_out = cls(
            meta=meta_out,
            instrument_name=meta_out.instrument_name,
            config=first_model.config,
        )

        # Forward data?
        if include_data:
            assert len(input) == 1, "Cannot include data when merging multiple models. Set include_data=False or provide a single model."
            first_model = datamodel_open(input[0])
            for extname in model_out.EXTENSIONS:
                if extname in ('PRIMARY', 'DATA_ANCESTORS', 'CALIBRATIONS', 'RECEIPT'):
                    continue
                extl = extname.lower()
                if extname in first_model.EXTENSIONS:
                    setattr(model_out, extl, getattr(first_model, extl, None))

        for _input in input:

            input_model = datamodel_open(_input)

            if append_receipts:
                model_out.append_receipts(input_model)

            if append_ancestors:
                model_out.append_ancestors(input_model)

            if append_calibrations:
                model_out.append_calibrations(input_model)

        return model_out
    
    def make_order_vec(self) -> np.ndarray:
        return self._make_order_vec(self.meta.order_start, self.meta.order_end)
        
    @staticmethod
    def _make_order_vec(order_start : int, order_end : int) -> np.ndarray:
        """
        Create an order vector based on the order start and end metadata.
        If order_end < order_start, the vector will be in descending order.

        Parameters
        ----------
        order_start : int
            The starting order number.
        order_end : int
            The ending order number.
        
        Returns
        -------
        np.ndarray
            An array of integers from order_start to order_end.
        """
        step = 1 if order_end >= order_start else -1
        return np.arange(order_start, order_end + step, step).astype(np.int16)

    @property
    def traces(self):
        """
        Convenience property to return a dictionary of traces for each fiber in the model.
        This is useful when the fiber is a dynamic variable and can be used in place of getattr(model, '<attr>_fib{fiber}').
        By default, any attribute with '_fib' in its name is included.
        This property may be removed in future versions unless it demonstrates significant utility.
        This property should not be used in large for loops if the rest of the computation is trivial as it adds some overhead.

        Examples:
            >>> model.traces # -> {1 : {'wave' : ..., 'spec' : ..., 'spec_err' : ...}, 2 : {...}, ...}
            >>> model.traces[1]['wave'] # Equivalent to model.wave_fib1

        Returns:
            dict: A dictionary where keys are fiber numbers and values are dictionaries of attributes without the '_fib#' suffix.
        """
        trace_data = {}
        for attr in self.EXTENSIONS:
            attrl = attr.lower()
            if '_fib' in attrl:
                try:
                    kind, fiber = attrl.rsplit('_fib', 1)
                    fiber = int(fiber)
                except ValueError:
                    continue
                trace_data.setdefault(fiber, {})
                trace_data[fiber][kind] = getattr(self, attr, None)
        return trace_data

    def load_calibration_from_role(
        self,
        cal_role : str,
    ) -> DataModel:
        """
        Load a calibration from the DataModel's calibrations table.

        Parameters
        ----------
        cal_role : str
            The role of the calibration to load.

        Returns
        -------
        dict
            The calibration record.
        """
        row = self.get_calibration_from_role(cal_role)
        cal_id = row['ID']
        with CalibrationStore(instrument_name=self.meta.instrument_name) as store:
            return store.get_calibration(calibration=cal_id)

    def get_calibration_from_role(self, cal_role : str) -> dict:
        """
        Return the row from the DataModel's calibrations table.

        Parameters
        ----------
        cal_role : str
            The role of the calibration to load.

        Returns
        -------
        dict
            The row from the DataModel's calibrations table.
        """
        # Match on cal_role
        mask = self.calibrations['CAL_ROLE'] == cal_role
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return None
        if len(indices) > 1:
            raise ValueError(f"Multiple calibrations found with role '{cal_role}' in {self.__class__.__name__}")
        # ! This can happen since we append calibration tables when say coadding data products.
        # TODO: Check if they are unique, raise warning if not?
        row_index = indices[0]
        return dict(self.calibrations[row_index])

    def update_calibration_table(self, calibration : CalibrationModel | str, note : str | None = None):
        cal_model = datamodel_open(calibration, meta_only=True)
        cal_orm = cal_model.to_orm(note=note)
        row = {}
        for cal_column in self.calibrations.colnames:
            if cal_column in cal_orm.__table__.columns:
                val = getattr(cal_orm, cal_column)
                if val is None:
                    val = ''
                row[cal_column] = val
        self.calibrations.add_row(row)

    def generate_filename(self, suffix: str | None = None) -> str:
        """
        Generate a filename for the datamodel.
        Format: {instrument_alias}.{YYYYMMDD}.{SSSSS.SS}_{suffix}.fits

            - Instrument aliases:
                
                - HISPEC (BSPEC): HB
                - HISPEC (RSPEC): HR
                - PARVI: PR

            - SSSSS.SS is the seconds of the day with two decimal places, calculated from the datetime_obs metadata attribute.

            - ``suffix`` is an optional string that can be appended to the end of the filename, separated by an underscore. This can be used to further specify any processing steps or other special information about the datamodel. The provided suffix suffix will automatically be appended to datamodel's ``data_level_suffix`` attribute with a hyphen. Ex: ``suffix='darksub'`` would produce a suffix 'rate-darksub' for a level 0 model.

        Stores the generated filename in `self.meta.filename`.
        Sets `self._filepath` to None since we do not know the 

        Parameters
        ----------
        suffix : str, optional
            An optional suffix to append to the filename. Defaults to None.
        """
        datetime_obs = self.meta.datetime_obs

        # Determine instrument alias
        instrument_alias = self.config.get_instrument_alias(self)
        assert instrument_alias is not None, \
        f"Instrument alias not found for instrument '{self.instrument_name}' and spectrograph '{self.meta.spectrograph}'."

        # Determine suffix
        if suffix is not None:
            suffix = f"{self.data_level_suffix}-{suffix}"
        else:
            suffix = self.data_level_suffix

        filename = self._generate_filename(
            instrument_alias=instrument_alias,
            datetime_obs=datetime_obs,
            suffix=suffix
        )

        self.meta.filename = filename

        if self._filepath is not None:
            input_dir = os.path.dirname(os.path.abspath(self._filepath))
            self._filepath = os.path.join(input_dir, self.meta.filename)

        return self.meta.filename

    @staticmethod
    def _generate_filename(
        instrument_alias : str,
        datetime_obs : str,
        suffix : str,
    ) -> str:
        """
        Generate a filename for a non-calibration datamodel.
        This method should not be called by the user. Instead, call self.generate_filename which will determine the instrument alias and suffix automatically based on the datamodel's metadata and config.
        Format: {instrument_alias}.{YYYYMMDD}.{SSSSS.SS}_{suffix}.fits

        Examples
        --------
        - HISPEC (BSPEC):
            'HB.20240924.00000.00_rate-darksub.fits'
        
        - HISPEC (RSPEC):
            'HR.20240924.00000.00_rate.fits'

        - PARVI:
            'PR.20240924.00000.00_rate.fits'

        Parameters
        ----------
        instrument_alias : str
            The two letter instrument alias.
        datetime_obs : str
            The ISO date and time of the observation.
        suffix : str
            The suffix to append to the filename. Required.

        Returns
        -------
        str
            The filename
        """
        datetime_str = get_koa_id_timestamp_from_datetime(datetime_obs)
        filename = f"{instrument_alias}.{datetime_str}.fits"
        if suffix:
            filename = "".join([filename[0:-5], "_", suffix, ".fits"])
        return filename
    
    @property
    def num_fibers(self) -> int:
        return self.config.num_fibers
    
    @property
    def instrument_name(self) -> str:
        return self.config.instrument_name
    
    def append_receipts(self, other : DataModel | list[DataModel] | dict[DataModel]) -> None:
        if isinstance(other, DataModel):
            others = [other]
        elif isinstance(other, dict):
            others = list(other.values())
        elif isinstance(other, list):
            others = other
        else:
            raise ValueError("Invalid type for other")
        for other_model in others:
            for row in other_model.receipt:
                self.receipt.add_row(row)

    def append_calibrations(self, other : DataModel | list[DataModel] | dict[DataModel]) -> None:
        if isinstance(other, DataModel):
            others = [other]
        elif isinstance(other, dict):
            others = list(other.values())
        elif isinstance(other, list):
            others = other
        else:
            raise ValueError("Invalid type for other")
        for other_model in others:
            for row in other_model.calibrations:
                self.calibrations.add_row(row)

    def append_ancestors(self, other : DataModel | list[DataModel] | dict[DataModel]) -> None:
        if isinstance(other, DataModel):
            others = [other]
        elif isinstance(other, dict):
            others = list(other.values())
        elif isinstance(other, list):
            others = other
        else:
            raise ValueError("Invalid type for other")
            
        for other_model in others:
            # Check for new columns to add to self
            new_cols = [c for c in other_model.data_ancestors.colnames if c not in self.data_ancestors.colnames]
            if new_cols:
                self._extend_data_ancestors_columns(new_cols)
                
            for row in other_model.data_ancestors:
                # astropy.table.row -> dict to handle column mapping
                row_dict = {c: row[c] for c in other_model.data_ancestors.colnames}
                self.data_ancestors.add_row(row_dict)


class CalibrationModel(DataModel):

    CAL_TYPE : ClassVar[str] = None # Type of calibration

    def on_init(self):
        self.meta.cal_type = self.CAL_TYPE
        if self.meta.cal_id is None:
            self.generate_calibration_id()
        super().on_init()

    def generate_calibration_id(self) -> str:
        """
        Generate the calibration ID which is reproducible UUID 7.

        Returns
        -------
        cal_id : str
            The calibration ID
        """
        self.meta.cal_id = str(make_uuid4())
        return self.meta.cal_id
    
    def generate_calibration_version(self, origin : str | None = None) -> str:
        if origin is not None:
            self.meta.origin = origin
        if self.meta.origin is None:
            self.meta.origin = os.getenv('KOA_CALIBRATION_ORIGIN', 'LOCAL')
        with HISPECCalibrationStore(
            instrument_name=self.meta.instrument_name,
            connect_remote=False
        ) as store:
            self.meta.cal_version = store.generate_calibration_version(
                self, origin=self.meta.origin
            )
        return self.meta.cal_version

    def generate_filename(
        self,
        suffix: str | None = None,
        origin : str | None = None,
        auto_version : bool = None,
    ) -> str:
        """

        Format: {instrument_alias}.{YYYYMMDD}.{SSSSS.SS}.[master-]{cal_type}.{cal_version}[_{suffix}].fits
            datetime_obs : YYYYMMDD.SSSSSS.SS. Where SSSSSS.SS is the number of seconds since midnight.
            cal_type : The type of calibration data.
            master_cal : Insert 'master-' before the cal_type if `master_cal=True`.
            cal_version : The version of the calibration file. If None, queries the local cache for the latest version if ``auto_version=True``.
            suffix : optional suffix

        Parameters
        ----------
        suffix : str, optional
            An optional suffix to append to the filename. Defaults to None.
        origin : str, optional
            The origin of the calibration, either 'LOCAL' or 'KECK'.
        auto_version : bool, optional
            If True, generate a new calibration version.

        Returns
        -------
        filename : str
            The generated filename.
        
        """
        
        # Determine instrument alias
        instrument_alias = self.config.get_instrument_alias(self)
        assert instrument_alias is not None, \
        f"Instrument alias not found for instrument '{self.instrument_name}' and spectrograph '{self.meta.spectrograph}.'"

        # Determine master_cal
        master_cal = getattr(self.meta, "master_cal")

        skip_version = get_env_var_bool('KOA_SKIP_CALIBRATION_VERSION', default=False)
        cache_defined = isinstance(os.getenv('KOA_CALIBRATION_CACHE'), str)
        if not skip_version and cache_defined and auto_version is not False:
            if self.meta.cal_version is None and auto_version:
                self.generate_calibration_version(origin=origin)

        filename = self._generate_filename(
            instrument_alias=instrument_alias,
            datetime_obs=self.meta.datetime_obs,
            cal_type=self.CAL_TYPE,
            master_cal=master_cal,
            cal_version=self.meta.cal_version,
            origin=self.meta.origin,
            suffix=suffix
        )

        self.meta.filename = filename

        if self._filepath is not None:
            input_dir = os.path.dirname(os.path.abspath(self._filepath))
            self._filepath = os.path.join(input_dir, self.meta.filename)

        return self.meta.filename

    @staticmethod
    def _generate_filename(
        instrument_alias : str,
        datetime_obs : str,
        cal_type : str,
        master_cal : bool = False,
        cal_version : str | None = None,
        origin : str | None = None,
        suffix : str | None = None,
    ) -> str:
        """
        Create a filename for a calibration data product.

        Format: {instrument_alias}.{YYYYMMDD}.{SSSSS.SS}.[master-]{cal_type}.{cal_version}[_{suffix}].fits
            datetime_obs : YYYYMMDD.SSSSSS.SS. Where SSSSSS.SS is the number of seconds since midnight.
            cal_type : The type of calibration data.
            master_cal : Insert 'master-' before the cal_type if `master_cal=True`.
            cal_version : The version of the calibration file. If None, queries the local cache for the latest version if ``auto_version=True``.
            suffix : optional suffix
        
        Examples
        --------
        'HB.20240924.00000.00.dark.001.fits'
        'HB.20240924.00000.00.dark.001-local.fits'
        'HB.20240924.00000.00.master-dark.001.fits'
        'HB.20240924.00000.00.master-dark.001_custom.fits'
        'HB.20240924.00000.00.master-dark.001-local_custom.fits'
        'HB.20240924.00000.00.dark.001_custom.fits'

        Parameters
        ----------
        datetime_obs : str
            The date and time of the observation.
        instrument_alias : str
            The two letter instrument alias.
        cal_type : str
            The type of calibration data.
        master_cal : bool, optional
            If True, insert 'master-' before the cal_type. Defaults to False.
        cal_version : str, optional
            The version of the calibration file. If None, queries the local cache.
        origin : str, optional
            The origin of the calibration file, either 'LOCAL' or 'KECK'. If None,
        suffix : str, optional
            An optional suffix to append to the filename. Defaults to None.

        Returns
        -------
        filename : str
            The generated filename.
        """

        # Convert date time string to KOA format
        datetime_str = get_koa_id_timestamp_from_datetime(datetime_obs)

        # Strings for master and origin
        if master_cal:
            cal_type = f"mastercal-{cal_type}"
        else:
            cal_type = f"cal-{cal_type}"

        # Construct starting filename
        filename = f"{instrument_alias}.{datetime_str}.{cal_type}"

        # Version
        if cal_version is not None:
            if isinstance(origin, str) and origin.lower() == 'local':
                cal_version = f"{cal_version}-local"
            filename = f"{filename}.{cal_version}"
        
        # Add suffix
        if suffix is not None:
            filename = f"{filename}_{suffix}"

        # File extension
        filename = f"{filename}.fits"

        return filename

    def to_record(self, note: str | None = None) -> dict:
        """
        Get the calibration record as a dictionary from the metadata.

        Parameters
        ----------
        note : str, optional
            An optional note to add to or update the record with.

        Returns
        -------
        dict
            The calibration record as a dictionary.
        """
        if note is not None:
            self.meta.note = note
        record = self.meta.to_record()
        return record