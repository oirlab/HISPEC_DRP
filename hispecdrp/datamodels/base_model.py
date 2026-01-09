from __future__ import annotations # NOTE: Not needed in Python 3.13+
import os
import astropy.table
from astropy.io import fits
from copy import deepcopy
from datetime import datetime

from .hispec_config import HISPECConfig
from .parvi_config import PARVIConfig
from .meta_data import MetaModel
from hispecdrp import __version__
import numpy as np
import warnings
from astropy.io.fits.verify import VerifyWarning
import uuid
from typing import Any, Sequence
from .utils import open as datamodel_open, merge_meta, get_config_from_hdulist
from ..utils import convert_endianness

from ..calibrations import get_calibration_orm_class, pydantic_field_to_sqlalchemy_column, get_orm_field_name

from .extensions import RECEIPT, CALIBRATIONS, DATA_ANCESTORS

from .dqflags import DQ_FLAGS

from koa_middleware import CalibrationORM
from koa_middleware import CalibrationStore


_VALID_HDU = fits.ImageHDU | fits.BinTableHDU | fits.PrimaryHDU

class DataModel:
    """
    Base class for data models.

    Class Attributes:
        EXTENSIONS (dict): Extensions and their properties.
        INHERIT_EXTENSIONS (bool): Whether to inherit extensions from parent class. Defaults to True.
        ALLOWED_ATTRIBUTES (list): List of attributes that can be set directly on the DataModel.
    """

    EXTENSIONS = {
        'PRIMARY' : {'hdu_type' : fits.PrimaryHDU, 'dtype' : None},
        **RECEIPT,
        **CALIBRATIONS,
        **DATA_ANCESTORS,
    }

    INHERIT_EXTENSIONS = True

    ALLOWED_ATTRIBUTES = [
        'EXTENSIONS', 'INHERIT_EXTENSIONS',
        'meta', '_data', '_filepath', 'ALLOWED_ATTRIBUTES',
        'DEFAULT_CONFIG', 'config',
    ]

    DEFAULT_CONFIG = HISPECConfig

    DQ_FLAGS = DQ_FLAGS

    def __init__(
        self,
        filepath : str | None = None,
        data : dict | None = None,
        meta : MetaModel | None = None,
        meta_only : bool = False,
        extensions : str | list[str] | None = None,
        correct_dtypes : bool = False,
        config : type | None = None,
    ):
        """
        Initialize a DataModel object.

        Parameters
        ----------
        filepath : str, optional
            Path to a FITS file to load.
        data : dict, optional
            Dictionary containing the data (numpy arrays or astropy tables). Keys should match the schema's extension names.
        meta : dict, optional
            Dictionary containing metadata. Keys should match the schema's property paths.
        meta_only : bool, optional
            If True, only metadata is loaded and no data is read from the FITS file. Defaults to False.
        extensions : str, list[str], optional
            List of extensions to load from the FITS file. Defaults to all extensions.
        correct_dtypes : bool, optional
            Whether to correct the data types of the data arrays upon reading them in. Defaults to False.
        config : Any, optional
            The instrument config class. If None, uses self.DEFAULT_CONFIG.
        """

        if config is not None:
            self.config = config
        else:
            if filepath is not None:
                with fits.open(filepath, memmap=True) as hdulist:
                    self.config = get_config_from_hdulist(hdulist)
                if self.config is None:
                    self.config = self.DEFAULT_CONFIG
            else:
                self.config = self.DEFAULT_CONFIG

        # Set the optional filepath
        if filepath is not None:
            if not os.path.isfile(filepath):
                raise FileNotFoundError(f"File not found: {filepath}")
            self._filepath = os.path.abspath(filepath)
            self.init_meta(meta=meta)
            if not meta_only:
                self.read_data(extensions=extensions, correct_dtypes=correct_dtypes)
            else:
                self._data = {}
        else:
            self._filepath = None
            self.init_meta(meta=meta)
            if not meta_only:
                self.init_data(data=data, correct_dtypes=correct_dtypes)
            else:
                self._data = {}

        # Hook
        self.on_init()
    
    def set_meta_from_headers(
        self,
        extensions : str | list[str] | None = None,
    ) -> dict[str, fits.Header]:
        """
        Set the metadata from the headers of the FITS file.
        
        Args:
            extensions (str, list[str], optional): List of extensions to read. Defaults to None.
        """
        extensions = self._resolve_extensions(extensions)
        with fits.open(self._filepath, memmap=True) as hdulist:
            for attr, field in self.meta.__class__.model_fields.items():
                fits_hdu = field.json_schema_extra['fits_hdu']
                fits_keyword = field.json_schema_extra['fits_keyword']
                header = hdulist[fits_hdu].header
                if fits_keyword in header:
                    self.meta[attr] = header[fits_keyword]
        return self.meta
    
    def read_data(
        self,
        extensions : str | list[str] | None = None,
        do_not_scale_image_data : bool | None = None,
        correct_dtypes : bool = False
    ) -> dict[str, _VALID_HDU]:
        """
        Read the data from the FITS file.

        Args:
            extensions (str, list[str], optional): List of extensions to read. Defaults to None.
            do_not_scale_image_data (bool | None, optional): Whether to not scale data according to BZERO and BSCALE. See astropy behavior.
            correct_dtypes (bool, optional): Whether to correct the data types of each data format to the specified data format.
        """
        extensions = self._resolve_extensions(extensions)
        self._data = {}
        with fits.open(
            self._filepath,
            memmap=False,
            do_not_scale_image_data=do_not_scale_image_data
        ) as hdulist:
            for extname in extensions:
                if extname in hdulist:
                    self._data[extname] = self.read_extension_data(hdulist[extname], correct_dtype=correct_dtypes)
        return self._data
    
    def read_extension_data(self, hdu : _VALID_HDU, correct_dtype : bool = False) -> np.ndarray | astropy.table.Table:
        hdu_type = type(hdu)
        data = hdu.data
        if hdu_type is fits.ImageHDU:
            dtype = self.EXTENSIONS[hdu.name]['dtype']
            if data is not None:
                if not np.issubdtype(data.dtype, dtype):
                    if correct_dtype:
                        warnings.warn(f"Converting data type of {hdu.name} from {hdu.data.dtype} to {dtype}")
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
        add_missing_metadata : bool = False,
        include_empty_extensions : bool = False,
        correct_dtypes : bool = False,
        **kwargs
    ) -> str:
        """
        Save the DataModel to a FITS file.

        Args:
            filepath (str, optional): The filepath to save to. Defaults to self._filepath.
            filename (str, optional): The filename to save to. Defaults to None.
            output_dir (str, optional): The directory to save to. Defaults to None.
            suffix (str, optional): An optional suffix to append to the filename. Defaults to None.
            add_missing_metadata (bool, optional): Whether to add missing metadata to the header. Defaults to False.
            include_empty_extensions (bool, optional): Whether to include empty extensions in the output. Defaults to False.
            correct_dtypes (bool, optional): Whether to correct the data types of the data arrays. Defaults to False.
            kwargs: Additional keyword arguments to pass to the fits.HDUList.writeto method.
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
        return self.to_fits(
            include_empty_extensions=include_empty_extensions,
            add_missing_metadata=add_missing_metadata,
            correct_dtypes=correct_dtypes,
            **kwargs
        )
    
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
            filepath = os.path.abspath(os.path.join(os.getcwd(), filename))
        else:
            filename = self.generate_filename(suffix=suffix)
            filepath = os.path.abspath(os.path.join(os.getcwd(), filename))
        return filepath

    def to_fits(self, include_empty_extensions : bool = False, add_missing_metadata : bool = False, correct_dtypes : bool = True, output_verify : str = 'fix', **kwargs):
        """
        Write the DataModel to a FITS file.
        
        Args:
            kwargs: Additional keyword arguments to pass to the fits.HDUList.writeto method.
        """
        if 'overwrite' not in kwargs:
            kwargs['overwrite'] = True
        if self._filepath is None:
            raise ValueError("No filepath specified. Set self._filepath to save the DataModel.")
        hdulist = fits.HDUList()
        extensions_list = list(self.EXTENSIONS.keys())
        extensions_list = ['PRIMARY'] \
                        + [e for e in self.EXTENSIONS.keys() if e not in {'PRIMARY', 'RECEIPT', 'DATA_ANCESTORS', 'CALIBRATIONS'}] \
                        + ['RECEIPT', 'DATA_ANCESTORS', 'CALIBRATIONS']
        for extname in extensions_list:
            if not include_empty_extensions \
                and (extname not in self._data or self._data[extname] is None) \
                and extname not in ('RECEIPT', 'PRIMARY'):
                continue
            hdu = self.make_hdu(extname, add_missing_metadata=add_missing_metadata, correct_dtypes=correct_dtypes)
            hdulist.append(hdu)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', VerifyWarning)
            hdulist.writeto(self._filepath, output_verify=output_verify, **kwargs)

        return self._filepath

    def on_init(self):
        """
        Hook called after the DataModel is initialized.
        """
        self.meta.model_type = self.__class__.__name__
        if self._filepath is not None:
            self.meta.filename = os.path.basename(self._filepath)
        self.meta.drp_version = __version__

    def on_save(self):
        """
        Hook called before a datamodel is saved. The default implementation:
            1. Updates `self.meta.filename` with base filename.
        """
        if self._filepath is None:
            raise ValueError("No filepath specified. Set self._filepath to save the DataModel.")
        self.meta.filename = os.path.basename(self._filepath)
        if self.meta.last_updated is None:
            self.meta.last_updated = datetime.now().isoformat(timespec="milliseconds")
        self.meta.model_type = self.__class__.__name__

    def make_hdu(self, extname : str, add_missing_metadata : bool = False, correct_dtypes : bool = True) -> _VALID_HDU:
        """
        Create an HDU for an extension name.
        
        Args:
            extname (str): The extension name.
            add_missing_metadata (bool, optional): Whether to add missing metadata to the header. Defaults to False.
            correct_dtypes (bool, optional): Whether to correct the data types of the data arrays. Defaults to False.
        """
        hdu_class = self.EXTENSIONS[extname]['hdu_type']
        header = self.meta.to_fits_header(extname=extname, add_missing_metadata=add_missing_metadata)
        if hdu_class is fits.PrimaryHDU:
            return hdu_class(header=header)
        if hdu_class is fits.ImageHDU:
            data = self._data.get(extname, None)
            if data is not None and correct_dtypes:
                dtype = self.EXTENSIONS[extname]['dtype']
                _data = data.astype(dtype)
            else:
                _data = data
            hdu = hdu_class(data=_data, name=extname, header=header)
        elif hdu_class is fits.BinTableHDU:
            data = self._data.get(extname, None)
            hdu = hdu_class(data=data, name=extname, header=header)
        return hdu

    def __repr__(self):
        if self._filepath is not None:
            return f"{self.__class__.__name__}('{os.path.basename(self._filepath)}')"
        else:
            return f"{self.__class__.__name__}()"
    
    def __init_subclass__(cls):
        subclasses = cls.mro()[::-1]
        for _cls in subclasses:
            if hasattr(_cls, "EXTENSIONS") and cls.INHERIT_EXTENSIONS:
                for extname in _cls.EXTENSIONS:
                    if extname not in cls.EXTENSIONS:
                        cls.EXTENSIONS[extname] = _cls.EXTENSIONS[extname]
            if 'PRIMARY' in cls.EXTENSIONS:
                cls.EXTENSIONS = {'PRIMARY': cls.EXTENSIONS['PRIMARY'], **{k: v for k, v in cls.EXTENSIONS.items() if k != 'PRIMARY'}}
            if 'RECEIPT' in cls.EXTENSIONS:
                cls.EXTENSIONS = {**{k: v for k, v in cls.EXTENSIONS.items() if k != 'RECEIPT'}, 'RECEIPT': cls.EXTENSIONS['RECEIPT']}

    def copy(self):
        """
        Create a deep copy of the object.
        """
        model_out = deepcopy(self)
        model_out._filepath = None
        return model_out
    
    def __deepcopy__(self, memo):
        result = self.__class__.__new__(self.__class__)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    def _resolve_extensions(self, extensions : str | list[str] | None = None):
        if extensions is None:
            return list(self.EXTENSIONS.keys())
        if isinstance(extensions, str):
            return [extensions.upper()]
        else:
            return [e.upper() for e in extensions]

    def __getattr__(self, attr): # This is only called if the attribute is not found in the instance's __dict__
        attru = attr.upper()
        if attru not in self.EXTENSIONS:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")
        if attru in self._data:
            return self._data[attru]
        else:
            return None
        
    def __setattr__(self, attr, value):
        attru = attr.upper()
        if attru in self.EXTENSIONS:
            self._data[attru] = value
        elif attr in self.ALLOWED_ATTRIBUTES:
            super().__setattr__(attr, value)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}' or is not allowed to be set directly")

    def init_data(self, data : dict | None = None, correct_dtypes : bool = False):
        """
        Initialize the data.

        Args:
            data (dict, optional): Dictionary containing data arrays (numpy) or tables (DataFrames).
                Keys should match the schema's extension names. Defaults to None.
        """
        self._data = {}
        if data is not None:
            for key in data:
                if key.upper() in self.EXTENSIONS:
                    if isinstance(data[key], astropy.table.Table):
                        self._data[key.upper()] = data[key]
                    elif isinstance(data[key], np.ndarray):
                        if correct_dtypes:
                            self._data[key.upper()] = data[key].astype(self.EXTENSIONS[key.upper()]['dtype'])
                        else:
                            self._data[key.upper()] = data[key]
                else:
                    raise ValueError(f"Unknown extension: {key.upper()}")
        if 'RECEIPT' not in self._data:
            self.init_receipt()

        # Initialize remaining extensions
        for extname in self.EXTENSIONS:
            if extname not in self._data:
                if 'columns' in self.EXTENSIONS[extname]:
                    self._data[extname] = astropy.table.Table({
                        name: np.array([], dtype=dtype)
                        for name, dtype in self.EXTENSIONS[extname]['columns'].items()
                    })
                else:
                    self._data[extname] = None # Keep Image HDUs as None

    def init_meta(self, meta : MetaModel | dict | None = None) -> MetaModel:
        """
        Initialize the metadata and optionally merge in provided metadata.

        Parameters
        ----------
        meta : MetaModel | dict, optional
            Existing metadata to store or merge.
        """
        
        if isinstance(meta, MetaModel):
            self.meta = meta
            return self.meta
        
        # Meta from file headers
        if self._filepath is None:
            meta_from_headers = {}
        else:
            meta_from_headers = self.read_meta_from_fits_headers()

        if isinstance(meta, dict):
            self.meta = self.config.meta_model(**(meta_from_headers | meta))
        elif meta is None:
            self.meta = self.config.meta_model(**meta_from_headers)
        else:
            raise ValueError(f"Unknown type for meta: {type(meta)}")
        return self.meta
    
    def read_meta_from_fits_headers(self) -> dict:
        """
        Read metadata from the FITS headers.

        Returns:
            dict: Metadata dictionary with keys matching the model fields, and values read in from the FITS headers.
        """
        meta_dict = {}
        meta_model_class = self.config.meta_model
        with fits.open(self._filepath, memmap=True) as hdulist:
            for attr, field in meta_model_class.model_fields.items():
                fits_hdu = field.json_schema_extra['fits_hdu']
                fits_keyword = field.json_schema_extra['fits_keyword']
                header = hdulist[fits_hdu].header
                if fits_keyword in header:
                    meta_dict[attr] = header[fits_keyword]
                else:
                    warnings.warn(f"Missing FITS keyword '{fits_keyword}' in HDU '{fits_hdu}' for metadata field '{attr}', using default value if available.")
        
        return meta_dict

    def init_receipt(self):
        """
        Initialize the receipt.
        """
        if 'RECEIPT' not in self._data:
            self._data['RECEIPT'] = astropy.table.Table({
                name: np.array([], dtype=dtype)
                for name, dtype in self.EXTENSIONS['RECEIPT']['columns'].items()
            })

    @property
    def extinfo(self):
        for extname, info in self.EXTENSIONS.items():
            hdu_type = info['hdu_type']
            if hdu_type is fits.PrimaryHDU:
                continue
            if hdu_type is fits.ImageHDU:
                if extname in self._data and self._data[extname] is not None:
                    dtype = self._data[extname].dtype
                    print(f"Extension: {extname}\n  dtype: {dtype}\n  shape: {self._data[extname].shape}")
            elif hdu_type is fits.BinTableHDU:
                if len(self._data[extname]) > 0:
                    print(f"Extension: {extname}\n  columns: {list(self._data[extname].columns)}\n  nrows: {len(self._data[extname])})")

    @property
    def data_level_suffix(self):
        return None

    @classmethod
    def from_datamodels(
        cls,
        input : Any,
        include_data: bool = True,
        copy_data: bool = True,
    ) -> 'DataModel':

        if input is None:
            raise ValueError("No models provided")
        
        # Normalize to iterable
        if isinstance(input, (str, DataModel)):
            input = [input]
        elif isinstance(input, dict):
            input = list(input.values())
        elif not isinstance(input, Sequence):
            raise ValueError("Input must be a DataModel, filepath, list of DataModels/filepaths, or dict of DataModels/filepaths.")

        # Merge metadata
        meta_out = merge_meta(input)

        # Create new model
        first_model = datamodel_open(input[0], meta_only=True)
        model_out = cls(meta=meta_out, config=first_model.config)

        # Merge data only if requested and single model
        if include_data and len(input) == 1:
            first_model = datamodel_open(input[0])
            data_out = {k: v for k, v in first_model._data.items() if k in model_out.EXTENSIONS}
            if copy_data:
                data_out = deepcopy(data_out)
            model_out._data.update(data_out)

        # Always append receipt, data ancestors, and calibrations tables
        for _input in input:
            input_model = datamodel_open(_input)
            model_out.append_receipts(input_model)
            model_out.append_ancestors(input_model)
            model_out.append_calibrations(input_model)

        return model_out
    
    def make_order_vec(self) -> np.ndarray:
        return self._make_order_vec(self.meta.order_start, self.meta.order_end)
        
    @staticmethod
    def _make_order_vec(order_start : int, order_end : int) -> np.ndarray:
        """
        Create an order vector based on the order start and end metadata. If order_end < order_start, the vector will be in descending order.
        
        Returns:
            np.ndarray: An array of integers from order_start to order_end.
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
        data = {}
        for attr in self._data:
            attr_lower = attr.lower()
            if '_fib' in attr_lower:
                try:
                    kind, fiber = attr_lower.rsplit('_fib', 1)
                    fiber = int(fiber)
                except ValueError:
                    continue
                if fiber not in data:
                    data[fiber] = {}
                data[fiber][kind] = getattr(self, attr)
        return data

    def add_calibration(
        self,
        calibration : MetaModel | DataModel | str | dict,
        cal_role : str,
        date_time : str,
        note : str | None = None,
        replace_existing : bool = True,
    ) -> None:
        """
        Add a calibration to the DataModel's calibrations table.
        
        Args:
            calibration (MetaModel | DataModel | str | dict): The calibration to add. Can be:
                - MetaModel object
                - DataModel object
                - String filepath (will load metadata to create entry)
                - Dict with calibration column values
            date_time (str): The datetime the calibration was used in ISO format.
            role (str): The role of this calibration. Defaults to 'SINGLE'.
        """

        # Check if role already exists
        remove_row_index = None
        if cal_role in self.calibrations['CAL_ROLE']:
            if replace_existing:
                remove_row_index = np.where(self.calibrations['CAL_ROLE']==cal_role)[0][0]
            else:
                warnings.warn(f"Calibration with role '{cal_role}' already exists for {self} and replace_existing=False, skipping.")
                return

        # Case 1: MetaModel object
        #   New row from ORM object
        if isinstance(calibration, MetaModel):
            new_row = {
                'ID' : calibration.cal_id,
                'FILENAME' : calibration.filename,
                'CAL_TYPE' : calibration.cal_type,
                'CAL_ROLE' : cal_role,
                'DATETIME_OBS' : calibration.datetime_obs,
                'DRP_VERSION' : calibration.drp_version,
                'NOTE' : calibration.note,
                'LAST_UPDATED' : calibration.last_updated,
                'TIME_APPLIED' : date_time,
            }

        # Case 2: filepath or DataModel
        # NOTE: This path should be used with caution
        #   and limited to dev purposes or limited user cases, never at Keck.
        elif isinstance(calibration, (str, DataModel)):
            from .utils import open as open_datamodel
            cal_model = open_datamodel(calibration, meta_only=True)
            new_row = {
                'ID' : cal_model.meta.cal_id or '',
                'FILENAME' : cal_model.meta.filename or '',
                'CAL_TYPE' : cal_model.CAL_TYPE,
                'CAL_ROLE' : cal_role,
                'DATETIME_OBS' : cal_model.meta.datetime_obs,
                'DRP_VERSION' : __version__,
                'NOTE' : note or '',
                'LAST_UPDATED' : '',
                'TIME_APPLIED' : date_time,
            }
        else:
            raise ValueError("Invalid calibration input")

        # Delete existing row if specified
        if remove_row_index is not None:
            self.calibrations.remove_row(remove_row_index)

        # Add new row
        self.calibrations.add_row(new_row)

        # Return new row
        return new_row

    def load_calibration_from_role(
        self,
        cal_role : str,
    ) -> DataModel:
        """
        Load a calibration from the DataModel's calibrations table.

        Args:
            cal_role (str): The role of the calibration to load.

        Returns:
            DataModel: The loaded calibration ORM object.
        """
        row = self.get_calibration_from_role(cal_role)
        cal_id = row['ID']
        with CalibrationStore(orm_class=self.META_MODEL) as store:
            return store.get_calibration_by_id(cal_id)

    def get_calibration_from_role(self, cal_role : str) -> DataModel:
        """
        Return the row from the DataModel's calibrations table.

        Args:
            cal_role (str): The role of the calibration to load.

        Returns:
            dict: The row from the DataModel's calibrations table.
        """
        # Match on cal_type and role
        mask = self.calibrations['CAL_ROLE'] == cal_role
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return None
        if len(indices) > 1:
            raise ValueError(f"Multiple calibrations found with role '{cal_role}' in {self.__class__.__name__}")
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
        See _generate_filename for details.
        """
        date_time = self.meta.datetime_obs

        # Determine instrument alias
        spectrograph = getattr(self.meta, "spectrograph", None)
        if spectrograph is not None:
            spectrograph = spectrograph.upper()
            instrument_alias = self.config.instrument_aliases[spectrograph]
        else:
            raise ValueError("meta.spectrograph is None, cannot determine spectrograph alias.")

        # Determine suffix
        if suffix is None:
            if getattr(self, "data_level_suffix", None) is not None:
                suffix = self.data_level_suffix
        else:
            if getattr(self, "data_level_suffix", None) is not None:
                suffix = f"{self.data_level_suffix}-{suffix}"

        return self._generate_filename(
            instrument_alias=instrument_alias,
            date_time=date_time,
            suffix=suffix
        )

    @staticmethod
    def _generate_filename(
        instrument_alias : str,
        date_time : str,
        suffix : str | None = None
    ) -> str:
        """
        Create a filename for a HISPEC data product.
        Format: {instrument_alias}.{YYYYMMDD}.{SSSSS.SS}_{suffix}.fits

        Args:
            instrument_alias (str): The two letter instrument alias.
            date_time (str): The properly formated datetime of the observation in .
            suffix (str, optional): An optional suffix to append to the filename. Defaults to None.

        Returns:
            str : The filename
        """
        dt = astropy.time.Time(date_time, precision=3, format='isot')
        dt = dt.datetime
        midnight = datetime.combine(dt.date(), datetime.min.time())
        seconds = (dt - midnight).total_seconds()
        date_time = dt.strftime("%Y%m%d") + '.' + f"{seconds:08.2f}"
        filename = f"{instrument_alias}.{date_time}.fits"
        if suffix:
            filename = "".join([filename[0:-5], "_", suffix, ".fits"])
        return filename
    
    @property
    def num_fibers(self) -> int:
        return self.config.num_fibers
    
    @property
    def instrument_name(self) -> str:
        return self.config.instrument_name
    
    def get_calibration_orm_class(self) -> type[CalibrationORM]:
        """
        Get the CalibrationORM class corresponding to this model's metadata.
        See hispecdrp.calibrations.orm.get_calibration_orm_class.
        """
        return get_calibration_orm_class(self.meta)
    
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
            for row in other_model.data_ancestors:
                self.data_ancestors.add_row(row)

class CalibrationModel(DataModel):

    CAL_TYPE = None # Type of calibration

    def on_init(self):
        super().on_init()
        self.meta.cal_type = self.CAL_TYPE
        if self.meta.cal_id is None:
            self.generate_calibration_id()

    def generate_calibration_id(self) -> str:
        """
        Generate the calibration ID.

        Returns
        -------
        cal_id : str
            The calibration ID
        """
        filename = self.meta.filename or 'unknown'
        drp_version = self.meta.drp_version or 'unknown'
        last_updated = self.meta.last_updated or datetime.now().isoformat(timespec="milliseconds")
        UUID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "HISPEC_DRP")
        self.meta.cal_id = str(uuid.uuid5(UUID_NAMESPACE, filename + drp_version + last_updated))
        return self.meta.cal_id
    
    def generate_filename(self, suffix: str | None = None) -> str:
        
        # Determine instrument alias
        spectrograph = getattr(self.meta, "spectrograph", None)
        instrument_alias = self.config.instrument_aliases.get(spectrograph)
        assert instrument_alias is not None, \
        f"Instrument alias not found for instrument '{self.instrument_name}' and spectrograph '{spectrograph}'"

        # Determine master_cal
        master_cal = getattr(self.meta, "master_cal")

        return self._generate_filename(
            instrument_alias=instrument_alias,
            date_time=self.meta.datetime_obs,
            cal_type=self.CAL_TYPE,
            master=master_cal,
            suffix=suffix
        )

    @staticmethod
    def _generate_filename(
        instrument_alias : str,
        date_time : str,
        cal_type : str,
        master : bool = False,
        suffix : str | None = None
    ) -> str:
        """
        Create a filename for a PARVI calibration data product.

        Format: {instrument_alias}_{date_time}_[master-]{cal_type}_{suffix}.fits
            datetime : YYYYMMDDTHHMMSS.
            cal_type : The type of calibration data.
            master : Insert 'master-' before the cal_type if `master=True`.
            suffix : optional suffix

        Parameters
        ----------
        date_time : str
            The date and time of the observation.
        instrument_alias : str
            The two letter instrument alias.
        cal_type : str
            The type of calibration data.
        master : bool, optional
            If True, insert 'master-' before the cal_type. Defaults to False.
        suffix : str, optional
            An optional suffix to append to the filename. Defaults to None.

        Returns
        -------
        filename : str
            The generated filename.
        """
        dt = astropy.time.Time(date_time, precision=3, format='isot')
        dt = dt.datetime
        midnight = datetime.combine(dt.date(), datetime.min.time())
        seconds = (dt - midnight).total_seconds()
        date_time = dt.strftime("%Y%m%d") + '.' + f"{seconds:07.2f}"
        if master:
            cal_type = f"master-{cal_type}"
        filename = f"{instrument_alias}.{date_time}.{cal_type}.fits"
        if suffix is not None:
            filename = "".join([filename[:-5], "_", suffix, ".fits"])
        return filename
    
    def to_orm(self, note : str | None = None) -> CalibrationORM:
        """
        Convert the CalibrationModel metadata to a CalibrationORM object. Keep for backwards compatibility with koa_middleware.

        Parameters
        ----------
        note (str, optional): An optional note to add to or update the ORM object with. If None, the field meta.note is used.

        Returns
        -------
        cal_orm : CalibrationORM
            The CalibrationORM object.
        """
        meta_vals = {}
        for attr, field in self.meta.__class__.model_fields.items():
            if field.json_schema_extra['orm_field']:
                sql_field_name = get_orm_field_name(attr, field)
                meta_vals[sql_field_name] = getattr(self.meta, attr)
        orm_class = get_calibration_orm_class(self.meta)
        meta_orm = orm_class(**meta_vals)
        if note is not None:
            meta_orm.note = note
        return meta_orm
