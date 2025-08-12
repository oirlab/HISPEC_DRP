from __future__ import annotations # NOTE: Not needed in Python 3.13+
import os
import astropy.table
from astropy.io import fits
from copy import deepcopy
from typing import TypeVar
import datetime
from .meta_dict import MetaDict
from hispecdrp import __version__
import numpy as np
import warnings
from sqlalchemy.inspection import inspect
from astropy.io.fits.verify import VerifyWarning

from koa_middleware import CalibrationORM
from koa_middleware import CalibrationStore

# Receipt columns
_RECEIPT_COLS = {
    'PROC_TIME' : str, # Processing time YYYY-MM-DDTHH:MM:SS.sss
    'DRP_VERSION' : str, # DRP semver
    'CLASS' : str, # Primitive class
    'FILENAME' : str, # Input base filename
    'ARGUMENTS' : str, # JSON rep of args ?
    'STATUS' : str, # COMPLETED, FAILED, SKIPPED
}

_CALIBRATION_COLS = {
    'ID': str, # UUID
    'FILENAME': str, # Base filename of the calibration
    'CAL_TYPE': str, # Type of calibration
    'CAL_ROLE' : str, # Role of calibration, typically the same as cal_type
    'DATETIME_OBS': str, # Observation date and time YYYY-MM-DDTHH:MM:SS.sss
    'DRP_VERSION': str, # DRP semver calibration was processed with (not necessarily current version)
    'NOTE': str, # Note for calibration matching ORM
    'LAST_UPDATED': str, # When calibration was last updated in the DB.
    'TIME_APPLIED' : str # Time calibration was applied YYYY-MM-DDTHH:MM:SS.sss
}

# So that dict[str, A] <: dict[str, B] when A <: B
_VALID_HDU = TypeVar('T', bound=fits.hdu.base._ValidHDU, covariant=True)


class DataModel:
    """
    Base class for data models.

    Class Attributes:
        EXTENSIONS (dict): Extensions and their properties.
        INHERIT_EXTENSIONS (bool): Whether to inherit extensions from parent class. Defaults to True.
        HEADER_SCHEMAS (list): List of schema files to load for the header.
        INHERIT_SCHEMAS (bool): Whether to inherit schemas from parent class. Defaults to True.
        ALLOWED_ATTRIBUTES (list): List of attributes that can be set directly on the DataModel.
        CAL_ORM_CLASS (type): The calibration ORM class corresponding to this datamodel.
    """

    EXTENSIONS = {
        'PRIMARY': {
            'hdu_type' : fits.PrimaryHDU,
        },
        'RECEIPT': {
            'hdu_type' : fits.BinTableHDU,
            'columns' : _RECEIPT_COLS
        },
        'CALIBRATIONS': {
            'hdu_type' : fits.BinTableHDU,
            'columns' : _CALIBRATION_COLS
        }
    }
    INHERIT_EXTENSIONS = True

    HEADER_SCHEMAS = []
    INHERIT_SCHEMAS = True

    CAL_ORM_CLASS = None

    ALLOWED_ATTRIBUTES = [
        'EXTENSIONS', 'INHERIT_EXTENSIONS', 'HEADER_SCHEMAS', 'INHERIT_SCHEMAS', 'CAL_ORM_CLASS',
        'meta', '_data', '_filepath',
    ]

    def __init__(
        self,
        filepath : str | None = None,
        data : dict | None = None,
        meta : MetaDict | None = None,
        meta_only : bool = False,
        extensions : str | list[str] | None = None,
        validate_metadata : bool = True,
        correct_dtypes : bool = False,
    ):
        """
        Initialize a DataModel object.

        Args:
            filepath (str, optional): Path to a FITS file to load.
            data (dict, optional): Dictionary containing the data (numpy arrays or astropy tables). Keys should match the schema's extension names.
            meta (dict, optional): Dictionary containing metadata. Keys should match the schema's property paths.
            meta_only (bool, optional): If True, only metadata is loaded and no data is read from the FITS file. Defaults to False.
            extensions (str, list[str], optional): List of extensions to load from the FITS file. Defaults to all extensions.
            validate_metadata (bool, optional): Whether to validate the metadata against the schema. Defaults to True.
            correct_dtypes (bool, optional): Whether to correct the data types of the data arrays upon reading them in. Defaults to False.
        """

        # Set the optional filepath
        if filepath is not None:
            if not os.path.isfile(filepath):
                raise FileNotFoundError(f"File not found: {filepath}")
            self._filepath = os.path.abspath(filepath)
            self.init_meta(validate_metadata=validate_metadata)
            self.set_meta_from_headers(extensions=extensions)
            if not meta_only:
                self.read_data(extensions=extensions, correct_dtypes=correct_dtypes)
            else:
                self._data = {}
        else:
            self._filepath = None
            self.init_meta(meta=meta, validate_metadata=validate_metadata)
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
            for path, info in self.meta.walk_schema(sections=False):
                fits_hdu = info.get('fits_hdu', self.meta._DEFAULTS['fits_hdu'])
                fits_keyword = info['keyword']
                header = hdulist[fits_hdu].header
                if fits_keyword in header:
                    val = header[fits_keyword]
                    self.meta[path] = val
        return self.meta
    
    def read_data(
        self,
        extensions : str | list[str] | None = None,
        do_not_scale_image_data : bool | None = None,
        correct_dtypes : bool = False,
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
                        data = data.astype(dtype)
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

    def to_fits(self, include_empty_extensions : bool = False, add_missing_metadata : bool = False, correct_dtypes : bool = True, **kwargs):
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
        for extname in self.EXTENSIONS:
            if not include_empty_extensions \
                and (extname not in self._data or self._data[extname] is None) \
                and extname not in ('RECEIPT', 'PRIMARY'):
                continue
            hdu = self.make_hdu(extname, add_missing_metadata=add_missing_metadata, correct_dtypes=correct_dtypes)
            hdulist.append(hdu)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', VerifyWarning)
            hdulist.writeto(self._filepath, **kwargs)

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
        self.meta.date_created = datetime.datetime.now().isoformat()
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
        header = self.make_header(extname=extname, add_missing_metadata=add_missing_metadata)
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
            if hasattr(_cls, "HEADER_SCHEMAS") and cls.INHERIT_SCHEMAS:
                for schema_file in _cls.HEADER_SCHEMAS:
                    if schema_file not in cls.HEADER_SCHEMAS:
                        cls.HEADER_SCHEMAS.append(schema_file)

    def copy(self):
        """
        Create a deep copy of the object.
        """
        return deepcopy(self)
    
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

    def init_meta(self, meta : MetaDict | None = None, validate_metadata : bool = True) -> MetaDict:
        """
        Initialize the metadata and optionally merge in provided metadata.

        Args:
            meta (dict, optional): Metadata to merge in. Defaults to None.
        """
        if isinstance(meta, MetaDict):
            self.meta = meta
        else:
            self.meta = MetaDict(schema=self.HEADER_SCHEMAS, validate=validate_metadata)
        return self.meta

    def init_receipt(self):
        """
        Initialize the receipt.
        """
        if 'RECEIPT' not in self._data:
            self._data['RECEIPT'] = astropy.table.Table({
                name: np.array([], dtype=dtype)
                for name, dtype in self.EXTENSIONS['RECEIPT']['columns'].items()
            })

    def make_header(
        self,
        extname : str = 'PRIMARY',
        add_missing_metadata : bool = False
    ) -> fits.Header:
        """
        Create an Astropy FITS header dynamically from a schema dict.
        Includes section headers as COMMENT cards.
        
        Args:
            meta (MetaDict, optional): Dictionary containing data values.
                Keys should match the schema's property paths.
                If None, the header will be created with default values.
            extname (str, optional): Name of the extension. Default is "PRIMARY".
            add_missing (bool, optional): If True, missing entries will be added with None values. Default is False.

        Returns:
            astropy.io.fits.Header : FITS header populated according to the schema
        """
        header = fits.Header()
        for path, info in self.meta.walk_schema(sections=True):

            # FITS HDU and kind of entry
            prop_extname = info.get('fits_hdu', self.meta._DEFAULTS['fits_hdu'])
            if extname != prop_extname:
                continue

            # Prop name
            prop_name = path[-1]

            # Section header
            #if dtype == 'object':
            if 'keyword' not in info:
                # Add section header as COMMENT
                header.append(('COMMENT', '=' * 40), end=True)
                header.append(('COMMENT', f' {prop_name.upper()}'), end=True)
                header.append(('COMMENT', '=' * 40), end=True)

            # Handle individual properties
            else:

                keyword = info['keyword']
                comment = info.get('comment', None)
                unit = info.get('unit', self.meta._DEFAULTS['unit'])
                if unit is not None:
                    if unit.lower() != 'none':
                        comment += f" ({unit})"
                value = self.meta[path]  # Get value from metadata
                
                # Skip if no value and not adding empty entries
                if value is None and not add_missing_metadata:
                    continue

                # Add to header
                header.append((keyword, value, comment), end=True)

        return header
    
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
    def from_datamodel(cls, model : 'DataModel', copy : bool = True):
        """
        Create a new DataModel from another DataModel by copying over any valid data and metadata entries. The _filepath is not copied.
        
        Args:
            other (DataModel): The DataModel to copy from.
        """
        _data = {k : v for k, v in model._data.items() if k in cls.EXTENSIONS}
        if copy:
            _data = deepcopy(_data)
        model_out = cls(data=_data)
        meta = deepcopy(model.meta) # if copy else model.meta
        exclude_keys_copy = ['cal_id']
        for key in exclude_keys_copy:
            if key in meta:
                del meta[key]
        model_out.meta.update(meta, override=False)
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
        calibration : CalibrationORM | DataModel | str | dict,
        cal_role : str,
        date_time : str,
    ) -> None:
        """
        Add a calibration to the DataModel's calibrations table.
        
        Args:
            calibration (CalibrationORM | str | dict): The calibration to add. Can be:
                - CalibrationORM object
                - String filepath (will load metadata to create entry)
                - Dict with calibration column values
            date_time (str): The datetime the calibration was used in ISO format.
            role (str): The role of this calibration. Defaults to 'SINGLE'.
        """
        # First, determine the cal_type to check for existing entries
        # TODO: REVIEW THIS FUNCTION
        
        cal_type = None
        
        if isinstance(calibration, CalibrationORM):
            cal_type = calibration.cal_type
        elif isinstance(calibration, (str, DataModel)):
            from .utils import open as open_datamodel
            cal_model = open_datamodel(calibration, meta_only=True)
            cal_orm = cal_model.to_orm()
            cal_type = cal_orm.cal_type
        elif isinstance(calibration, dict): # WHAT IS THIS? meta data?
            cal_type = calibration.get('CAL_TYPE') or calibration.get('cal_type')
        else:
            raise TypeError(f"calibration must be CalibrationORM, str, or dict, got {type(calibration)}")

        # Check for existing entry with same cal_type AND role
        if cal_type:
            mask = (self.calibrations['CAL_TYPE'] == cal_type) & (self.calibrations['CAL_ROLE'] == cal_role.upper())
            indices = np.where(mask)[0]
            
            if len(indices) > 0:
                # Update existing row
                assert len(indices) == 1, f"Multiple entries found for calibration type {cal_type} with role {cal_role}: {self}"
                existing_index = indices[0]
                
                # Update the existing row with new data
                new_row = {}
                new_row['TIME_APPLIED'] = date_time
                new_row['CAL_ROLE'] = cal_role
                
                if isinstance(calibration, CalibrationORM):
                    cal_orm_fields = {c.key.lower(): c.key for c in inspect(calibration).mapper.column_attrs}
                    for col_name in self.calibrations.colnames:
                        if col_name.lower() in cal_orm_fields:
                            value = getattr(calibration, cal_orm_fields[col_name.lower()])
                            if value is None:
                                value = ''
                            new_row[col_name] = value
                elif isinstance(calibration, (str, DataModel)):
                    from .utils import open as open_datamodel
                    cal_model = open_datamodel(calibration, meta_only=True)
                    cal_orm = cal_model.to_orm()
                    cal_orm_fields = {c.key.lower(): c.key for c in inspect(cal_orm).mapper.column_attrs}
                    for col_name in self.calibrations.colnames:
                        if col_name.lower() in cal_orm_fields:
                            value = getattr(cal_orm, cal_orm_fields[col_name.lower()])
                            if value is None:
                                value = ''
                            new_row[col_name] = value
                elif isinstance(calibration, dict):
                    for col_name in self.calibrations.colnames:
                        if col_name in calibration:
                            new_row[col_name] = calibration[col_name]
                        elif col_name.lower() in calibration:
                            new_row[col_name] = calibration[col_name.lower()]
                
                # Update the existing row
                for key, value in new_row.items():
                    self.calibrations[key][existing_index] = value
                return

        # No existing entry found, create new row
        new_row = {}
        new_row['TIME_APPLIED'] = date_time
        new_row['CAL_ROLE'] = cal_role

        if isinstance(calibration, CalibrationORM):
            # Extract from ORM object
            cal_orm_fields = {c.key.lower(): c.key for c in inspect(calibration).mapper.column_attrs}
            for col_name in self.calibrations.colnames:
                if col_name.lower() in cal_orm_fields:
                    value = getattr(calibration, cal_orm_fields[col_name.lower()])
                    if value is None:
                        value = ''
                    new_row[col_name] = value
        
        elif isinstance(calibration, (str, DataModel)):
            # Load from filepath
            from .utils import open as open_datamodel
            cal_model = open_datamodel(calibration, meta_only=True)
            cal_orm = cal_model.to_orm()
            cal_orm_fields = {c.key.lower(): c.key for c in inspect(cal_orm).mapper.column_attrs}
            for col_name in self.calibrations.colnames:
                if col_name.lower() in cal_orm_fields:
                    value = getattr(cal_orm, cal_orm_fields[col_name.lower()])
                    if value is None:
                        value = ''
                    new_row[col_name] = value
        
        elif isinstance(calibration, dict):
            # Direct column mapping
            for col_name in self.calibrations.colnames:
                if col_name in calibration:
                    new_row[col_name] = calibration[col_name]
                elif col_name.lower() in calibration:
                    new_row[col_name] = calibration[col_name.lower()]
        
        # Add new row
        self.calibrations.add_row(new_row)

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
        with CalibrationStore(orm_class=self.CAL_ORM_CLASS) as store:
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
        

    