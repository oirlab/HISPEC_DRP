import os
import astropy.table
from astropy.io import fits
import pandas as pd
import copy
from typing import TypeVar
import datetime
from . import schema
from .meta_dict import MetaDict
from hispecdrp import __version__
import numpy as np
import warnings
from astropy.io.fits.verify import VerifyWarning

# Receipt columns
_RECEIPT_COLS = {
    'Time' : str,
    'HISPECDRP_Version' : str,
    'Class' : str,
    'Filename' : str,
    'Arguments' : str,
    'Status' : str,
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
    """

    EXTENSIONS = {
        'PRIMARY': {
            'hdu_type' : fits.PrimaryHDU,
        },
        'RECEIPT': {
            'hdu_type' : fits.BinTableHDU,
            'columns' : _RECEIPT_COLS
        }
    }
    INHERIT_EXTENSIONS = True

    HEADER_SCHEMAS = []
    INHERIT_SCHEMAS = True

    def __init__(
        self,
        filepath : str | None = None,
        data : dict | None = None,
        extensions : str | list[str] | None = None,
        meta : MetaDict | None = None,
        meta_only : bool = False,
        validate_metadata : bool = True,
        correct_dtypes : bool = False,
    ):
        """
        Initialize a DataModel object.

        Args:
            data (dict, optional): Dictionary containing data arrays (numpy) or tables (DataFrames).
                Keys should match the schema's extension names. Defaults to None.

            meta (dict, optional): Dictionary containing metadata.
                Keys should match the schema's property paths. Defaults to None.
                
            filepath (str, optional): Path to a FITS file to load. Defaults to None.
            extensions (str, list[str], optional): List of extensions to load from the FITS file. Defaults to None.
            validate_metadata (bool, optional): Whether to validate the metadata against the schema. Defaults to True.
            correct_dtypes (bool, optional): Whether to correct the data types of the data arrays. Defaults to False.
        """

        # TODO: Implement hdu_list input.

        # Set the optional filepath
        if filepath is not None:
            if not os.path.isfile(filepath):
                raise FileNotFoundError(f"File not found: {filepath}")
            self._filepath = os.path.abspath(filepath)
            self.init_meta(validate_metadata=validate_metadata)
            self.set_meta_from_headers(extensions=extensions, validate=validate_metadata)
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
        validate : bool = False,
    ) -> dict[str, fits.Header]:
        """
        Set the metadata from the headers of the FITS file.
        
        Args:
            extensions (str, list[str], optional): List of extensions to read. Defaults to None.
            validate (bool, optional): Whether to validate the metadata against the schema. Defaults to False.
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
        # TODO: Implement validation as optional input keyword, here we apply blind conversion
        extensions = self._resolve_extensions(extensions)
        self._data = {}
        with fits.open(
            self._filepath,
            memmap=False,
            do_not_scale_image_data=do_not_scale_image_data
        ) as hdulist:
            for extname in extensions:
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
            correct_dtypes (bool, optional): Whether to correct the data types of the data arrays. Defaults to False.
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
        return self.to_fits(add_missing_metadata=add_missing_metadata, correct_dtypes=correct_dtypes, **kwargs)
    
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

    def to_fits(self, add_missing_metadata : bool = False, correct_dtypes : bool = True, **kwargs):
        """
        Write the DataModel to a FITS file.
        
        Args:
            **kwargs: Additional keyword arguments to pass to the fits.HDUList.writeto method.
        """
        if 'overwrite' not in kwargs:
            kwargs['overwrite'] = True
        if self._filepath is None:
            raise ValueError("No filepath specified. Set self._filepath to save the DataModel.")
        hdulist = fits.HDUList()
        for extname in self.EXTENSIONS:
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
        return copy.deepcopy(self)
    
    def __deepcopy__(self, memo):
        result = self.__class__.__new__(self.__class__)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result

    def _resolve_extensions(self, extensions : str | list[str] | None = None):
        if extensions is None:
            return list(self.EXTENSIONS.keys())
        if isinstance(extensions, str):
            return [extensions.upper()]
        else:
            return [e.upper() for e in extensions]

    def __getattr__(self, attr):
        attr = attr.upper()
        if attr not in self.EXTENSIONS:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")
        if attr in self._data:
            return self._data[attr]
        else:
            return None
        
    def __setattr__(self, attr, value):
        attru = attr.upper()
        if attru in self.EXTENSIONS:
            self._data[attru] = value
        else:
            super().__setattr__(attr, value)

    def validate_metadata(self):
        """
        Validate the metadata against the schema.
        """
        schema.validate(self.schema, self.meta)

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
            - meta (MetaDict, optional): Dictionary containing data values.
                Keys should match the schema's property paths.
                If None, the header will be created with default values.
            - extname (str, optional): Name of the extension. Default is "PRIMARY".
            - add_missing (bool, optional): If True, missing entries will be added with None values. Default is False.

        Returns:
            astropy.io.fits.Header : FITS header populated according to the schema
        """
        header = fits.Header()
        for path, info in self.meta.walk_schema(sections=True):

            # FITS HDU and kind of entry
            prop_extname = info.get('fits_hdu', self.meta._DEFAULTS['fits_hdu'])
            if extname != prop_extname:
                continue
            dtype = info.get('datatype', self.meta._DEFAULTS['datatype'])

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
    def data_level_suffix(self):
        """
        Datamodel level suffix to override for each datamodel.
        """
        return None
    
    @property
    def dirname(self):
        """
        Returns the directory of the DataModel if present.
        """
        return os.path.dirname(self._filepath) if self._filepath else None

    @property
    def basename(self):
        """
        Returns the base filename of the DataModel if present.
        """
        return os.path.basename(self._filepath) if self._filepath else None