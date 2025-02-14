import os
import astropy.table
from astropy.io import fits
import pandas as pd
import copy
from typing import TypeVar
from . import schema
from .meta_dict import MetaDict
import numpy as np

import warnings
from astropy.io.fits.verify import VerifyWarning

# Receipt columns
_RECEIPT_COLS = {
    'Time' : str,
    'Code_Release' : str,
    'Branch_Name' : str,
    'Commit_Hash' : str,
    'Class' : str,
    'Datamodel' : str,
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

    _ALLOWED_ATTRIBUTES = []

    def __init__(
        self,
        filename : str | None = None,
        data : dict | None = None,
        meta : MetaDict | None = None,
        extensions : str | list[str] | None = None,
        validate_metadata : bool = True
    ):
        """
        Initialize a DataModel object.

        Args:
            data (dict, optional): Dictionary containing data arrays (numpy) or tables (DataFrames).
                Keys should match the schema's extension names. Defaults to None.
            meta (dict, optional): Dictionary containing metadata.
                Keys should match the schema's property paths. Defaults to None.
            filename (str, optional): Path to a FITS file to load. Defaults to None.
            extensions (str, list[str], optional): List of extensions to load from the FITS file. Defaults to None.
            validate_metadata (bool, optional): Whether to validate the metadata against the schema. Defaults to True.
        """

        # TODO: Implement hdu_list input.

        # Set the optional filename
        self.filename = filename

        # Load and merge the schemas
        self.load_schemas()

        # If given a file, load the header and data from that
        if self.filename is not None and os.path.isfile(self.filename):
            self.set_meta_from_headers(extensions=extensions, validate=validate_metadata)
            self.read_data(extensions=extensions)
        else:
            # If given starting (meta)data, set that.
            self.init_meta(meta)
            self.init_data(data)

        # Initialize the receipt
        self.init_receipt()

        # Defaults
        if self.meta.data_level is None:
            self.meta.data_level = self.default_level


    @property
    def dirname(self):
        """
        Returns the directory of the DataModel if present.
        """
        return os.path.dirname(self.filename) if self.filename else None


    @property
    def basename(self):
        """
        Returns the base filename of the DataModel if present.
        """
        return os.path.basename(self.filename) if self.filename else None
    

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
        self.meta = MetaDict(_schema=self.schema, _validate=validate)
        with fits.open(self.filename, memmap=True) as hdulist:
            for path, entry in schema.walk_schema(self.schema):
                fits_hdu = entry.get('fits_hdu', 'PRIMARY')
                fits_keyword = entry.get('fits_keyword')
                header = hdulist[fits_hdu].header
                if fits_keyword in header:
                    val = header[fits_keyword]
                    self.meta[path] = val
        return self.meta
    

    def read_data(
        self,
        extensions : str | list[str] | None = None,
        do_not_scale_image_data : bool | None = None,
    ) -> dict[str, _VALID_HDU]:
        """
        Read the data from the FITS file.

        Args:
            extensions (str, list[str], optional): List of extensions to read. Defaults to None.
            do_not_scale_image_data (bool, optional): Whether to not scale data according to BZERO and BSCALE. Defaults to True.
        """
        # TODO: Implement validation as optional input keyword, here we apply blind conversion
        extensions = self._resolve_extensions(extensions)
        self._data = {}
        with fits.open(
            self.filename,
            memmap=False,
            do_not_scale_image_data=do_not_scale_image_data
        ) as hdulist:
            for extname in extensions:
                hdu = hdulist[extname]
                hdu_type = type(hdu)
                if hdu_type is fits.ImageHDU:
                    dtype = self.EXTENSIONS[extname]['dtype']
                    self._data[extname] = hdu.data.astype(dtype)
                elif hdu_type is fits.BinTableHDU:
                    self._data[extname] = pd.DataFrame(hdu.data)
        return self._data
    

    def save(
        self,
        filename : str | None = None,
        add_missing_metadata : bool = False
    ):
        """
        Save the DataModel to a FITS file.

        Args:
            filename (str, optional): The filename to save to. Defaults to self.filename.
            add_missing_metadata (bool, optional): Whether to add missing metadata to the header. Defaults to False.
        """
        self.on_save(filename)
        if self.filename is None:
            raise ValueError("No filename provided")
        hdulist = fits.HDUList()
        for extname in self.EXTENSIONS:
            hdu = self.make_hdu_for_extension(extname, add_missing_metadata=add_missing_metadata)
            hdulist.append(hdu)
        with warnings.catch_warnings(): # NOTE: This (at minimum) suppresses the 80 char limit warning
            warnings.simplefilter('ignore', VerifyWarning)
            hdulist.writeto(self.filename, overwrite=True)

    def on_save(
        self,
        filename : str | None = None,
    ):
        """
        Hook called when a datamodel is saved. The default implementation:
            1. Updates `self.filename` with full path.
            2. Updates `self.meta.filename` with base filename.
            3. Updates `self.meta.model_type` with class name.

        Args:
            filename (str, optional): The filename to save to.
        """
        if filename is not None:
            self.filename = filename
            self.meta.filename = os.path.basename(filename)

        if self.meta.model_type is None:
            self.meta.model_type = self.__class__.__name__


    def make_hdu_for_extension(self, extname : str, add_missing_metadata : bool = False) -> _VALID_HDU:
        """
        Create an HDU for an extension name.
        
        Args:
            extname (str): The extension name.
            add_missing_metadata (bool, optional): Whether to add missing metadata to the header. Defaults to False.
        """
        hdu_class = self.EXTENSIONS[extname]['hdu_type']
        header = self._make_header(self.meta, extname=extname, add_missing=add_missing_metadata)
        if hdu_class is fits.PrimaryHDU:
            return hdu_class(header=header)
        if hdu_class is fits.ImageHDU:
            data = self._data.get(extname, None)
            if data is not None:
                dtype = self.EXTENSIONS[extname]['dtype']
            hdu = hdu_class(data=data.astype(dtype), name=extname, header=header)
            if np.issubdtype(dtype, np.unsignedinteger):
                hdu.scale(dtype.__name__)
        elif hdu_class is fits.BinTableHDU:
            data = self._data.get(extname, None)
            if data is not None:
                data = astropy.table.Table.from_pandas(data)
            hdu = hdu_class(data=data, name=extname, header=header)
        return hdu


    def get_calibration_file(self, cal_type : str, **kwargs) -> str:
        """
        Get the appropriate calibration file for a DataModel from the local cache or online database.
        To be implemented later.
        This method might be moved to the base primitive class.
        """
        return NotImplementedError(f"Method get_calibration_file not implemented for class {self.__class__.__name__}")


    def __repr__(self):
        if self.filename is not None:
            return f"{self.__class__.__name__}('{os.path.basename(self.filename)}')"
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
    

    def _resolve_extensions(self, extensions : str | list[str] | None = None):
        if extensions is None:
            return list(self.EXTENSIONS.keys())
        if isinstance(extensions, str):
            return [extensions.upper()]
        else:
            return [e.upper() for e in extensions]


    def __getattr__(self, attr):
        if attr.startswith('_') or attr in self._ALLOWED_ATTRIBUTES:
           return super().__getattr__(attr)
        if attr in self.meta:
           return self.meta.__getattr__(attr)
        if attr in self.schema['properties']:
            return None
        if attr in self._data:
            return self._data[attr]
        if attr.upper() in self._data:
            return self._data[attr.upper()]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")


    def validate_metadata(self):
        """
        Validate the metadata against the schema.
        """
        schema.validate(self.schema, self.meta)

    
    def load_schemas(self):
        """
        Load the schemas from the schema files and merges them into one schema.

        Returns:
            dict : The merged schema.
        """
        self.schemas = {}
        self.schema = {}
        for schema_file in self.HEADER_SCHEMAS:
            self.schemas[schema_file] = schema.load_schema(schema_file)
            self.schema = schema.merge_schemas(self.schema, self.schemas[schema_file])
        return self.schema
    

    def init_data(self, data : dict | None):
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
                    self._data[key.upper()] = data[key].astype(self.EXTENSIONS[key.upper()]['dtype'])
                else:
                    raise ValueError(f"Unknown extension: {key.upper()}")
    

    def init_meta(self, meta : MetaDict | None = None):
        """
        Initialize the metadata and optionally merge in provided metadata.

        Args:
            meta (dict, optional): Metadata to merge in. Defaults to None.
        """
        self.meta = MetaDict(_schema=self.schema)
        if meta is not None:
            self.meta.update(meta)


    def init_receipt(self):
        """
        Initialize the receipt.
        """
        if 'RECEIPT' not in self._data:
            self._data['RECEIPT'] = pd.DataFrame({
                name : pd.Series(dtype=dtype)
                for name, dtype in self.EXTENSIONS['RECEIPT']['columns'].items()
            })

    @property
    def level(self):
        """
        Datamodel level to override for each datamodel.
        If level is None, it is not included when creating filenames unless specified.
        """
        return None


    @staticmethod
    def _make_header(meta : MetaDict, extname : str = 'PRIMARY', add_missing : bool = False) -> fits.Header:
        """
        Create a fits header for the given extension.
        
        Args:
            extname (str, optional): The extension name. Defaults to 'PRIMARY'.
            add_missing (bool, optional): Whether to add missing metadata to the header. Defaults to False.
        Returns:
            astropy.io.fits.Header: The header.
        """
        return schema.create_header(meta, extname=extname, add_missing=add_missing)