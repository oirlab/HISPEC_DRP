from __future__ import annotations # NOTE: Not needed in Python 3.13+
import importlib
import inspect
import os
import re
from astropy.table import Table, Column
from astropy.io import fits
from copy import deepcopy

from hispecdrp import __version__
import numpy as np
import warnings
from astropy.io.fits.verify import VerifyWarning
from uuid import uuid4 as make_uuid4
from typing import Any, Sequence, Self, ClassVar, TYPE_CHECKING

if TYPE_CHECKING:
    from ..primitives.base_primitive import BasePrimitive

from hispecdrp.calibrations.hispec_store import HISPECCalibrationStore

from .utils import (
    open as datamodel_open,
    get_instrument_config,
    append_ancestors as _append_ancestors,
    append_receipts as _append_receipts,
    append_calibrations as _append_calibrations,
    migrate_receipt_table,
)

from .meta_data import MetaModel, merge_meta, get_meta_schema
from ..utils import convert_endianness
from .extensions import RECEIPT, CALIBRATIONS, DATA_ANCESTORS
from .utils import _set_datetime_meta, _set_koa_meta, _set_target_meta

from koa_middleware import CalibrationStore
from koa_middleware.utils import get_koa_id_timestamp_from_datetime, get_env_var_bool

import logging
logger = logging.getLogger(__name__)

_VALID_HDU = fits.ImageHDU | fits.BinTableHDU | fits.PrimaryHDU

# Extensions every model inherits from DataModel itself (see DataModel.EXTENSIONS)
# -- never model-specific, so left out of a generated extensions table.
_DOC_TABLE_SPECIAL_EXTENSIONS = ('PRIMARY', 'RECEIPT', 'DATA_ANCESTORS', 'CALIBRATIONS')

# What '#' means for each recognized `enumerate=<tag>` (see .extensions),
# spelled out under a generated extensions table wherever it's actually used.
# An unrecognized tag still gets a (less polished) generic fallback -- see
# DataModel.extensions_table -- so this only needs entries worth wordsmithing.
_DOC_TABLE_ENUMERATE_MEANING = {
    'FIB': 'the fiber number (1..model.num_fibers)',
}

# Display name for each recognized shape-symbol string (see .extensions),
# used only to render a friendly "Dimensions" column in a generated
# extensions table -- has no bearing on actual shape resolution. Axes with
# no single fixed answer (N_ORDERS varies by config; N_FIBERS/N_READS/ANY
# have no doc-time source at all) stay symbolic; DETECTOR_SIZE_Y/X are
# resolved to real numbers where a detector size is available (see
# extensions_table), falling back to these names otherwise.
_DOC_TABLE_AXIS_DISPLAY = {
    'DETECTOR_SIZE_Y': 'Ny',
    'DETECTOR_SIZE_X': 'Nx',
    'N_ORDERS': 'Nord',
    'N_FIBERS': 'Nfib',
    'N_READS': 'Nreads',
    'ANY': 'N',
}


def _doc_table_render_axis(axis, detector_size : tuple[int, int] | None = None) -> str:
    if isinstance(axis, int):
        return str(axis)
    axis = axis.upper()
    if detector_size is not None and axis in ('DETECTOR_SIZE_Y', 'DETECTOR_SIZE_X'):
        return str(detector_size[0 if axis == 'DETECTOR_SIZE_Y' else 1])
    return _DOC_TABLE_AXIS_DISPLAY.get(axis, axis)


def _doc_table_render_shape(shape, detector_size : tuple[int, int] | None = None) -> str:
    if not shape:
        return ''
    return ' x '.join(_doc_table_render_axis(axis, detector_size) for axis in shape)


def _doc_table_render_dtype_name(dtype : type[np.generic]) -> str:
    match = re.match(r'([a-z]+)(\d*)', dtype.__name__)
    kind, bits = match.group(1), match.group(2)
    kind_display = {'float': 'Float', 'uint': 'UInt', 'int': 'Int', 'bool': 'Bool', 'complex': 'Complex'}
    return kind_display.get(kind, kind.capitalize()) + bits


def _doc_table_render_dtype(dtype) -> str:
    if dtype is None:
        return ''
    dtypes = dtype if isinstance(dtype, tuple) else (dtype,)
    return ' | '.join(_doc_table_render_dtype_name(d) for d in dtypes)


class _classproperty:
    """
    A read-only property accessible on the class itself, not just instances.

    Deliberately a non-data descriptor (no ``__set__``): an instance's own
    ``__dict__`` entry (e.g. ``self.config`` set in ``__init__``) always takes
    priority over this on that instance. This is only consulted for
    class-level access (``SomeModel.config``) or from within a classmethod
    (``cls.config``), where there is no instance to hold a per-instance value.
    """
    def __init__(self, fget):
        self.fget = fget

    def __get__(self, obj, owner):
        return self.fget(owner)


def _build_empty_table_with_cols(extdef : dict) -> Table:
    """
    Build a 0-row :class:`astropy.table.Table` from an extension's ``'columns'`` spec.
    """
    cols = [
        Column(name=name, data=np.empty(0, dtype=dtype))
        for name, dtype in extdef["columns"].items()
    ]
    return Table(cols)


def _canonical_dtype(dtype_spec):
    """The dtype to construct/cast to: the first entry if `dtype_spec` is a
    tuple (see :func:`.extensions.IMAGE_HDU`), else `dtype_spec` itself."""
    return dtype_spec[0] if isinstance(dtype_spec, tuple) else dtype_spec


def _dtype_family(dtype) -> str:
    """
    Broad numeric family of a dtype, for kind-aware auto-casting.

    Numpy distinguishes signed ('i') and unsigned ('u') integers, but for
    auto-casting purposes any integer array should match a declared integer
    dtype, and any float array should match a declared float dtype.
    """
    kind = np.dtype(dtype).kind
    if kind in ('i', 'u'):
        return 'integer'
    if kind == 'f':
        return 'float'
    return kind


def _correct_image_dtype(data : np.ndarray, dtype_spec, extname : str = '') -> np.ndarray:
    """
    Cast an image extension's array to its declared dtype, if needed.

    ``dtype_spec`` is an extension's declared ``'dtype'`` entry (see
    :func:`.extensions.IMAGE_HDU`): a single numpy scalar type, or a
    precedence-ordered tuple of them for extensions with more than one
    legitimate dtype (e.g. ``RampModel.DATA``: uint16 for real detector data,
    float32 for simulations run with ``convert_to_uint16=False``).

    If ``data``'s dtype already matches one of the declared choices
    (subtype-wise), it's returned unchanged. Otherwise the declared choice
    sharing ``data``'s broad numeric family (see :func:`_dtype_family` --
    e.g. any integer array matches a declared integer dtype, any float array
    matches a declared float dtype) is used as the cast target, falling back
    to the first (canonical) declared dtype if none share that family.
    """
    choices = dtype_spec if isinstance(dtype_spec, tuple) else (dtype_spec,)
    if any(np.issubdtype(data.dtype, d) for d in choices):
        return data

    target = next(
        (d for d in choices if _dtype_family(d) == _dtype_family(data.dtype)),
        choices[0],
    )
    logger.warning(
        f"Converting data type of {extname or 'extension'} from {data.dtype} to {np.dtype(target)}"
    )
    return data.astype(target, copy=False)


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

    # Process-wide default instrument config, settable via
    # `datamodels.utils.set_instrument`. Only consulted by the `config`
    # classproperty below when there's no instance (or no per-instance
    # `self.config` override) to read from -- see `_classproperty`.
    _config : ClassVar[type | None] = None

    @_classproperty
    def config(cls) -> type | None:
        return cls._config

    @classmethod
    def _doc_default_detector_size(cls) -> tuple[int, int] | None:
        """
        Resolve ``(ny, nx)`` purely to render real numbers in a generated
        extensions table's Dimensions column (see :meth:`extensions_table`)
        -- from ``cls._DEFAULT_INSTRUMENT``'s config, since this runs from
        :meth:`__init_subclass__` at class-definition time, before any
        instance (and so any live ``self.config``/``self.default_detector_size``,
        which a multi-detector model could have overridden) exists.

        Deliberately does *not* go through :func:`.utils.get_instrument_config`
        (which reads ``INSTRUMENT_CONFIGS``, a dict this very package's
        ``__init__.py`` only builds *after* importing every model class --
        every ``__init_subclass__`` call happens before it exists). Instead
        imports ``<instrument>_config`` directly by name, matching this
        package's own ``hispec_config.py``/``parvi_config.py`` convention --
        that submodule only needs ``base_model``/``meta_data``, already
        loaded by the time any model class is being defined, regardless of
        how far this package's own ``__init__.py`` has otherwise gotten.

        Never raises; ``None`` (axes stay symbolic, e.g. ``Ny``/``Nx``) if
        the default instrument's config can't be resolved this way or has
        no ``DETECTOR_SIZE``.
        """
        try:
            module = importlib.import_module(f'.{cls._DEFAULT_INSTRUMENT.lower()}_config', package=__package__)
            config = getattr(module, f'{cls._DEFAULT_INSTRUMENT.upper()}Config')
        except Exception:
            return None
        return getattr(config, 'DETECTOR_SIZE', None)

    @classmethod
    def _extensions_rows(cls) -> tuple[list[str], list[list[str]], list[str]]:
        """
        Shared data behind both :meth:`extensions_table` (plain aligned text,
        for standalone/interactive use, e.g. ``help()``) and the RST table
        spliced into the class docstring by :meth:`__init_subclass__` --
        one source of truth read straight from ``cls.EXTENSIONS``, so
        neither can drift from it the way a hand-written docstring table
        could. ``title=``/``units=`` (see :func:`.extensions.IMAGE_HDU`/
        :func:`.extensions.TABLE_HDU`) fill the Description/Units columns;
        everything else (name, HDU type, dtype, shape) is read directly off
        the extension. ``DETECTOR_SIZE_Y``/``DETECTOR_SIZE_X`` render as
        real numbers (see :meth:`_doc_default_detector_size`); other
        symbolic axes (``N_ORDERS``, etc.) stay symbolic since they have no
        single fixed doc-time answer.

        Returns
        -------
        headers : list[str]
        rows : list[list[str]]
        tag_notes : list[str]
            One explanatory line per enumerate tag in use, e.g. "'_FIB#' in
            an HDU name denotes the fiber number."
        """
        detector_size = cls._doc_default_detector_size()
        rows = []
        tags = set()
        # _populate_shape_info fills in 'shape' for an extension that
        # omitted it (see IMAGE_HDU's shape=None) from its family's
        # primary, purely for an accurate Dimensions column here -- the
        # same fill-in auto-init does per-instance at construction time.
        for name, info in cls._populate_shape_info(cls.EXTENSIONS).items():
            if name in _DOC_TABLE_SPECIAL_EXTENSIONS:
                continue
            tag = info.get('enumerate')
            hdu_name = f'{name}_{tag}#' if tag else name
            if tag:
                tags.add(tag)
            if info['hdu_type'] is fits.BinTableHDU:
                hdu_type, dtype, dims = 'Table', 'See docs', 'See docs'
            else:
                hdu_type = 'Image'
                dtype = _doc_table_render_dtype(info.get('dtype'))
                dims = _doc_table_render_shape(info.get('shape'), detector_size)
            rows.append([
                hdu_name, hdu_type, dtype, dims,
                info.get('units') or '', info.get('title') or '',
            ])

        headers = ['HDU Name', 'HDU Type', 'Data Type', 'Dimensions', 'Units', 'Description']
        # Explain what '#' means for every enumerate tag actually used below,
        # e.g. '_FIB#' -> the fiber number -- read before the table that uses it.
        tag_notes = [
            f"'_{tag}#' in an HDU name denotes "
            f"{_DOC_TABLE_ENUMERATE_MEANING.get(tag, f'the {tag.lower()} number')}."
            for tag in sorted(tags)
        ]
        return headers, rows, tag_notes

    @classmethod
    def extensions_table(cls) -> str:
        """
        Render the "HDU Name / HDU Type / Data Type / Dimensions / Units /
        Description" table from :meth:`_extensions_rows` as plain,
        whitespace-aligned text -- the layout ``help()`` and other
        interactive tools show. Spliced into the class docstring (as a
        proper RST table, not this plain-text form) by
        :meth:`__init_subclass__`; call this directly only if you need the
        plain-text table standalone (e.g. to print it, or embed it
        elsewhere). Returns ``''`` when there are no non-special extensions
        to show, e.g. for an abstract intermediate base class.
        """
        headers, rows, tag_notes = cls._extensions_rows()
        if not rows:
            return ''

        widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]

        def fmt_row(cells):
            return '    '.join(cell.ljust(width) for cell, width in zip(cells, widths)).rstrip()

        lines = list(tag_notes)
        if lines:
            lines.append('')

        lines.append(fmt_row(headers))
        lines.append(fmt_row(['-' * width for width in widths]))
        lines.extend(fmt_row(row) for row in rows)

        return '\n'.join(f'    {line}' for line in lines)

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
        shape : tuple[int, ...] | None = None,
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
        shape : tuple[int, ...] | None, optional
            Explicit shape to use for ``model.shape`` when the primary extension has no data of its own yet. Once the primary extension is actually assigned real data, its array shape takes precedence over this override.
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

        # Explicit shape override (see `shape` property); only used as a fallback
        # when the primary extension has no real array to derive a shape from.
        self._shape = shape

        # Get the extension names now that we know the instrument
        self.EXTENSIONS = self._get_extensions(self.num_fibers)

        # Set the optional filepath
        if filepath is not None:
            if not os.path.isfile(filepath):
                raise FileNotFoundError(f"File not found: {filepath}")
            self._filepath = os.path.abspath(filepath)
            self.init_meta(meta=meta, validate=validate_meta)
            self.meta.filename = os.path.basename(self._filepath)
            if not meta_only:
                self.read_data(
                    extensions=extensions,
                    correct_dtypes=correct_dtypes
                )
        else:
            self._filepath = None
            self.init_meta(meta=meta, validate=validate_meta)
            if not meta_only:
                self.init_data(correct_dtypes=correct_dtypes, **kwargs)

        # Hook
        self.on_init()

    @classmethod
    def _get_extensions(cls, num_fibers : int) -> dict:
        extensions = cls._expand_enumerated_extensions(cls.EXTENSIONS, num_fibers)
        return cls._populate_shape_info(extensions)

    @classmethod
    def _populate_shape_info(cls, extensions : dict) -> dict:
        """
        Fill in ``'shape'`` for any image extension that omitted it (see
        :func:`.extensions.IMAGE_HDU`'s ``shape=None``), copying the primary
        extension's declared shape -- the first key not in the reserved
        special names (``PRIMARY``, ``RECEIPT``, ``DATA_ANCESTORS``,
        ``CALIBRATIONS``), matching the base :attr:`primary_extension_name`
        default. A copy of the info dict is only made where a fill-in is
        actually needed, so shared info dicts (e.g. ``DQ_L0``, reused as-is
        across many models) are never mutated in place.

        A model with a dynamic ``primary_extension_name`` override
        (returning ``None`` until some fiber's data is actually set) can't
        rely on this -- it always declares every extension's own ``shape=``
        explicitly, so there's nothing left here to fill in for it.
        """
        special = ('PRIMARY', 'RECEIPT', 'DATA_ANCESTORS', 'CALIBRATIONS')
        primary_name = next((name for name in extensions if name not in special), None)
        primary_shape = extensions[primary_name].get('shape') if primary_name is not None else None
        if primary_shape is None:
            return extensions

        populated = {}
        for name, info in extensions.items():
            if name != primary_name and 'shape' in info and info.get('shape') is None:
                info = {**info, 'shape': primary_shape}
            populated[name] = info
        return populated

    @classmethod
    def _expand_enumerated_extensions(cls, extensions : dict, num_fibers : int) -> dict:
        """
        Expand any extension marked ``enumerate=<tag>`` (see
        :func:`.extensions.IMAGE_HDU` / :func:`.extensions.TABLE_HDU`) into
        one ``<NAME>_<tag>{n}`` entry per fiber -- the format is always
        ``prefix_{enumerate}#`` -- for ``n in 1..num_fibers``. Extensions
        without ``enumerate`` pass through unchanged.
        """
        expanded = {}
        for name, info in extensions.items():
            tag = info.get('enumerate')
            if tag is not None:
                for fiber in range(1, num_fibers + 1):
                    expanded[f'{name}_{tag}{fiber}'] = info
            else:
                expanded[name] = info
        return expanded

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
                # PRIMARY never gets a data attribute -- it's header-only and
                # handled specially by make_hdu.
                if extname == 'PRIMARY':
                    continue
                extl = extname.lower()
                if extname in hdulist:
                    data = self.read_extension_data(
                        hdulist[extname],
                        correct_dtype=correct_dtypes
                    )
                    setattr(self, extl, data)
                elif extname not in self.EXTENSIONS:
                    raise ValueError(f"Extension {extname} not found in FITS file: {self._filepath}")

        # Initialize remaining extensions not set by user or FITS file.
        # Table HDUs always get an empty Table now. Everything else (image
        # extensions, and PRIMARY, which never gets an attribute at all) is
        # left unset, so that accessing an image extension lazily
        # auto-initializes a default array (see DataModel.__getattr__ /
        # _make_default_array), or resolves to None via __getattr__.
        for extname in self.EXTENSIONS:
            extl = extname.lower()
            if extl in self.__dict__:
                continue
            info = self.EXTENSIONS[extname]
            if 'columns' in info:
                setattr(self, extl, _build_empty_table_with_cols(info))
    
    def read_extension_data(
        self, hdu : _VALID_HDU,
        correct_dtype : bool = False
    ) -> np.ndarray | Table:
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
        data : np.ndarray or Table
            If an ImageHDU, returns a numpy ndarray.
            If a BinTableHDU, returns an astropy Table.
        """
        hdu_type = type(hdu)
        data = hdu.data
        if hdu_type is fits.ImageHDU:
            if data is not None:
                if correct_dtype:
                    dtype = self.EXTENSIONS[hdu.name]['dtype']
                    data = _correct_image_dtype(data, dtype, extname=hdu.name)
                data = convert_endianness(data, output_endianness='little')
        elif hdu_type is fits.BinTableHDU:
            data = Table(data)
            if getattr(hdu, "name", None) == "RECEIPT":
                data = migrate_receipt_table(data)
        return data

    def migrate(self) -> Self:
        """
        Apply schema migrations to this model's current data, in place.

        Useful for upgrading an already-constructed model -- e.g. one built
        from kwargs or merged via :meth:`from_datamodels` -- not just data
        freshly read from a FITS file.

        Returns
        -------
        Self
            The updated model.
        """
        if self.receipt is not None:
            self.receipt = migrate_receipt_table(self.receipt)
        return self

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
        **kwargs
            Additional keyword arguments to pass to the fits.HDUList.writeto method.
        """
        # TODO: Reconsider how _filepath is set.
        if filepath is not None:
            self._filepath = filepath
        else:
            self._filepath = self._generate_output_path(
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
    
    def _generate_output_path(
        self,
        filename : str | None = None,
        output_dir : str | None = None,
        suffix : str | None = None,
    ) -> str:
        if output_dir is not None and filename is not None:
            self.meta.filename = filename
            return os.path.join(
                os.path.abspath(output_dir),
                filename
            )
        elif output_dir is not None and filename is None:
            filename = self.generate_filename(suffix=suffix)
            return os.path.join(
                os.path.abspath(output_dir),
                filename
            )
        elif output_dir is None and filename is not None:
            return os.path.join(os.getcwd(), filename)
        else:
            filename = self.generate_filename(suffix=suffix)
            return os.path.join(os.getcwd(), filename)

    def get_output_path(
        self,
        filename: str | None = None,
        output_dir: str | None = None,
        suffix: str | None = None,
    ) -> str:
        """Return the FITS path that :meth:`save` would write for ``output_dir`` / ``suffix``."""
        return self._generate_output_path(
            filename=filename,
            output_dir=output_dir,
            suffix=suffix,
        )
    
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
                and self._extension_missing_or_empty(extname) \
                and extname not in ('RECEIPT', 'PRIMARY'):
                continue
            hdu = self.make_hdu(
                extname,
                include_missing_metadata=include_missing_metadata,
                correct_dtypes=correct_dtypes
            )
            if hdu is not None:
                hdulist.append(hdu)
        
        return hdulist

    def _extension_missing_or_empty(self, extname: str) -> bool:
        """
        True if extension data is absent or an empty BinTable.

        Empty :class:`astropy.table.Table` (0 rows) must be treated like missing when
        building FITS output with ``include_empty_extensions=False``: Astropy cannot
        serialize 0-row tables that include object columns (e.g. variable-length
        ``CCF`` arrays) because no row exists to infer inner dtypes
        (:func:`astropy.io.fits.table_to_hdu` / ``ColDefs._init_from_array``).

        Reads the raw (un-auto-initialized) value: this is only called when
        ``include_empty_extensions=False``, in which case an image extension
        that was never explicitly set should be treated as missing rather than
        lazily materialized into a default array just to decide whether to save it.
        """
        val = self._raw_extension_value(extname)
        if val is None:
            return True
        if isinstance(val, Table) and len(val) == 0:
            return True
        return False

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
        _set_target_meta(self)

        if self.meta.last_processed is None:
            self.meta.last_processed = self.meta.datetime_obs

        if self.meta.instrument_era is None:
           self.meta.instrument_era = "0.0.1"

        # Apply instrument metadata defaults
        self.config.set_meta_defaults(self)

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
            if data is not None:
                if correct_dtypes:
                    dtype = self.EXTENSIONS[extname]['dtype']
                    data = _correct_image_dtype(data, dtype, extname=extname)
                data = convert_endianness(data, output_endianness='little')
            hdu = hdu_class(data=data, name=extname)
        
        # Nothing special for BIN tables
        elif hdu_class is fits.BinTableHDU:
            data = getattr(self, extl, None)
            if data is None:
                return None
            elif len(data) == 0:
                dtypes = {name: data[name].dtype for name in data.colnames}
                if any(dtype == object for dtype in dtypes.values()):
                    return None
                else:
                    return hdu_class(data=data, name=extname)
            else:
                return hdu_class(data=data, name=extname)
            #if data is None:
            #    hdu = hdu_class(data=data, name=extname)
            #elif isinstance(data, Table):


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

        # Splice a generated "Extensions" table (see _extensions_rows) onto
        # the docstring now that EXTENSIONS is in its final, fully-merged
        # form -- keeps the docstring in sync with EXTENSIONS automatically
        # instead of relying on a hand-maintained table that can drift.
        headers, rows, tag_notes = cls._extensions_rows()
        if rows:
            # inspect.cleandoc (not just .rstrip()) matters here: Python's compiler
            # only started dedenting docstrings at compile time in 3.13, so on
            # 3.11/3.12 cls.__doc__ still carries its raw 4-space source indent.
            # Without normalizing that away first, appending our own lines below
            # at a *guessed* indent goes wrong on whichever Python version guessed
            # wrong -- e.g. appending at 4 spaces when the base doc is already
            # dedented to 0 makes our lines look more indented than the paragraph
            # above them, which RST reads as a block quote (hence the stray left
            # border). Normalize once here so both this text and the appended
            # block always share the same, known 0-indent baseline.
            doc = inspect.cleandoc(cls.__doc__ or '').rstrip()
            # A raw "Extensions\n----------" section title isn't one of the
            # numpydoc sections napoleon rewrites (Parameters, Returns, ...), so it
            # reaches docutils as a literal RST section title -- which autodoc's
            # "Unexpected section title" rejects, since a directive's content may
            # only hold body elements, not structural ones. Plain **bold** text is
            # a body element instead (and, unlike `.. rubric::`, isn't italicized
            # by the theme). The table itself is a real `list-table` (relative
            # column widths from the same character counts extensions_table()
            # uses for alignment) so it renders as an actual bordered table, not a
            # preformatted/code-block one.
            widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]

            def rst_row(cells):
                first, *rest = cells
                out = [f'   * - {first}']
                out.extend(f'     - {cell}' for cell in rest)
                return out

            block = ['**Extensions**', '']
            block += list(tag_notes)
            if tag_notes:
                block.append('')
            block += [
                '.. list-table::',
                '   :header-rows: 1',
                '   :widths: ' + ' '.join(str(w) for w in widths),
                '',
            ]
            block += rst_row(headers)
            for row in rows:
                block += rst_row(row)

            cls.__doc__ = f"{doc}\n\n" + '\n'.join(block)

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

        attru = attr.upper()
        info = self.EXTENSIONS.get(attru)
        if info is not None and info.get('hdu_type') is fits.ImageHDU:
            default = self._make_default_array(attru, info)
            if default is not None:
                setattr(self, attrl, default)
                return default
            return None

        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")

    def _infer_shape_symbols(self) -> dict:
        """
        Scan every currently-set image extension and, for each, match its
        real array shape positionally against its own declared ``shape=``
        spec (see :func:`.extensions.IMAGE_HDU`; populated for every
        extension, even one that omitted it itself, by
        :meth:`_populate_shape_info` at construction time), recording
        ``{symbol: size}`` for every non-literal, non-``ANY`` symbol
        position.

        This is what lets e.g. an explicitly-assigned
        ``model.bias = np.ones((10, 10))`` inform the shape used to
        auto-init ``model.ktc``, ``model.dq``, etc. on that same model --
        nothing is hardcoded about "bias" being special; *any* extension
        whose declared shape shares a symbol with another can inform it, as
        long as it's actually been set. Also the only source ``N_READS`` (no
        independent resolver -- see :meth:`_resolve_shape_axis`) ever has.
        """
        inferred = {}
        for extname, info in self.EXTENSIONS.items():
            spec = info.get('shape')
            if spec is None:
                continue
            val = self._raw_extension_value(extname)
            if not isinstance(val, np.ndarray) or val.ndim != len(spec):
                continue
            for axis, size in zip(spec, val.shape):
                if isinstance(axis, str) and axis.upper() != 'ANY':
                    inferred[axis.upper()] = size
        return inferred

    @property
    def default_detector_size(self) -> tuple[int, ...] | None:
        """
        This model's ``(ny, nx)`` detector plane size -- the last-resort
        source for the ``DETECTOR_SIZE_Y``/``DETECTOR_SIZE_X`` shape-symbol
        resolvers (see :meth:`_resolve_shape_axis`) when no extension's
        already-set real data and no explicit constructor ``shape=``
        override provide it instead.

        Defaults to ``config.DETECTOR_SIZE``, which is all a single-detector
        instrument ever needs. An instrument with multiple, differently
        sized detectors has no single automatic answer -- override this
        property (in a subclass, or an intermediate base shared by that
        detector's models) to compute the right one from whatever instance
        state actually distinguishes them, e.g.::

            @property
            def default_detector_size(self):
                return self.config.DETECTOR_SIZES.get(self.meta.instrument_mode)
        """
        return getattr(self.config, 'DETECTOR_SIZE', None)

    def _resolve_shape_axis(self, axis, inferred : dict) -> int | None:
        """
        Resolve one axis of a ``shape=`` spec (see :func:`.extensions.IMAGE_HDU`)
        to a concrete size, or ``None`` if it can't be resolved on this model
        right now.

        A string axis is matched case-insensitively against the recognized
        symbolic names (``DETECTOR_SIZE_Y``, ``DETECTOR_SIZE_X``,
        ``N_ORDERS``, ``N_FIBERS``, ``N_READS``, ``ANY``). Checked in order:
        (1) ``inferred`` (see :meth:`_infer_shape_symbols`); (2) the symbol's
        own independent source, if it has one -- ``DETECTOR_SIZE_Y``/
        ``DETECTOR_SIZE_X`` fall back to the constructor's ``shape=``
        override, then :attr:`default_detector_size`; ``N_ORDERS`` to
        ``meta.order_start``/``order_end``; ``N_FIBERS`` to
        ``config.num_fibers``. ``N_READS`` and any other symbol with no
        independent source are only ever resolvable via ``inferred``. ``ANY``
        is never resolvable at all, by design.
        """
        if isinstance(axis, int):
            return axis
        axis = axis.upper()
        if axis == 'ANY':
            return None
        if axis in inferred:
            return inferred[axis]
        if axis == 'DETECTOR_SIZE_Y' or axis == 'DETECTOR_SIZE_X':
            size = self.__dict__.get('_shape') or self.default_detector_size
            if size is None:
                return None
            return size[-2] if axis == 'DETECTOR_SIZE_Y' else size[-1]
        if axis == 'N_ORDERS':
            start, end = self.meta.order_start, self.meta.order_end
            if start is None or end is None:
                return None
            return abs(end - start) + 1
        if axis == 'N_FIBERS':
            return self.config.num_fibers
        return None

    def _resolve_shape(self, shape_spec) -> tuple[int, ...] | None:
        """
        Resolve a full ``shape=`` spec (see :func:`.extensions.IMAGE_HDU`) to
        a concrete shape tuple, or ``None`` if any axis can't be resolved.
        """
        if shape_spec is None:
            return None
        inferred = self._infer_shape_symbols()
        resolved = []
        for axis in shape_spec:
            size = self._resolve_shape_axis(axis, inferred)
            if size is None:
                return None
            resolved.append(size)
        return tuple(resolved)

    def _make_default_array(self, extname : str, info : dict) -> np.ndarray | None:
        """
        Lazily build the default array for an image extension, if eligible.

        If the extension declares ``'default'`` (see :func:`.extensions.IMAGE_HDU`)
        and its ``'shape'`` spec (populated for every extension by
        :meth:`_populate_shape_info` at construction time) fully resolves
        (see :meth:`_resolve_shape`), the array is built directly as
        ``np.full(resolved_shape, default, dtype=...)``. Otherwise, building
        it is delegated to :meth:`make_default_array` -- tried
        unconditionally, whether or not ``default`` was given -- which
        subclasses can override for extensions whose shape can't be expressed
        with the available symbols (e.g. polynomial coefficients, a
        per-instance fit result).
        """
        default = info.get('default')
        if default is not None:
            shape = self._resolve_shape(info.get('shape'))
            if shape is not None:
                return np.full(shape, default, dtype=_canonical_dtype(info['dtype']))

        return self.make_default_array(extname, info)

    def make_default_array(self, extname : str, info : dict) -> np.ndarray | None:
        """
        Build a default array for an image extension whose shape can't be
        resolved from its declared ``shape=`` spec alone.

        Called by :meth:`_make_default_array` whenever the automatic
        ``np.full(resolved_shape, default, dtype=...)`` path doesn't apply --
        e.g. an extension containing ``ANY`` in its ``shape=`` (like
        polynomial coefficients, whose axis sizes are a per-instance fit
        result with no symbol at all), or one with no ``default`` given.

        Override in a subclass to handle specific extension names, falling
        back to ``super().make_default_array(extname, info)``. The base
        implementation returns ``None`` (no custom default), so models that
        don't override this see no behavior change.

        Parameters
        ----------
        extname : str
            The extension name (uppercase).
        info : dict
            The extension's declared descriptor (see :func:`.extensions.IMAGE_HDU`).
        """
        return None

    def _raw_extension_value(self, extname : str):
        """Read an extension's current value without triggering lazy auto-initialization."""
        return self.__dict__.get(extname.lower())

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
                if isinstance(val, Table):
                    if keyu == "RECEIPT":
                        val = migrate_receipt_table(val)
                    setattr(self, keyl, val)
                elif isinstance(val, np.ndarray):
                    if correct_dtypes:
                        val = _correct_image_dtype(val, self.EXTENSIONS[keyu]['dtype'], extname=keyu)
                    val = convert_endianness(val, output_endianness='little')
                    setattr(self, keyl, val)
                else:
                    raise ValueError(f"Invalid data type for extension {keyu}: {type(val)}")
            else:
                raise ValueError(f"Unknown extension: {key.upper()}")
            
        # Initialize remaining extensions. Table HDUs always get an empty Table now.
        # Everything else (image extensions, and PRIMARY, which never gets an
        # attribute at all) is left unset, so that accessing an image extension
        # lazily auto-initializes a default array (see DataModel.__getattr__ /
        # _make_default_array), or resolves to None via __getattr__.
        for extname in self.EXTENSIONS:
            extl = extname.lower()
            if extl in self.__dict__:
                continue
            info = self.EXTENSIONS[extname]
            if 'columns' in info:
                setattr(self, extl, _build_empty_table_with_cols(info))

    def init_meta(self, meta: dict | MetaModel | None = None, validate : bool = True):
        """
        Initialize the metadata and optionally merge in provided metadata.

        Parameters
        ----------
        meta : dict or MetaModel, optional
            Existing metadata to store or merge.
        validate : bool, optional
            Whether the resulting :class:`MetaModel` validates values against
            the schema on every assignment. Only applies when a new
            ``MetaModel`` is built here (from a dict or ``None``) -- an
            already-built ``MetaModel`` passed in via ``meta`` keeps its own
            ``validate`` setting as-is.
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

            self.meta = MetaModel(meta_schema, merged, validate=validate)

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
                self.data_ancestors.add_column(Column(
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
                data = self._raw_extension_value(extl)
                if data is not None:
                    dtype = data.dtype
                    print(f"{extname}\n  dtype: {dtype}\n  shape: {data.shape}")
            elif hdu_type is fits.BinTableHDU:
                data = self._raw_extension_value(extl)
                if data is not None:
                    print(f"{extname}\n  columns: {list(data.columns)}\n  nrows: {len(data)})")

    @property
    def data_dict(self):
        data = {}
        for extname, info in self.EXTENSIONS.items():
            extl = extname.lower()
            hdu_type = info['hdu_type']
            if hdu_type is fits.PrimaryHDU:
                continue
            data[extl] = self._raw_extension_value(extl)
        return data

    @property
    def data_level_suffix(self):
        return None

    @classmethod
    def from_datamodels(
        cls,
        dataset : Any,
        append_receipts: bool = True,
        ancestor_rule : str = 'these', # 'these', 'append', 'none'
        append_calibrations: bool = True,
        include_data : bool = False,
        primitive : 'BasePrimitive' | None = None,
        **kwargs
    ) -> 'DataModel':
        """
        Create a new DataModel by merging metadata from multiple input models.

        Parameters
        ----------
        dataset : DataModel | str | list[DataModel | str] | dict[str, DataModel | str]
            Input DataModel(s) or file path(s) to merge.
        append_receipts : bool, optional
            Whether to append receipts from input models. Defaults to True.
        ancestor_rule : str, optional
            Rule for handling data ancestors. Defaults to ``these``.

            - ``these``: Add the input models themselves to the data ancestors of the output model, but do not copy their ancestors.
            - ``append``: Copy the data ancestors from the input models to the output model, but do not include the input models themselves as ancestors.
            - ``none``: Do not copy ancestors or add input models as ancestors.

        append_calibrations : bool, optional
            Whether to append calibrations from input models. Defaults to True.
        include_data : bool, optional
            Whether to include data from the input model (must only be one). Defaults to False.

        Returns
        -------
        DataModel
            A new DataModel with merged metadata and optional data.
        """

        if dataset is None:
            raise ValueError("No models provided")
        
        # Normalize to iterable
        if isinstance(dataset, (str, DataModel)):
            dataset = [dataset]
        elif isinstance(dataset, dict):
            dataset = [
                item
                for v in dataset.values()
                for item in (v if isinstance(v, list) else [v])
            ]
        elif not isinstance(dataset, Sequence):
            raise ValueError("Input must be a DataModel, filepath, list of DataModels/filepaths, or dict of DataModels/filepaths.")
        
        # Sort inputs by datetime_obs
        dataset = sorted(dataset, key=lambda x: datamodel_open(x, meta_only=True).meta.datetime_obs)

        # Merge metadata
        meta_out = merge_meta(dataset)

        # Get first model
        first_model = datamodel_open(dataset[0])

        # Create new model
        model_out = cls(
            meta=meta_out,
            instrument_name=meta_out.instrument_name,
            config=first_model.config,
        )

        # Forward data?
        if include_data:
            assert len(dataset) == 1, "Cannot include data when merging multiple models. Set include_data=False or provide a single model."
            first_model = datamodel_open(dataset[0])
            for extname in model_out.EXTENSIONS:
                if extname in ('PRIMARY', 'DATA_ANCESTORS', 'CALIBRATIONS', 'RECEIPT'):
                    continue
                if extname in first_model.EXTENSIONS:
                    extl = extname.lower()
                    setattr(model_out, extl, getattr(first_model, extl, None))

        ancestor_rule = ancestor_rule.lower()
        for _input in dataset:

            input_model = datamodel_open(_input)

            if append_receipts:
                _append_receipts(model_out, input_model)

            if ancestor_rule == 'append':
                _append_ancestors(model_out, input_model)
            elif ancestor_rule == 'these':
                # Add THIS model to the ancestors table, not its ancestors
                primitive.add_data_ancestors(model_out, input_model)
            elif ancestor_rule == 'none':
                # Do nothing
                pass
            else:
                raise ValueError(f"Unknown ancestor_rule: {ancestor_rule}")

            if append_calibrations:
                _append_calibrations(model_out, input_model)

        for k in kwargs:
            if hasattr(model_out, k):
                setattr(model_out, k, kwargs[k])

        return model_out
    
    def make_order_vec(self) -> np.ndarray:
        return self._make_order_vec(self.meta.order_start, self.meta.order_end)

    @property
    def n_orders(self) -> int:
        """
        Return the number of orders in the model based on the order_start and order_end metadata.
        """
        return abs(self.meta.order_end - self.meta.order_start) + 1
        
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
    def orders(self) -> np.ndarray:
        """Echelle order vector from ``meta.order_start`` / ``order_end``."""
        return self.make_order_vec()

    def order_index(self, order: int) -> int:
        """Return the row index of ``order`` in :meth:`make_order_vec`."""
        orders = self.orders
        hits = np.where(np.asarray(orders) == int(order))[0]
        if hits.size == 0:
            raise ValueError(f"Order {order} is not in the order vector: {orders}")
        return int(hits[0])

    # @property
    # def traces(self):
    #     """
    #     Convenience property to return a dictionary of traces for each fiber in the model.
    #     This is useful when the fiber is a dynamic variable and can be used in place of getattr(model, '<attr>_fib{fiber}').
    #     By default, any attribute with '_fib' in its name is included.
    #     This property may be removed in future versions unless it demonstrates significant utility.
    #     This property should not be used in large for loops if the rest of the computation is trivial as it adds some overhead.

    #     Examples:
    #         >>> model.traces # -> {1 : {'wave' : ..., 'spec' : ..., 'spec_err' : ...}, 2 : {...}, ...}
    #         >>> model.traces[1]['wave'] # Equivalent to model.wave_fib1

    #     Returns:
    #         dict: A dictionary where keys are fiber numbers and values are dictionaries of attributes without the '_fib#' suffix.
    #     """
    #     trace_data = {}
    #     for attr in self.EXTENSIONS:
    #         attrl = attr.lower()
    #         if '_fib' in attrl:
    #             try:
    #                 kind, fiber = attrl.rsplit('_fib', 1)
    #                 fiber = int(fiber)
    #             except ValueError:
    #                 continue
    #             trace_data.setdefault(fiber, {})
    #             trace_data[fiber][kind] = getattr(self, attr, None)
    #     return trace_data

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
    
    @classmethod
    def has_extension(cls, extname: str) -> bool:
        """
        Whether ``extname`` (case-insensitive) is a valid extension for this class.

        Cheap: never rebuilds the enumerated-expanded ``EXTENSIONS`` dict (see
        :meth:`_expand_enumerated_extensions`) -- pattern-matches
        ``<BASE>_<tag>{n}`` (the format is always ``prefix_{enumerate}#``)
        against ``cls.EXTENSIONS``'s ``enumerate``-marked base entries instead.

        A ``_<tag>{n}`` suffix is bounds-checked against ``cls.config.num_fibers``
        when ``cls.config`` is set (see the ``config`` classproperty and
        :func:`datamodels.utils.set_instrument`); otherwise any ``_<tag>{n}``
        suffix on an ``enumerate``-marked base is accepted without a bound.
        """
        extname = extname.upper()

        info = cls.EXTENSIONS.get(extname)
        if info is not None:
            # An enumerate-marked base name (e.g. 'SPEC') is never itself a
            # real extension -- only its _<tag>{n} expansions are.
            return info.get('enumerate') is None

        for base, base_info in cls.EXTENSIONS.items():
            tag = base_info.get('enumerate')
            if tag is None:
                continue
            prefix = f'{base}_{tag}'
            if not extname.startswith(prefix):
                continue
            suffix = extname[len(prefix):]
            if not suffix.isdigit():
                continue
            config = cls.config
            return config is None or 1 <= int(suffix) <= config.num_fibers
        return False

    def has_extension_value(self, extname: str) -> bool:
        """
        Whether ``extname`` (case-insensitive) is both a valid extension for
        this model and currently holds a value.

        Reads the raw (un-auto-initialized) value. This must not trigger lazy
        auto-initialization: several ``primary_extension_name`` overrides call this
        to find which fiber has actually been set, and lazy auto-initialization
        itself depends on ``primary_extension_name`` (via ``model.shape``), which
        would recurse infinitely.
        """
        if not self.has_extension(extname):
            return False
        return self._raw_extension_value(extname) is not None

    @property
    def primary_extension_name(self):
        """
        Return the name of the primary extension for this datamodel. By default, this is the first extension declared within the EXTENSIONS dictionary that is not one of the special extensions: 'PRIMARY', 'RECEIPT', 'DATA_ANCESTORS', or 'CALIBRATIONS'.
        """
        for extname in self.EXTENSIONS:
            if extname not in ('PRIMARY', 'RECEIPT', 'DATA_ANCESTORS', 'CALIBRATIONS'):
                return extname
        return None

    @property
    def shape(self):
        """
        The shape of the model's primary extension.

        Reads the primary extension's *raw* value (never triggers lazy
        auto-initialization). If the primary extension has no real array yet,
        falls back to the ``shape=`` override passed to the constructor, else
        ``None``.

        Note this is unrelated to auto-init's own shape resolution (see
        :meth:`_resolve_shape_axis`), which no longer depends on this
        property or ``primary_extension_name`` at all -- it resolves each
        symbol (``DETECTOR_SIZE_Y``, ``N_ORDERS``, etc.) independently, so
        this property stays a simple, general-purpose reflection of "what's
        actually been assigned to the primary extension" for any other code
        that reads ``model.shape`` directly.
        """
        ext = self.primary_extension_name
        if ext is not None:
            val = self._raw_extension_value(ext)
            if isinstance(val, np.ndarray):
                return val.shape

        return self.__dict__.get('_shape')

class CalibrationModel(DataModel):

    CAL_TYPE : ClassVar[str] = None # Type of calibration

    def on_init(self):
        self.meta.cal_type = self.CAL_TYPE
        if self.meta.master_cal is None:
            self.meta.master_cal = True
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

        _set_koa_meta(self)

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

    @property
    def orders(self) -> np.ndarray:
        """
        Return the order vector for the calibration model based on the order_start and order_end metadata.
        """
        return self.make_order_vec()

    def order_index(self, order : int) -> int:
        """
        Return the index of the given order in the order vector.

        Parameters
        ----------
        order : int
            The order number to find the index for.

        Returns
        -------
        int
            The index of the order in the order vector.
        """
        orders = self.orders
        if order not in orders:
            raise ValueError(f"Order {order} is not in the order vector: {orders}")
        return np.where(orders == order)[0][0]