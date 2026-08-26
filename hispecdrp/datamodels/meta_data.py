from pathlib import Path
import yaml
import copy
import numpy as np
import jsonschema
import jsonref
from typing import MutableMapping, Self
from astropy.io import fits
import os

from .utils import get_instrument_config

import logging
logger = logging.getLogger(__name__)

_META_SCHEMA_DEFAULTS = {
    'title' : '', # Human-readable title, used as FITS comment
    'fits_section' : 'core', # Section in FITS header
    'fits_hdu' : 'PRIMARY', # FITS HDU name
    'type' : 'string', # JSON schema type
    'unit' : None, # Physical unit
    'cal_db' : None, # Column name in calibration DB. True -> Use property name, str -> use provided string, None -> not included in DB
    'merge_rule' : 'first', # Rule for merging metadata from multiple files
    'include_ancestors' : False, # Include in ancestors table when merging data models?
}

# Additional JSON schema fields:
# - enum: List of allowed values
# - default: Default value

class MetaModel(MutableMapping):
    """
    Small helper class for handling metadata based on JSON schema.

    Parameters
    ----------
    schema : dict
        The metadata schema.
    meta : dict | None, optional
        Initial metadata values. Defaults to None.
    validate : bool, optional
        Whether to validate metadata values against the schema. Defaults to True.
    """

    def __init__(self, schema: dict, meta: dict | None = None, validate : bool = True):
        self._validate = validate
        self._schema = schema
        self._data = {}
        if meta is not None:
            for k, v in meta.items():
                self[k] = v
        self.set_default_values()

    def __getitem__(self, key):
        if key in self._schema['properties']:
            return self._data.get(key)
        else:
            raise KeyError(f"Property '{key}' not found in metadata schema.")

    def __setitem__(self, key, value):
        if key not in self._schema['properties']:
            raise KeyError(f"Property '{key}' not found in metadata schema.")
        if self._validate:
            self.validate(key, value)
        self._data[key] = value

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self[name] = value
    
    def __dir__(self):
         return sorted(list(self._schema['properties'].keys()))
    
    def __repr__(self) -> str:
        return f"MetaModel({self._data})"
    
    def __deepcopy__(self, memo) -> Self:
        cls = self.__class__
        result = cls.__new__(cls)  # blank instance
        memo[id(self)] = result

        # Copy internal attributes directly
        result._validate = self._validate
        result._schema = copy.deepcopy(self._schema, memo)
        result._data = copy.deepcopy(self._data, memo)

        return result
    
    def validate(
        self,
        prop_name : str,
        value
    ):
        """
        Validate a metadata field against the schema.

        Parameters
        ----------
        prop_name : str
            The property name to validate.
        value : Any
            The value to validate.

        Raises
        ------
        ValueError
            If the value does not conform to the schema.
        """
        
        # Get the sub-schema for the property
        sub_schema = self._schema['properties'].get(prop_name)
        if sub_schema is None:
            raise KeyError(f"Property '{prop_name}' not found in metadata schema.")
        
        # Accept None values
        if value is None:
            return
        
        try:
            jsonschema.validate(
                instance=value,
                schema=sub_schema,
            )
        except jsonschema.ValidationError as e:
            raise ValueError(f"Invalid value for '{prop_name}': {e.message}") from None
    
    def to_fits_header(
        self,
        extname : str = 'PRIMARY',
        include_missing_metadata : bool = False
    ) -> fits.Header:
        """
        Convert the metadata to a FITS header.

        Parameters
        ----------
        extname : str, optional
            The extension name for which to create the header. Defaults to 'PRIMARY'.
        include_missing_metadata : bool, optional
            Whether to include None (unset) metadata values in the header. Defaults to False.

        Returns
        -------
        header : fits.Header
            The FITS header containing the metadata.
        """

        # Init header
        header = fits.Header()

        # Helper to make comments in FITS heaers
        def make_comment(sub_schema : dict) -> str:
            comment = sub_schema['title']
            unit = sub_schema['unit']
            if unit is not None:
                if unit.lower() != 'none':
                    comment += f" ({unit})"
            return comment
        
        # Group based on section names
        meta_sections = self.group_meta_by_section()

        for section_name, section_attrs in meta_sections.items():

            if len(section_attrs) == 0:
                continue

            section_vals = [self[attr] for attr in section_attrs]

            if not any(section_vals) and not include_missing_metadata:
                continue

            # Add section header as COMMENT
            header.append(('COMMENT', '=' * 40), end=True)
            header.append(('COMMENT', f'  {section_name}'), end=True)
            header.append(('COMMENT', '=' * 40), end=True)

            for attr in section_attrs:

                # Sub schema and value
                sub_schema = self._schema['properties'][attr]
                value = self[attr]

                # Skip if no value and not adding empty entries
                if value is None and not include_missing_metadata:
                    continue

                # FITS HDU
                prop_ext = sub_schema['fits_hdu']
                if extname != prop_ext:
                    continue

                # Info and value
                keyword = sub_schema['fits_keyword']
                comment = make_comment(sub_schema)

                # Add to header
                header.append((keyword, value, comment), end=True)

        return header
    
    def to_record(self) -> dict:
        """
        Convert the metadata to a dictionary record.

        Returns
        -------
        record : dict
            The metadata as a dictionary.
        """
        record = {}
        for prop_name, sub_schema in self._schema['properties'].items():
            if sub_schema['cal_db']:
                db_col = sub_schema['cal_db']
                if db_col is True:
                    db_col = prop_name
                elif isinstance(db_col, str):
                    db_col = db_col
                record[db_col] = self[prop_name]
        return record
    
    def set_default_values(self):
        """
        Set default values for metadata fields based on the schema.
        """
        for prop_name, sub_schema in self._schema['properties'].items():
            if self._data.get(prop_name) is None:
                default_value = sub_schema.get('default')
                if default_value is not None:
                    self[prop_name] = default_value

    def set_default(self, prop_name : str, value):
        """
        Set a default value for a specific metadata field.

        Parameters
        ----------
        prop_name : str
            The property name for which to set the default value.
        value : Any
            The default value to set if the property is not already set.
        """
        if self[prop_name] is None:
            self[prop_name] = value
    
    def group_meta_by_section(self, flat: bool = False) -> dict:
        """
        Group metadata properties by their FITS section.

        Parameters
        ----------
        flat : bool, optional
            Whether to return a flat dictionary instead of grouped by section. Defaults to False.
        
        Returns
        -------
        dict
            Dictionary of metadata grouped by FITS section.
            Keys are section names and values are dictionaries of metadata properties.
        """

        sections = {}
        for name, sub in self._schema["properties"].items():
            sec = sub.get("fits_section", "Core")
            sections.setdefault(sec, {})[name] = self._data.get(name)

        # Core first, then alphabetical
        ordered = {}
        if "Core" in sections:
            ordered["Core"] = sections["Core"]

        for sec in sorted(sections):
            if sec != "Core":
                ordered[sec] = sections[sec]

        if not flat:
            return ordered

        # Flatten
        out = {}
        for sec_dict in ordered.values():
            out.update(sec_dict)

        return out
    
    def boolean_columns(self) -> list[str]:
        """
        Return a set of columns that are logically booleans (for PostgreSQL)
        """
        return [
            prop_name
            for prop_name, sub_schema in self._schema["properties"].items()
            if sub_schema.get("cal_db") and sub_schema.get("type") == "boolean"
        ]

    def pprint(self, show_none : bool = False, header_style : bool = True):
        """
        Pretty-print the metadata.

        Parameters
        ----------
        show_none : bool, optional
            Whether to show metadata fields with None values. Defaults to False.
        """
        for attr, sub_schema in self._schema['properties'].items():
            val = self[attr]
            if not val is None or show_none:

                unit = sub_schema['unit']

                if header_style:
                    s = f"{attr} = {val}"
                    if unit is not None:
                        s += f" ({unit})"
                    s += f"  # {sub_schema['title']}"
                    print(s)
                else:
                    if unit is not None:
                        print(f"{attr}: {val} {unit}")
                    else:
                        print(f"{attr}: {val}")
                    print(f"  title: {sub_schema['title']}")
                    print(f"  type: {sub_schema['type']}")
                    print(f"  fits_keyword: {sub_schema['fits_keyword']}")
                    print(f"  fits_hdu: {sub_schema['fits_hdu']}")

    def __delitem__(self, key):
        del self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)
    
    @property
    def fields(self) -> list[str]:
        """
        Return a view of metadata fields names, including those with None values.
        """
        return self._schema['properties'].keys()


def get_meta_schema(
    instrument_name : str | None = None,
    config : type | None = None,
) -> dict:
    """
    Get the metadata schema for a given instrument.

    Parameters
    ----------
    instrument_name : str | None
        Name of the instrument. Optional.
    config : type | None
        Instrument configuration class. Optional.
    
    Returns
    -------
    meta : dict
        The metadata schema with defaults applied.
    """

    # Determine instrument config
    if config is None:
        assert instrument_name is not None, "Either instrument_name or config must be provided."
        config = get_instrument_config(instrument_name)

    # Load metadata schema for this instrument config
    meta_schema = config.get_metadata_schema()
    
    return meta_schema

def read_meta_schema(filepath : str) -> dict:
    def _loader(uri, **_):
        return yaml.safe_load(
            Path(
                uri.replace("file://", "")
            ).read_text()
        )
    
    schema = jsonref.load_uri(
        Path(filepath).absolute().as_uri(),
        loader=_loader
    )
    
    properties = {}
    if 'allOf' in schema:
        parts = schema.pop('allOf')
        for part in parts:
            fits_section = os.path.splitext(os.path.basename(part['id']))
            for k, v in part.get('properties', {}).items():
                properties[k] = v
                properties[k]['fits_section'] = fits_section[0].upper()
    else:
        raise ValueError("Metadata schema must contain 'allOf' at the top level.")
    
    # Fix any nested allOf
    for k, v in properties.items():
        if 'allOf' in v:
            parts = v.pop('allOf')
            for part in parts:
                for pk, pv in part.items():
                    properties[k][pk] = pv

    for p in properties.values():
        for k, v in _META_SCHEMA_DEFAULTS.items():
            p.setdefault(k, v)

    for prop_name, sub_schema in properties.items():
        if sub_schema['cal_db']:
            db_col = sub_schema['cal_db']
            if db_col is True:
                db_col = prop_name
            assert isinstance(db_col, str), f"cal_db must be a string or True for property '{prop_name}'"
            sub_schema['cal_db'] = db_col

    schema['properties'] = properties
            
    return schema

def merge_meta(models: list[dict]) -> dict:
    """
    Merge the meta information from multiple DataModel instances.

    Parameters
    ----------
    models : list[DataModel]
        List of DataModel instances to merge meta information from.

    Returns
    -------
    meta_merged : MetaModel
        Merged metadata.
    """
    merged_data = {}

    # Sort models
    models = sorted(models, key=lambda m: m.meta.datetime_obs)

    for prop_name, sub_schema in models[0].meta._schema['properties'].items():
        values = [getattr(model.meta, prop_name) for model in models]
        values = [v for v in values if v is not None]
        if not values:
            continue
        merge_rule = sub_schema['merge_rule']
        if merge_rule == 'drop': # Drop the value
            merged_value = None
        elif merge_rule == 'first': # First value in time
            merged_value = values[0]
        elif merge_rule == 'last': # Last value in time
            merged_value = values[-1]
        elif merge_rule == 'sum': # Sum the values
            merged_value = np.sum(values)
        elif merge_rule == 'max': # Max of the values
            merged_value = max(values)
        elif merge_rule == 'min': # Min of the values
            merged_value = min(values)
        elif merge_rule == 'mean': # Mean of the values
            merged_value = np.mean(values)
        elif merge_rule == 'fiber_state': # Special rule for fibers
            merged_value = _merge_fiber_state(values)
        else:
            raise ValueError(f"Unknown merge rule '{merge_rule}' for meta field '{prop_name}'")
        merged_data[prop_name] = merged_value

    meta_merged = MetaModel(
        meta=merged_data,
        schema=models[0].meta._schema
    )

    return meta_merged


_FIBER_EMPTY_STATES = ["BLOCKED", "NONE"]

def _merge_fiber_state(values) -> str:
    values = [v for v in values if v is not None]

    if not values:
        return "NONE"

    illuminated = [v for v in values if v not in _FIBER_EMPTY_STATES]

    if illuminated:
        return illuminated[0]

    if "BLOCKED" in values:
        return "BLOCKED"

    return "NONE"