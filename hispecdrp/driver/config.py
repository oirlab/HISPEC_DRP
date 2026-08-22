from __future__ import annotations
import importlib.resources
import os
import yaml
from typing import Any, Callable
from .. import datamodels

from koa_middleware import CalibrationSelector

from packaging.version import Version

_PKG_NAME = __name__.split('.')[0]

import logging
logger = logging.getLogger(__name__)

__all__ = [
    'load_config_file',
    'get_config_filepath',
    'get_config_dir',
    'ConfigFile',
    'ConfigStore',
    'ConfigSelector'
]

class FlowSeqDumper(yaml.SafeDumper):
    pass

def _flow_seq(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

FlowSeqDumper.add_representer(list, _flow_seq)


class ConfigFile:
    """
    A class representing a configuration file.

    Attributes
    ----------
    name : str
        Unique name for a config file.
    class_name : str
        The name of the primitive or pipeline class (with or without the module path).
    filepath : str
        Full filepath to the config file.
    context : dict
        A dictionary of metadata for this config file.
    """

    _DEFAULT_CONTEXT = {
        'quicklook': None,
        'instrument_era': ['0.0.1', None],
        'instrument_mode': None,
        'processing_mode': None,

    }

    def __init__(
        self,
        name: str,
        class_name: str,
        description: str | None = None,
        filepath: str | None = None,
        context: dict | None = None,
        parameters : dict | None = None,
        primitives : dict | None = None,
        logging : dict | None = None,
    ):
        """
        Initialize a ConfigFile object.
        
        Parameters
        ----------
        name : str
            Unique name for all config files = filename without extension.
        class_name : str
            The name of the primitive or pipeline class (with or without the module path).
        description : str, optional
            A description of the config file.
        filepath : str, optional
            Full filepath to the config file.
        context : dict, optional
            A dictionary of metadata for this config file.
        parameters : dict, optional
            A dictionary of parameters for this config file.
        primitives : dict, optional
            A dictionary of primitives for this config file.
        logging : dict, optional
            A dictionary of logging configuration for this config file.
        """
        if filepath is not None:
            self.filepath = os.path.abspath(filepath)
        else:
            self.filepath = None
        self._name = name
        self.class_name = class_name
        self.description = description
        self.context = context or {}
        for key, default_value in self._DEFAULT_CONTEXT.items():
            self.context.setdefault(key, default_value)
        self.parameters = parameters or {}
        self.primitives = primitives or {}
        self.logging = logging or {}

    @property
    def name(self):
        return self._name or os.path.splitext(self.filename)[0] or self.class_name

    @property
    def class_(self):
        if not hasattr(self, '_class') or self._class is not None:
            try:
                class_ = get_primitive_class(self.class_name)
            except KeyError:
                class_ = None
            if class_ is None:
                try:
                    class_ = get_pipeline_class(self.class_name)
                except KeyError:
                    class_ = None
            if class_ is None:
                raise ValueError(f"Could not find class for {self.class_name}.")
            self._class = class_
        return self._class

    @property
    def filename(self):
        if self.filepath is not None:
            return os.path.basename(self.filepath)
        return None

    @classmethod
    def from_file(cls, filepath: str, **kwargs):
        config_data = load_config_file(filepath)
        name = config_data.get('name', os.path.splitext(os.path.basename(filepath))[0])
        return cls(
            name=name,
            class_name=config_data['class'], # Required field
            description=config_data.get('description'), # Optional field
            filepath=filepath,
            context=config_data.get('context', {}), # Optional section
            parameters=config_data.get('parameters', {}), # Optional section
            primitives=config_data.get('primitives', {}), # Optional section
            **kwargs
        )

    @classmethod
    def from_dict(cls, config: dict):
        return cls(
            name=config.get('name'),
            filepath=None,
            class_name=config['class'],
            description=config.get('description'),
            context=config.get('context'),
            parameters=config.get('parameters'),
            primitives=config.get('primitives'),
        )

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'class': self.class_name,
            'description': self.description,
            'context': self.context,
            'parameters': self.parameters,
            'primitives': self.primitives,
        }
    
    def save(self, output_path : str, output_dir : str | None = None) -> str:
        if output_path is None:
            if output_dir is None:
                raise ValueError("Either output_path or output_dir must be specified.")
            output_path = os.path.join(output_dir, f"{self.name}.yaml")
        config_dict = self.to_dict()
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, Dumper=FlowSeqDumper, sort_keys=False)
        return output_path

    @property
    def is_pipeline(self):
        from ..pipelines.base_pipeline import BasePipeline
        class_ = self.class_
        if class_ is None:
            return False
        return issubclass(class_, BasePipeline)

    def __repr__(self):
        return (
            f"ConfigFile(name={self.name!r}, description={self.description!r}, class_name={self.class_name!r}, "
            f"filepath={self.filepath!r}, context={self.context!r}, "
            f"parameters={self.parameters!r}, primitives={self.primitives!r})"
        )


class ConfigStore:
    """
    A store for managing configuration files.

    Examples
    --------
    >>> with ConfigStore() as store:
            config_file : ConfigFile = store.select_and_get_config(
                dataset,
                selector
            )
            filepath = config_file.filepath
    """

    _cache: list[ConfigFile] | None = None

    def __init__(self, keep_in_memory: bool = True, config_dir: str | None = None):
        self.config_dir = config_dir
        # Instances pointed at a custom config_dir never touch the shared
        # cache -- it exists only for the real, installed configs directory.
        self.keep_in_memory = keep_in_memory and config_dir is None
        if self.keep_in_memory and ConfigStore._cache is not None:
            self.configs = ConfigStore._cache
        else:
            self.configs = self.populate()
            if self.keep_in_memory:
                ConfigStore._cache = self.configs

    @classmethod
    def clear_cache(cls):
        cls._cache = None

    def populate(self):
        configs = []
        config_dir = self.config_dir or get_config_dir()
        for filename in os.listdir(config_dir):
            if filename.endswith(('.yaml', '.yml')):
                filepath = os.path.join(config_dir, filename)
                try:
                    config_file = ConfigFile.from_file(filepath)
                except KeyError:
                    #logger.warning(f"Skipping non-config YAML file in {config_dir}: {filename}")
                    continue
                configs.append(config_file)
        return configs

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if not self.keep_in_memory:
            self.configs = None
    
    # Main method called to retrieve a config file for a dataset and primitives class
    def select_and_get_config(
        self,
        dataset: Any,
        selector : ConfigSelector,
    ) -> ConfigFile:
        """
        Given an input dataset and a primitive class to run, select a config file.

        Parameters
        ----------
        dataset : Any
            The dataset for which to select a config.
        selector : ConfigSelector
            The config selector to use for selecting the config.

        Returns
        -------
        config_file : ConfigFile
            The selected config file object.
        """
        model = datamodels.utils.get_first_model(dataset, meta_only=True)
        meta = model.meta

        # Select the config file
        config_file = selector.select(meta, self.configs)

        return config_file

    # Dict interface
    def __iter__(self):
        return iter(self.configs)
    
    def __len__(self):
        return len(self.configs)

    def __getitem__(self, key):
        return self.configs[key]

    def __setitem__(self, key, value):
        self.configs[key] = value

    def __delitem__(self, key):
        del self.configs[key]

    def __contains__(self, config_name : str):
        return config_name in self.configs


class AmbiguousConfigError(ValueError):
    pass

class NoConfigFoundError(ValueError):
    pass


def _get_nested(meta : Any, path : str) -> Any:
    """Resolve a dotted path against dict-like or attribute-like metadata."""
    value = meta
    for part in path.split('.'):
        if value is None:
            return None
        if isinstance(value, dict):
            value = value.get(part)
        else:
            value = getattr(value, part, None)
    return value


class ConfigSelector(CalibrationSelector):

    _CUSTOM_FILTERS: dict[str, Callable[[Any, dict], bool]] = {}

    def __init__(
        self,
        class_name : str | type,
        context : dict | None = None,
        skip_filters : list[str] = None,
    ):
        self.class_name = class_name
        self.context = context or {}
        self.skip_filters = skip_filters or []

    #### API ####

    def select(self, meta, candidates : list[ConfigFile]) -> ConfigFile | None:
        """
        Select the most appropriate config file for the given metadata.

        Parameters
        ----------
        meta : dict
            Data model metadata.
        candidates : list[ConfigFile]
            List of candidate config files to consider.

        Returns
        -------
        result : ConfigFile | None
            The selected config file if one is found, otherwise None.
        """
        result = self._select(meta, candidates)
        if result is None:
            raise NoConfigFoundError(f"No config file could be selected for class {self.class_name} and the given metadata.")
        return result

    def _select(self, meta, candidates : list[ConfigFile]) -> ConfigFile | None:
        """
        Internal method to perform the core config file selection logic.

        This method first retrieves a set of candidate config files using ``get_candidates``
        and then selects the best one (or ones) from these candidates using ``select_best``.
        Subclasses can override this method to implement custom selection workflows,
        but it's often more appropriate to override ``get_candidates`` or ``select_best``.

        Parameters
        ----------
        meta : dict
            The metadata for which a config file is to be selected.
        candidates : list[ConfigFile]
            The list of candidate config files to consider.

        Returns
        -------
        result : ConfigFile | None
            The selected config file if one is found, otherwise None.
        """
        candidates = self.get_candidates(meta, candidates)
        result = self.select_best(meta, candidates)
        return result

    def get_candidates(self, meta, candidates : list[ConfigFile]) -> ConfigFile:
        """
        Select the most appropriate config file for the given metadata.

        Parameters
        ----------
        meta : dict
            Data metadata to match against the ``meta`` blocks of the
            candidate config files.
        candidates : list[ConfigFile]
            List of candidate config files to consider.

        Returns
        -------
        config_file : ConfigFile
        """
        if not candidates:
            raise NoConfigFoundError(f"No config files available for class {self.class_name}.")

        # Run through the standard filters
        candidates = self._filter_class(candidates)
        if "quicklook" not in self.skip_filters:
            candidates = self._filter_quicklook(candidates)
        if "instrument_era" not in self.skip_filters:
            candidates = self._filter_instrument_era(meta, candidates)
        if "instrument_mode" not in self.skip_filters:
            candidates = self._filter_instrument_mode(meta, candidates)
        if "processing_mode" not in self.skip_filters:
            candidates = self._filter_processing_mode(candidates)

        # Run additional filters
        for filter_name, filter_func in self._CUSTOM_FILTERS.items():
            if filter_name not in self.skip_filters:
                candidates = filter_func(meta, candidates)

        return candidates

    def select_best(self, meta, candidates : list[ConfigFile]) -> ConfigFile:

        if len(candidates) == 0:
            raise NoConfigFoundError(f"No config files available for class {self.class_name}.")

        if len(candidates) > 1:
            raise AmbiguousConfigError(f"Multiple config files for class {self.class_name} match the given dataset and context. Must specify a more specific config.")
        
        # NOTE: Config files are versioned by the DRP,
        # so only one should remain
        candidate = candidates[0]

        return candidate

    #### Filters ####

    def _filter_class(self, candidates : list[ConfigFile]) -> list[ConfigFile]:
        """
        Filter config files by the primitive or pipeline class they are meant to configure.

        Parameters
        ----------
        candidates : list[ConfigFile]
            The list of candidate config files to filter.

        Returns
        -------
        filtered_candidates : list[ConfigFile]
            The filtered list of config files that match the class.
        """
        if isinstance(self.class_name, type):
            class_name = self.class_name.__name__
        else:
            class_name = self.class_name.rsplit('.', 1)[-1]
        candidates_out = [
            cf for cf in candidates
            if cf.class_name.rsplit('.', 1)[-1] == class_name
        ]

        if len(candidates_out) == 0:
            raise ValueError(f"No config files match the class '{class_name}'.")
        
        return candidates_out

    def _filter_instrument_era(self, meta, candidates : list[ConfigFile]) -> list[ConfigFile]:
        """
        Filter candidates by the valid instrument_era range.

        Parameters
        ----------
        meta : dict
            Datamodel metadata
        candidates : list[ConfigFile]
            List of candidate config files to filter.

        Returns
        -------
        filtered_candidates : list[ConfigFile]
            The filtered list of config files that match the instrument era.
        """
        era = _get_nested(meta, 'instrument_era')
        if era is None:
            return candidates
        era = Version(era)
        candidates_out = []
        for cf in candidates:

            # The valid instrument era for this config file
            valid_era = cf.context.get('instrument_era')

            # No era specified
            if (
                valid_era is None
                or (
                    isinstance(valid_era, str)
                    and valid_era.lower() == 'any'
                )
            ):
                candidates_out.append(cf)
                continue

            # era bounds
            bounds = valid_era if isinstance(valid_era, list) else [valid_era, valid_era]
            assert len(bounds) == 2, f"Invalid instrument_era in config {cf.name}: {valid_era}"

            def _to_version(bound):
                """None or 'any' means unbounded; otherwise parse as a Version."""
                if bound is None or (isinstance(bound, str) and bound.lower() == 'any'):
                    return None
                return Version(bound)

            era_min, era_max = (_to_version(b) for b in bounds)

            # Check if the dataset's era falls within the bounds
            if (era_min is None or era >= era_min) and (era_max is None or era <= era_max):
                candidates_out.append(cf)

        return candidates_out

    def _filter_instrument_mode(self, meta, candidates : list[ConfigFile]) -> list[ConfigFile]:
        """
        Filter candidates by instrument mode.

        Parameters
        ----------
        meta : dict
            Datamodel metadata
        candidates : list[ConfigFile]
            List of candidate config files to filter.

        Returns
        -------
        filtered_candidates : list[ConfigFile]
            The filtered list of config files that match the instrument mode.
        """
        inst_mode = _get_nested(meta, 'instrument_mode')
        if inst_mode is None:
            return candidates
        inst_mode = inst_mode.lower()
        candidates_out = []
        for cf in candidates:

            # Valid instrument modes for this config file
            inst_mode_valid = cf.context.get('instrument_mode')

            # Applies to all modes
            if inst_mode_valid is None:
                candidates_out.append(cf)
                continue

            if isinstance(inst_mode_valid, str):
                inst_mode_valid = [inst_mode_valid.lower()]
            elif isinstance(inst_mode_valid, list):
                inst_mode_valid = [m.lower() for m in inst_mode_valid]
            else:
                raise ValueError(f"Invalid instrument_mode type in config {cf.name}: {inst_mode_valid}")

            # Check if the dataset's instrument mode matches any of the valid modes
            if (
                inst_mode in inst_mode_valid
                or 'any' in inst_mode_valid
            ):
                candidates_out.append(cf)

        return candidates_out

    def _filter_ifs_mode(self, meta, candidates : list[ConfigFile]):
        """
        Filter candidates by IFS mode.

        Parameters
        ----------
        meta : dict
            Datamodel metadata
        candidates : list[ConfigFile]
            List of candidate config files to filter.

        Returns
        -------
        filtered_candidates : list[ConfigFile]
            The filtered list of config files that match the IFS mode.
        """
        ifs_mode = getattr(meta.instrument, 'ifs_mode', None)
        if ifs_mode is None:
            return candidates
        ifs_mode = ifs_mode.lower()
        if ifs_mode != 'ifs':
            return candidates
        candidates_out = []
        for cf in candidates:

            # Valid instrument modes for this config file
            inst_mode_valid = cf.context.get('ifs_mode')

            # Applies to all modes
            if inst_mode_valid is None:
                candidates_out.append(cf)
                continue

            if isinstance(inst_mode_valid, str):
                inst_mode_valid = [inst_mode_valid.lower()]
            elif isinstance(inst_mode_valid, list):
                inst_mode_valid = [m.lower() for m in inst_mode_valid]
            else:
                raise ValueError(f"Invalid instrument_mode type in config {cf.name}: {inst_mode_valid}")

            # Check if the dataset's instrument mode matches any of the valid modes
            if (
                ifs_mode in inst_mode_valid
                or 'any' in inst_mode_valid
            ):
                candidates_out.append(cf)

        return candidates_out

    def _filter_quicklook(self, candidates : list[ConfigFile]):
        """
        Filter candidates by the quicklook flag (``self.quicklook``).

        A config file that omits ``quicklook`` (context value ``None``) is a
        wildcard: it matches any requested value. It's only used as a
        fallback when no candidate explicitly matches the requested value,
        so an explicit ``True``/``False`` config always takes precedence
        over a wildcard one for the same class.

        Cases
        -----
        1.  Input context = None
            ConfigFile contexts = (None, ...)
            -> All
            ConfigFile contexts = (True, False)
            -> False
        2. Input context = False
            ConfigFile contexts = (True, False)
            -> False
            ConfigFile contexts = (None, ...)
            -> All
        3. Input context = True
            ConfigFile contexts = (True, None)
            -> True

        Parameters
        ----------
        candidates : list[ConfigFile]
            List of candidate config files to filter.

        Returns
        -------
        filtered_candidates : list[ConfigFile]
            The filtered list of config files that match the quicklook criteria.
        """
        ql_input = self.context.get('quicklook')
        if ql_input is None:
            ql_input = False

        matched = [cf for cf in candidates if cf.context['quicklook'] is ql_input]
        if matched:
            return matched

        return [cf for cf in candidates if cf.context['quicklook'] is None]

    def _filter_processing_mode(self, candidates : list[ConfigFile]) -> list[ConfigFile]:
        """
        Filter candidates by the processing mode (``self.context.processing_mode``).

        Unlike ``quicklook``, ``processing_mode`` has no natural default value,
        so an unspecified input (``None``) leaves the candidates unconstrained.
        When a value *is* specified, a config file that explicitly matches it
        always wins; a config file that omits ``processing_mode`` (``None``, a
        wildcard) is only used as a fallback when nothing explicitly matches.

        Parameters
        ----------
        candidates : list[ConfigFile]
            List of candidate config files to filter.

        Returns
        -------
        filtered_candidates : list[ConfigFile]
            The filtered list of config files that match the processing mode criteria.
        """
        pm_input = self.context.get('processing_mode')
        if pm_input is None:
            return candidates

        matched = [cf for cf in candidates if cf.context['processing_mode'] == pm_input]
        if matched:
            return matched

        return [cf for cf in candidates if cf.context['processing_mode'] is None]

    @classmethod
    def register_filter(cls, key: str):
        """Register custom match logic for a selection key with non-equality semantics."""
        def _register(fn):
            cls._CUSTOM_FILTERS[key] = fn
            return fn
        return _register

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._CUSTOM_FILTERS = dict(cls._CUSTOM_FILTERS)  # copy-on-inherit


#### Utility methods for config file handling ####

def get_config_dir():
    config_dir = importlib.resources.files(_PKG_NAME).joinpath("configs")
    if config_dir.is_dir():
        return str(config_dir)
    else:
        raise FileNotFoundError("Config directory not found in hispecdrp.configs")


def get_config_filepath(config_file : str) -> str:
    """
    Resolve the path to a configuration file.

    Parameters
    ----------
    config_file : str
        The base filename of the configuration file or a valid file path.

    Returns
    -------
    config_filepath : str
        The resolved path to the configuration file.
    """

    # If already a ConfigFile object, get its filepath
    if isinstance(config_file, ConfigFile):
        config_file = config_file.filepath

    # If already valid file or filepath, return it.
    if os.path.isfile(config_file):
        return os.path.abspath(config_file)

    # Get full path
    config_filepath = os.path.join(get_config_dir(), config_file)

    # Check if valid filepath.
    if os.path.isfile(config_filepath):
        return config_filepath
    raise FileNotFoundError(f"No valid config file found for: {config_file}")


def _load_yaml_config_file(config_filepath : str) -> dict:
    with open(config_filepath, 'r') as f:
        return yaml.safe_load(f)


def load_config_file(config_filepath : str) -> dict:
    """
    Read a configuration file and return its contents as a dictionary."""
    ext = os.path.splitext(config_filepath)[1].lower()
    is_file = os.path.isfile(config_filepath)
    if not is_file:
        raise FileNotFoundError(f"Config file not found: {config_filepath}")
    if config_filepath.endswith(('.yaml', '.yml')):
        return _load_yaml_config_file(config_filepath)
    raise ValueError(f"Unsupported config file extension: {ext}. Only .yaml and .yml are supported.")

def get_primitive_class(name: str) -> type | None:
    """
    Lazily resolve and return a primitive or pipeline class.
    
    Accepts:
    - Short class name: "AssignWCSStep"
    - Full dot path: "some_external_pkg.module.CustomStep"
    """
    from ..primitives import PRIMITIVE_REGISTER
    return PRIMITIVE_REGISTER[name]


def get_pipeline_class(name: str) -> type | None:
    """
    Lazily resolve and return a primitive or pipeline class.
    
    Accepts:
    - Short class name: "AssignWCSStep"
    - Full dot path: "some_external_pkg.module.CustomStep"
    """
    from ..pipelines import PIPELINE_REGISTER
    return PIPELINE_REGISTER[name]