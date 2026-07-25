import importlib.resources

import copy
from .base_model import DataModel
from .meta_data import read_meta_schema

import scipy.constants
from typing import ClassVar

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .base_model import DataModel

__all__ = ['PARVIConfig']

class PARVIConfig:

    instrument_name : ClassVar[str] = 'PARVI'
    num_fibers : ClassVar[int] = 4
    metadata_schema_filename : ClassVar[str] = 'hispec_metadata_schema.yaml'

    @classmethod
    def get_metadata_schema(cls) -> dict:
        """
        This allows parvi to specify its own metadata but reuse hispec's schema.

        Returns
        -------
        meta_schema : dict
            The metadata schema dictionary.
        """
        
        # Determine filepath to YAML schema file
        metadata_dir = importlib.resources.files("hispecdrp.datamodels.metadata_schemas")
        filepath = str(metadata_dir / cls.metadata_schema_filename)

        # Load schema from YAML file
        meta_schema = read_meta_schema(filepath)

        props = meta_schema['properties']

        # Modify schema as needed for PARVI
        props['instrument_name']['enum'] = ['PARVI']
        props['instrument_name']['default'] = 'PARVI'
        props['telescope']['default'] = 'Hale 200'
        props['telescope']['enum'] = ['Hale 200']
        
        props['object_fiber4'] = copy.deepcopy(
            props['object_fiber3']
        )
        props['object_fiber4']['title'] = "Object name in fiber 4"
        props['object_fiber4']['fits_keyword'] = 'OBJFIB4'

        for fiber in range(1, cls.num_fibers + 1):
            key = f'object_fiber{fiber}'
            props[key]['enum'] += [
                'SCIENCE',
                'WHITELIGHT',
                'LFC',
                'ETALON',
                'UNe',
                'GASCELL-QUAD', 'GASCELL-HF4',
                'SKY',
                'BLOCKED',
                'NONE',
                'MINICOMB'
            ]
            props[key]['enum'] = list(set(props[key]['enum']))

        props['instrument_mode']['enum'] = ['JH']
        props['instrument_mode']['default'] = 'JH'

        props['spectrograph']['enum'] = ['JH']
        props['spectrograph']['default'] = 'JH'
        props['n_channels']['default'] = 32

        props['ref_channel_position']['default'] = 'NONE'
        
        return meta_schema

    @classmethod
    def set_meta_defaults(cls, model : 'DataModel'):
        """
        Apply defaults dynamically.
        These will continue to evolve as the DRP is developed.
        """

        # Instrument
        if model.meta.order_start is None:
            model.meta.order_start = 128
        if model.meta.order_end is None:
            model.meta.order_end = 84
        if model.meta.clock_rate is None:
            model.meta.clock_rate = 150_000
        if model.meta.lfc_nu0 is None:
            model.meta.lfc_nu0 = scipy.constants.c / (1559.91370 * 1E-9) # Hz
        if model.meta.lfc_delta_nu is None:
            model.meta.lfc_delta_nu = 10.000000174E9 # Hz
        
        return model
    
    @classmethod
    def get_instrument_alias(cls, *args, **kwargs) -> str:
        return 'PR'