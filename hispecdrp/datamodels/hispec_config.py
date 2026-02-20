import importlib.resources
from .base_model import DataModel
from .meta_data import read_meta_schema

import scipy.constants
from typing import ClassVar

__all__ = ['HISPECConfig']

class HISPECConfig:

    # hispec_metadata_schema.yaml

    instrument_name : ClassVar[str] = 'HISPEC'
    instrument_aliases : ClassVar[dict[str, str]] = {'BSPEC': 'HB', 'RSPEC': 'HR'}
    num_fibers : ClassVar[int] = 3

    # TODO: Update LFC params, eventually in calibration DB
    lfc_params : ClassVar[dict[str, float]] = {
        'LFC-B' : {
            'nu0' : scipy.constants.c / (1559.91370 * 1E-9), # Hz
            'delta_nu' : 10.000000174E9, # Hz
        },
        'LFC-R' : {
            'nu0' : scipy.constants.c / (1559.91370 * 1E-9), # Hz
            'delta_nu' : 10.000000174E9, # Hz
        },
    }

    echelle_orders : dict[str, dict[str, int]] = {
        'BSPEC': {
            'order_start' : 148,
            'order_end' : 110
        },
        'RSPEC': {
            'order_start' : 97,
            'order_end' : 60
        }
    }

    metadata_schema_filename : ClassVar[str] = 'hispec_metadata_schema.yaml'

    @classmethod
    def get_metadata_schema(cls) -> dict:
        
        # Determine filepath to YAML schema file
        metadata_dir = importlib.resources.files("hispecdrp.datamodels.metadata_schemas")
        filepath = str(metadata_dir / cls.metadata_schema_filename)

        # Load schema from YAML file with ref resolution
        meta_schema = read_meta_schema(filepath)
        
        return meta_schema

    @classmethod
    def set_meta_defaults(cls, model : 'DataModel'):
        """
        Apply defaults dynamically.
        These will continue to evolve as the DRP is developed.

        Returns
        -------
        meta_schema : dict
            The metadata schema dictionary.
        """

        # Calibration
        if hasattr(model, 'CAL_TYPE'):
            if model.meta.master_cal is None:
                model.meta.master_cal = True

        # Instrument
        if model.meta.instrument_era is None:
           model.meta.instrument_era = "0.0.1"
        if model.meta.instrument_mode is None:
            model.meta.instrument_mode = 'YJHK'
        if model.meta.spectrograph is None:
            model.meta.spectrograph = 'BSPEC'
        if model.meta.spectrograph in cls.echelle_orders:
            model.meta.order_start = cls.echelle_orders[model.meta.spectrograph]['order_start']
            model.meta.order_end = cls.echelle_orders[model.meta.spectrograph]['order_end']
        if model.meta.clock_rate is None:
            model.meta.clock_rate = 150_000
        # TODO: Defaults need to know which LFC, ignore until we have a more robust way to determine this from the metadata
        # if model.meta.lfc_nu0 is None:
        #     model.meta.lfc_nu0 = cls.lfc_params[model.meta.spectrograph]['nu0']
        # if model.meta.lfc_delta_nu is None:
        #     model.meta.lfc_delta_nu = cls.lfc_params[model.meta.spectrograph]['delta_nu']
        
        return model
    
    @classmethod
    def get_instrument_alias(
        cls,
        model : DataModel | None = None,
        spectrograph : str | None = None
    ) -> str:
        
        if spectrograph is not None:
            instrument_alias = cls.instrument_aliases.get(spectrograph.upper())
            return instrument_alias
        
        if model is not None:
            spectrograph = model.meta.spectrograph
            instrument_alias = cls.instrument_aliases.get(spectrograph.upper())
            return instrument_alias