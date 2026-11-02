import importlib.resources
import logging

from astropy.units import Hz
from .base_model import DataModel
from .meta_data import read_meta_schema

import scipy.constants
from typing import ClassVar

logger = logging.getLogger(__name__)

__all__ = ['HISPECConfig']

class HISPECConfig:

    # hispec_metadata_schema.yaml

    instrument_name : ClassVar[str] = 'HISPEC'
    instrument_aliases : ClassVar[dict[str, str]] = {'BSPEC': 'HB', 'RSPEC': 'HR'}
    num_fibers : ClassVar[int] = 3

    # (ny, nx) of the detector plane. Read by DataModel.default_detector_size
    # (the base implementation just returns config.DETECTOR_SIZE) to resolve
    # the DETECTOR_SIZE_Y/DETECTOR_SIZE_X symbols in an IMAGE_HDU `shape=`
    # spec (see DataModel._resolve_shape_axis), as a last resort when no
    # extension's already-set real data and no explicit `shape=` constructor
    # override provide it instead. HISPEC has one detector, so this default
    # is all that's needed; an instrument with multiple, differently sized
    # detectors would instead give this config class one attribute per
    # detector (e.g. IFS_DETECTOR_SIZE, IMG_DETECTOR_SIZE) and override
    # `default_detector_size` on the relevant model classes to pick between
    # them from instance state (e.g. self.meta.instrument_mode).
    DETECTOR_SIZE : ClassVar[tuple[int, int]] = (4096, 4160)

    # TODO: Move LFC params to the calibration DB.
    # Mode spacings per HISPEC.DRP.DDD.001 (2024-12-05): the BSPEC
    # Menlo/KPF comb is Fabry-Perot filtered to 20 GHz (Sec. 2.4.5.1) and
    # the RSPEC electro-optic astrocomb runs at 16 GHz (Sec. 2.4.5.2).
    # The DDD does not specify the absolute offset frequency, so ``nu0``
    # is unchanged.
    lfc_params : ClassVar[dict[str, float]] = {
        # NOTE: From https://github.com/Keck-DataReductionPipelines/KPF-Pipeline/blob/master/docs/source/info/data_format.rst
        # LFCFRREF	20000000000.0	LFC filtered Offset Freq RR Filter setpoint(Hz)
        # LFCCEOFR	5220000000.0	LFC CEO Filtered Setpoint Freq (Hz)
        'LFC-B' : {
            'nu0' : 5.22E9, # Hz
            'delta_nu' : 20E9, # Hz
        },
        'LFC-R' : {
            'nu0' : scipy.constants.c / (1559.91370 * 1E-9), # Hz
            'delta_nu' : 16E9, # Hz
        },
    }

    echelle_orders : dict[str, dict[str, int]] = {
        'BSPEC': {
            'order_start' : 110,
            'order_end' : 148
        },
        'RSPEC': {
            'order_start' : 60,
            'order_end' : 97
        }
    }

    @classmethod
    def get_order_list(cls, spectrograph: str) -> list[int]:
        """
        Return the list of echelle order numbers for a spectrograph (single source of truth).
        Used by benchmarks and trace finding so order count and indices match model meta.
        """
        sp = (spectrograph or "").strip().upper()
        if sp not in cls.echelle_orders:
            return []
        o = cls.echelle_orders[sp]
        start, end = o['order_start'], o['order_end']
        step = 1 if end >= start else -1
        return list(range(start, end + step, step))

    @classmethod
    def sync_echelle_orders_for_spectrograph(cls, model: 'DataModel', spectrograph: str) -> None:
        """
        Set ``meta.spectrograph``, ``meta.order_start``, and ``meta.order_end`` from
        :attr:`echelle_orders` for the given spectrograph.

        Use this when constructing or reloading simulated data so echelle order labels
        match :meth:`get_order_list` and primitives that rely on ``make_order_vec()``
        (e.g. spectral extraction, stellar RV), including after FITS headers carried
        stale ORDERST/ORDEREND from an older run.
        """
        sp = (spectrograph or "").strip().upper()
        if sp not in cls.echelle_orders:
            logger.warning(
                "Unknown spectrograph %r; leaving echelle order metadata unchanged on model.",
                spectrograph,
            )
            return
        o = cls.echelle_orders[sp]
        model.meta.spectrograph = sp
        model.meta.order_start = o["order_start"]
        model.meta.order_end = o["order_end"]

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

        # Instrument
        if model.meta.instrument_mode is None:
            model.meta.instrument_mode = 'YJHK'
        if model.meta.spectrograph is None:
            model.meta.spectrograph = 'BSPEC'
        # Only fill echelle order bounds when unset. Unconditional overwrite breaks
        # models whose 2-D spectra cover a subset of orders: meta must stay aligned
        # with ``wave_fib*`` axis 0 (e.g. L2Model.from_datamodels + StellarRVPrimitive).
        if model.meta.spectrograph in cls.echelle_orders:
            if model.meta.order_start is None:
                model.meta.order_start = cls.echelle_orders[model.meta.spectrograph]['order_start']
            if model.meta.order_end is None:
                model.meta.order_end = cls.echelle_orders[model.meta.spectrograph]['order_end']
        if model.meta.clock_rate is None:
            model.meta.clock_rate = 150_000
        if model.meta.lfc_nu0 is None:
            key = 'LFC-B' if model.meta.spectrograph == 'BSPEC' else 'LFC-R'
            model.meta.lfc_nu0 = cls.lfc_params[key]['nu0']
        if model.meta.lfc_delta_nu is None:
            key = 'LFC-B' if model.meta.spectrograph == 'BSPEC' else 'LFC-R'
            model.meta.lfc_delta_nu = cls.lfc_params[key]['delta_nu']
        
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