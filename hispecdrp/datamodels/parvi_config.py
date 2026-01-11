from .meta_data import FitsField, MetaModel, InstrumentMeta, TargetMeta

import scipy.constants
from typing import ClassVar, Literal, Any, Self
from pydantic import model_validator, ValidatorFunctionWrapHandler

__all__ = ['PARVIConfig']

num_fibers = 2
instrument_name = 'PARVI'
instrument_aliases = {'JH': 'PR'}
clock_rate = 150_000
echelle_orders : dict[str, dict[str, int]] = {
    'JH': {
        'order_start' : 125,
        'order_end' : 85
    }
}

lfc_nu0 = scipy.constants.c / (1559.91370 * 1E-9) # Hz
lfc_delta_nu = 10.000000174E9 # Hz

class PARVIInstrumentMeta(InstrumentMeta):
    instrument_name: Literal['PARVI'] = FitsField(
        fits_keyword="INSTRUME",
        title="Instrument name",
        orm_field=True,
        default='PARVI',
        merge_rule='coalesce',
    )
    spectrograph: Literal['JH'] = FitsField(
        fits_keyword="SPECTROG",
        title="Spectrograph name",
        orm_field=True,
        default='JH',
        merge_rule='coalesce',
    )

_valid_fiber_objects = [
    'SCIENCE',
    'FIBERFLAT',
    'LFC',
    'ETALON',
    'UNe',
    'GASCELL-QUAD', 'GASCELL-HF4',
    'SKY',
    'BLOCKED',
    'NONE'
]

_valid_modes = ['JH']

PARVITargetMeta = TargetMeta.create(instrument_name=instrument_name, num_fibers=num_fibers, valid_fiber_objects=_valid_fiber_objects)
PARVIInstrumentMeta = InstrumentMeta.create(instrument_name=instrument_name, valid_spectrographs=list(instrument_aliases.keys()), valid_modes=_valid_modes)

class PARVIMetaModel(PARVIInstrumentMeta, PARVITargetMeta, MetaModel):
    
    @model_validator(mode='wrap')
    @classmethod
    def set_parvi_defaults(cls, data: Any, handler: ValidatorFunctionWrapHandler) -> Self:
        instance = handler(data)
        if instance.spectrograph in echelle_orders:
            if instance.order_start is None:
                object.__setattr__(instance, 'order_start', echelle_orders[instance.spectrograph]['order_start'])
            if instance.order_end is None:
                object.__setattr__(instance, 'order_end', echelle_orders[instance.spectrograph]['order_end'])
        if instance.lfc_nu0 is None:
            object.__setattr__(instance, 'lfc_nu0', lfc_nu0)
        if instance.lfc_delta_nu is None:
            object.__setattr__(instance, 'lfc_delta_nu', lfc_delta_nu)
        if instance.clock_rate is None:
            object.__setattr__(instance, 'clock_rate', clock_rate)
        if instance.n_channels is None:
            object.__setattr__(instance, 'n_channels', 32)
        if instance.n_refpixs is None:
            object.__setattr__(instance, 'n_refpixs', 4)
        if instance.n_refpixs is None:
            object.__setattr__(instance, 'channels_are_vertical', True)
        return instance

class PARVIConfig:

    instrument_name : ClassVar[str] = instrument_name
    instrument_aliases : ClassVar[dict[str, str]] = {'JH': 'PR'}
    num_fibers : ClassVar[int] = num_fibers
    clock_rate : ClassVar[int] = clock_rate
    lfc_nu0 = lfc_nu0
    lfc_delta_nu = lfc_delta_nu
    meta_model : ClassVar[type[MetaModel]] = PARVIMetaModel