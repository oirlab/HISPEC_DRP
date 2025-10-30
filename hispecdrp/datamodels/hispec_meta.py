from .meta_data import FitsField, TargetMeta, InstrumentMeta, CoreMeta
from typing import Literal, ClassVar, Any
from pydantic import model_validator
import warnings

__all__ = ['HISPECMeta']

_HISPEC_FIBER_OBJECTS = Literal[
    'SCIENCE', 'FLAT', 'LFCR', 'LFCB', 'ETALONR', 'ETALONB', 'UNe', 'GASCELL', 'SKY', 'SPECKLE', 'BLOCKED'
]

class HISPECTargetMeta(TargetMeta):
    object_fiber1: _HISPEC_FIBER_OBJECTS | None = FitsField(
        fits_keyword="OBJFIB1",
        title="Object name in fiber 1",
        orm_field=True,
        default='BLOCKED'
    )
    object_fiber2: _HISPEC_FIBER_OBJECTS | None = FitsField(
        fits_keyword="OBJFIB2",
        title="Object name in fiber 2",
        orm_field=True,
        default='BLOCKED'
    )
    object_fiber3: _HISPEC_FIBER_OBJECTS | None = FitsField(
        fits_keyword="OBJFIB3",
        title="Object name in fiber 3",
        orm_field=True,
        default='BLOCKED'
    )

class HISPECInstrumentMeta(InstrumentMeta):
    spectrograph: Literal['BSPEC', 'RSPEC'] | None = FitsField(
        fits_keyword="SPECTROG",
        title="Name of the spectrograph (BSPEC, RSPEC)",
        orm_field=True,
    )
    instrument_mode: Literal['YJ', 'HK', 'YJHK'] | None = FitsField(
        fits_keyword="INSTMODE",
        title="Instrument mode (YJ, HK, YJHK)",
        default="YJHK",
        orm_field=True,
    )

class HISPECMeta(HISPECTargetMeta, HISPECInstrumentMeta, CoreMeta):
    ECHELLE_ORDERS : ClassVar[dict] = {
        # NOTE: 39 orders in current sims
        'BSPEC': {
            'order_start' : 148, # NOTE: This is 149 from latest list of echelle orders
            'order_end' : 110
        },
        # NOTE: 38 orders in current sims
        'RSPEC': {
            'order_start' : 97,
            'order_end' : 60 # NOTE: This is 59 from latest list of echelle orders
        }
    }

    @model_validator(mode='before')
    @classmethod
    def set_hispec_defaults(cls, data: Any) -> Any:
        if data.get('instrument_name') is None:
            data['instrument_name'] = "HISPEC"
        if data.get('telescope') is None:
            data['telescope'] = "Keck-II"
        if data.get('clock_rate') is None:
            data['clock_rate'] = 150000
        if data.get('spectrograph') is None:
            warnings.warn("meta.spectrograph not set, defaulting to BSPEC.")
            data['spectrograph'] = "BSPEC" # NOTE: KEEP EYE ON THIS DEFAULT!
        if data.get('instrument_mode') is None:
            data['instrument_mode'] = 'YJHK'
        if data.get('order_start') is None:
            if data.get('spectrograph') is not None:
                data['order_start'] = cls.ECHELLE_ORDERS[data['spectrograph']]['order_start']
        if data.get('order_end') is None:
            if data.get('spectrograph') is not None:
                data['order_end'] = cls.ECHELLE_ORDERS[data['spectrograph']]['order_end']
        return data