# from .meta_data import FitsField, TargetMeta, InstrumentMeta, CoreMeta, ExposureMeta
# from typing import Literal, ClassVar, Self, Any
# from pydantic import model_validator
# import scipy.constants

# __all__ = ['PARVIMeta']

# _PARVI_FIBER_OBJECTS = Literal[
#     'SCIENCE', 'FLAT', 'LFC', 'ETALON', 'UNe', 'GASCELL_QUAD', 'BLOCKED', 'NONE'
# ]

# class PARVITargetMeta(TargetMeta):
#     object_fiber1: _PARVI_FIBER_OBJECTS | None = FitsField(
#         fits_keyword="OBJFIB1",
#         title="Object name in fiber 1",
#         orm_field=True,
#         default='NONE',
#         merge_rule='coalesce',
#     )
#     object_fiber2: _PARVI_FIBER_OBJECTS | None = FitsField(
#         fits_keyword="OBJFIB2",
#         title="Object name in fiber 2",
#         orm_field=True,
#         default='NONE',
#         merge_rule='coalesce',
#     )

# class PARVIInstrumentMeta(InstrumentMeta):
#     instrument_name: Literal['PARVI'] = FitsField(
#         fits_keyword="INSTRUME",
#         title="Instrument name",
#         default="PARVI",
#     )
#     spectrograph: Literal['JH'] = FitsField(
#         fits_keyword="SPECTROG",
#         title="Name of the spectrograph (JH)",
#         orm_field=True,
#         default='JH',
#     )
#     instrument_mode: Literal['JH'] = FitsField(
#         fits_keyword="INSTMODE",
#         title="Instrument mode",
#         default="JH",
#         orm_field=True,
#     )

# class PARVIExposureMeta(ExposureMeta):
#     clock_rate: float | None = FitsField(
#         fits_keyword="CLOCKRAT",
#         title="Clock rate of the readout of the detector.",
#         unit="Hz",
#         default=150000
#     )


# class PARVIMeta(PARVIExposureMeta, PARVITargetMeta, PARVIInstrumentMeta, CoreMeta):
#     ECHELLE_ORDERS : ClassVar[dict] = {'order_start' : 125, 'order_end' : 85}
    
#     @model_validator(mode='before')
#     @classmethod
#     def set_parvi_defaults(cls, data: Any) -> Any:
#         if data.get('instrument_name') is None:
#             data['instrument_name'] = "PARVI"
#         if data.get('telescope') is None:
#             data['telescope'] = "Palomar"
#         if data.get('clock_rate') is None:
#             data['clock_rate'] = 150000
#         if data.get('spectrograph') is None:
#             data['spectrograph'] = "JH"
#         if data.get('order_start') is None:
#             if data.get('spectrograph') is not None:
#                 data['order_start'] = cls.ECHELLE_ORDERS['order_start']
#         if data.get('order_end') is None:
#             if data.get('spectrograph') is not None:
#                 data['order_end'] = cls.ECHELLE_ORDERS['order_end']
#         if data.get('lfc_delta_nu') is None:
#             data['lfc_delta_nu'] = 10.000000174E9
#         if data.get('lfc_nu0') is None:
#             data['lfc_nu0'] = scipy.constants.c / (1559.91370 * 1E-9)
#         return data