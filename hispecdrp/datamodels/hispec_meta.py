# from .meta_data import FitsField, TargetMeta, InstrumentMeta, CoreMeta, ExposureMeta
# from typing import Literal, ClassVar, Any
# from pydantic import model_validator

# __all__ = ['HISPECMeta']

# _HISPEC_FIBER_OBJECTS = Literal[
#     'SCIENCE',
#     'FLAT',
#     'LFCR', 'LFCB',
#     'ETALONR', 'ETALONB',
#     'UNe',
#     'GASCELL',
#     'SKY',
#     'SPECKLE',
#     'BLOCKED',
#     'NONE'
# ]

# class HISPECTargetMeta(TargetMeta):
#     object_fiber1: _HISPEC_FIBER_OBJECTS | None = FitsField(
#         fits_keyword="OBJFIB1",
#         title="Object name in fiber 1",
#         orm_field=True,
#         default='NONE',
#         merge_rule='coalesce',
#     )
#     object_fiber2: _HISPEC_FIBER_OBJECTS | None = FitsField(
#         fits_keyword="OBJFIB2",
#         title="Object name in fiber 2",
#         orm_field=True,
#         default='NONE',
#         merge_rule='coalesce',
#     )
#     object_fiber3: _HISPEC_FIBER_OBJECTS | None = FitsField(
#         fits_keyword="OBJFIB3",
#         title="Object name in fiber 3",
#         orm_field=True,
#         default='NONE',
#         merge_rule='coalesce',
#     )

# class HISPECExposureMeta(ExposureMeta):
#     clock_rate: float | None = FitsField(
#         fits_keyword="CLOCKRAT",
#         title="Clock rate of the readout of the detector.",
#         unit="Hz",
#         default=150000
#     )

# class HISPECInstrumentMeta(InstrumentMeta):
#     instrument_name: Literal['HISPEC'] = FitsField(
#         fits_keyword="INSTRUME",
#         title="Instrument name",
#         default="HISPEC",
#     )
#     spectrograph: Literal['BSPEC', 'RSPEC'] = FitsField(
#         fits_keyword="SPECTROG",
#         title="Name of the spectrograph (BSPEC, RSPEC)",
#         orm_field=True,
#         default="BSPEC",
#     )
#     instrument_mode: Literal['YJ', 'HK', 'YJHK'] = FitsField(
#         fits_keyword="INSTMODE",
#         title="Instrument mode (YJ, HK, YJHK)",
#         default="YJHK",
#         orm_field=True,
#     )

# class HISPECMeta(HISPECExposureMeta, HISPECTargetMeta, HISPECInstrumentMeta, CoreMeta):
#     ECHELLE_ORDERS : ClassVar[dict] = {
#         # NOTE: 39 orders in current sims
#         'BSPEC': {
#             'order_start' : 148, # NOTE: This is 149 from latest list of echelle orders
#             'order_end' : 110
#         },
#         # NOTE: 38 orders in current sims
#         'RSPEC': {
#             'order_start' : 97,
#             'order_end' : 60 # NOTE: This is 59 from latest list of echelle orders
#         }
#     }

#     @model_validator(mode='before')
#     @classmethod
#     def set_hispec_defaults(cls, data: Any) -> Any:
#         #if data.get('instrument_name') is None:
#         #     data['instrument_name'] = "HISPEC"
#         # if data.get('telescope') is None:
#         #     data['telescope'] = "Keck-II"
#         if data.get('instrument_mode') is None:
#             data['instrument_mode'] = 'YJHK'
#         if data.get('order_start') is None:
#             if data.get('spectrograph') is not None:
#                 data['order_start'] = cls.ECHELLE_ORDERS[data['spectrograph']]['order_start']
#         if data.get('order_end') is None:
#             if data.get('spectrograph') is not None:
#                 data['order_end'] = cls.ECHELLE_ORDERS[data['spectrograph']]['order_end']
#         return data