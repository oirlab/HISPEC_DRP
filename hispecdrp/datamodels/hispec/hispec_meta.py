from ..meta_data import FitsField, TargetMeta, CoreMeta, CalibrationCoreMeta

__all__ = ['HISPECCoreMeta', 'HISPECCalibrationMeta']

class HISPECTargetMeta(TargetMeta):
    object_fiber1: str | None = FitsField(
        fits_keyword="OBJFIB1",
        title="Object name in fiber 1"
    )
    object_fiber2: str | None = FitsField(
        fits_keyword="OBJFIB2",
        title="Object name in fiber 2"
    )
    object_fiber3: str | None = FitsField(
        fits_keyword="OBJFIB3",
        title="Object name in fiber 3"
    )

class HISPECCoreMeta(HISPECTargetMeta, CoreMeta):
    ORM_TABLE_NAME = 'HISPEC'

    koa_id: str | None = FitsField(
        fits_keyword="KOAID",
        title="Keck Observatory Archive (KOA) ID"
    )
    koa_filepath: str | None = FitsField(
        fits_keyword="KOAFPATH",
        title="KOA file path"
    )

class HISPECCalibrationMeta(HISPECCoreMeta, CalibrationCoreMeta):
    pass