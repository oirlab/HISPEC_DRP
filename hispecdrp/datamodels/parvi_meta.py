from .meta_data import FitsField, TargetMeta, CoreMeta, CalibrationCoreMeta

__all__ = ['PARVIMeta', 'PARVICalibrationMeta']


class PARVITargetMeta(TargetMeta):
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
    object_fiber4: str | None = FitsField(
        fits_keyword="OBJFIB4",
        title="Object name in fiber 4"
    )


class PARVIMeta(CoreMeta, PARVITargetMeta):
    ORM_TABLE_NAME = 'PARVI'
    pass


class PARVICalibrationMeta(PARVIMeta, CalibrationCoreMeta):
    pass