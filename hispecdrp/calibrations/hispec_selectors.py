from .. import datamodels
from koa_middleware import CalibrationDB
from sqlalchemy import func

from koa_middleware.selector_base import CalibrationSelector

__all__ = [
    'HISPECDarkSelector',
    'HISPECNonlinearCorrectionSelector',
    'HISPECDetectorFlatSelector'
]

class HISPECDarkSelector(CalibrationSelector):

    def get_candidates(self, input : datamodels.HISPECDataModel, db : CalibrationDB):
        with db.session_manager() as session:
            return session.query(db.orm_class).filter(
                db.orm_class.cal_type == 'dark',
                db.orm_class.instera == input.meta.instrument.era,
                db.orm_class.spectrograph == input.meta.instrument.spectrograph,
                db.orm_class.master == True
            ).order_by(func.abs(db.orm_class.mjd_start - input.meta.exposure.mjd_start)).all()

class HISPECNonlinearCorrectionSelector(CalibrationSelector):

    def get_candidates(self, input : datamodels.HISPECDataModel, db : CalibrationDB):
        with db.session_manager() as session:
            return session.query(db.orm_class).filter(
                db.orm_class.cal_type == 'nonlin',
                db.orm_class.instera == input.meta.instrument.era,
                db.orm_class.spectrograph == input.meta.instrument.spectrograph,
                db.orm_class.master == True
            ).order_by(func.abs(db.orm_class.mjd_start - input.meta.exposure.mjd_start)).all()
        

class HISPECDetectorFlatSelector(CalibrationSelector):

    def get_candidates(self, input : datamodels.PARVIDataModel, db : CalibrationDB):
        with db.session_manager() as session:
            return session.query(db.orm_class).filter(
                db.orm_class.cal_type == 'detflat',
                db.orm_class.instera == input.meta.instrument.era,
                db.orm_class.spectrograph == input.meta.instrument.spectrograph,
                db.orm_class.master == True
            ).order_by(func.abs(db.orm_class.mjd_start - input.meta.exposure.mjd_start)).all()