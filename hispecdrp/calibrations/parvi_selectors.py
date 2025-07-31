from .. import datamodels
from koa_middleware import CalibrationDB
from sqlalchemy import func

from koa_middleware.selector_base import CalibrationSelector

__all__ = [
    'PARVIDarkSelector',
    'PARVINonlinearCorrectionSelector',
    'PARVIDetectorFlatSelector'
]

class PARVIDarkSelector(CalibrationSelector):

    def get_candidates(self, input : datamodels.PARVIDataModel, db : CalibrationDB):
        with db.session_manager() as session:
            return session.query(db.orm_class).filter(
                db.orm_class.cal_type == 'dark',
                db.orm_class.instera == input.meta.instrument.era,
            ).order_by(func.abs(db.orm_class.mjd_start - input.meta.exposure.mjd_start)).all()
        

class PARVINonlinearCorrectionSelector(CalibrationSelector):

    def get_candidates(self, input : datamodels.PARVIDataModel, db : CalibrationDB):
        with db.session_manager() as session:
            return session.query(db.orm_class).filter(
                db.orm_class.cal_type == 'nonlin',
                db.orm_class.instera == input.meta.instrument.era,
            ).order_by(func.abs(db.orm_class.mjd_start - input.meta.exposure.mjd_start)).all()
        

class PARVIDetectorFlatSelector(CalibrationSelector):

    def get_candidates(self, input : datamodels.PARVIDataModel, db : CalibrationDB):
        with db.session_manager() as session:
            return session.query(db.orm_class).filter(
                db.orm_class.cal_type == 'detflat',
                db.orm_class.instera == input.meta.instrument.era,
            ).order_by(func.abs(db.orm_class.mjd_start - input.meta.exposure.mjd_start)).all()