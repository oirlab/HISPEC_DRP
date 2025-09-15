from koa_middleware import CalibrationDB, CalibrationORM
from sqlalchemy import func

from koa_middleware.selector_base import CalibrationSelector

__all__ = [
    'DQSelector',
    'SaturationSelector',
    'DarkSelector',
    'BiasSelector',
    'ReadNoiseSelector',
    'NonlinearCorrectionSelector',
    'DetectorFlatSelector',
    'DetectorGainSelector',
    'WavelengthSolutionSelector',
    'ReferenceEtalonSelector',
]

class DQSelector(CalibrationSelector):
    def get_candidates(self, meta : dict, db : CalibrationDB):
        with db.session_manager() as session:
            if meta['instrument_name'].lower() == 'hispec':
                return session.query(db.orm_class).filter(
                    db.orm_class.cal_type == 'dq',
                    db.orm_class.instrument_era == meta['instrument_era'],
                    db.orm_class.spectrograph == meta['spectrograph'],
                    db.orm_class.master_cal == True,
                ).order_by(func.abs(db.orm_class.mjd_start - meta['mjd_start'])).all()
            elif meta['instrument_name'].lower() == 'parvi':
                return session.query(db.orm_class).filter(
                    db.orm_class.cal_type == 'dq',
                    db.orm_class.instrument_era == meta['instrument_era'],
                ).order_by(func.abs(db.orm_class.mjd_start - meta['mjd_start'])).all()

class SaturationSelector(CalibrationSelector):

    def get_candidates(self, meta : dict, db : CalibrationDB):
        with db.session_manager() as session:
            if meta['instrument_name'].lower() == 'hispec':
                return session.query(db.orm_class).filter(
                    db.orm_class.cal_type == 'saturation',
                    db.orm_class.instrument_era == meta['instrument_era'],
                    db.orm_class.spectrograph == meta['spectrograph'],
                    db.orm_class.master_cal == True,
                ).order_by(func.abs(db.orm_class.mjd_start - meta['mjd_start'])).all()
            elif meta['instrument_name'].lower() == 'parvi':
                return session.query(db.orm_class).filter(
                    db.orm_class.cal_type == 'saturation',
                    db.orm_class.instrument_era == meta['instrument_era'],
                ).order_by(func.abs(db.orm_class.mjd_start - meta['mjd_start'])).all()


class DarkSelector(CalibrationSelector):

    def get_candidates(self, meta : dict, db : CalibrationDB):
        with db.session_manager() as session:
            if meta['instrument_name'].lower() == 'hispec':
                return session.query(db.orm_class).filter(
                    db.orm_class.cal_type == 'dark',
                    db.orm_class.instrument_era == meta['instrument_era'],
                    db.orm_class.spectrograph == meta['spectrograph'],
                    db.orm_class.master_cal == True
                ).order_by(func.abs(db.orm_class.mjd_start - meta['mjd_start'])).all()
            elif meta['instrument_name'].lower() == 'parvi':
                return session.query(db.orm_class).filter(
                    db.orm_class.cal_type == 'dark',
                    db.orm_class.instrument_era == meta['instrument_era'],
                ).order_by(func.abs(db.orm_class.mjd_start - meta['mjd_start'])).all()


class BiasSelector(CalibrationSelector):

    def get_candidates(self, meta : dict, db : CalibrationDB):
        with db.session_manager() as session:
            if meta['instrument_name'].lower() == 'hispec':
                return session.query(db.orm_class).filter(
                    db.orm_class.cal_type == 'bias',
                    db.orm_class.instrument_era == meta['instrument_era'],
                    db.orm_class.spectrograph == meta['spectrograph'],
                    db.orm_class.master_cal == True,
                    db.orm_class.mjd_start <= meta['mjd_start']
                ).order_by(func.abs(db.orm_class.mjd_start - meta['mjd_start'])).all()
            elif meta['instrument_name'].lower() == 'parvi':
                return session.query(db.orm_class).filter(
                    db.orm_class.cal_type == 'bias',
                    db.orm_class.instrument_era == meta['instrument_era'],
                    db.orm_class.mjd_start <= meta['mjd_start']
                ).order_by(func.abs(db.orm_class.mjd_start - meta['mjd_start'])).all()


class ReadNoiseSelector(CalibrationSelector):

    def get_candidates(self, meta : dict, db : CalibrationDB):
        with db.session_manager() as session:
            if meta['instrument_name'].lower() == 'hispec':
                return session.query(db.orm_class).filter(
                    db.orm_class.cal_type == 'rn',
                    db.orm_class.instrument_era == meta['instrument_era'],
                    db.orm_class.spectrograph == meta['spectrograph'],
                    db.orm_class.master_cal == True,
                ).order_by(func.abs(db.orm_class.mjd_start - meta['mjd_start'])).all()
            elif meta['instrument_name'].lower() == 'parvi':
                return session.query(db.orm_class).filter(
                    db.orm_class.cal_type == 'rn',
                    db.orm_class.instrument_era == meta['instrument_era'],
                ).order_by(func.abs(db.orm_class.mjd_start - meta['mjd_start'])).all()

class NonlinearCorrectionSelector(CalibrationSelector):

    def get_candidates(self, meta : dict, db : CalibrationDB):
        with db.session_manager() as session:
            if meta['instrument_name'].lower() == 'hispec':
                return session.query(db.orm_class).filter(
                    db.orm_class.cal_type == 'nonlin',
                    db.orm_class.instrument_era == meta['instrument_era'],
                    db.orm_class.spectrograph == meta['spectrograph'],
                    db.orm_class.master_cal == True
                ).order_by(func.abs(db.orm_class.mjd_start - meta['mjd_start'])).all()
            elif meta['instrument_name'].lower() == 'parvi':
                return session.query(db.orm_class).filter(
                    db.orm_class.cal_type == 'nonlin',
                    db.orm_class.instrument_era == meta['instrument_era'],
                    db.orm_class.master_cal == True
                ).order_by(func.abs(db.orm_class.mjd_start - meta['mjd_start'])).all()


class DetectorFlatSelector(CalibrationSelector):

    def get_candidates(self, meta : dict, db : CalibrationDB):
        with db.session_manager() as session:
            if meta['instrument_name'].lower() == 'hispec':
                return session.query(db.orm_class).filter(
                    db.orm_class.cal_type == 'detflat',
                    db.orm_class.instrument_era == meta['instrument_era'],
                    db.orm_class.spectrograph == meta['spectrograph'],
                    db.orm_class.master_cal == True
                ).order_by(func.abs(db.orm_class.mjd_start - meta['mjd_start'])).all()
            elif meta['instrument_name'].lower() == 'parvi':
                return session.query(db.orm_class).filter(
                    db.orm_class.cal_type == 'detflat',
                    db.orm_class.instrument_era == meta['instrument_era'],
                    db.orm_class.master_cal == True
                ).order_by(func.abs(db.orm_class.mjd_start - meta['mjd_start'])).all()


class DetectorGainSelector(CalibrationSelector):

    def get_candidates(self, meta : dict, db : CalibrationDB):
        with db.session_manager() as session:
            if meta['instrument_name'].lower() == 'hispec':
                return session.query(db.orm_class).filter(
                    db.orm_class.cal_type == 'detgain',
                    db.orm_class.instrument_era == meta['instrument_era'],
                    db.orm_class.spectrograph == meta['spectrograph'],
                    db.orm_class.master_cal == True,
                    db.orm_class.mjd_start <= meta['mjd_start']
                ).order_by(func.abs(db.orm_class.mjd_start - meta['mjd_start'])).all()
            elif meta['instrument_name'].lower() == 'parvi':
                return session.query(db.orm_class).filter(
                    db.orm_class.cal_type == 'detgain',
                    db.orm_class.instrument_era == meta['instrument_era'],
                    db.orm_class.master_cal == True,
                    db.orm_class.mjd_start <= meta['mjd_start']
                ).order_by(func.abs(db.orm_class.mjd_start - meta['mjd_start'])).all()

class WavelengthSolutionSelector(CalibrationSelector):

    def __init__(self, cal_type : str = 'lfc_wavesol'):
        super().__init__()
        self.cal_type = cal_type

    def get_candidates(self, meta : dict, db : CalibrationDB):
        with db.session_manager() as session:
            if meta['instrument_name'].lower() == 'hispec':
                return session.query(db.orm_class).filter(
                    db.orm_class.cal_type == self.cal_type,
                    db.orm_class.instrument_era == meta['instrument_era'],
                    db.orm_class.spectrograph == meta['spectrograph'],
                ).order_by(func.abs(db.orm_class.mjd_start - meta['mjd_start'])).all()
            elif meta['instrument_name'].lower() == 'parvi':
                return session.query(db.orm_class).filter(
                    db.orm_class.cal_type == self.cal_type,
                    db.orm_class.instrument_era == meta['instrument_era'],
                ).order_by(func.abs(db.orm_class.mjd_start - meta['mjd_start'])).all()

class ReferenceEtalonSelector(CalibrationSelector):

    def get_candidates(self, meta : dict, db : CalibrationDB):
        with db.session_manager() as session:
            if meta['instrument_name'].lower() == 'hispec':
                return session.query(db.orm_class).filter(
                    db.orm_class.cal_type == 'ref_etalon',
                    db.orm_class.instrument_era == meta['instrument_era'],
                    db.orm_class.spectrograph == meta['spectrograph'],
                ).order_by(func.abs(db.orm_class.mjd_start - meta['mjd_start'])).all()
            elif meta['instrument_name'].lower() == 'parvi':
                return session.query(db.orm_class).filter(
                    db.orm_class.cal_type == 'ref_etalon',
                    db.orm_class.instrument_era == meta['instrument_era'],
                ).order_by(func.abs(db.orm_class.mjd_start - meta['mjd_start'])).all()