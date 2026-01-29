from koa_middleware import LocalCalibrationDB
from koa_middleware.selector_base import CalibrationSelector

__all__ = [
    'DQSelector',
    'SaturationSelector',
    'DarkSelector',
    'BiasSelector',
    'ReadNoiseSelector',
    'NonlinearCorrectionSelector',
    'DetectorFlatSelector',
    'GainSelector',
    'WavelengthSolutionSelector',
    'ReferenceEtalonSelector',
    'TracePosSelector',
    'TraceProfSelector',
]

class DQSelector(CalibrationSelector):
    def get_candidates(self, meta : dict, db : LocalCalibrationDB):
        instrument = meta['instrument_name'].lower()
        if instrument == 'hispec':
            rows = db.rows_where(
                """
                cal_type = :cal_type AND
                instrument_era = :era AND
                spectrograph = :spec AND
                master_cal = 1
                """,
                {
                    "cal_type": "dq",
                    "era": meta["instrument_era"],
                    "spec": meta["spectrograph"]
                },
                order_by="datetime_obs"
            )
        elif instrument == 'parvi':
            rows = db.rows_where(
                """
                cal_type = :cal_type AND
                instrument_era = :era
                """,
                {
                    "cal_type": "dq",
                    "era": meta["instrument_era"]
                },
                order_by="datetime_obs"
            )

        return list(rows)

        

class SaturationSelector(CalibrationSelector):

    def get_candidates(self, meta : dict, db : LocalCalibrationDB):
        instrument = meta['instrument_name'].lower()
        if instrument == 'hispec':
            rows = db.rows_where(
                """
                cal_type = :cal_type AND
                instrument_era = :era AND
                spectrograph = :spec AND
                master_cal = 1
                """,
                {
                    "cal_type": "saturation",
                    "era": meta["instrument_era"],
                    "spec": meta["spectrograph"]
                },
                order_by="abs(mjd_start - {})".format(meta["mjd_start"])
            )
        elif instrument == 'parvi':
            rows = db.rows_where(
                """
                cal_type = :cal_type AND
                instrument_era = :era
                """,
                {
                    "cal_type": "saturation",
                    "era": meta["instrument_era"]
                },
                order_by="abs(mjd_start - {})".format(meta["mjd_start"])
            )

        return list(rows)

class DarkSelector(CalibrationSelector):

    def get_candidates(self, meta : dict, db : LocalCalibrationDB):
        instrument = meta['instrument_name'].lower()
        if instrument == 'hispec':
            rows = db.rows_where(
                """
                cal_type = :cal_type AND
                instrument_era = :era AND
                spectrograph = :spec AND
                master_cal = 1
                """,
                {
                    "cal_type": "dark",
                    "era": meta["instrument_era"],
                    "spec": meta["spectrograph"]
                },
                order_by="abs(mjd_start - {})".format(meta["mjd_start"])
            )
        elif instrument == 'parvi':
            rows = db.rows_where(
                """
                cal_type = :cal_type AND
                instrument_era = :era
                """,
                {
                    "cal_type": "dark",
                    "era": meta["instrument_era"]
                },
                order_by="abs(mjd_start - {})".format(meta["mjd_start"])
            )

        return list(rows)

class TracePosSelector(CalibrationSelector):

    def get_candidates(self, meta : dict, db : LocalCalibrationDB):
        instrument = meta['instrument_name'].lower()
        if instrument == 'hispec':
            rows = db.rows_where(
                """
                cal_type = :cal_type AND
                instrument_era = :era AND
                spectrograph = :spec AND
                master_cal = 1
                """,
                {
                    "cal_type": "tracepos",
                    "era": meta["instrument_era"],
                    "spec": meta["spectrograph"]
                },
                order_by="abs(mjd_start - {})".format(meta["mjd_start"])
            )
        elif instrument == 'parvi':
            rows = db.rows_where(
                """
                cal_type = :cal_type AND
                instrument_era = :era AND
                spectrograph = :spec
                """,
                {
                    "cal_type": "tracepos",
                    "era": meta["instrument_era"],
                    "spec": meta["spectrograph"]
                },
                order_by="abs(mjd_start - {})".format(meta["mjd_start"])
            )

        return list(rows)

class TraceProfSelector(CalibrationSelector):

    def get_candidates(self, meta : dict, db : LocalCalibrationDB):
        instrument = meta['instrument_name'].lower()
        if instrument == 'hispec':
            rows = db.rows_where(
                """
                cal_type = :cal_type AND
                instrument_era = :era AND
                spectrograph = :spec AND
                master_cal = 1
                """,
                {
                    "cal_type": "traceprof",
                    "era": meta["instrument_era"],
                    "spec": meta["spectrograph"]
                },
                order_by="abs(mjd_start - {})".format(meta["mjd_start"])
            )
        elif instrument == 'parvi':
            rows = db.rows_where(
                """
                cal_type = :cal_type AND
                instrument_era = :era AND
                spectrograph = :spec
                """,
                {
                    "cal_type": "traceprof",
                    "era": meta["instrument_era"],
                    "spec": meta["spectrograph"]
                },
                order_by="abs(mjd_start - {})".format(meta["mjd_start"])
            )

        return list(rows)

class BiasSelector(CalibrationSelector):

    def get_candidates(self, meta : dict, db : LocalCalibrationDB):
        instrument = meta['instrument_name'].lower()
        if instrument == 'hispec':
            rows = db.rows_where(
                """
                cal_type = :cal_type AND
                instrument_era = :era AND
                spectrograph = :spec AND
                master_cal = 1
                """,
                {
                    "cal_type": "bias",
                    "era": meta["instrument_era"],
                    "spec": meta["spectrograph"]
                },
                order_by="abs(mjd_start - {})".format(meta["mjd_start"])
            )
        elif instrument == 'parvi':
            rows = db.rows_where(
                """
                cal_type = :cal_type AND
                instrument_era = :era
                """,
                {
                    "cal_type": "bias",
                    "era": meta["instrument_era"]
                },
                order_by="abs(mjd_start - {})".format(meta["mjd_start"])
            )

        return list(rows)

class ReadNoiseSelector(CalibrationSelector):

    def get_candidates(self, meta : dict, db : LocalCalibrationDB):
        instrument = meta['instrument_name'].lower()
        if instrument == 'hispec':
            rows = db.rows_where(
                """
                cal_type = :cal_type AND
                instrument_era = :era AND
                spectrograph = :spec AND
                master_cal = 1
                """,
                {
                    "cal_type": "rn",
                    "era": meta["instrument_era"],
                    "spec": meta["spectrograph"]
                },
                order_by="abs(mjd_start - {})".format(meta["mjd_start"])
            )
        elif instrument == 'parvi':
            rows = db.rows_where(
                """
                cal_type = :cal_type AND
                instrument_era = :era
                """,
                {
                    "cal_type": "rn",
                    "era": meta["instrument_era"]
                },
                order_by="abs(mjd_start - {})".format(meta["mjd_start"])
            )

        return list(rows)

class NonlinearCorrectionSelector(CalibrationSelector):

    def get_candidates(self, meta : dict, db : LocalCalibrationDB):
        instrument = meta['instrument_name'].lower()
        if instrument == 'hispec':
            rows = db.rows_where(
                """
                cal_type = :cal_type AND
                instrument_era = :era AND
                spectrograph = :spec AND
                master_cal = 1
                """,
                {
                    "cal_type": "nonlin",
                    "era": meta["instrument_era"],
                    "spec": meta["spectrograph"]
                },
                order_by="abs(mjd_start - {})".format(meta["mjd_start"])
            )
        elif instrument == 'parvi':
            rows = db.rows_where(
                """
                cal_type = :cal_type AND
                instrument_era = :era AND
                master_cal = 1
                """,
                {
                    "cal_type": "nonlin",
                    "era": meta["instrument_era"]
                },
                order_by="abs(mjd_start - {})".format(meta["mjd_start"])
            )

        return list(rows)

class DetectorFlatSelector(CalibrationSelector):

    def get_candidates(self, meta : dict, db : LocalCalibrationDB):
        instrument = meta['instrument_name'].lower()
        if instrument == 'hispec':
            rows = db.rows_where(
                """
                cal_type = :cal_type AND
                instrument_era = :era AND
                spectrograph = :spec AND
                master_cal = 1
                """,
                {
                    "cal_type": "detflat",
                    "era": meta["instrument_era"],
                    "spec": meta["spectrograph"]
                },
                order_by="abs(mjd_start - {})".format(meta["mjd_start"])
            )
        elif instrument == 'parvi':
            rows = db.rows_where(
                """
                cal_type = :cal_type AND
                instrument_era = :era AND
                master_cal = 1
                """,
                {
                    "cal_type": "detflat",
                    "era": meta["instrument_era"]
                },
                order_by="abs(mjd_start - {})".format(meta["mjd_start"])
            )

        return list(rows)

class GainSelector(CalibrationSelector):

    def get_candidates(self, meta : dict, db : LocalCalibrationDB):
        instrument = meta['instrument_name'].lower()
        if instrument == 'hispec':
            rows = db.rows_where(
                """
                cal_type = :cal_type AND
                instrument_era = :era AND
                spectrograph = :spec AND
                master_cal = 1
                """,
                {
                    "cal_type": "gain",
                    "era": meta["instrument_era"],
                    "spec": meta["spectrograph"]
                },
                order_by="abs(mjd_start - {})".format(meta["mjd_start"])
            )
        elif instrument == 'parvi':
            rows = db.rows_where(
                """
                cal_type = :cal_type AND
                instrument_era = :era AND
                master_cal = 1
                """,
                {
                    "cal_type": "gain",
                    "era": meta["instrument_era"]
                },
                order_by="abs(mjd_start - {})".format(meta["mjd_start"])
            )

        return list(rows)

class WavelengthSolutionSelector(CalibrationSelector):

    def __init__(self, cal_type : str = 'lfc_wavesol'):
        super().__init__()
        self.cal_type = cal_type

    def get_candidates(self, meta : dict, db : LocalCalibrationDB):
        instrument = meta['instrument_name'].lower()
        if instrument == 'hispec':
            rows = db.rows_where(
                """
                cal_type = :cal_type AND
                instrument_era = :era AND
                spectrograph = :spec
                """,
                {
                    "cal_type": self.cal_type,
                    "era": meta["instrument_era"],
                    "spec": meta["spectrograph"]
                },
                order_by="abs(mjd_start - {})".format(meta["mjd_start"])
            )
        elif instrument == 'parvi':
            rows = db.rows_where(
                """
                cal_type = :cal_type AND
                instrument_era = :era
                """,
                {
                    "cal_type": self.cal_type,
                    "era": meta["instrument_era"]
                },
                order_by="abs(mjd_start - {})".format(meta["mjd_start"])
            )

        return list(rows)

class ReferenceEtalonSelector(CalibrationSelector):

    def get_candidates(self, meta : dict, db : LocalCalibrationDB):
        instrument = meta['instrument_name'].lower()
        if instrument == 'hispec':
            rows = db.rows_where(
                """
                cal_type = :cal_type AND
                instrument_era = :era AND
                spectrograph = :spec
                """,
                {
                    "cal_type": "ref_etalon",
                    "era": meta["instrument_era"],
                    "spec": meta["spectrograph"]
                },
                order_by="abs(mjd_start - {})".format(meta["mjd_start"])
            )
        elif instrument == 'parvi':
            rows = db.rows_where(
                """
                cal_type = :cal_type AND
                instrument_era = :era
                """,
                {
                    "cal_type": "ref_etalon",
                    "era": meta["instrument_era"]
                },
                order_by="abs(mjd_start - {})".format(meta["mjd_start"])
            )

        return list(rows)