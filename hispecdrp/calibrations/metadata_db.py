from contextlib import contextmanager
from sqlalchemy.orm import Mapped, Session
from sqlalchemy.orm import mapped_column
from sqlalchemy import create_engine
from sqlalchemy import DateTime, String, Float, func, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
import uuid
from sqlalchemy.exc import SQLAlchemyError
import datetime
import os
from ..keck import koa
import glob
import warnings

__all__ = ['CalibrationDB', 'Calibration', 'RemoteCalibrationDB', 'LocalCalibrationDB']

from hispecdrp import __version__
from .. import datamodels


# Use the environment variable for database connection
_Base = declarative_base()

# Generate a UUID for the namespace using a hash of the unique string
UUID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "HISPEC_DRP")


class Calibration(_Base):
    """Table to store calibration information."""
    __tablename__ = "calibrations"

    # ID
    id : Mapped[uuid.UUID] = mapped_column(String(36), primary_key=True)

    # Filename in cache dir
    filename : Mapped[str] = mapped_column(String, nullable=False)

    # KOA ID, not same as internal filenames
    koa_id : Mapped[str] = mapped_column(String, nullable=True)

    # Location on KOA (/KPF/2024/20240926/lev0/filename.fits - change for HISPEC once determined)
    koa_filepath : Mapped[str] = mapped_column(String, nullable=True)

    # Calibration type (dark, flat)
    cal_type : Mapped[str] = mapped_column(String, nullable=False)

    # Master?
    master : Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    # Instrument era (0.0.1)
    instera : Mapped[str] = mapped_column(String, nullable=False)

    # Spectrograph (BSPEC/RSPEC)
    spectrograph : Mapped[str] = mapped_column(String, nullable=False)

    # JD Start of observation (or first observation if coadd)
    jd_start : Mapped[float] = mapped_column(Float, nullable=False)

    # Datetime of observation start, should match jd_start (ISO 8601 format)
    datetime_obs : Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # Read mode
    read_mode : Mapped[str] = mapped_column(String, nullable=False)

    # Objects
    target_name : Mapped[str] = mapped_column(String, nullable=False)
    object_fiber1 : Mapped[str] = mapped_column(String, nullable=True)
    object_fiber2 : Mapped[str] = mapped_column(String, nullable=True)
    object_fiber3 : Mapped[str] = mapped_column(String, nullable=True)

    # DRP Version (0.0.1)
    drp_version : Mapped[str] = mapped_column(String, nullable=False)

    # Custom note
    note : Mapped[str] = mapped_column(String, nullable=True)

    # How to keep track of last update for efficient syncing?
    last_updated : Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),   # runs on insert
        onupdate=func.now(),         # runs on update
        nullable=False
    )

    def to_dict(self) -> dict:
        return dict(
            id=self.id,
            filename=self.filename,
            koa_id=self.koa_id,
            koa_filepath=self.koa_filepath,
            cal_type=self.cal_type,
            instera=self.instera,
            spectrograph=self.spectrograph,
            jd_start=self.jd_start,
            datetime_obs=self.datetime_obs,
            read_mode=self.read_mode,
            target_name=self.target_name,
            object_fiber1=self.object_fiber1,
            object_fiber2=self.object_fiber2,
            object_fiber3=self.object_fiber3,
            drp_version=self.drp_version,
            note=self.note,
            last_updated=self.last_updated,
        )
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Calibration':
        return cls(
            id=data['id'],
            filename=data['filename'],
            koa_id=data['koa_id'],
            koa_filepath=data['koa_filepath'],
            cal_type=data['cal_type'],
            instera=data['instera'],
            spectrograph=data['spectrograph'],
            jd_start=data['jd_start'],
            datetime_obs=data['datetime_obs'],
            read_mode=data['read_mode'],
            target_name=data['target_name'],
            object_fiber1=data['object_fiber1'],
            object_fiber2=data['object_fiber2'],
            object_fiber3=data['object_fiber3'],
            drp_version=data['drp_version'],
            note=data['note'],
            last_updated=data['last_updated'],
        )

    @staticmethod
    def generate_uuid(filename : str) -> str:
        # NOTE: Revisit this with Max+KOA
        return str(uuid.uuid5(UUID_NAMESPACE, filename + __version__))

    @classmethod
    def from_datamodel(
        cls,
        input : str | datamodels.HISPECCalibrationModel,
        note : str | None = None,
    ) -> 'Calibration':
        cal_model = datamodels.open(input, copy=False, meta_only=True)
        filename = cal_model.meta.filename
        datetime_obs = cal_model.meta.date_obs + 'T' + cal_model.meta.utc
        datetime_obs = datetime.datetime.strptime(datetime_obs, "%Y%m%dT%H:%M:%S.%f")
        _id = cls.generate_uuid(filename)
        koa_filepath = koa.generate_koa_filepath(cal_model)
        koa_id = koa_filepath.split('/')[-1]
        return cls(
            id=_id,
            filename=filename,
            koa_id=koa_id,
            koa_filepath=koa_filepath,
            cal_type=cal_model.meta.cal_type.lower(),
            instera=cal_model.meta.instrument.era,
            read_mode=cal_model.meta.exposure.read_mode,
            target_name=cal_model.meta.target.target_name,
            object_fiber1=cal_model.meta.target.object_fiber1,
            object_fiber2=cal_model.meta.target.object_fiber2,
            object_fiber3=cal_model.meta.target.object_fiber3,
            spectrograph=cal_model.meta.instrument.spectrograph,
            jd_start=cal_model.meta.exposure.jd_start,
            datetime_obs=datetime_obs,
            drp_version=__version__, # NOTE: Or cal_model.meta.drp_version??
            note=note,
            last_updated=datetime.datetime.now(),
        )

    def __repr__(self):
        s = self.__class__.__name__ + '\n'
        s += f"  id: {self.id}\n"
        s += f"  filename: {self.filename}\n"
        s += f"  koa_id: {self.koa_id}\n"
        s += f"  koa_filepath: {self.koa_filepath}\n"
        s += f"  cal_type: {self.cal_type}\n"
        s += f"  instera: {self.instera}\n"
        s += f"  spectrograph: {self.spectrograph}\n"
        s += f"  jd_start: {self.jd_start}\n"
        s += f"  datetime_obs: {self.datetime_obs}\n"
        s += f"  read_mode: {self.read_mode}\n"
        s += f"  target_name: {self.target_name}\n"
        s += f"  object_fiber1: {self.object_fiber1}\n"
        s += f"  object_fiber2: {self.object_fiber2}\n"
        s += f"  object_fiber3: {self.object_fiber3}\n"
        s += f"  drp_version: {self.drp_version}\n"
        s += f"  note: {self.note}\n"
        s += f"  last_updated: {self.last_updated}\n"
        return s


class CalibrationDB:
    """Generic database interface for calibration storage and retrieval. The remote and local DBs inherit from this object."""

    
    def __init__(self, url : str):
        self.engine = self.get_engine(url=url)
        _Base.metadata.create_all(self.engine) # Ensure tables exist
        self.Session = sessionmaker(bind=self.engine)

    def get_engine(self, url : str, echo : bool = True, pool_size : int = 10, max_overflow : int = 20):
        """Connect to DB"""
        return create_engine(url, echo=echo, pool_size=pool_size, max_overflow=max_overflow)

    def close(self):
        """Close the database engine."""
        self.engine.dispose()

    @contextmanager
    def session_manager(self, external_session: Session | None = None):
        own_session = external_session is None
        session = self.Session() if own_session else external_session
        try:
            yield session
        except SQLAlchemyError as e:
            session.rollback()
            raise e
        finally:
            if own_session:
                session.close()
    
    def calibration_in_db(
        self,
        calibration : Calibration,
        session : Session | None = None,
    ) -> Calibration | None:
        """
        Check if a calibration is in the database.
        """
        with self.session_manager(session) as session:
            return session.query(Calibration).filter(
                #Calibration.id == calibration.id,
                Calibration.filename == calibration.filename,
            ).first()
    
    def get_deltas(
        self,
        calibrations : list[Calibration],
        session : Session | None = None,
    ) -> Calibration | list[Calibration] | None:
        """
        Based on a provided list of calibrations, returns entries which are not present in this DB.
        """
        missing = []
        with self.session_manager(session) as session:
            for calibration in calibrations:
                existing_calibration = self.calibration_in_db(calibration, session)
                if existing_calibration is None:
                    missing.append(calibration)
        session.expunge_all()
        return missing

    def query(
        self,
        session : Session | None = None,
        cal_type : str | None = None,
        date_time_start : datetime.datetime = datetime.datetime.min,
        date_time_end : datetime.datetime = datetime.datetime.max,
        fetch : str = 'all',
    ) -> list[Calibration]:
        """
        Higher level query method to retrieve calibrations based on a specified calibration type and datetime start/end.
        The utility of this method will be revisited as the DRP is developed.
        """
        with self.session_manager(session) as session:
            _query = session.query(Calibration).filter(
                Calibration.datetime_obs >= date_time_start,
                Calibration.datetime_obs <= date_time_end
            )
            if cal_type is not None:
                _query = _query.filter(Calibration.cal_type == cal_type)
            if fetch == 'all':
                result = _query.all()
            elif fetch == 'first':
                result = _query.first()
        return result

    def add(
        self,
        calibration : Calibration | datamodels.HISPECCalibrationModel | str | list,
        note : str | None = None,
        session : Session | None = None,
        commit : bool = True,
    ):
        """
        Add one or many calibrations to the database.
        """
        if not isinstance(calibration, list):
            calibration = [calibration]
        with self.session_manager(session) as session:
            for item in calibration:
                if isinstance(item, (str, datamodels.HISPECCalibrationModel)):
                    cal_model = datamodels.open(item, copy=False, meta_only=True)
                    obj = Calibration.from_datamodel(cal_model, note=note)
                else:
                    obj = item
                session.merge(obj)
            if commit:
                session.commit()

        
class RemoteCalibrationDB(CalibrationDB):
    """
    Interface to database at KOA, nothing specific done here yet.
    """
    
    def __init__(self, url : str):
        super().__init__(url)


class LocalCalibrationDB(CalibrationDB):
    """
    Interface to local SQLite DB.
    """

    def __init__(self, db_path : str):
        url = f"sqlite:///{db_path}"
        super().__init__(url)

    def update_from_cache(self, cache_dir : str):
        """
        Update the local SQLite DB based on the cache dir.
        """
        calibrations = []
        cal_dir = f'{cache_dir}calibrations{os.sep}'
        cal_files = [os.path.abspath(f) for f in glob.glob(cal_dir + '*.fits')]
        for filepath in cal_files:
            try:
                calibration = Calibration.from_datamodel(filepath)
                calibrations.append(calibration)
            except Exception as e:
                warnings.warn(f"Failed to parse {filepath}: {e}")

        if calibrations:
            self.add(calibrations)