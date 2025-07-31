from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy import String, Float,  Boolean
from sqlalchemy.orm import declarative_base
import uuid
import datetime
from .. import datamodels
from ..keck import koa

__all__ = ['HISPECCalibrationORM']

from koa_middleware import CalibrationORM

# Use the environment variable for database connection
_Base = declarative_base()

# Generate a UUID for the namespace using a hash of the unique string
UUID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "HISPEC_DRP")

class HISPECCalibrationORM(CalibrationORM, _Base):
    __tablename__ = "calibrations"

    id: Mapped[uuid.UUID] = mapped_column(String(36), primary_key=True)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    koa_id: Mapped[str] = mapped_column(String, nullable=True)
    koa_filepath: Mapped[str] = mapped_column(String, nullable=True)
    master: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    cal_type: Mapped[str] = mapped_column(String, nullable=False)
    instera: Mapped[str] = mapped_column(String, nullable=False)
    spectrograph: Mapped[str] = mapped_column(String, nullable=False)
    mjd_start: Mapped[float] = mapped_column(Float, nullable=False)
    datetime_obs: Mapped[str] = mapped_column(String, nullable=False)
    read_mode: Mapped[str] = mapped_column(String, nullable=False)
    target_name: Mapped[str] = mapped_column(String, nullable=False)
    object_fiber1: Mapped[str] = mapped_column(String, nullable=True)
    object_fiber2: Mapped[str] = mapped_column(String, nullable=True)
    object_fiber3: Mapped[str] = mapped_column(String, nullable=True)
    drp_version: Mapped[str] = mapped_column(String, nullable=False)
    note: Mapped[str] = mapped_column(String, nullable=True)
    last_updated: Mapped[str] = mapped_column(String, nullable=False)

    def to_dict(self) -> dict:
        return {col.name: getattr(self, col.name) for col in self.__table__.columns}

    @classmethod
    def from_dict(cls, data: dict) -> 'HISPECCalibrationORM':
        return cls(**data)

    @staticmethod
    def generate_uuid(filename: str, drp_version: str, last_updated: str) -> str:
        return str(uuid.uuid5(UUID_NAMESPACE, filename + drp_version + last_updated))

    def __repr__(self):
        fields = '\n'.join(f"  {col.name}: {getattr(self, col.name)}" for col in self.__table__.columns)
        return f"{self.__class__.__name__}\n{fields}"

    @classmethod
    def from_datamodel(
        cls,
        input : str | datamodels.HISPECCalibrationModel,
        note : str | None = None,
    ) -> 'HISPECCalibrationORM':
        cal_model = datamodels.open(input, meta_only=True)
        datetime_obs = cal_model.meta.exposure.datetime_start
        filename = cal_model.meta.filename
        drp_version = cal_model.meta.drp_version
        koa_filepath = koa.generate_koa_filepath(cal_model)
        koa_id = koa_filepath.split('/')[-1]
        last_updated = datetime.datetime.now().isoformat()
        _id = cls.generate_uuid(filename, drp_version, last_updated)
        return cls(
            id=_id,
            filename=filename,
            koa_id=koa_id,
            koa_filepath=koa_filepath,
            cal_type=cal_model.meta.cal_type.lower(),
            instera=cal_model.meta.instrument.era,
            read_mode=cal_model.meta.exposure.read_mode,
            target_name=cal_model.meta.target.name,
            object_fiber1=cal_model.meta.target.object_fiber1,
            object_fiber2=cal_model.meta.target.object_fiber2,
            object_fiber3=cal_model.meta.target.object_fiber3,
            spectrograph=cal_model.meta.instrument.spectrograph,
            mjd_start=cal_model.meta.exposure.mjd_start,
            datetime_obs=datetime_obs,
            drp_version=drp_version,
            note=note,
            last_updated=last_updated,
        )
