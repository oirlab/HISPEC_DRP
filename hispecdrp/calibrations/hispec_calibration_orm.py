from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy import String, Float, Boolean
from sqlalchemy.orm import declarative_base
import uuid

from koa_middleware import CalibrationORM

__all__ = ['HISPECCalibrationORM']

# Use the environment variable for database connection
_Base = declarative_base()

class HISPECCalibrationORM(CalibrationORM, _Base):
    __tablename__ = "HISPEC"

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
