from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy import String, Float
from sqlalchemy.orm import declarative_base
import uuid

from koa_middleware import CalibrationORM

__all__ = ['PARVICalibrationORM']

# Use the environment variable for database connection
_Base = declarative_base()

class PARVICalibrationORM(CalibrationORM, _Base):
    __tablename__ = "PARVI"

    id: Mapped[uuid.UUID] = mapped_column(String(36), primary_key=True)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    cal_type: Mapped[str] = mapped_column(String, nullable=False)
    instera: Mapped[str] = mapped_column(String, nullable=False)
    mjd_start: Mapped[float] = mapped_column(Float, nullable=False)
    datetime_obs: Mapped[str] = mapped_column(String, nullable=False)
    read_mode: Mapped[str] = mapped_column(String, nullable=False)
    target_name: Mapped[str] = mapped_column(String, nullable=False)
    object_fiber1: Mapped[str] = mapped_column(String, nullable=True)
    object_fiber2: Mapped[str] = mapped_column(String, nullable=True)
    object_fiber3: Mapped[str] = mapped_column(String, nullable=True)
    object_fiber4: Mapped[str] = mapped_column(String, nullable=True)
    drp_version: Mapped[str] = mapped_column(String, nullable=False)
    note: Mapped[str] = mapped_column(String, nullable=True)
    last_updated: Mapped[str] = mapped_column(String, nullable=False)