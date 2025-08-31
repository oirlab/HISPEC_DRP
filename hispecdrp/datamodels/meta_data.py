from pydantic import BaseModel, Field
from typing import Any, ClassVar, Literal, Union, get_args, get_origin
from sqlalchemy import Column, Integer, String, Float, Boolean
from sqlalchemy.orm import declarative_base

__all__ = [
    'FitsField', 'MetaModel',
    'CoreMeta',
    'InstrumentMeta', 'ExposureMeta', 'TargetMeta', 'ProgramMeta',
    'CalibrationCoreMeta', 'CoreMeta',
]

Base = declarative_base()

def FitsField(*,
    default : Any = None,
    fits_keyword: str, unit : str | None = None,
    fits_hdu : str = 'PRIMARY',
    orm_field : bool | str = False,
    **kwargs
):
    extra = kwargs.pop("json_schema_extra", {})
    extra["fits_keyword"] = fits_keyword
    extra["unit"] = unit
    extra["fits_hdu"] = fits_hdu
    extra["orm_field"] = orm_field
    return Field(default=default, json_schema_extra=extra, **kwargs)


class MetaModel(BaseModel):

    SECTION_NAME : ClassVar[str | None] = None

    ORM_TABLE_NAME : ClassVar[str | None] = None

    model_config = {
        'validate_assignment' : True,
        'extra' : 'forbid',
    }
    # __pydantic_extra__ = {
    #     'SECTION_NAME' : ClassVar[str | None],
    #     'ORM_TABLE_NAME' : ClassVar[str | None],
    # }

    def __setitem__(self, item, value):
        setattr(self, item, value)
    
    @classmethod
    def group_meta_by_section(cls, flat: bool = True) -> dict:
        meta_sections = {}
        for attr, _ in cls.model_fields.items():
            section_name = None
            for _class in cls.__mro__:
                if issubclass(_class, MetaModel) and attr in getattr(_class, '__annotations__', {}):
                    section_name = _class.SECTION_NAME
                    break
            
            assert section_name is not None, f"No Meta section found for {attr}"
            meta_sections[attr] = section_name
        
        if flat:
            return meta_sections
        else:
            out = {}
            for k, v in meta_sections.items():
                out.setdefault(v, []).append(k)
            return out

    def pprint(self, show_none : bool = False):
        """
        Pretty-print the metadata.
        """
        for attr in self.__class__.model_fields.keys():
            val = getattr(self, attr)
            field = self.__class__.model_fields[attr]
            if not val is None or show_none:
                print(f"{attr}: {val}")
                print(f"  title: {field.title}")
                print(f"  fits_keyword: {field.json_schema_extra['fits_keyword']}")
                print(f"  fits_hdu: {field.json_schema_extra['fits_hdu']}")
                print(f"  annotation: {field.annotation}")
                unit = field.json_schema_extra['unit']
                if unit is not None:
                    print(f"  unit: {unit}")

    @classmethod
    def get_orm_class(cls) -> dict:
        model_attrs = {}
        model_attrs['__tablename__'] = cls.ORM_TABLE_NAME
        model_attrs['__table_args__'] = {'extend_existing': True}
        for attr, field in cls.model_fields.items():
            if field.json_schema_extra['orm_field']:
                field_name = cls.get_orm_field_name(attr, field)
                model_attrs[field_name] = cls.field_to_mapped(attr, field)
        orm_class = type(f"{cls.__name__}ORM", (Base,), model_attrs)
        return orm_class

    def to_orm(self) -> dict:
        meta_vals = {}
        for attr, field in self.__class__.model_fields.items():
            if field.json_schema_extra['orm_field']:
                field_name = self.get_orm_field_name(attr, field)
                meta_vals[field_name] = getattr(self, attr)
        orm_class = self.get_orm_class()
        meta_orm = orm_class(**meta_vals)
        return meta_orm
    
    @classmethod
    def field_to_mapped(cls, attr : str, field : Field) -> dict:
        TYPE_MAP = {
            int: Integer,
            str: String,
            float: Float,
            bool: Boolean,
        }
        field_name = cls.get_orm_field_name(attr, field)
        py_type = cls.remove_none_type(field.annotation)
        col_type = TYPE_MAP.get(py_type, String)
        default = field.default
        comment = field.title

        if field_name == 'id':
            return Column(field_name, col_type, primary_key=True, comment=comment)
        else:
            return Column(field_name, col_type, default=default, comment=comment)
    
    @staticmethod
    def remove_none_type(t: type) -> type:
        origin = get_origin(t)
        if origin is Union:
            args = [a for a in get_args(t) if a is not type(None)]
            if len(args) == 1:
                return args[0]
        return t
    
    @staticmethod
    def get_orm_field_name(attr: str, field: Field) -> str | None:
        orm_field = field.json_schema_extra.get('orm_field', False)
        if not orm_field:
            return None
        return orm_field if isinstance(orm_field, str) else attr


class _CoreMeta(MetaModel):

    SECTION_NAME = 'Core'

    last_updated: str | None = FitsField(
        fits_keyword="LASTUPD",
        title="ISO 8601 datetime file was last updated YYYYMMDDTHHMMSS",
        orm_field=True,
    )
    date_obs: str | None = FitsField(
        fits_keyword="DATE-OBS",
        title="ISO 8601 datetime of the observation YYYYMMDD for KOA",
    )
    utc_obs: str | None = FitsField(
        fits_keyword="UTC",
        title="UTC time of the observation HHMMSS.SS for KOA"
    )
    filename: str | None = FitsField(
        fits_keyword="FILENAME",
        title="Name of the file",
        orm_field=True,
    )
    file_type: str | None = FitsField(
        fits_keyword="FILETYPE",
        title="Type of file",
        default="FITS"
    )
    telescope: str | None = FitsField(
        fits_keyword="TELESCOP",
        title="Telescope used to acquire the data",
        default="Keck-II"
    )
    data_level: str | None = FitsField(
        fits_keyword="DATALVL",
        title="Data level"
    )
    model_type: str | None = FitsField(
        fits_keyword="DATAMODL",
        title="Type of data model"
    )
    drp_version: str | None = FitsField(
        fits_keyword="DRPVER",
        title="DRP version used to create this file",
        orm_field=True,
    )
    note: str | None = FitsField(
        fits_keyword="NOTE",
        title="Custom note",
        orm_field=True,
    )
    order_start : int | None = FitsField(
        fits_keyword="ORDERST",
        title="Starting echelle order of the data array",
    )
    order_end: int | None = FitsField(
        fits_keyword="ORDEREND",
        title="End echelle order of the data array",
    )


class InstrumentMeta(MetaModel):
    SECTION_NAME = 'Core'
    instrument_name: str | None = FitsField(
        fits_keyword="INSTRUME",
        title="Name of the instrument used to acquire the data",
    )
    spectrograph: Literal['BSPEC', 'RSPEC'] | None = FitsField(
        fits_keyword="SPECTROG",
        title="Name of the spectrograph (BSPEC or RSPEC)",
        orm_field=True,
    )
    instrument_era: str | None = FitsField(
        fits_keyword="INSTERA",
        title="Instrument era specified via semver",
        default="0.0.1",
        orm_field=True,
    )
    instrument_mode: str | None = FitsField(
        fits_keyword="INSTMODE",
        title="Instrument mode (YJ, HK, YJHK)",
        default="YJHK",
        orm_field=True,
    )


class ExposureMeta(MetaModel):
    SECTION_NAME = 'Core'
    exposure_type : str | None = FitsField(
        fits_keyword="IMTYPE",
        title="Type of exposure",
        orm_field=True,
        default='SCI'
    )
    data_quality: str | None = FitsField(
        fits_keyword="DATAQUAL",
        title="Data quality flag at the exposure level",
        default="GOOD"
    )
    datetime_obs: str | None = FitsField(
        fits_keyword="DATEOBS",
        title="Exposure observation datetime in format YYYY-MM-DDTHH:MM:SS.SSS",
        orm_field=True,
    )
    mjd_start: float | None = FitsField(
        fits_keyword="EXPMJDST",
        title="Exposure start Modified Julian Date",
        orm_field=True,
    )
    read_mode: str | None = FitsField(
        fits_keyword="READMODE",
        title="Readout mode",
        default='DEFAULT',
        orm_field=True,
    )
    exposure_time: float | None = FitsField(
        fits_keyword="EXPTIME",
        title="Effective exposure time",
        unit="sec"
    )


class TargetMeta(MetaModel):
    SECTION_NAME = 'Core'
    target_name: str | None = FitsField(
        fits_keyword="OBJECT",
        title="Target or object name, internal or on-sky"
    )
    target_ra: float | None = FitsField(
        fits_keyword="RA",
        title="Target Right Ascension",
        unit="degrees"
    )
    target_dec: float | None = FitsField(
        fits_keyword="DEC",
        title="Target Declination",
        unit="degrees"
    )
    target_ra_uncertainty: float | None = FitsField(
        fits_keyword="URA",
        title="Target RA uncertainty",
        unit="degrees"
    )
    target_dec_uncertainty: float | None = FitsField(
        fits_keyword="UDEC",
        title="Target Dec uncertainty",
        unit="degrees"
    )
    target_mu_ra: float | None = FitsField(
        fits_keyword="MURA",
        title="Target proper motion in RA",
        unit="degrees"
    )
    target_mu_dec: float | None = FitsField(
        fits_keyword="MUDEC",
        title="Target proper motion in Dec",
        unit="degrees"
    )
    airmass_start: float | None = FitsField(
        fits_keyword="AIRMAST",
        title="Airmass at start of exposure"
    )
    airmass_end: float | None = FitsField(
        fits_keyword="AIRMAEND",
        title="Airmass at end of exposure"
    )
    azimuth_start: float | None = FitsField(
        fits_keyword="AZST",
        title="Azimuth at start of exposure"
    )
    azimuth_end: float | None = FitsField(
        fits_keyword="AZEND",
        title="Azimuth at end of exposure"
    )
    elevation_start: float | None = FitsField(
        fits_keyword="ELSTART",
        title="Elevation at start of exposure"
    )
    elevation_end: float | None = FitsField(
        fits_keyword="ELEND",
        title="Elevation at end of exposure"
    )
    equinox: str | None = FitsField(
        fits_keyword="EQUINOX",
        title="Equinox of coordinates",
        default="2000"
    )


class ProgramMeta(MetaModel):
    SECTION_NAME = 'Core'
    program_name: str | None = FitsField(
        fits_keyword="PROGNAME",
        title="Program name"
    )


class CoreMeta(_CoreMeta, InstrumentMeta, ExposureMeta, TargetMeta, ProgramMeta):
    pass


class CalibrationCoreMeta(MetaModel):
    SECTION_NAME = 'Core'
    cal_id: str | None = FitsField(
        fits_keyword="CALID",
        title="Unique identifier",
        orm_field='id',
    )
    cal_type: str | None = FitsField(
        fits_keyword="CALTYPE",
        title="Type of calibration",
        orm_field=True,
    )
    master_cal: bool | None = FitsField(
        fits_keyword="CALMAST",
        title="Appropriate for calibrating data?",
        default=True,
        orm_field=True,
    )