from astropy.io import fits
import astropy.time
from pydantic import BaseModel, Field, create_model, model_validator, ValidatorFunctionWrapHandler, ConfigDict
from typing import Any, Literal, Callable, Self, Annotated

from ..utils.koa import generate_koa_id, generate_koa_filehandle


__all__ = [
    'FitsField', 'MetaModel',
    'CoreMeta',
    'InstrumentMeta', 'ExposureMeta', 'TargetMeta', 'ProgramMeta',
]

def FitsField(*,
    default : Any = None,
    fits_keyword: str, unit : str | None = None,
    fits_hdu : str = 'PRIMARY',
    orm_field : bool | str = False,
    title: str | None = None,
    merge_rule : str | Callable | None = 'first',
    **kwargs
):
    """
    Special constructor for creating pydantic Field objects based on Fits metadata.

    Parameters
    ----------
    default : Any
        The default value for the field.
    fits_keyword : str
        The FITS keyword associated with the field.
    unit : str or None, optional
        The unit of the field.
    fits_hdu : str, optional
        The FITS HDU (Header Data Unit) associated with the field. Defaults to 'PRIMARY'.
    orm_field : bool or str
        Whether the field should be included in the ORM mapping. If `True`, the field
        will be included using the field's attribute name. If a string is provided,
        it will be used as the ORM column name. If `False`, the field will not be included
        in the ORM mapping.
    title : str or None, optional
        The title of the Pydantic Field, also used for the FITS comment. Defaults to None.
    merge_rule : str or Callable, optional
        Rule for merging multiple metadata values. Can be a string indicating a predefined rule
        (e.g., 'first', 'sum') or a custom callable function. Defaults to 'first'.
    **kwargs
        Additional keyword arguments accepted by Pydantic's `Field` constructor.

    Returns
    -------
    Field
        A pydantic Field object with additional FITS metadata stored in json_schema_extra.
    """
    extra = kwargs.pop("json_schema_extra", {})
    extra["fits_keyword"] = fits_keyword
    extra["unit"] = unit
    extra["fits_hdu"] = fits_hdu
    extra["orm_field"] = orm_field
    extra["merge_rule"] = merge_rule
    return Field(default=default, title=title, json_schema_extra=extra, **kwargs)

class CoreMeta(BaseModel):
    last_updated: str | None = FitsField(
        fits_keyword="LASTUPD",
        title="ISO 8601 datetime file was last updated YYYY-MM-DDTHH:MM:SS.SSS",
        orm_field=True,
    )
    datetime_obs: str | None = FitsField(
        fits_keyword="DATETOBS",
        title="Exposure observation datetime in format YYYY-MM-DDTHH:MM:SS.SSS",
        orm_field=True,
        merge_rule='min_datetime',
        default='2024-09-24T00:00:00.000' # NOTE: Could be dangerous
    )
    filename: str | None = FitsField(
        fits_keyword="FILENAME",
        title="Name of the file",
        orm_field=True,
        merge_rule='drop'
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
    model_type: str | None = FitsField(
        fits_keyword="DATAMODL",
        title="Type of data model",
        merge_rule='drop',
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

class InstrumentMeta(BaseModel):
    instrument_name: str | None = FitsField(
        fits_keyword="INSTRUME",
        title="Instrument name",
        orm_field=True,
    )
    instrument_era: str | None = FitsField(
        fits_keyword="INSTERA",
        title="Instrument era specified with semver",
        default="0.0.1",
        orm_field=True,
    )
    instrument_mode: str | None = FitsField(
        fits_keyword="INSTMODE",
        title="Instrument mode",
        orm_field=True,
    )
    lfc_nu0: float | None = FitsField(
        fits_keyword="LFCNU0",
        title="LFC nu0",
        unit="GHz",
    )
    lfc_delta_nu: float | None = FitsField(
        fits_keyword="LFCDNU",
        title="LFC delta nu",
        unit="GHz",
    )
    order_start : int | None = FitsField(
        fits_keyword="ORDERST",
        title="Starting echelle order of the data array",
    )
    order_end: int | None = FitsField(
        fits_keyword="ORDEREND",
        title="End echelle order of the data array",
    )

    @classmethod
    def create(cls, instrument_name : str, valid_spectrographs : list[str], valid_modes : list[str]):
        _fields = {
            'instrument_name' : (
                Literal[instrument_name],
                FitsField(
                    fits_keyword=f"INSTRUME",
                    title=f"Instrument name",
                    orm_field=True,
                    default=instrument_name,
                    merge_rule='coalesce',
                )
            ),
            'spectrograph' : (
                Annotated[str, Literal[tuple(valid_spectrographs)]],
                FitsField(
                    fits_keyword="SPECTROG",
                    title="Spectrograph name",
                    orm_field=True,
                    default=valid_spectrographs[0],
                    merge_rule='coalesce',
                )
            ),
            'instrument_mode' : (
                Annotated[str, Literal[tuple(valid_modes)]],
                FitsField(
                    fits_keyword="INSTMODE",
                    title="Instrument mode",
                    orm_field=True,
                    default=valid_modes[0],
                    merge_rule='coalesce',
                )
            )
        }

        model_class = create_model(
            f"{instrument_name}{cls.__name__}",
            **_fields,
            __base__=cls
        )

        return model_class

VALID_IMTYPES = Literal['DARK', 'BIAS', 'DETFLAT', 'SPEC', 'NONE']

class ExposureMeta(BaseModel):
    exposure_type : VALID_IMTYPES | None = FitsField(
        fits_keyword="IMTYPE",
        title="Type of exposure",
        orm_field=True,
        default='NONE'
    )
    data_quality: str | None = FitsField(
        fits_keyword="DATAQUAL",
        title="Data quality flag at the exposure level",
        default="GOOD"
    )
    mjd_start: float | None = FitsField(
        fits_keyword="EXPMJDST",
        title="Exposure start Modified Julian Date of first read",
        merge_rule='min',
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
        unit="sec",
        merge_rule='sum',
        default=0.0,
    )
    clock_rate: float | None = FitsField(
        fits_keyword="CLOCKRAT",
        title="Clock rate of the readout of the detector.",
        unit="Hz"
    )

class TargetMeta(BaseModel):
    target_name: str | None = FitsField(
        fits_keyword="OBJECT",
        title="Target or object name, internal or on-sky",
        default='NONE',
        orm_field=True,
    )
    target_ra: float | None = FitsField(
        fits_keyword="RA",
        title="Target Right Ascension",
        unit="degrees",
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
        title="Airmass at start of exposure",
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

    @classmethod
    def create(cls, instrument_name: str, num_fibers: int, valid_fiber_objects: list[str]):
        object_fiber_fields = {
            f"object_fiber{i}": (
                Annotated[str, Literal[tuple(valid_fiber_objects)]] | None,
                FitsField(
                    fits_keyword=f"OBJFIB{i}",
                    title=f"Object name in fiber {i}",
                    orm_field=True,
                    default='NONE',
                    merge_rule='coalesce',
                )
            )
            for i in range(1, num_fibers + 1)
        }

        model = create_model(
            f"{instrument_name}{cls.__name__}",
            **object_fiber_fields,
            __base__=cls
        )
   
        return model

class ProgramMeta(BaseModel):
    program_name: str | None = FitsField(
        fits_keyword="PROGNAME",
        title="Program name"
    )

class KOAMeta(BaseModel):
    koa_id: str | None = FitsField(
        fits_keyword="KOAID",
        title="KOA ID",
        merge_rule='drop',
        orm_field=True,
    )
    koa_filehandle: str | None = FitsField(
        fits_keyword="KOAPATH",
        title="KOA file path",
        merge_rule='drop',
        orm_field=True,
    )
    koa_date_obs : str | None = FitsField(
        fits_keyword="DATE-OBS",
        title="KOA observation date YYYY-MM-DD",
        merge_rule='coalesce',
    )
    koa_utc_obs : str | None = FitsField(
        fits_keyword="UT",
        title="KOA observation UTC HH:MM:SS.S",
    )

class CalibrationMeta(BaseModel):
    cal_id: str | None = FitsField(
        fits_keyword="CALID",
        title="Unique calibration identifier",
        orm_field='id',
        merge_rule='drop',
    )
    cal_type: str | None = FitsField(
        fits_keyword="CALTYPE",
        title="Type of calibration",
        orm_field=True,
    )
    master_cal: bool = FitsField(
        fits_keyword="MASTCAL",
        title="Appropriate for calibrating data?",
        default=True,
        orm_field=True,
    )


class MetaModel(KOAMeta, ProgramMeta, CalibrationMeta, InstrumentMeta, ExposureMeta, TargetMeta, CoreMeta):

    model_config = ConfigDict(extra='forbid', validate_assignment=True)

    def to_fits_header(self, extname: str = "PRIMARY", add_missing_metadata: bool = False) -> fits.Header:
        """Create a FITS header from flattened metadata fields."""
        
        # Create empty FITS header
        header = fits.Header()

        def make_comment(field : Field) -> str:
            comment = field.title
            unit = field.json_schema_extra['unit']
            if unit is not None:
                if unit.lower() != 'none':
                    comment += f" ({unit})"
            return comment

        meta_sections = self.group_meta_by_section(flat=False)

        for section_name, section_attrs in meta_sections.items():

            # Add section header as COMMENT
            header.append(('COMMENT', '=' * 40), end=True)
            header.append(('COMMENT', f'  {section_name}'), end=True)
            header.append(('COMMENT', '=' * 40), end=True)

            for attr in section_attrs:

                # Current value and field
                value = getattr(self, attr)
                field = self.__class__.model_fields[attr]

                # Skip if no value and not adding empty entries
                if value is None and not add_missing_metadata:
                    continue

                # FITS HDU
                field_extname = field.json_schema_extra['fits_hdu']
                if extname != field_extname:
                    continue

                # Info and value
                keyword = field.json_schema_extra['fits_keyword']
                comment = make_comment(field)

                # Add to header
                header.append((keyword, value, comment), end=True)

        return header
    
    def pprint(self, show_none : bool = False):
        """
        Pretty-print the metadata.
        """
        for attr, field in self.__class__.model_fields.items():
            val = getattr(self, attr)
            if not val is None or show_none:
                unit = field.json_schema_extra['unit']
                if unit is not None:
                    print(f"{attr}: {val} {unit}")
                else:
                    print(f"{attr}: {val}")
                print(f"  title: {field.title}")
                print(f"  fits_keyword: {field.json_schema_extra['fits_keyword']}")
                print(f"  fits_hdu: {field.json_schema_extra['fits_hdu']}")
                print(f"  annotation: {field.annotation}")
    
    # @classmethod
    # def group_meta_by_section(cls, flat: bool = True) -> dict:
    #     meta_sections = {}
    #     for attr, _ in cls.model_fields.items():
    #         section_name = None
    #         for _class in cls.__mro__:
    #             if issubclass(_class, BaseModel) and attr in getattr(_class, '__annotations__', {}):
    #                 section_name = _class.__name__.replace('Meta', '').replace('_', '').replace('HISPEC', '').replace('PARVI', '')
    #                 break
            
    #         assert section_name is not None, f"No Meta section found for {attr}"
    #         meta_sections[attr] = section_name

    #     if flat:
    #         return meta_sections
    #     else:
    #         out = {}
    #         for k, v in meta_sections.items():
    #             out.setdefault(v, []).append(k)
    #         out = dict(reversed(out.items()))
    #         return out

    @classmethod
    def group_meta_by_section(cls, flat: bool = True) -> dict:
        def section_name_from_class(_class):
            name = _class.__name__
            for t in ("Meta", "_", "HISPEC", "PARVI"):
                name = name.replace(t, "")
            return name

        # Map field → section
        field_to_section = {}
        for attr in cls.model_fields:
            for _class in cls.__mro__:
                if issubclass(_class, BaseModel) and attr in getattr(_class, "__annotations__", {}):
                    field_to_section[attr] = section_name_from_class(_class)
                    break

        # Group fields by section (this keeps fields grouped correctly)
        sections = {}
        for f in cls.model_fields:
            sec = field_to_section[f]
            sections.setdefault(sec, []).append(f)

        # -------- the actual fix --------
        # Section order: Core first, then alphabetical (or any deterministic grouping)
        section_order = []
        if "Core" in sections:
            section_order.append("Core")

        section_order.extend(
            sec for sec in sorted(sections.keys())
            if sec != "Core"
        )
        # --------------------------------

        if not flat:
            return {sec: sections[sec] for sec in section_order}

        # Flatten, preserving contiguous section grouping
        out = {}
        for sec in section_order:
            for f in sections[sec]:
                out[f] = sec
        return out




    @model_validator(mode='wrap')
    @classmethod
    def set_datetime(cls, data: Any, handler: ValidatorFunctionWrapHandler) -> Self:
        instance = handler(data)
        if instance.datetime_obs is not None:
            dt = astropy.time.Time(instance.datetime_obs, format="isot", scale="utc")
            if instance.koa_date_obs is None:
                object.__setattr__(instance, 'koa_date_obs', dt.strftime("%Y-%m-%d"))
            if instance.koa_utc_obs is None:
                object.__setattr__(instance, 'koa_utc_obs', dt.strftime("%H:%M:%S.%f").rstrip("0").rstrip("."))
            if instance.mjd_start is None:
                object.__setattr__(instance, 'mjd_start', dt.mjd)
        return instance
    
    @model_validator(mode='wrap')
    @classmethod
    def set_koa(cls, data : Any, handler : ValidatorFunctionWrapHandler) -> Self:
        instance = handler(data)
        if instance.koa_id is None and instance.filename is not None:
            koa_id = generate_koa_id(instance)
            object.__setattr__(instance, 'koa_id', koa_id)
        if instance.koa_filehandle is None:
            koa_filehandle = generate_koa_filehandle(instance)
            object.__setattr__(instance, 'koa_filehandle', koa_filehandle)
        return instance