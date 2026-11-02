from koa_middleware import LocalCalibrationDB
from koa_middleware.selector_base import CalibrationSelector
from ..datamodels import MetaModel

import os

__all__ = [
    'DQSelector',
    'SaturationSelector',
    'DarkSelector',
    'BiasSelector',
    'ReadNoiseSelector',
    'NonlinearCorrectionSelector',
    'DetectorFlatSelector',
    'FiberFlatSelector',
    'GainSelector',
    'WavelengthSolutionSelector',
    'ReferenceEtalonSelector',
    'TracePosSelector',
    'TraceProfSelector',
    'LSFSelector',
]

class HISPECCalibrationSelector(CalibrationSelector):
    """
    Base selector for HISPEC calibrations.
    Implements common selection logic based on origin, version, and master calibration status
    """

    def __init__(self, origin : str = "ANY", cal_version : str | None = 'LATEST', master_cal : bool | str | None = None, **kwargs):
        """
        Initialize the selector with optional filtering parameters.

        Parameters
        ----------
        origin : str, optional
            The origin to filter by. Can be "ANY", "LOCAL", or "REMOTE". Default is "ANY".
        cal_version : str, optional
            The calibration version to filter by. Can be a specific version string (e.g., "001"), "ANY", "LATEST", or a negative integer string (e.g., "-1" for previous version). Default is "LATEST".
        master_cal : bool | str | None, optional
            Whether to filter for master calibrations. Can be True, False, "PREFER" (prefer master but allow non-master if no master available), or None/"ANY" (no filtering). Default is None.
        **kwargs
            Additional keyword arguments for the selector that are set as attributes.
        """
        super().__init__(origin=origin, **kwargs)
        if cal_version is not None:
            self.cal_version = cal_version
        else:
            self.cal_version = os.getenv('HISPECDRP_CALIBRATION_VERSION')
        if master_cal is not None:
            self.master_cal = master_cal
        else:
            self.master_cal = os.getenv('HISPECDRP_MASTER_CAL')

    def select_best(self, input, candidates : list[dict] | dict, **kwargs) -> dict | None:
        """
        Select the best calibration(s) based on the candidates.
        The default implementation does the following:

            - If `candidates` is a list, it applies the origin and versions filters and returns the first candidate if available, or `None` if the list is empty.
            - If `candidates` is not a list (e.g., a single dict), it returns the element as is.
        
        The input ordering is preserved.
        This can optionally be overridden by subclasses, but care should be taken in doing so to maintain the expected behavior.
        
        Parameters
        ----------
        input
            The input object for which a calibration is to be selected.
            The exact type depends on the specific selector implementation.
        candidates : list[dict] | dict
            Candidate calibrations returned from `get_candidates()`.
        **kwargs
            Additional filtering parameters as necessary.
        
        Returns
        -------
        dict | None
            Selected calibration metadata record, or None if no candidates available.
        """

        if len(candidates) == 0:
            return None
        if type(candidates) in (dict, None):
            return candidates
        
        candidates = self._filter_origin(candidates)
        if len(candidates) == 0:
            return None
        candidates = self._filter_master(candidates)
        if len(candidates) == 0:
            return None
        
        cal0 = candidates[0]
        candidates_family = []
        for cal in candidates:
            # Check if next cal is in same version family
            # HACK: Spectrograph, cal_type, master_cal already filtered correctly, only need to check date time obs
            # This is a temporary workaround since get_version_family_column_names is implemented in the store and not accessible here.
            if cal0['datetime_obs'] == cal['datetime_obs']:
                candidates_family.append(cal)

        # Narrowed down to single version family, now safely filter by version
        candidates = self._filter_version(candidates_family)
        result = candidates[0] if len(candidates) > 0 else None

        return result

    def _filter_origin(self, rows : list[dict], origin : str | None = None) -> list[dict]:
        """
        Filter the candidate rows based on the origin.

        Parameters
        ----------
        rows : list[dict]
            List of candidate calibration metadata records, each containing an 'origin' key.
        origin : str, optional
            The origin filter to apply. Can be "ANY", "LOCAL", or "REMOTE".
            If None, defaults to `self.origin`.

        Returns
        -------
        list[dict]
            Filtered list of calibration metadata records.
        """
        if origin is None:
            origin = self.origin
        if origin in ("ANY", None):
            return rows
        elif origin == "LOCAL":
            return [row for row in rows if row['origin'] == "LOCAL"]
        elif origin == "REMOTE":
            return [row for row in rows if row['origin'] == "REMOTE"]
        else:
            raise ValueError(f"Invalid origin filter: {origin}. Must be 'ANY', 'LOCAL', or 'REMOTE'.")
        
    def _filter_version(self, rows : list[dict], cal_version : str | None = None) -> list[dict]:
        """
        Filter the candidate rows based on the calibration version.

        Parameters
        ----------
        rows : list[dict]
            List of candidate calibration metadata records, each containing a 'cal_version' key.
        cal_version : str, optional
            The calibration version to filter by. If None, no version filtering is applied.

        Returns
        -------
        list[dict]
            Filtered list of calibration metadata records.
        """

        if cal_version is None:
            cal_version = self.cal_version
        if isinstance(cal_version, str):
            cal_version = cal_version.upper()

        # Extract all versions. Remote / legacy rows may have cal_version=None;
        # treat missing versions as older than any explicit string so LATEST /
        # rollback comparisons never mix str and NoneType.
        versions = [row['cal_version'] for row in rows]

        def _version_sort_key(v):
            if v is None:
                return (0, "")
            return (1, str(v))

        # If None, pick latest version
        if cal_version in (None, "ANY", "LATEST"):
            vmax = max(versions, key=_version_sort_key)
            return [row for row, v in zip(rows, versions) if v == vmax]
        
        # If specific version provided, filter by that version
        if (
            isinstance(cal_version, str)
            and len(cal_version) == 3
            and cal_version.isdigit()
        ):
            return [row for row in rows if row['cal_version'] == cal_version]

        # If version is a negative integer, roll back to previous version
        if (
            isinstance(cal_version, str)
            and cal_version.startswith("-")
            and cal_version[1:].isdigit()
        ):
            n = int(cal_version)
            unique_versions = sorted(set(versions), key=_version_sort_key, reverse=True)
            if -n > len(unique_versions):
                return []  # Not enough versions available
            target_version = unique_versions[-n]
            return [row for row in rows if row['cal_version'] == target_version]
        
        if (
            isinstance(cal_version, int)
            and cal_version < 0
        ):
            n = cal_version
            unique_versions = sorted(set(versions), key=_version_sort_key, reverse=True)
            if -n > len(unique_versions):
                return []  # Not enough versions available
            target_version = unique_versions[-n]
            return [row for row in rows if row['cal_version'] == target_version]
    
    def _filter_master(self, rows : list[dict], master_cal : bool | None = None) -> list[dict]:
        """
        Filter the candidate rows to only include master calibrations.

        Parameters
        ----------
        rows : list[dict]
            List of candidate calibration metadata records, each containing a 'master_cal' key.

        Returns
        -------
        list[dict]
            Filtered list of calibration metadata records where 'master_cal' is True.
        """
        if master_cal is None:
            master_cal = self.master_cal

        if master_cal is True:
            return [row for row in rows if row['master_cal'] == 1]
        elif master_cal is False:
            return [row for row in rows if row['master_cal'] == 0]
        elif master_cal in ("ANY", None):
            return rows
        elif isinstance(master_cal, str):
            master_cal = master_cal.upper()
            if master_cal == "PREFER":
                master_cals = [row for row in rows if row['master_cal'] == 1]
                if master_cals:
                    return master_cals
                else:
                    non_master_cals = [row for row in rows if row['master_cal'] == 0]
                    return non_master_cals
            else:
                raise ValueError(f"Invalid master_cal filter: {master_cal}. Must be 'PREFER', 'ANY', True, or False.")
        else:
            return rows

class DQSelector(HISPECCalibrationSelector):
    """
    Selector for data quality calibration files.

    **Calibration Type:** dq

    **Selection Logic:**

    - Match instrument era
    - Match spectrograph
    - Closest in observation time
    """

    def get_candidates(self, meta : MetaModel, db : LocalCalibrationDB):
        rows = db.rows_where(
            """
            cal_type = :cal_type AND
            instrument_era = :era AND
            spectrograph = :spec
            """,
            {
                "cal_type": "dq",
                "era": meta["instrument_era"],
                "spec": meta["spectrograph"],
                "mjd_start": meta["mjd_start"]
            },
            order_by="ABS(mjd_start - :mjd_start)"
        )

        return list(rows)

class SaturationSelector(HISPECCalibrationSelector):
    """
    Selector for saturation calibration files.

    **Calibration Type:** saturation

    **Selection Logic:**

    - Match instrument era
    - Match spectrograph
    - Require master calibration
    - Closest in observation time
    """

    def get_candidates(self, meta : MetaModel, db : LocalCalibrationDB):
        rows = db.rows_where(
            """
            cal_type = :cal_type AND
            instrument_era = :era AND
            spectrograph = :spec
            """,
            {
                "cal_type": "saturation",
                "era": meta["instrument_era"],
                "spec": meta["spectrograph"],
                "mjd_start": meta["mjd_start"]
            },
            order_by="ABS(mjd_start - :mjd_start)"
        )

        return list(rows)

class DarkSelector(HISPECCalibrationSelector):
    """
    Selector for dark calibration files.

    **Calibration Type:** dark

    **Selection Logic:**

    - Match instrument era
    - Match spectrograph
    - Closest in observation time
    """

    def get_candidates(self, meta : MetaModel, db : LocalCalibrationDB):
        rows = db.rows_where(
            """
            cal_type = :cal_type AND
            instrument_era = :era AND
            spectrograph = :spec
            """,
            {
                "cal_type": "dark",
                "era": meta["instrument_era"],
                "spec": meta["spectrograph"],
                "mjd_start": meta["mjd_start"],
            },
            order_by="ABS(mjd_start - :mjd_start)"
        )

        return list(rows)

class TracePosSelector(HISPECCalibrationSelector):
    """
    Selector for trace position calibration files.

    **Calibration Type:** tracepos

    **Selection Logic:**

    - Match instrument era
    - Match spectrograph
    - Require master calibration
    - Closest in observation time
    """

    def get_candidates(self, meta : MetaModel, db : LocalCalibrationDB):
        rows = db.rows_where(
            """
            cal_type = :cal_type AND
            instrument_era = :era AND
            spectrograph = :spec
            """,
            {
                "cal_type": "tracepos",
                "era": meta["instrument_era"],
                "spec": meta["spectrograph"],
                "mjd_start": meta["mjd_start"]
            },
            order_by="ABS(mjd_start - :mjd_start)"
        )

        return list(rows)

class TraceProfSelector(HISPECCalibrationSelector):
    """
    Selector for trace profile calibration files.

    **Calibration Type:** traceprof

    **Selection Logic:**

    - Match instrument era
    - Match spectrograph
    - Require master calibration
    - Closest in observation time
    """

    def get_candidates(self, meta : MetaModel, db : LocalCalibrationDB):
        rows = db.rows_where(
            """
            cal_type = :cal_type AND
            instrument_era = :era AND
            spectrograph = :spec
            """,
            {
                "cal_type": "traceprof",
                "era": meta["instrument_era"],
                "spec": meta["spectrograph"],
                "mjd_start": meta["mjd_start"]
            },
            order_by="ABS(mjd_start - :mjd_start)"
        )

        return list(rows)

class BiasSelector(HISPECCalibrationSelector):
    """
    Selector for bias calibration files.

    **Calibration Type:** bias

    **Selection Logic:**

    - Match instrument era
    - Match spectrograph
    - Require master calibration
    - Closest in observation time
    """

    def get_candidates(self, meta : MetaModel, db : LocalCalibrationDB):
        rows = db.rows_where(
            """
            cal_type = :cal_type AND
            instrument_era = :era AND
            spectrograph = :spec
            """,
            {
                "cal_type": "bias",
                "era": meta["instrument_era"],
                "spec": meta["spectrograph"],
                "mjd_start": meta["mjd_start"]
            },
            order_by="ABS(mjd_start - :mjd_start)"
        )

        return list(rows)

class ReadNoiseSelector(HISPECCalibrationSelector):
    """
    Selector for read noise calibration files.

    **Calibration Type:** rn

    **Selection Logic:**

    - Match instrument era
    - Match spectrograph
    - Require master calibration
    - Closest in observation time
    """

    def get_candidates(self, meta : MetaModel, db : LocalCalibrationDB):
        rows = db.rows_where(
            """
            cal_type = :cal_type AND
            instrument_era = :era AND
            spectrograph = :spec
            """,
            {
                "cal_type": "rn",
                "era": meta["instrument_era"],
                "spec": meta["spectrograph"],
                "mjd_start": meta["mjd_start"]
            },
            order_by="ABS(mjd_start - :mjd_start)"
        )

        return list(rows)

class NonlinearCorrectionSelector(HISPECCalibrationSelector):
    """
    Selector for nonlinear correction calibration files.

    **Calibration Type:** nonlin

    **Selection Logic:**

    - Match instrument era
    - Match spectrograph
    - Require master calibration
    - Closest in observation time
    - Prefer sim-matched nonlin (``meta.note`` / ``sim_matched_nonlin``) when
      present among candidates, so UTR sims that set ``nonlin_coeffs`` do not
      silently pick a competing KOA master
    """

    def get_candidates(self, meta : MetaModel, db : LocalCalibrationDB):
        rows = db.rows_where(
            """
            cal_type = :cal_type AND
            instrument_era = :era AND
            spectrograph = :spec
            """,
            {
                "cal_type": "nonlin",
                "era": meta["instrument_era"],
                "spec": meta["spectrograph"],
                "mjd_start": meta["mjd_start"]
            },
            order_by="ABS(mjd_start - :mjd_start)"
        )

        return list(rows)

    def select_best(self, input, candidates : list[dict] | dict, **kwargs) -> dict | None:
        """Prefer sim-matched nonlin when it is among the candidates."""
        if isinstance(candidates, list) and candidates:
            tag = "sim_matched_nonlin"
            preferred = [
                c
                for c in candidates
                if tag in str(c.get("note") or "")
                or tag in str(c.get("filename") or "")
            ]
            if preferred:
                return super().select_best(input, preferred, **kwargs)
        return super().select_best(input, candidates, **kwargs) 
class FiberFlatSelector(HISPECCalibrationSelector):
    """
    Selector for fiber-flat calibration files.

    Prefers flats whose ``object_fiber{N}`` is ``WHITELIGHT`` for every science
    fiber on the input (``object_fiber{N} == SCIENCE``), so a fiber-2-only flat
    is not chosen for a fiber-1 / fiber-3 exposure on the same night.
    """

    def get_candidates(self, meta : MetaModel, db : LocalCalibrationDB):
        rows = db.rows_where(
            """
            cal_type = :cal_type AND
            instrument_era = :era AND
            spectrograph = :spec
            """,
            {
                "cal_type": "fiberflat",
                "era": meta["instrument_era"],
                "spec": meta["spectrograph"],
                "mjd_start": meta["mjd_start"]
            },
            order_by="ABS(mjd_start - :mjd_start)"
        )

        return list(rows)

    @staticmethod
    def _science_fibers_from_input(input) -> list[int]:
        """
        Resolve SCIENCE fibers from a DataModel or bare ``meta``.

        ``HISPECCalibrationStore.select_and_get_calibration`` passes
        ``input.meta`` into ``selector.select``, so ``input`` is often already a
        ``MetaModel`` (no nested ``.meta``).
        """
        src = getattr(input, "meta", None)
        if src is None:
            src = input
        fibers: list[int] = []
        for fib in (1, 2, 3):
            key = f"object_fiber{fib}"
            val = getattr(src, key, None)
            if val is None and hasattr(src, "get"):
                try:
                    val = src.get(key)
                except Exception:
                    val = None
            if str(val or "").upper() == "SCIENCE":
                fibers.append(fib)
        return fibers

    def select_best(self, input, candidates : list[dict] | dict, **kwargs) -> dict | None:
        rows = candidates if isinstance(candidates, list) else [candidates]
        science_fibers = self._science_fibers_from_input(input)
        if science_fibers:
            preferred = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                if all(
                    str(row.get(f"object_fiber{fib}") or "").upper() == "WHITELIGHT"
                    for fib in science_fibers
                ):
                    preferred.append(row)
            if preferred:
                return super().select_best(input, preferred, **kwargs)
        return super().select_best(input, candidates, **kwargs)

class DetectorFlatSelector(HISPECCalibrationSelector):
    """
    Selector for detector flat calibration files.

    **Calibration Type:** detflat

    **Selection Logic:**

    - Match instrument era
    - Match spectrograph
    - Require master calibration
    - Closest in observation time
    """

    def get_candidates(self, meta : MetaModel, db : LocalCalibrationDB):
        rows = db.rows_where(
            """
            cal_type = :cal_type AND
            instrument_era = :era AND
            spectrograph = :spec
            """,
            {
                "cal_type": "detflat",
                "era": meta["instrument_era"],
                "spec": meta["spectrograph"],
                "mjd_start": meta["mjd_start"]
            },
            order_by="ABS(mjd_start - :mjd_start)"
        )

        return list(rows)

class GainSelector(HISPECCalibrationSelector):
    """
    Selector for detector gain calibration files.

    **Calibration Type:** gain

    **Selection Logic:**

    - Match instrument era
    - Match spectrograph
    - Require master calibration
    - Closest in observation time
    """

    def get_candidates(self, meta : MetaModel, db : LocalCalibrationDB):
        rows = db.rows_where(
            """
            cal_type = :cal_type AND
            instrument_era = :era AND
            spectrograph = :spec
            """,
            {
                "cal_type": "gain",
                "era": meta["instrument_era"],
                "spec": meta["spectrograph"],
                "mjd_start": meta["mjd_start"],
            },
            order_by="ABS(mjd_start - :mjd_start)"
        )

        return list(rows)

class WavelengthSolutionSelector(HISPECCalibrationSelector):
    """
    Selector for wavelength solution calibration files.

    **Calibration Type:** Configurable (default: lfc_wavesol)

    **Selection Logic:**

    - Match instrument era
    - Match spectrograph
    - Closest in observation time
    - Prefer packaged sim-HDF line-ZP WAVE (``meta.note`` /
      ``sim_hdf_wavesol.line_zp``) when present among candidates, so simulation
      Stage 1 does not silently pick a competing geometric ``lfc_wavesol``

    Parameters
    ----------
    cal_type : str, optional
        The type of wavelength calibration to select. Default: lfc_wavesol
    fallback_cal_type : str | None, optional
        The type of wavelength calibration to select as a fallback. Default: None
    """

    def __init__(self, cal_type : str = 'lfc_wavesol', fallback_cal_types : str | None = None):
        super().__init__()
        self.cal_type = cal_type
        if fallback_cal_types is not None:
            self.fallback_cal_types = fallback_cal_types
        else:
            self.fallback_cal_types = ['arclamp_wavesol', 'gascell_wavesol', 'sim_wavesol']

    def _rows_for_cal_type(self, cal_type : str, meta : MetaModel, db : LocalCalibrationDB) -> list[dict]:
        rows = db.rows_where(
            """
            cal_type = :cal_type AND
            instrument_era = :era AND
            spectrograph = :spec
            """,
            {
                "cal_type": cal_type,
                "era": meta["instrument_era"],
                "spec": meta["spectrograph"],
                "mjd_start": meta["mjd_start"]
            },
            order_by="ABS(mjd_start - :mjd_start)"
        )
        return list(rows)

    def get_candidates(self, meta : MetaModel, db : LocalCalibrationDB):
        for cal_type in (self.cal_type, *self.fallback_cal_types):
            rows = self._rows_for_cal_type(cal_type, meta, db)
            if rows:
                return rows
        return []

    def select_best(self, input, candidates : list[dict] | dict, **kwargs) -> dict | None:
        """Prefer packaged sim-HDF line-ZP WAVE when it is among the candidates."""
        if isinstance(candidates, list) and candidates:
            # Keep this string match local (no simulations import) to avoid cycles.
            tag = "sim_hdf_wavesol.line_zp"
            preferred = [
                c
                for c in candidates
                if tag in str(c.get("note") or "")
                or tag in str(c.get("filename") or "")
            ]
            if preferred:
                return super().select_best(input, preferred, **kwargs)
        return super().select_best(input, candidates, **kwargs)

class ReferenceEtalonSelector(HISPECCalibrationSelector):
    """
    Selector for reference etalon calibration files.

    **Calibration Type:** ref_etalon

    **Selection Logic:**

    - Match instrument era
    - Match spectrograph
    - Closest in observation time
    """

    def get_candidates(self, meta : MetaModel, db : LocalCalibrationDB):
        rows = db.rows_where(
            """
            cal_type = :cal_type AND
            instrument_era = :era AND
            spectrograph = :spec
            """,
            {
                "cal_type": "ref_etalon",
                "era": meta["instrument_era"],
                "spec": meta["spectrograph"],
                "mjd_start": meta["mjd_start"]
            },
            order_by="ABS(mjd_start - :mjd_start)"
        )

        return list(rows)
    
class LSFSelector(HISPECCalibrationSelector):
    """
    Selector for LSF calibration files.

    **Calibration Type:** Configurable (default: lsf)

    **Selection Logic:**

    - Match instrument era
    - Match spectrograph
    - Closest in observation time

    Parameters
    ----------
    cal_type : str, optional
        The type of wavelength calibration to select. Default: lsf
    """

    def __init__(self, cal_type : str = 'lsf'):
        super().__init__()
        self.cal_type = cal_type

    def get_candidates(self, meta : MetaModel, db : LocalCalibrationDB):
        rows = db.rows_where(
            """
            cal_type = :cal_type AND
            instrument_era = :era AND
            spectrograph = :spec
            """,
            {
                "cal_type": self.cal_type,
                "era": meta["instrument_era"],
                "spec": meta["spectrograph"],
                "mjd_start": meta["mjd_start"]
            },
            order_by="ABS(mjd_start - :mjd_start)"
        )

        return list(rows)