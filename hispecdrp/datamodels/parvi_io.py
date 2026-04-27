from __future__ import annotations

from pathlib import Path
import glob
import os
import logging
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from astropy.io import fits

from .parvi_config import PARVIConfig

__all__ = ["ParviUTRPayload", "load_parvi_utr_cube"]

logger = logging.getLogger(__name__)


@dataclass
class ParviUTRPayload:
    """Container with all inputs required to build a RampModel."""

    data: np.ndarray
    dq_raw: np.ndarray
    dq: np.ndarray
    meta: dict[str, Any]
    config: type = PARVIConfig
    filename: str | None = None
    filepaths: list[str] | None = None

    def as_ramp_kwargs(self) -> dict[str, Any]:
        return {
            "data": self.data,
            "dq_raw": self.dq_raw,
            "dq": self.dq,
            "meta": self.meta,
            "config": self.config,
        }


def _extract_indexed_time_vector(header: fits.Header, prefix: str) -> list[float]:
    pattern = re.compile(rf"^{prefix}(\d+)$")
    values: list[tuple[int, float]] = []
    for key in header:
        m = pattern.match(key)
        if m is None:
            continue
        try:
            values.append((int(m.group(1)), float(header[key])))
        except (TypeError, ValueError):
            continue
    values.sort(key=lambda x: x[0])
    return [v for _, v in values]


def _resolve_parvi_filepaths(
    source: str | Path | list[str] | list[Path] | tuple[str, ...] | tuple[Path, ...],
    n_reads: int | None,
    start_index: int,
    read_id_width: int,
    exposure_index: int | None,
    exposure_id_width: int,
) -> list[str]:
    if isinstance(source, (list, tuple)):
        filepaths = [str(Path(p)) for p in source]
        if len(filepaths) == 0:
            raise ValueError("No PARVI input filepaths were provided.")
        return filepaths

    source_str = str(source)

    if any(ch in source_str for ch in "*?["):
        filepaths = sorted(glob.glob(source_str))
        if len(filepaths) == 0:
            raise ValueError(f"No files matched glob pattern: {source_str}")
        if n_reads is not None:
            filepaths = filepaths[:n_reads]
        return filepaths

    if "{read}" in source_str or "{exp}" in source_str:
        if n_reads is None:
            raise ValueError("n_reads must be provided when source contains read placeholders.")
        filepaths = []
        exp_token = None
        if "{exp}" in source_str:
            if exposure_index is None:
                raise ValueError("exposure_index must be provided when source contains '{exp}' placeholder.")
            exp_token = f"{exposure_index:0{exposure_id_width}d}"
        for idx in range(start_index, start_index + n_reads):
            read_token = f"{idx:0{read_id_width}d}"
            filepaths.append(source_str.format(read=read_token, exp=exp_token))
        return filepaths

    if "{}" in source_str:
        if n_reads is None:
            raise ValueError("n_reads must be provided when source contains '{}' placeholders.")
        n_placeholders = source_str.count("{}")
        if n_placeholders > 2:
            raise ValueError("Only up to 2 '{}' placeholders are supported (exposure, read).")
        filepaths = []
        exp_token = None
        if n_placeholders == 2:
            if exposure_index is None:
                raise ValueError(
                    "exposure_index must be provided when source contains two '{}' placeholders (exposure, read)."
                )
            exp_token = f"{exposure_index:0{exposure_id_width}d}"
        for idx in range(start_index, start_index + n_reads):
            token = f"{idx:0{read_id_width}d}"
            if n_placeholders == 1:
                filepaths.append(source_str.format(token))
            else:
                filepaths.append(source_str.format(exp_token, token))
        return filepaths

    # Single file fallback.
    return [source_str]


def load_parvi_utr_cube(
    source: str | Path | list[str] | list[Path] | tuple[str, ...] | tuple[Path, ...],
    n_reads: int | None = None,
    *,
    start_index: int = 1,
    read_id_width: int = 2,
    exposure_index: int | None = None,
    exposure_id_width: int = 2,
    clock_rate: float | None = None,
    instrument_era: str | None = None,
    strict_timei00: bool = True,
    sort_by_timei00: bool = True,
    repair_non_monotonic_timei00: bool = True,
    nominal_readtime_sec: float = 1.467,
    use_timing_vector_header: bool = True,
    timing_header_filepath: str | Path | None = None,
    timing_reference: str = "start",
) -> ParviUTRPayload:
    """
    Load PARVI single-read FITS files into a UTR payload.

    Parameters
    ----------
    source
        One of:
        - list/tuple of per-read filepaths,
        - glob pattern,
        - format pattern with '{}' for read index token (requires n_reads).
    n_reads
        Number of reads to load for pattern/glob sources. Ignored for explicit filepath list.
    start_index
        Starting read index when using '{}' patterns.
    read_id_width
        Zero-padding width for read index token in '{}' patterns.
    exposure_index
        Exposure index token for patterns containing exposure placeholders.
    exposure_id_width
        Zero-padding width for exposure token.
    clock_rate
        Optional clock rate override in pixels/s.
    instrument_era
        Optional instrument era metadata override.
    strict_timei00
        If True, missing TIMEI00 header raises ValueError. If False, fills NaN.
    sort_by_timei00
        If True, reorder frames by TIMEI00 timestamps to enforce monotonic read timing.
    repair_non_monotonic_timei00
        If True, replace non-monotonic/repeated TIMEI00 sequence with a synthetic
        monotonic series anchored at the first timestamp using nominal_readtime_sec cadence.
    nominal_readtime_sec
        Nominal seconds per read used when repairing non-monotonic TIMEI00.
    use_timing_vector_header
        If True, read TIMEI{} / TIMEF{} vectors from a timing header file (default: last read file)
        and use those values as the authoritative read times.
    timing_header_filepath
        Optional path to the file containing TIMEI{} / TIMEF{} vectors. If None, uses last filepath in sequence.
    timing_reference
        Which timing vector to use for read timestamps: 'start' -> TIMEI{}, 'end' -> TIMEF{},
        'mid' -> 0.5 * (TIMEI{} + TIMEF{}).

    Returns
    -------
    ParviUTRPayload
        Payload containing data arrays, metadata, config, and source filename needed
        to instantiate a PARVI RampModel.
    """
    filepaths = _resolve_parvi_filepaths(
        source=source,
        n_reads=n_reads,
        start_index=start_index,
        read_id_width=read_id_width,
        exposure_index=exposure_index,
        exposure_id_width=exposure_id_width,
    )

    cube_frames: list[np.ndarray] = []
    unix_ns: list[float] = []
    expected_shape: tuple[int, int] | None = None

    for fp in filepaths:
        if not os.path.isfile(fp):
            raise FileNotFoundError(f"PARVI read file not found: {fp}")

        with fits.open(fp, memmap=False) as hdul:
            frame = hdul[0].data
            header = hdul[0].header

        if frame is None:
            raise ValueError(f"Primary HDU has no image data: {fp}")

        frame = np.asarray(frame)
        if frame.ndim != 2:
            raise ValueError(f"Expected 2D PARVI frame in primary HDU, got shape={frame.shape} for file {fp}")

        if expected_shape is None:
            expected_shape = frame.shape
        elif frame.shape != expected_shape:
            raise ValueError(
                f"Inconsistent frame shape: expected {expected_shape}, got {frame.shape} in file {fp}"
            )

        cube_frames.append(frame.astype(np.uint16, copy=False))

        if "TIMEI00" in header:
            unix_ns.append(float(header["TIMEI00"]))
        elif strict_timei00:
            raise ValueError(f"Missing TIMEI00 header keyword in file: {fp}")
        else:
            logger.warning("TIMEI00 missing in %s. Filling NaN timestamp.", fp)
            unix_ns.append(np.nan)

    n_loaded = len(cube_frames)

    timing_timef_vec: list[float] | None = None
    vector_applied = False
    if use_timing_vector_header:
        timing_fp = str(timing_header_filepath) if timing_header_filepath is not None else filepaths[-1]
        if not os.path.isfile(timing_fp):
            raise FileNotFoundError(f"Timing header file not found: {timing_fp}")
        with fits.open(timing_fp, memmap=False) as hdul:
            timing_header = hdul[0].header

        timei_vec = _extract_indexed_time_vector(timing_header, "TIMEI")
        timef_vec = _extract_indexed_time_vector(timing_header, "TIMEF")
        timing_timef_vec = timef_vec

        if timing_reference not in ("start", "end", "mid"):
            raise ValueError("timing_reference must be one of: 'start', 'end', 'mid'.")

        if timing_reference == "start":
            if len(timei_vec) >= n_loaded:
                unix_ns = list(np.asarray(timei_vec[:n_loaded], dtype=np.float64))
                vector_applied = True
        elif timing_reference == "end":
            if len(timef_vec) >= n_loaded:
                unix_ns = list(np.asarray(timef_vec[:n_loaded], dtype=np.float64))
                vector_applied = True
        else:  # mid
            if len(timei_vec) >= n_loaded and len(timef_vec) >= n_loaded:
                unix_ns = list((np.asarray(timei_vec[:n_loaded], dtype=np.float64)
                                + np.asarray(timef_vec[:n_loaded], dtype=np.float64)) / 2.0)
                vector_applied = True

        if not vector_applied:
            logger.warning(
                "Could not build timing vector from TIMEI/TIMEF header keywords for all loaded reads. "
                "Falling back to per-file TIMEI00 values."
            )

    if sort_by_timei00 and len(unix_ns) > 1 and np.all(np.isfinite(unix_ns)):
        order = np.argsort(np.asarray(unix_ns, dtype=np.float64), kind="stable")
        cube_frames = [cube_frames[i] for i in order]
        unix_ns = [unix_ns[i] for i in order]

    if len(unix_ns) > 1 and np.all(np.isfinite(unix_ns)):
        delta_ns = np.diff(np.asarray(unix_ns, dtype=np.float64))
        if np.any(delta_ns <= 0):
            if repair_non_monotonic_timei00:
                anchor_ns = float(np.asarray(unix_ns, dtype=np.float64)[0])
                cadence_ns = float(nominal_readtime_sec) * 1e9
                unix_ns = (anchor_ns + np.arange(len(unix_ns), dtype=np.float64) * cadence_ns).tolist()
                logger.warning(
                    "Non-monotonic TIMEI00 sequence detected; repaired using nominal cadence %.6fs.",
                    nominal_readtime_sec,
                )
            else:
                raise ValueError(
                    "Non-monotonic TIMEI00 sequence detected. "
                    "Set repair_non_monotonic_timei00=True to synthesize a monotonic sequence."
                )

    data_cube = np.stack(cube_frames, axis=0)
    ny, nx = data_cube.shape[1:]

    meta = {
        "instrument_name": "PARVI",
        "parvi_timei00_unix_ns": unix_ns,
        "parvi_readtime_sec": float(nominal_readtime_sec),
    }

    if timing_timef_vec is not None and len(timing_timef_vec) >= len(unix_ns):
        meta["parvi_timef_unix_ns"] = list(np.asarray(timing_timef_vec[:len(unix_ns)], dtype=np.float64))
    if clock_rate is not None:
        meta["clock_rate"] = float(clock_rate)
    if instrument_era is not None:
        meta["instrument_era"] = instrument_era

    return ParviUTRPayload(
        data=data_cube,
        dq_raw=np.zeros_like(data_cube, dtype=np.uint8),
        dq=np.zeros((ny, nx), dtype=np.uint32),
        meta=meta,
        config=PARVIConfig,
        filename=os.path.basename(filepaths[0]),
        filepaths=filepaths,
    )
