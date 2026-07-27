"""
Path helpers and Stage 1/2 simulation runners for end-to-end HISPEC tests.

Used by :mod:`hispecdrp.simulations.e2e_pipeline` and
:mod:`hispecdrp.simulations.simulate_up_the_ramp`.
"""

from __future__ import annotations

import copy
import logging
import os
import sys
from dataclasses import dataclass
from logging.handlers import QueueHandler
from pathlib import Path
from typing import Any

import numpy as np

from .. import datamodels
from ..pipelines.stage0 import stage0_pipeline as _stage0_pipeline
from ..pipelines.stage1 import stage1_pipeline as _stage1_pipeline
from ..pipelines.stage2 import stage2_pipeline as _stage2_pipeline
from ..primitives import (
    BarycenterCorrectionPrimitive,
    BlazeCorrectionPrimitive,
    StellarRVPrimitive,
)
from ..utils.line_mask_utils import packaged_default_line_mask_path

logger = logging.getLogger(__name__)

DEFAULT_TEMPLATE_FILENAME = "lte049.0-2.5-0.0a+0.0.BT-Settl.spec.7.h5"
DEFAULT_RATE_MAP_FIBERS: list[int] = [1, 2, 3]
# HISPEC detector (ny, nx); matches datamodel ramp/calibration image sizes.
DEFAULT_HISPEC_DETECTOR_SHAPE: tuple[int, int] = (4096, 4160)

# E2E Stage 2 runs barycenter correction between blaze correction and stellar RV.
E2E_STAGE2_PRIMITIVES: dict[str, type] = {
    "blaze_corr": BlazeCorrectionPrimitive,
    "barycorr": BarycenterCorrectionPrimitive,
    "stellar_rv": StellarRVPrimitive,
}


def repository_root() -> Path:
    """HISPEC_DRP repository root (parent of the ``hispecdrp`` package)."""
    return Path(__file__).resolve().parents[2]


def templates_data_dir() -> str:
    """Bundled BT-Settl templates under ``hispecdrp/data/templates``."""
    import hispecdrp

    return os.path.join(os.path.dirname(hispecdrp.__file__), "data", "templates")


def ensure_output_directory(output_dir: str) -> str:
    """Create ``output_dir`` if needed and return its absolute path."""
    out = os.path.abspath(output_dir)
    os.makedirs(out, exist_ok=True)
    return out


def calibration_dir(output_dir: str) -> str:
    """Absolute path to ``<output_dir>/cals``."""
    return ensure_output_directory(os.path.join(os.path.abspath(output_dir), "cals"))


def ensure_koa_calibration_cache_env(output_dir: str) -> str:
    """
    If ``KOA_CALIBRATION_CACHE`` is unset, set it to
    ``<output_dir>/cals/KOA_CALIBRATION_CACHE`` and create that directory.
    """
    existing = (os.environ.get("KOA_CALIBRATION_CACHE") or "").strip()
    if existing:
        return existing
    cal_root = calibration_dir(output_dir)
    koa_path = os.path.join(cal_root, "KOA_CALIBRATION_CACHE")
    os.makedirs(koa_path, exist_ok=True)
    os.environ["KOA_CALIBRATION_CACHE"] = koa_path
    logger.info(
        "KOA_CALIBRATION_CACHE was unset; using default under output directory: %s",
        koa_path,
    )
    return koa_path


def dummy_cal_path(output_dir: str, key: str) -> str:
    """File path for a dummy calibration model under ``output_dir/cals``."""
    return os.path.join(calibration_dir(output_dir), f"sim_{key}.fits")


def identity_nonlin_calibration_model(ny: int, nx: int) -> datamodels.NonlinearCorrectionModel:
    """Per-pixel identity nonlinearity map for Stage 0 when no master nonlin is available."""
    coeffs = np.zeros((2, ny, nx), dtype=np.float32)
    coeffs[1, :, :] = 1.0
    nonlin_thresh = np.zeros((ny, nx), dtype=np.uint16)
    nonlin_max = np.full((ny, nx), np.iinfo(np.uint16).max, dtype=np.uint16)
    nonlin_dq = np.zeros((ny, nx), dtype=np.uint32)
    return datamodels.NonlinearCorrectionModel(
        coeffs=coeffs,
        nonlin_thresh=nonlin_thresh,
        nonlin_max=nonlin_max,
        dq=nonlin_dq,
    )


_CALIBRATION_SHAPE_ARRAY_NAMES: tuple[str, ...] = (
    "dq",
    "gain",
    "flat",
    "bias",
    "rate",
    "rn",
    "sat_thresh",
    "coeffs",
    "DQ",
    "GAIN",
    "FLAT",
    "BIAS",
    "RATE",
    "RN",
    "SAT_THRESH",
    "COEFFS",
)


def calibration_model_detector_shape(model: datamodels.CalibrationModel) -> tuple[int, int]:
    """Return ``(ny, nx)`` for a Stage 0 calibration model."""
    for name in _CALIBRATION_SHAPE_ARRAY_NAMES:
        if hasattr(model, name):
            arr = getattr(model, name)
        elif isinstance(model, dict) and name in model:
            arr = model[name]
        else:
            continue
        data = np.asarray(arr)
        if data.ndim == 2:
            return int(data.shape[0]), int(data.shape[1])
        if data.ndim == 3:
            return int(data.shape[-2]), int(data.shape[-1])
    raise ValueError(f"Could not determine detector shape for {type(model)!r}")


def stage0_calibration_shape_matches(
    model: datamodels.CalibrationModel,
    expected_shape: tuple[int, int],
) -> bool:
    """Return True when ``model`` image arrays match ``expected_shape`` ``(ny, nx)``."""
    try:
        return calibration_model_detector_shape(model) == expected_shape
    except ValueError:
        return False


def open_cached_stage0_calibration(
    filepath: str,
    expected_shape: tuple[int, int],
    *,
    key: str | None = None,
) -> datamodels.CalibrationModel | None:
    """
    Open a cached simulated calibration if its detector shape matches ``expected_shape``.

    Returns ``None`` when the file is missing or stale (wrong shape).
    """
    if not os.path.isfile(filepath):
        return None
    model = datamodels.open(filepath)
    if stage0_calibration_shape_matches(model, expected_shape):
        return model
    label = key or os.path.basename(filepath)
    logger.warning(
        "Ignoring cached Stage 0 calibration %s (shape=%s != expected %s).",
        label,
        calibration_model_detector_shape(model),
        expected_shape,
    )
    return None


def apply_e2e_save_and_preview_flags(
    prim_args: dict,
    primitives: dict[str, type],
) -> dict:
    """
    Enable ``save_result`` and ``save_preview`` on every primitive that supports them.

    Used by E2E simulation runners so each pipeline step writes FITS products and
    generates available previews under ``output_dir``.
    """
    out = copy.deepcopy(prim_args)
    for name, cls in primitives.items():
        flags: dict[str, bool] = {}
        if "save_result" in cls.ARGUMENTS:
            flags["save_result"] = True
        if "save_preview" in cls.ARGUMENTS:
            flags["save_preview"] = True
        if name == "stellar_rv":
            existing = out.get(name, {})
            if "save_overlay_debug" in cls.ARGUMENTS and "save_overlay_debug" not in existing:
                flags["save_overlay_debug"] = True
            # HTML preview only: Kaleido PNG of multi-fiber line-mask CCF figures (hundreds of
            # traces) can hang or run for many minutes in headless Chrome. Always override the
            # primitive default (True) even when get_default_stage*_prim_args already copied it.
            if "preview_write_png" in cls.ARGUMENTS:
                flags["preview_write_png"] = False
        if flags:
            out.setdefault(name, {}).update(flags)
    return out


def apply_max_cores_to_stage0_prim_args(prim_args: dict, max_cores: int) -> dict:
    out = copy.deepcopy(prim_args)
    for name, cls in _stage0_pipeline.Stage0Pipeline.PRIMITIVES.items():
        if "max_cores" not in cls.ARGUMENTS:
            continue
        out.setdefault(name, {})["max_cores"] = max_cores
    return out

def apply_max_cores_to_stage1_prim_args(prim_args: dict, max_cores: int) -> dict:
    out = copy.deepcopy(prim_args)
    for name, cls in _stage1_pipeline.Stage1Pipeline.PRIMITIVES.items():
        if "max_cores" not in cls.ARGUMENTS:
            continue
        out.setdefault(name, {})["max_cores"] = max_cores
    return out

def apply_max_cores_to_stage2_prim_args(prim_args: dict, max_cores: int) -> dict:
    out = copy.deepcopy(prim_args)
    for name, cls in E2E_STAGE2_PRIMITIVES.items():
        if "max_cores" not in getattr(cls, "ARGUMENTS", {}):
            continue
        out.setdefault(name, {})["max_cores"] = max_cores
    return out

def get_default_stage1_prim_args(*, max_cores: int | None = None) -> dict:
    """Default ``prim_args`` for :class:`~hispecdrp.pipelines.stage1.stage1_pipeline.Stage1Pipeline`.

    Optional ``max_cores`` is written into each primitive's args when provided.
    """
    _skip = frozenset({"input", "output_path", "output_dir"})
    out: dict[str, dict] = {}
    for name, cls in _stage1_pipeline.Stage1Pipeline.PRIMITIVES.items():
        args = {k: v for k, v in cls.ARGUMENTS.items() if k not in _skip}
        if max_cores is not None and "max_cores" in args:
            args["max_cores"] = max_cores
        out[name] = args
    return out


def get_default_stage2_prim_args(*, max_cores: int | None = None) -> dict:
    """Default ``prim_args`` for :class:`~hispecdrp.pipelines.stage2.stage2_pipeline.Stage2Pipeline`.

    Optional ``max_cores`` is written into each primitive's args when provided.
    """
    _skip = frozenset({"input", "output_path", "output_dir"})
    out: dict[str, dict] = {}
    for name, cls in _stage2_pipeline.Stage2Pipeline.PRIMITIVES.items():
        args = {k: v for k, v in cls.ARGUMENTS.items() if k not in _skip}
        if max_cores is not None and "max_cores" in args:
            args["max_cores"] = max_cores
        out[name] = args
    if "barycorr" not in out:
        out["barycorr"] = {}
    if max_cores is not None and "max_cores" in BarycenterCorrectionPrimitive.ARGUMENTS:
        out["barycorr"]["max_cores"] = max_cores
    if "stellar_rv" in out:
        out["stellar_rv"]["mask_path"] = packaged_default_line_mask_path()
    return out


def run_stage1_simulation(
    rate_model: datamodels.DataModel,
    trace_pos_model: datamodels.DataModel,
    trace_prof_model: datamodels.DataModel,
    prim_args: dict | None = None,
    *,
    wavesol_model: datamodels.DataModel | None = None,
    output_dir: str | None = None,
    max_cores: int | None = None,
    extraction_aperture: int = 5,
    do_orders: list | None = None,
    do_fibers: list[str] | None = None,
) -> datamodels.DataModel:
    """Run Stage 1 on a rate/slope model with trace calibrations and optional wavelength solution."""
    if do_orders is None and hasattr(rate_model, "make_order_vec"):
        do_orders = rate_model.make_order_vec()
    if do_orders is None:
        do_orders = []
    if do_fibers is None:
        do_fibers = ["1", "2", "3"]

    if prim_args is not None:
        method = prim_args.get("spec_extract", {}).get("method", "box")
    else:
        method = "box"

    logger.info(
        "Running Stage 1 simulation (do_orders=%s, do_fibers=%s, method=%s).",
        do_orders,
        do_fibers,
        method,
    )

    if prim_args is None:
        prim_args = get_default_stage1_prim_args(max_cores=max_cores)
        prim_args["spec_extract"]["method"] = method
        prim_args["spec_extract"]["extraction_aperture"] = extraction_aperture
        prim_args["spec_extract"]["do_orders"] = do_orders
        prim_args["spec_extract"]["do_fibers"] = do_fibers
    else:
        prim_args = copy.deepcopy(prim_args)
        if max_cores is not None:
            prim_args = apply_max_cores_to_stage1_prim_args(prim_args, max_cores)

    prim_args.setdefault("spec_extract", {})
    prim_args["spec_extract"]["tracepos"] = trace_pos_model
    prim_args["spec_extract"]["traceprof"] = trace_prof_model

    prim_args.setdefault("set_wavesol", {})
    if wavesol_model is None:
        prim_args["set_wavesol"]["run"] = False
    else:
        prim_args["set_wavesol"].pop("run", None)
        prim_args["set_wavesol"]["wavesol"] = wavesol_model

    prim_args = apply_e2e_save_and_preview_flags(
        prim_args, _stage1_pipeline.Stage1Pipeline.PRIMITIVES
    )

    pipe = _stage1_pipeline.Stage1Pipeline(prim_args=prim_args)
    apply_kw: dict = {}
    if output_dir is not None:
        apply_kw["output_dir"] = ensure_output_directory(output_dir)
    return pipe.apply(rate_model, **apply_kw)


def _init_e2e_primitive(prim_cls: type, prim_arg_dict: dict, output_dir: str | None):
    """Construct a primitive with E2E ``prim_args`` and optional output directory."""
    prim = prim_cls()
    valid = prim.ARGUMENTS | prim.CALIBRATIONS
    prim.init_args(**{k: v for k, v in prim_arg_dict.items() if k in valid})
    if output_dir is not None and (prim.save_result or prim.save_preview):
        prim.set_args(output_dir=output_dir)
    return prim


def run_stage2_simulation(
    stage1_output: datamodels.DataModel,
    *,
    fiberflat_model: datamodels.DataModel,
    prim_args: dict | None = None,
    output_dir: str | None = None,
    max_cores: int | None = None,
    stellar_rv_mask_path: str | None = None,
    stellar_rv_ccf_input_velocity_m_s: float | None = None,
) -> datamodels.DataModel:
    """Run Stage 2 (blaze + barycenter correction + stellar RV) on the Stage 1 product."""
    if prim_args is None:
        prim_args = get_default_stage2_prim_args(max_cores=max_cores)
    else:
        prim_args = copy.deepcopy(prim_args)
        if max_cores is not None:
            prim_args = apply_max_cores_to_stage2_prim_args(prim_args, max_cores)

    if stellar_rv_mask_path is not None and "stellar_rv" in prim_args:
        prim_args["stellar_rv"]["mask_path"] = stellar_rv_mask_path
    if (
        stellar_rv_ccf_input_velocity_m_s is not None
        and np.isfinite(float(stellar_rv_ccf_input_velocity_m_s))
        and "stellar_rv" in prim_args
    ):
        prim_args["stellar_rv"]["ccf_input_velocity"] = float(stellar_rv_ccf_input_velocity_m_s)

    prim_args = apply_e2e_save_and_preview_flags(prim_args, E2E_STAGE2_PRIMITIVES)

    out_dir = ensure_output_directory(output_dir) if output_dir is not None else None
    model = stage1_output
    for prim_name, prim_cls in E2E_STAGE2_PRIMITIVES.items():
        prim = _init_e2e_primitive(prim_cls, prim_args.get(prim_name, {}), out_dir)
        if prim_name == "blaze_corr":
            prim.fiberflat = fiberflat_model
        model = prim.apply(model)
    return model


def ensure_e2e_console_logging() -> None:
    """
    Ensure E2E summary messages reach the terminal after pipeline logging stops.

    Pipeline primitives use :func:`~hispecdrp.utils.logging.logging_context`, which
    routes the root logger through a multiprocessing queue. When the last pipeline
    exits, the queue listener is stopped, so later ``logger.info`` calls on
    ``hispecdrp.simulations.*`` would otherwise be discarded.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        if isinstance(handler, QueueHandler):
            root.removeHandler(handler)
    has_stdout = any(
        isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is sys.stdout
        for h in root.handlers
    )
    if not has_stdout:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s][%(name)s][%(levelname)s]: %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )
        root.addHandler(handler)
    if root.level > logging.INFO:
        root.setLevel(logging.INFO)


def _calibration_model_filepath(model: datamodels.DataModel | None) -> str:
    """Best-effort absolute path (or label) for a calibration data model."""
    if model is None:
        return "—"
    fp = getattr(model, "_filepath", None)
    if fp:
        return os.path.abspath(str(fp))
    meta = getattr(model, "meta", None)
    fn = getattr(meta, "filename", None) if meta is not None else None
    if fn:
        return str(fn)
    return "(in-memory)"


def _format_calibration_source_label(source: str) -> str:
    """Human-readable provenance label from internal ``sources`` values."""
    if source == "koa":
        return "KOA"
    if source == "simulated":
        return "Simulated (local)"
    if source == "simulated_hdf":
        return "Simulated (HDF template)"
    if source == "fiberflat_pipeline":
        return "Generated (fiberflat pipeline)"
    if source == "simulated_dummy_rate":
        return "Simulated (dummy fiberflat rate)"
    if source.startswith("simulated_cached:"):
        return "Simulated (cached local)"
    if source.startswith("local_cache:"):
        return "Local cache"
    if source.startswith("testing_data:"):
        return "Test data (Google Drive)"
    if source.startswith("config:"):
        return "User config"
    if source.startswith("simulated_dummy_rate:"):
        return "Simulated (dummy fiberflat rate)"
    return source


def _calibration_summary_filepath(source: str, model: datamodels.DataModel | None) -> str:
    if source.startswith("config:"):
        return os.path.abspath(source.split(":", 1)[1])
    if source.startswith("simulated_cached:"):
        return os.path.abspath(source.split(":", 1)[1])
    if source.startswith("local_cache:"):
        return os.path.abspath(source.split(":", 1)[1])
    if source.startswith("testing_data:"):
        return os.path.abspath(source.split(":", 1)[1])
    if source.startswith("simulated_dummy_rate:"):
        return os.path.abspath(source.split(":", 1)[1])
    return _calibration_model_filepath(model)


def calibration_source_category(source: str | None) -> str:
    """Map internal ``sources`` tag to a coarse provenance category."""
    if not source or source == "—":
        return "Unresolved"
    if source == "koa":
        return "KOA"
    if source.startswith("config:"):
        return "User config"
    if source in ("simulated", "simulated_hdf"):
        return "Simulated"
    if source.startswith("simulated_cached:"):
        return "Simulated (cached)"
    if source.startswith("local_cache:"):
        return "Local cache"
    if source.startswith("testing_data:"):
        return "Test data"
    if source == "fiberflat_pipeline":
        return "Generated (fiberflat pipeline)"
    if source.startswith("simulated_dummy_rate"):
        return "Simulated (fiberflat rate)"
    return "Other"


@dataclass
class CalibrationProvenanceRow:
    """One calibration product in the E2E dataset summary."""

    product: str
    category: str
    source_tag: str
    filepath: str
    file_exists: bool
    cal_id: str | None = None
    origin: str | None = None
    notes: str | None = None


@dataclass
class CalibrationDatasetSummary:
    """Full calibration dataset report for an E2E or calibrations-only run."""

    output_dir: str
    rows: list[CalibrationProvenanceRow]
    expected_products: list[str]
    missing_products: list[str]
    validation_warnings: list[str]
    koa_cache_dir: str | None = None


def _model_provenance_fields(model: datamodels.DataModel | None) -> tuple[str | None, str | None]:
    if model is None:
        return None, None
    meta = getattr(model, "meta", None)
    if meta is None:
        return None, None
    cal_id = getattr(meta, "cal_id", None)
    origin = getattr(meta, "origin", None)
    return (
        str(cal_id) if cal_id is not None else None,
        str(origin) if origin is not None else None,
    )


def build_calibration_dataset_summary(
    *,
    output_dir: str,
    calibration_models: dict[str, datamodels.CalibrationModel],
    sources: dict[str, str],
    fiberflat_bundle: dict[str, datamodels.DataModel] | None = None,
    wavesol_model: datamodels.DataModel | None = None,
    expected_products: list[str] | None = None,
) -> CalibrationDatasetSummary:
    """Build a validated provenance summary for all calibration products."""
    from .e2e_calibrations import STAGE0_CAL_KEYS, STAGE1_CAL_KEYS

    stage1_models: dict[str, datamodels.DataModel | None] = {
        "trace_pos": None,
        "trace_prof": None,
        "fiberflat": None,
        "wavesol": wavesol_model,
    }
    if fiberflat_bundle:
        for key in ("trace_pos", "trace_prof", "fiberflat"):
            stage1_models[key] = fiberflat_bundle.get(key)

    if expected_products is None:
        expected_products = list(STAGE0_CAL_KEYS)
        rate_keys = sorted(k for k in sources if k.startswith("fiberflat_rate_fib"))
        expected_products.extend(rate_keys)
        for key in STAGE1_CAL_KEYS:
            if key not in expected_products:
                expected_products.append(key)

    ordered_keys: list[str] = list(STAGE0_CAL_KEYS)
    ordered_keys.extend(sorted(k for k in sources if k.startswith("fiberflat_rate_fib")))
    for key in STAGE1_CAL_KEYS:
        if key not in ordered_keys:
            ordered_keys.append(key)
    ordered_keys.extend(sorted(k for k in sources if k not in ordered_keys))

    rows: list[CalibrationProvenanceRow] = []
    resolved_keys: set[str] = set()
    validation_warnings: list[str] = []

    for key in ordered_keys:
        source = sources.get(key)
        model = calibration_models.get(key) or stage1_models.get(key)
        if source is None and model is None:
            continue
        if source is None:
            source = "simulated" if key in calibration_models else "—"
        resolved_keys.add(key)
        category = calibration_source_category(source)
        fpath = _calibration_summary_filepath(source, model)
        file_exists = (
            os.path.isfile(fpath)
            if fpath not in ("—", "(in-memory)") and not fpath.startswith("(")
            else model is not None
        )
        cal_id, origin = _model_provenance_fields(model)
        notes: list[str] = []
        if fpath in ("(in-memory)", "—") or fpath.startswith("("):
            notes.append("in-memory model")
        elif not file_exists:
            notes.append("file not found on disk")
            validation_warnings.append(
                f"{key}: source={source!r} but file missing: {fpath}"
            )
        if source == "koa" and origin is None:
            notes.append("KOA product (check meta.origin in FITS)")
        rows.append(
            CalibrationProvenanceRow(
                product=key,
                category=category,
                source_tag=source,
                filepath=fpath,
                file_exists=file_exists,
                cal_id=cal_id,
                origin=origin,
                notes="; ".join(notes) if notes else None,
            )
        )

    missing_products = [k for k in expected_products if k not in resolved_keys]
    for key in missing_products:
        validation_warnings.append(f"Expected calibration {key!r} was not resolved.")

    koa_cache = (os.environ.get("KOA_CALIBRATION_CACHE") or "").strip() or None

    return CalibrationDatasetSummary(
        output_dir=os.path.abspath(output_dir),
        rows=rows,
        expected_products=list(expected_products),
        missing_products=missing_products,
        validation_warnings=validation_warnings,
        koa_cache_dir=koa_cache,
    )


def log_calibration_dataset_summary(
    summary: CalibrationDatasetSummary,
    *,
    log: logging.Logger | None = None,
) -> None:
    """Log the full calibration dataset table, missing products, and validation warnings."""
    ensure_e2e_console_logging()
    log = log or logger

    if not summary.rows:
        log.info("Calibration dataset summary: no products recorded.")
    else:
        headers = ("Product", "Category", "Origin", "File", "Exists")
        entries = [
            (
                row.product,
                row.category,
                row.origin or row.source_tag,
                row.filepath,
                "yes" if row.file_exists else "no",
            )
            for row in summary.rows
        ]
        w = [max(len(h), *(len(e[i]) for e in entries)) for i, h in enumerate(headers)]
        sep = "-" * (sum(w) + 2 * (len(w) - 1))
        log.info(
            "Calibration dataset summary (output_dir=%s)\n%s\n%s\n%s",
            summary.output_dir,
            sep,
            "  ".join(f"{h:<{w[i]}}" for i, h in enumerate(headers)),
            "\n".join(
                "  ".join(f"{e[i]:<{w[i]}}" for i in range(len(headers))) for e in entries
            ),
        )

    if summary.koa_cache_dir:
        log.info("KOA calibration cache: %s", summary.koa_cache_dir)

    if summary.missing_products:
        log.warning(
            "Missing expected calibration product(s): %s",
            ", ".join(summary.missing_products),
        )

    if summary.validation_warnings:
        log.warning(
            "Calibration validation (%d issue(s)):\n  %s",
            len(summary.validation_warnings),
            "\n  ".join(summary.validation_warnings),
        )
    else:
        log.info(
            "Calibration validation: all %d resolved product(s) passed file checks.",
            len(summary.rows),
        )


def write_calibration_manifest(
    output_dir: str,
    summary: CalibrationDatasetSummary,
    *,
    filename: str = "calibration_manifest.json",
) -> str:
    """Write ``summary`` to ``<output_dir>/cals/calibration_manifest.json``."""
    import json

    cal_dir = calibration_dir(output_dir)
    path = os.path.join(cal_dir, filename)
    payload = {
        "output_dir": summary.output_dir,
        "koa_cache_dir": summary.koa_cache_dir,
        "expected_products": summary.expected_products,
        "missing_products": summary.missing_products,
        "validation_warnings": summary.validation_warnings,
        "products": [
            {
                "product": row.product,
                "category": row.category,
                "source_tag": row.source_tag,
                "filepath": row.filepath,
                "file_exists": row.file_exists,
                "cal_id": row.cal_id,
                "origin": row.origin,
                "notes": row.notes,
            }
            for row in summary.rows
        ],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    logger.info("Wrote calibration manifest to %s", path)
    return path


def log_e2e_calibration_summary(
    *,
    calibration_models: dict[str, datamodels.DataModel],
    sources: dict[str, str],
    fiberflat_bundle: dict[str, datamodels.DataModel] | None = None,
    wavesol_model: datamodels.DataModel | None = None,
    log: logging.Logger | None = None,
) -> None:
    """
    Log a table of calibration products used in the E2E run and their provenance.

    Parameters
    ----------
    calibration_models : dict
        Stage 0 calibration models (``dq``, ``bias``, etc.).
    sources : dict
        Map of product key → source tag (``koa``, ``simulated``, ``config:path``, …).
    fiberflat_bundle : dict or None
        Optional ``trace_pos``, ``trace_prof``, ``fiberflat`` models used for Stage 1/2.
    wavesol_model : DataModel or None
        Wavelength solution passed to Stage 1, if applicable.
    """
    from .e2e_calibrations import STAGE0_CAL_KEYS, STAGE1_CAL_KEYS

    ensure_e2e_console_logging()
    log = log or logger

    stage1_models: dict[str, datamodels.DataModel | None] = {
        "trace_pos": None,
        "trace_prof": None,
        "fiberflat": None,
        "wavesol": wavesol_model,
    }
    if fiberflat_bundle:
        for key in ("trace_pos", "trace_prof", "fiberflat"):
            stage1_models[key] = fiberflat_bundle.get(key)

    ordered_keys: list[str] = list(STAGE0_CAL_KEYS)
    rate_keys = sorted(
        k for k in sources if k.startswith("fiberflat_rate_fib")
    )
    ordered_keys.extend(rate_keys)
    for key in STAGE1_CAL_KEYS:
        if key not in ordered_keys:
            ordered_keys.append(key)

    extra_keys = sorted(k for k in sources if k not in ordered_keys)
    ordered_keys.extend(extra_keys)

    col_product = "Product"
    col_source = "Source"
    col_file = "File"
    entries: list[tuple[str, str, str]] = []

    for key in ordered_keys:
        source = sources.get(key)
        if source is None and key not in calibration_models and stage1_models.get(key) is None:
            continue
        if source is None:
            source = "simulated" if key in calibration_models else "—"
        label = _format_calibration_source_label(source)
        model = calibration_models.get(key) or stage1_models.get(key)
        fpath = _calibration_summary_filepath(source, model)
        entries.append((key, label, fpath))

    if not entries:
        log.info("E2E calibration summary: no calibration products recorded.")
        return

    w_product = max(len(col_product), *(len(e[0]) for e in entries))
    w_source = max(len(col_source), *(len(e[1]) for e in entries))
    w_file = max(len(col_file), *(len(e[2]) for e in entries))
    formatted_rows = [
        f"{key:<{w_product}}  {label:<{w_source}}  {fpath}"
        for key, label, fpath in entries
    ]
    sep = "-" * (w_product + w_source + w_file + 4)
    log.info(
        "E2E calibration summary\n%s\n%-*s  %-*s  %s\n%s",
        sep,
        w_product,
        col_product,
        w_source,
        col_source,
        col_file,
        "\n".join(formatted_rows),
    )


def read_rvccfavg(
    l2_model: datamodels.DataModel, fiber: int
) -> tuple[float, float] | None:
    """Return ``(RV, RV_ERR)`` in m/s from ``RVCCFAVG_FIB#``, or None if unavailable."""
    tbl = getattr(l2_model, f"rvccfavg_fib{fiber}", None)
    if tbl is None or len(tbl) == 0:
        return None
    rv = float(tbl["RV"][0])
    rv_err = float(tbl["RV_ERR"][0])
    if not (np.isfinite(rv) and np.isfinite(rv_err)):
        return None
    return rv, rv_err


@dataclass
class E2ERVPerformanceRow:
    """Per-fiber RV recovery metrics for E2E summary tables."""

    fiber: int
    injected_rv_km_s: float
    measured_rv_km_s: float | None
    measured_rv_m_s: float | None
    rv_err_m_s: float | None
    bias_km_s: float | None
    bias_m_s: float | None
    berv_km_s: float | None
    inlier_scatter_m_s: float | None
    n_inlier_chunks: int
    n_total_chunks: int


def read_rvccf_inlier_scatter_m_s(
    l2_model: datamodels.DataModel,
    fiber: int,
    *,
    rv_mean_m_s: float | None = None,
) -> tuple[float | None, int, int]:
    """
    RMS scatter of inlier chunk RVs about ``RVCCFAVG`` (m/s), matching the stellar-RV preview.

    Returns ``(scatter_m_s, n_inlier, n_total)``.
    """
    tbl = getattr(l2_model, f"rvccf_fib{fiber}", None)
    if tbl is None or len(tbl) == 0:
        return None, 0, 0
    rv_all = np.asarray(tbl["RV"], dtype=np.float64)
    n_total = int(rv_all.size)
    if "INLIER" in tbl.colnames:
        inlier = np.asarray(tbl["INLIER"], dtype=np.uint8).astype(bool)
    else:
        inlier = np.isfinite(rv_all)
    valid = inlier & np.isfinite(rv_all)
    n_inlier = int(np.sum(valid))
    if n_inlier == 0:
        return None, 0, n_total
    if rv_mean_m_s is None:
        avg = read_rvccfavg(l2_model, fiber)
        rv_mean_m_s = avg[0] if avg is not None else float(np.nanmean(rv_all[valid]))
    if not np.isfinite(rv_mean_m_s):
        return None, n_inlier, n_total
    biases = rv_all[valid] - float(rv_mean_m_s)
    scatter = float(np.sqrt(np.sum(biases**2) / n_inlier))
    return scatter, n_inlier, n_total


def collect_e2e_rv_performance_rows(
    stage2_model: datamodels.DataModel | None,
    *,
    injected_rv_km_s: float,
    injected_berv_km_s: float | None = None,
    fibers: list[int] | None = None,
) -> list[E2ERVPerformanceRow]:
    """Build per-fiber RV performance rows for logging and HTML summaries."""
    if stage2_model is None:
        return []
    fiber_list = sorted({int(f) for f in (fibers if fibers is not None else DEFAULT_RATE_MAP_FIBERS)})
    injected_m_s = float(injected_rv_km_s) * 1000.0
    rows: list[E2ERVPerformanceRow] = []
    for fiber in fiber_list:
        berv_km_s = read_berv_km_s(stage2_model, fiber)
        if berv_km_s is None and injected_berv_km_s is not None and np.isfinite(injected_berv_km_s):
            berv_km_s = float(injected_berv_km_s)
        pair = read_rvccfavg(stage2_model, fiber)
        if pair is None:
            scatter, n_in, n_tot = read_rvccf_inlier_scatter_m_s(stage2_model, fiber)
            rows.append(
                E2ERVPerformanceRow(
                    fiber=fiber,
                    injected_rv_km_s=float(injected_rv_km_s),
                    measured_rv_km_s=None,
                    measured_rv_m_s=None,
                    rv_err_m_s=None,
                    bias_km_s=None,
                    bias_m_s=None,
                    berv_km_s=berv_km_s,
                    inlier_scatter_m_s=scatter,
                    n_inlier_chunks=n_in,
                    n_total_chunks=n_tot,
                )
            )
            continue
        meas_m_s, err_m_s = pair
        bias_m_s = meas_m_s - injected_m_s
        scatter, n_in, n_tot = read_rvccf_inlier_scatter_m_s(
            stage2_model, fiber, rv_mean_m_s=meas_m_s
        )
        rows.append(
            E2ERVPerformanceRow(
                fiber=fiber,
                injected_rv_km_s=float(injected_rv_km_s),
                measured_rv_km_s=meas_m_s / 1000.0,
                measured_rv_m_s=meas_m_s,
                rv_err_m_s=err_m_s,
                bias_km_s=bias_m_s / 1000.0,
                bias_m_s=bias_m_s,
                berv_km_s=berv_km_s,
                inlier_scatter_m_s=scatter,
                n_inlier_chunks=n_in,
                n_total_chunks=n_tot,
            )
        )
    return rows


def read_berv_km_s(l2_model: datamodels.DataModel, fiber: int) -> float | None:
    """Return measured BERV (km/s) from ``BARYCORR_FIB#``, or None if unavailable."""
    tbl = getattr(l2_model, f"barycorr_fib{fiber}", None)
    if tbl is None or len(tbl) == 0:
        return None
    berv_m_s = float(tbl["BERV"][0])
    if not np.isfinite(berv_m_s):
        return None
    return berv_m_s / 1000.0


def log_e2e_rv_performance_summary(
    stage2_model: datamodels.DataModel | None,
    *,
    injected_rv_km_s: float,
    gaia_target: Any | None = None,
    fibers: list[int] | None = None,
    log: logging.Logger | None = None,
) -> None:
    """
    Log measured ``RVCCFAVG`` versus injected Doppler RV for E2E performance assessment.

    Parameters
    ----------
    stage2_model : L2 model or None
        Stage 2 pipeline output. When None, logs that RV measurement was not run.
    injected_rv_km_s : float
        Radial velocity applied to the input template before rate-map generation (km/s).
    fibers : list of int or None
        Fiber numbers to report. Default is ``[1, 2, 3]``.
    """
    ensure_e2e_console_logging()
    log = log or logger
    fiber_list = fibers if fibers is not None else list(DEFAULT_RATE_MAP_FIBERS)
    injected_m_s = float(injected_rv_km_s) * 1000.0

    if stage2_model is None:
        log.info(
            "E2E RV summary: Stage 2 was not run; no measured RV (injected=%.3f km/s = %.1f m/s).",
            injected_rv_km_s,
            injected_m_s,
        )
        return

    if gaia_target is not None:
        log.info(
            "E2E Gaia target %s (source_id=%s): Gaia RV=%.3f km/s, reference BERV=%.3f km/s, "
            "injected stellar RV=%.3f km/s (BERV via CCF ccf_zb only)",
            gaia_target.target_meta.get("target_name"),
            gaia_target.source_id,
            gaia_target.gaia_rv_km_s,
            gaia_target.berv_km_s,
            gaia_target.injected_rv_km_s,
        )
        berv_rows: list[str] = []
        for fiber in sorted({int(f) for f in fiber_list}):
            meas_berv = read_berv_km_s(stage2_model, fiber)
            if meas_berv is None:
                berv_rows.append(f"  fib{fiber}: (no BARYCORR)")
            else:
                delta = meas_berv - gaia_target.berv_km_s
                berv_rows.append(
                    f"  fib{fiber}: measured BERV={meas_berv:.3f} km/s "
                    f"(Δ vs reference {delta:+.3f} km/s)"
                )
        log.info("E2E barycenter correction summary\n%s", "\n".join(berv_rows))

    header = (
        "E2E RV performance summary "
        f"(injected stellar RV = {injected_rv_km_s:.3f} km/s = {injected_m_s:.1f} m/s; "
        "barycentric, excludes BERV)"
    )
    perf_rows = collect_e2e_rv_performance_rows(
        stage2_model,
        injected_rv_km_s=injected_rv_km_s,
        injected_berv_km_s=(
            float(gaia_target.berv_km_s)
            if gaia_target is not None and np.isfinite(gaia_target.berv_km_s)
            else None
        ),
        fibers=fiber_list,
    )
    col_fiber = "Fiber"
    col_meas = "Measured RV (m/s)"
    col_err = "RV_ERR (m/s)"
    col_bias = "Bias (m/s)"
    col_bias_km = "Bias (km/s)"
    w_fiber = max(len(col_fiber), 5)
    w_meas = len(col_meas)
    w_err = len(col_err)
    w_bias = len(col_bias)
    w_bias_km = len(col_bias_km)

    rows: list[str] = []
    for row in perf_rows:
        if row.measured_rv_m_s is None:
            rows.append(
                f"{row.fiber:>{w_fiber}}  "
                f"{'—':>{w_meas}}  "
                f"{'—':>{w_err}}  "
                f"{'—':>{w_bias}}  "
                f"{'(no RVCCFAVG)':>{w_bias_km}}"
            )
            continue
        rows.append(
            f"{row.fiber:>{w_fiber}}  "
            f"{row.measured_rv_m_s:>{w_meas}.3f}  "
            f"{row.rv_err_m_s:>{w_err}.3f}  "
            f"{row.bias_m_s:>{w_bias}.3f}  "
            f"{row.bias_km_s:>{w_bias_km}.6f}"
        )

    sep = "-" * (w_fiber + w_meas + w_err + w_bias + w_bias_km + 8)
    body = "\n".join(rows) if rows else "(no fibers requested)"
    if rows and all("(no RVCCFAVG)" in row for row in rows):
        body += (
            "\n\nNote: CCF line-mask returned 0 chunks for every fiber (see "
            "n_chunks=0 in StellarRV logs). RVCCFAVG is empty or non-finite; "
            "check spectrum quality, wavelength solution, and mask overlap."
        )
    log.info(
        "%s\n%s\n%s %s %s %s %s\n%s",
        header,
        sep,
        col_fiber.rjust(w_fiber),
        col_meas.rjust(w_meas),
        col_err.rjust(w_err),
        col_bias.rjust(w_bias),
        col_bias_km.rjust(w_bias_km),
        body,
    )
