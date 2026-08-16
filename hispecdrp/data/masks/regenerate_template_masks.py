#!/usr/bin/env python3
"""
Regenerate CCF line masks from packaged BT-Settl templates.

For each ``*.h5`` under ``hispecdrp/data/templates/``, writes:

- ``<stem>.delta.mas`` — two-column **relaxed** seed mask from
  :func:`hispecdrp.utils.line_mask_utils.build_line_mask_from_template` using
  :data:`hispecdrp.utils.line_mask_utils.RELAXED_TEMPLATE_MASK_BUILD_KW`
  (depth ≥ 10%; no deblend / telluric / saturated-core rejection).
- ``<stem>.tophat.mas`` — three-column segment mask: each delta line center becomes a
  finite rest segment with half-width
  ``half_width_nm = slope * λ_center_nm + intercept`` using
  :data:`hispecdrp.utils.line_mask_utils.DEFAULT_WIDTH_CALIBRATION_K2_ALL`, same weights
  as the delta table (sum to 1).

Run from the repository root (after ``pip install -e .`` or with ``PYTHONPATH`` set)::

    python hispecdrp/data/masks/regenerate_template_masks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> None:
    import numpy as np

    from hispecdrp.utils.line_mask_utils import MaskLines
    from hispecdrp.utils.line_mask_utils import (
        DEFAULT_WIDTH_CALIBRATION_K2_ALL,
        RELAXED_TEMPLATE_MASK_BUILD_KW,
        build_line_mask_from_template,
        read_mask_generation_meta,
    )

    masks_dir = _HERE.parent
    templates_dir = _REPO_ROOT / "hispecdrp" / "data" / "templates"
    h5_list = sorted(templates_dir.glob("*.h5"))
    if not h5_list:
        raise SystemExit(f"No *.h5 templates under {templates_dir}")

    masks_dir.mkdir(parents=True, exist_ok=True)
    slope, intercept = DEFAULT_WIDTH_CALIBRATION_K2_ALL

    for tpl in h5_list:
        stem = tpl.stem
        out_delta = masks_dir / f"{stem}.delta.mas"
        print(f"delta (relaxed): {tpl.name} -> {out_delta.name}")
        _path, entries = build_line_mask_from_template(
            str(tpl),
            output_path=str(out_delta),
            masks_directory=str(masks_dir),
            **RELAXED_TEMPLATE_MASK_BUILD_KW,
        )
        centers_nm = np.array([e.wavelength_nm for e in entries], dtype=np.float64)
        weights = np.array([e.weight for e in entries], dtype=np.float64)
        half_nm = slope * centers_nm + intercept
        half_nm = np.maximum(half_nm, 1.0e-12)
        lo_nm = centers_nm - half_nm
        hi_nm = centers_nm + half_nm

        meta_delta = read_mask_generation_meta(str(out_delta)) or {}
        md: dict = dict(meta_delta)
        for k in (
            "disk_wavelength_columns",
            "hispecdrp_wavelength_columns",
            "mask_style",
            "mask_line_geometry",
        ):
            md.pop(k, None)
        md["generator"] = (
            "hispecdrp.data.masks.regenerate_template_masks:delta_table_to_tophat_segments"
        )
        md["tophat_segment_half_width_law_nm_vs_center_nm"] = {
            "slope": float(slope),
            "intercept_nm": float(intercept),
            "half_width_nm_floor": 1.0e-12,
            "calibration_note": (
                "Linear law from :data:`hispecdrp.utils.line_mask_utils.DEFAULT_WIDTH_CALIBRATION_K2_ALL` "
                "(historically fit to SPIROU K2_all-style segment lists)."
            ),
        }
        md["source_delta_mask_path"] = str(out_delta.resolve())

        ml = MaskLines.from_tophat_segment_edges_um(
            lo_nm / 1000.0, hi_nm / 1000.0, weights, metadata=md
        )
        out_tophat = masks_dir / f"{stem}.tophat.mas"
        print(f"tophat: {tpl.name} -> {out_tophat.name}  (n={len(entries)})")
        ml.save(output_path=str(out_tophat))


if __name__ == "__main__":
    main()
