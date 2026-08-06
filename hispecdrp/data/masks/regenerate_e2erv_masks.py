#!/usr/bin/env python3
"""
Regenerate RV-aligned e2e delta masks (``*.e2erv.delta.mas``) from packaged templates.

Uses parabolic template centroids + resolution-element deblend so mask λ matches
the same truth used by the sim-HDF line WAVE correction (injected RV = 0).

Run from the repository root::

    python hispecdrp/data/masks/regenerate_e2erv_masks.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)


def main() -> None:
    from hispecdrp.utils.line_mask_utils import build_e2erv_line_mask_from_template

    masks_dir = _HERE.parent
    templates_dir = _REPO_ROOT / "hispecdrp" / "data" / "templates"
    # Prefer the e2e default template; fall back to all packaged .h5
    targets = sorted(templates_dir.glob("lte049*.h5"))
    if not targets:
        targets = sorted(templates_dir.glob("*.h5"))
    if not targets:
        raise SystemExit(f"No templates under {templates_dir}")

    n = 0
    for tpl in targets:
        out = masks_dir / f"{tpl.stem}.e2erv.delta.mas"
        print(f"e2erv: {tpl.name} -> {out.name}")
        path, entries = build_e2erv_line_mask_from_template(
            str(tpl),
            output_path=str(out),
            masks_directory=str(masks_dir),
        )
        print(f"  wrote {len(entries)} lines -> {path}")
        n += 1
    print(f"done: {n} mask(s)")


if __name__ == "__main__":
    main()
