#!/usr/bin/env python3
"""
Regenerate CCF-contrast–optimized delta line masks (template-only path).

For each ``*.h5`` under ``hispecdrp/data/templates/`` that has a paired
``<stem>.delta.mas`` seed, writes:

- ``<stem>.ccfopt.delta.mas`` — two-column mask from
  :func:`hispecdrp.utils.line_mask_utils.build_ccfopt_line_mask_from_template_and_seed`.

This is a **fast** refine on the 1D BT-Settl template (injected RV = 0). The
canonical E2E regenerator (ZEMAX → Stage 0–2 → line-by-line stellar RV) is:

    hispecdrp/simulations/notebooks/ccfopt_mask_from_sim.ipynb

Run from the repository root (after ``pip install -e .`` or with ``PYTHONPATH`` set)::

    python hispecdrp/data/masks/regenerate_ccfopt_masks.py
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
    from hispecdrp.utils.line_mask_utils import (
        build_ccfopt_line_mask_from_template_and_seed,
    )

    masks_dir = _HERE.parent
    templates_dir = _REPO_ROOT / "hispecdrp" / "data" / "templates"
    h5_list = sorted(templates_dir.glob("*.h5"))
    if not h5_list:
        raise SystemExit(f"No *.h5 templates under {templates_dir}")

    masks_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0
    for tpl in h5_list:
        stem = tpl.stem
        seed = masks_dir / f"{stem}.delta.mas"
        if not seed.is_file():
            print(f"skip (no seed delta mask): {tpl.name}")
            continue
        out = masks_dir / f"{stem}.ccfopt.delta.mas"
        print(f"ccfopt: {tpl.name} seed={seed.name} -> {out.name}")
        path, entries = build_ccfopt_line_mask_from_template_and_seed(
            str(tpl),
            str(seed),
            output_path=str(out),
            masks_directory=str(masks_dir),
        )
        print(f"  wrote {len(entries)} lines -> {path}")
        n_written += 1

    if n_written == 0:
        raise SystemExit("No CCF-opt masks written (missing paired *.delta.mas seeds?)")
    print(f"done: {n_written} mask(s)")


if __name__ == "__main__":
    main()
