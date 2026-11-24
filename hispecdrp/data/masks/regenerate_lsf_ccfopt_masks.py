#!/usr/bin/env python3
"""
Regenerate LSF-matched CCF-opt delta masks (``*.lsf.ccfopt.delta.mas``).

Paints an injected-RV=0 L2 with :func:`build_l2_lsf_matched`, then runs the
same line-by-line ccfopt builder as :func:`build_ccfopt_mask`.

This path needs the ZEMAX / pyechelle HDF model (same as Stage-1 sim). It is
slower than ``regenerate_ccfopt_masks.py`` (template-only). Prefer a narrow
order range for smoke checks::

    python hispecdrp/data/masks/regenerate_lsf_ccfopt_masks.py \\
        --template lte040.0-4.5-0.0a+0.0.BT-Settl.spec.7.h5 \\
        --fiber 2 --order-start 110 --order-end 111

Full-band packaged products (all BSPEC orders) are expensive; run on a
workstation and commit the resulting ``*.lsf.ccfopt.delta.mas``.
"""

from __future__ import annotations

import argparse
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
    from hispecdrp.simulations.simulate_spectra import build_ccfopt_mask_lsf_matched

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--template",
        default="lte040.0-4.5-0.0a+0.0.BT-Settl.spec.7.h5",
        help="Template basename under hispecdrp/data/templates/",
    )
    p.add_argument("--spectrograph", default="BSPEC")
    p.add_argument("--fiber", type=int, default=2)
    p.add_argument("--order-start", type=int, default=None)
    p.add_argument("--order-end", type=int, default=None)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_HERE.parent,
        help="Directory for seed + LSF ccfopt masks (default: packaged masks/)",
    )
    args = p.parse_args()

    seed, ccfopt, _l2 = build_ccfopt_mask_lsf_matched(
        args.template,
        args.output_dir,
        spectrograph=args.spectrograph,
        fiber=args.fiber,
        order_start=args.order_start,
        order_end=args.order_end,
        rebuild_seed=True,
    )
    print(f"seed:   {seed}")
    print(f"lsf:    {ccfopt}")


if __name__ == "__main__":
    main()
