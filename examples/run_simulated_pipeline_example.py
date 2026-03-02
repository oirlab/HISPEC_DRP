#!/usr/bin/env python
"""
Example executable to run the simulated pipeline.

Run from the project root:

    python examples/run_simulated_pipeline_example.py

With no arguments: fiber 1 only, master calibrations, output to ~/Downloads,
spectrum normalized to J=11. To override, pass any run_simulated_pipeline options, e.g.:

    python examples/run_simulated_pipeline_example.py --datadir /path/to/output --fibers 1,2,3

Equivalent module invocation:

    python -m hispecdrp.scripts.run_simulated_pipeline -d /Users/ckilpatrick/Downloads -f 1 --J 11 --use-master-calibrations
"""

from __future__ import annotations

import sys
import os

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hispecdrp.scripts.run_simulated_pipeline import main

if __name__ == "__main__":
    # Use example defaults only when no CLI args are given
    if len(sys.argv) == 1:
        sys.argv.extend([
            "--datadir", "/Users/ckilpatrick/Downloads",
            "--fibers", "1",
            "--J", "11",
            "--use-master-calibrations",
        ])
    main()
