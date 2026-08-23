#!/usr/bin/env python
"""
Download HISPEC/Cals and HISPEC/L0 files from GDRIVE_FILES to a given directory.

Reads hispecdrp/utils/testing_data.py and uses gdown. Requires: pip install gdown

Usage:
    python -m hispecdrp.scripts.download_hispec_test_data [output_dir]
    Default output_dir is ~/Downloads.
"""

from __future__ import annotations

import logging
import os
import re
import sys

logger = logging.getLogger(__name__)

# Repo root (parent of hispecdrp package) when this module lives in hispecdrp/scripts/
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_ROOT = os.path.dirname(_SCRIPT_DIR)
_PROJECT_ROOT = os.path.dirname(_PACKAGE_ROOT)

try:
    import gdown
except ImportError:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
    logger.error("This script requires gdown. Install with: pip install gdown")
    sys.exit(1)

from hispecdrp.driver.logging import LogConfig, ROOT_LOGGER


def _load_gdrive_files(project_root: str) -> dict[str, str]:
    path = os.path.join(project_root, "hispecdrp", "utils", "testing_data.py")
    with open(path) as f:
        content = f.read()
    start = content.rfind("GDRIVE_FILES = {") + len("GDRIVE_FILES = {")
    depth = 1
    i = start
    while depth and i < len(content):
        if content[i] == "{":
            depth += 1
        elif content[i] == "}":
            depth -= 1
        i += 1
    dict_str = content[start : i - 1]
    pairs = re.findall(r"\s*'([^']+)'\s*:\s*'([^']+)'", dict_str)
    return dict(pairs)


def main():
    config = LogConfig(level=logging.INFO, enable_stdout=True, enable_file=False)
    config.apply([ROOT_LOGGER])

    output_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Downloads")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    gdrive_files = _load_gdrive_files(_PROJECT_ROOT)
    count = 0
    for filepath in sorted(gdrive_files):
        if filepath.startswith("HISPEC/Cals/") or filepath.startswith("HISPEC/L0/"):
            file_id = gdrive_files[filepath]
            fname = os.path.basename(filepath)
            out_path = os.path.join(output_dir, fname)
            url = f"https://drive.google.com/uc?id={file_id}"
            logger.info("Downloading %s ...", fname)
            gdown.download(url, out_path, quiet=False)
            count += 1

    logger.info("Done. Downloaded %d files to %s", count, output_dir)


if __name__ == "__main__":
    main()
