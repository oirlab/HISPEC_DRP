"""
CLI and runnable scripts for HISPEC DRP.

Run from the command line with python -m hispecdrp.scripts.<script_name>:

- download_hispec_test_data : Download HISPEC test data (Cals, L0) from Google Drive.
- upload_cals_to_koa       : Register calibration FITS from a directory and upload to KOA.
- test_koa_caldb           : Dummy BSPEC/RSPEC L0 + KOA caldb sync/association smoke test.

These modules are not imported here so that ``python -m hispecdrp.scripts.<name>``
loads the script only as __main__, avoiding the RuntimeWarning about the module
being in sys.modules prior to execution.
"""

__all__ = []
