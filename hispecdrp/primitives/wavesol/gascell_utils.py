"""Helpers for the gas cell wavelength solution."""
from __future__ import annotations

import importlib.resources

import h5py

__all__ = [
    "GASCELL_FTS_FILENAME",
    "read_gascell_fts",
]

# JPL FTS scans of the HISPEC gas cells, bundled in hispecdrp/data/templates
GASCELL_FTS_FILENAME = "B0216.HISPEC.gascells.JPL-FTS.h5"


def read_gascell_fts(filepath : str | None = None) -> dict:
    """
    Read the JPL FTS gas-cell transmission spectra from an HDF5 file.

    Parameters
    ----------
    filepath : str | None
        Path to the HDF5 file. Default is the bundled `GASCELL_FTS_FILENAME` in
        ``hispecdrp/data/templates``.

    Returns
    -------
    dict
        ``"wavenumber"`` : np.ndarray of shape (Nwave,), in cm-1, shared by all scans.
        ``"attrs"`` : dict of file-level metadata (instrument setup, experimental conditions, the readme).
        ``"scans"`` : dict keyed by scan name (e.g. ``"B0216.1a"``), each a dict with
        ``"transmission"`` (np.ndarray of shape (Nwave,)) and ``"attrs"`` (dict of per-scan metadata,
        e.g. ``gases``, ``pressures_Torr``, ``temperature_K``).
    """
    if filepath is None:
        resource = importlib.resources.files("hispecdrp.data.templates") / GASCELL_FTS_FILENAME
        with importlib.resources.as_file(resource) as path:
            return _read_gascell_fts(path)
    return _read_gascell_fts(filepath)


def _read_gascell_fts(filepath) -> dict:
    with h5py.File(filepath, "r") as f:
        out = {
            "wavenumber": f["wavenumber"][:],
            "attrs": dict(f.attrs),
            "scans": {},
        }
        for name, g in f["scans"].items():
            out["scans"][name] = {
                "transmission": g["transmission"][:],
                "attrs": dict(g.attrs),
            }
    return out
