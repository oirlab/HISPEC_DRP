"""Helpers for wavelength-solution primitives."""
from __future__ import annotations

import h5py

__all__ = [
    "GASCELL_FTS_PATH",
    "read_gascell_fts",
]

# TODO: make the DRP aware of this file instead of hard coding a local path.
GASCELL_FTS_PATH = (
    "/Users/cale/Desktop/HISPEC_GasCell_Data/B0216.GasCells2ndFour.Caltech.HISPEC/"
    "B0216.HISPEC.gascells.JPL-FTS.h5"
)


def read_gascell_fts(filepath : str = GASCELL_FTS_PATH) -> dict:
    """
    Read the JPL FTS gas-cell transmission spectra from an HDF5 file.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file. Default is `GASCELL_FTS_PATH`.

    Returns
    -------
    dict
        ``"wavenumber"`` : np.ndarray of shape (Nwave,), in cm-1, shared by all scans.
        ``"attrs"`` : dict of file-level metadata (instrument setup, experimental conditions, the readme).
        ``"scans"`` : dict keyed by scan name (e.g. ``"B0216.1a"``), each a dict with
        ``"transmission"`` (np.ndarray of shape (Nwave,)) and ``"attrs"`` (dict of per-scan metadata,
        e.g. ``gases``, ``pressures_Torr``, ``temperature_K``).
    """
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
