"""Helpers for the gas cell wavelength solution."""
from __future__ import annotations

import importlib.resources

import h5py
import numpy as np

__all__ = [
    "GASCELL_FTS_FILENAME",
    "read_gascell_fts",
]

# JPL FTS transmission spectra of the HISPEC gas cells (session B0215, the "a" scan of each cell and the
# no-cell scan), bundled in hispecdrp/data/templates
GASCELL_FTS_FILENAME = "B0215.HISPEC.gascells.JPL-FTS.h5"


def read_gascell_fts(filepath : str | None = None) -> dict:
    """
    Read the JPL FTS gas-cell transmission spectra from an HDF5 file.

    All arrays are instantaneous flux densities: each value is the transmission at its wavelength sample,
    not an average or integral over a bin. Evaluate them at other wavelengths by interpolation (never
    flux-conserving rebinning), and multiply them pointwise (as for ``transmission_tot``).

    Parameters
    ----------
    filepath : str | None
        Path to the HDF5 file. Default is the bundled `GASCELL_FTS_FILENAME` in
        ``hispecdrp/data/templates``.

    Returns
    -------
    dict
        ``"wavelength"`` : np.ndarray of shape (Nwave,), vacuum wavelength in microns, increasing,
        shared by all arrays.
        One np.ndarray of shape (Nwave,) per gas cell, keyed by its gases: ``"N2O"``, ``"CO2"``,
        ``"CH4+H2O"``, ``"NH3"``. Each is the cell scan ratioed to the background and
        baseline-corrected.
        ``"noCell"`` : np.ndarray of shape (Nwave,), the no-cell scan, processed the same way.
        ``"transmission_tot"`` : np.ndarray of shape (Nwave,), the product of all gas cells (not noCell).
        ``"meta"`` : dict of the file's metadata from the readme, e.g. ``"resolution"`` (the FTS
        spectral resolution in cm-1, defined as 0.5/MOPD) and ``"readme"`` (the full readme text).
        ``meta["cells"]`` is the list of gas cell keys, and ``meta[key]`` is a dict of each array's
        readme table row (e.g. ``meta["N2O"]["pressures_Torr"]``).
    """
    if filepath is None:
        filepath = importlib.resources.files("hispecdrp.data.templates") / GASCELL_FTS_FILENAME
    with h5py.File(filepath, "r") as f:
        out = {"wavelength": f["wavelength"][:], "meta": dict(f.attrs)}
        out["meta"]["cells"] = list(f["cells"])
        for key in out["meta"]["cells"]:
            out[key] = f["cells"][key][:]
            out["meta"][key] = dict(f["cells"][key].attrs)
        out["noCell"] = f["noCell"][:]
        out["meta"]["noCell"] = dict(f["noCell"].attrs)

    out["transmission_tot"] = np.prod([out[key] for key in out["meta"]["cells"]], axis=0)
    return out
