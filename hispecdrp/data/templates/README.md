# BT-Settl (CIFIST) spectral templates

This directory holds **BT-Settl (CIFIST)** theoretical spectra retrieved from the
Spanish Virtual Observatory (SVO) theoretical spectra service. The CIFIST grid uses
the BT-Settl model atmospheres with a cloud model and Caffau et al. (2011) solar
abundances; **wavelengths in those products are given in air**.

**Service and grid:**  
[BT-Settl (CIFIST) — SVO Theoretical Spectra](http://svo2.cab.inta-csic.es/theory/newov2/index.php?models=bt-settl-cifist)

Use that page to choose effective temperature, surface gravity, metallicity, and
other grid parameters; download the HDF5 spectrum for the stellar parameters you need.
The SVO interface documents the full parameter coverage and references (Allard et al.,
Caffau et al., etc.).

---

## File naming and astrophysical parameters

Files follow the usual **BT-Settl / PHOENIX-style** basename pattern, e.g.:

`lte025.0-5.5-0.0a+0.0.BT-Settl.spec.7.h5`  
`lte049.0-2.5-0.0a+0.0.BT-Settl.spec.7.h5`

Typical fields (separated by `-`), in order:

| Segment | Meaning (typical) |
|--------|-------------------|
| `lte025.0`, `lte049.0`, … | **Effective temperature** encoding in the basename (usually **T_eff / 100** in the numeric part). **Prefer** the `temperature` attribute inside the HDF5 file (see below). |
| `5.5`, `2.5`, … | **log g** (cgs). **Prefer** `surface_gravity` in the file attributes. |
| `0.0a+0.0`, … | **Metallicity** and **alpha** in the basename; **metallicity** is stored as a file attribute. Alpha / `[α/Fe]` may appear only in the filename for some downloads. |
| `BT-Settl.spec.7` | Model family and spectrum version identifier from the archive. |
| `.h5` | HDF5 container. |

For **T_eff**, **log g**, and **[M/H]**, read the **HDF5 root attributes** (next section)—they are the explicit model parameters bundled with the spectrum. Use the basename for archive/version strings and for any quantity not duplicated as an attribute (e.g. some alpha-enhancement tags).

---

## HDF5 layout (HISPEC expectation)

Each template is an HDF5 file with at least:

| Dataset | Description |
|---------|-------------|
| `wavelength` | Wavelength in **Angstrom**, on an **air** scale for BT-Settl (CIFIST) SVO products (see SVO model description). |
| `flux` | Flux density in **10⁻²³ erg cm⁻² s⁻¹ Å⁻¹** (stored numeric values × 1e-23 erg cm⁻² s⁻¹ Å⁻¹ when loaded with HISPEC). |

### Model parameters (HDF5 attributes on the root group)

BT-Settl (CIFIST) spectra from SVO include the sampled atmosphere parameters as **attributes
on the file root** (not on the `wavelength` / `flux` datasets in the files checked here):

| Attribute | Meaning | Units / convention |
|-----------|---------|-------------------|
| `temperature` | Effective temperature **T_eff** | Kelvin |
| `surface_gravity` | **log g** | cgs (log₁₀ of surface gravity in cm s⁻²) |
| `metallicity` | Metallicity **[M/H]** | Linear offset (e.g. `0.0` = solar scaled as in the grid) |

Read them with **h5py** (or any HDF5 library), for example:

```python
import h5py

path = "lte025.0-5.5-0.0a+0.0.BT-Settl.spec.7.h5"
with h5py.File(path, "r") as f:
    teff_K = float(f.attrs["temperature"])
    logg_cgs = float(f.attrs["surface_gravity"])
    metallicity = float(f.attrs["metallicity"])
    # optional: list all root attrs
    # print(dict(f.attrs))
```

For the bundled HISPEC examples, these match the basename (e.g. `lte025.0-5.5-0.0a+0.0…`
→ 2500 K, log g 5.5, [M/H] = 0.0). Other SVO downloads may expose additional attributes;
inspect `f.attrs.keys()` if you need more.

---

## Air vs vacuum wavelengths

**CIFIST spectra from SVO are documented as air wavelengths.** For cross-correlation or
comparison to HISPEC L2 data (vacuum wavelength solution) or to masks documented in
vacuum nanometres, convert **air → vacuum** before or during use.

In HISPEC DRP, :func:`hispecdrp.primitives.utils.spectral_utils.load_btsettl_spectrum` does
this by default (`wavelengths_are_air=True`): Angstrom → nanometre (air), then
:func:`hispecdrp.primitives.utils.spectral_utils.air_wavelength_nm_to_vacuum_nm` (Edlé /
standard dispersion; same Cauchy-style form as used elsewhere in the package), then to
**vacuum microns** for template CCF. The inverse mapping (vacuum nm → air nm), e.g. to
compare HISPEC vacuum λ to archive air tables, is
:func:`hispecdrp.primitives.utils.spectral_utils.vacuum_wavelength_nm_to_air_nm`.

If you read the HDF5 yourself:

1. `λ_nm_air = wavelength_angstrom / 10`
2. Apply a vacuum correction in nm (e.g. the same function as above, or your preferred
   air-to-vacuum recipe consistent with your data).
3. `λ_vac_um = λ_vac_nm / 1000`

Do **not** assume vacuum if you bypass the loader—keep air vs vacuum consistent with
your science wavelengths.

---

## Citations

Cite **BT-Settl / CIFIST** and the SVO service appropriately for your work, using the
references listed on the
[SVO BT-Settl (CIFIST) page](http://svo2.cab.inta-csic.es/theory/newov2/index.php?models=bt-settl-cifist)
(e.g. Allard et al., Caffau et al. 2011, and related papers linked there).
