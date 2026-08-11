# Atmospheric transmission (Gemini `mktrans`)

## `mktrans_zm_10_10.dat`

This file is the **Mauna Kea zenith transmission** model **`mktrans_zm_10_10.dat`**
from Gemini Observatory’s observing constraints / atmospheric models page.

- **Source URL:**  
  https://www.gemini.edu/sciops/ObsProcess/obsConstraints/atm-models/mktrans_zm_10_10.dat  
- **Format:** ASCII. The file begins with **`#` comment lines** documenting the source URL,
  column meanings (**wavelength in microns**, **dimensionless transmission**), and the fact
  that numeric rows use a **fixed decimal format** (eight digits after the decimal for each
  column). Data lines are whitespace-separated floats.
- **Use in HISPEC DRP:** :mod:`hispecdrp.utils.line_mask_utils` averages transmission over a
  user-defined **± velocity** interval (default ±500 km/s) around each candidate mask line
  to reject lines that fall in strong telluric absorption.

When publishing, cite the **Gemini Observatory** / your observing site’s transmission
model policy as appropriate; this file is **not** produced by HISPEC DRP.

# Atmospheric sky background (Gemini `skybg`)

This file is the **Mauna Kea sky background** data file **`mktrans_zm_10_10.dat`**
from Gemini Observatory’s observing constraints / atmospheric models page.

- **Source URL:**  
https://www.gemini.edu/sciops/ObsProcess/obsConstraints/atm-models/mktrans_zm_10_10.dat
- **Format:** ASCII. No header or comment lines.
  Columns are (**wavelength in nanometers**, **sky emission photons/s/m^2/arcsecond^2/nm**). 
  Data lines are whitespace-separated floats.
- **Use in HISPEC DRP:** The model is currently only used for generating simulated data.

Any use of the data in these tables should reference Lord (1992) and acknowledge Gemini Observatory.
