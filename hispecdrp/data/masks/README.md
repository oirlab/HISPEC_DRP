# CCF line masks (SPIROU-style `.mas` format)

## Contents of this directory

Line masks here are **generated from packaged BT-Settl templates** under
`hispecdrp/data/templates/*.h5`. Regenerate all of them with:

```bash
python hispecdrp/data/masks/regenerate_template_masks.py
```

(from the repository root, with the package on `PYTHONPATH` or installed editable).

For each template stem, that script writes:

| File pattern | Role |
|--------------|------|
| `<template_stem>.delta.mas` | Two columns: vacuum line-center (nm), weight (see :func:`hispecdrp.utils.line_mask_utils.build_line_mask_from_template`). |
| `<template_stem>.tophat.mas` | Three columns: vacuum segment start (nm), end (nm), weight — finite-width lines derived from the delta table using the linear half-width law :data:`hispecdrp.utils.line_mask_utils.DEFAULT_WIDTH_CALIBRATION_K2_ALL`. |

Each file includes a `# HISPEC_DRP_MASK_META` JSON line; parse it with
:func:`hispecdrp.utils.line_mask_utils.read_mask_generation_meta`.

**Default mask** when `StellarRVPrimitive(mask_path=None)` (via
:func:`hispecdrp.utils.line_mask_utils.resolve_mask_path`) is the lte049
template-derived delta mask — basename
:data:`hispecdrp.utils.line_mask_utils.DEFAULT_PACKAGED_LINE_MASK_BASENAME`, absolute path
:func:`hispecdrp.utils.line_mask_utils.packaged_default_line_mask_path`.

## On-disk layout and loaders

Numeric layouts follow SPIROU / OPERA conventions as implemented in
:meth:`hispecdrp.utils.line_mask_utils.MaskLines.load` (``.mas`` files): **two** columns
(center λ, weight) for delta-style masks; **three** columns (start λ, end λ, weight)
for tophat-style segment masks. Wavelengths in these generated files are **vacuum**
nanometers (see mask JSON and module docstrings).

The **mask-shifted CCF** algorithm is discussed relative to SPIROU / KPF-style workflows
in :mod:`hispecdrp.primitives.utils.ccf.ccf_utils` and the references there.

### Segment widths (tophat / empirical SPIROU masks)

For each line, ``width_nm = λ_end − λ_start`` and ``mean_nm = (λ_start + λ_end)/2``.
On typical SPIROU segment lists, **width in nm increases roughly linearly with mean
wavelength** (approximately constant Δln λ). The template-derived **tophat** files here
use the packaged linear calibration explicitly. Regenerate the diagnostic scatter figure
with ``python hispecdrp/data/masks/plot_mask_widths.py`` (writes
``mask_width_vs_wavelength.png`` from ``*.tophat.mas`` only).

---

## Relationship to KPF mask-line logic

The **mask-shifted CCF** *algorithm* implemented in HISPEC DRP (velocity loop,
Doppler-shifted segments, pixel overlap, Gaussian fit to the CCF) is **inspired by
public descriptions** of the Keck **KPF** pipeline’s mask-line RV approach, in particular
the structure of mask line tables and barycentric/velocity bookkeeping discussed in:

- **KPF-Pipeline** (Keck Data Reduction Pipelines), radial velocity module — e.g.  
  `alg_rv_mask_line.py` at commit `81609b0ac0ab6d12536fb771bfdd0112001340c3`:  
  <https://github.com/Keck-DataReductionPipelines/KPF-Pipeline/blob/81609b0ac0ab6d12536fb771bfdd0112001340c3/modules/radial_velocity/src/alg_rv_mask_line.py>  
- **Same module tree:**  
  <https://github.com/Keck-DataReductionPipelines/KPF-Pipeline/tree/81609b0ac0ab6d12536fb771bfdd0112001340c3/modules/radial_velocity/src>

HISPEC DRP’s Python/Numba code is an **independent implementation** for HISPEC L2 data;
it is **not** a copy of KPF proprietary source. KPF-style **two-column** text masks (center wavelength + weight) are also supported via
:meth:`hispecdrp.utils.line_mask_utils.MaskLines.load` (``.txt`` / ``.dat`` / ``.csv``; wavelength
unit inferred from the column median).

---

## Comment lines in `.mas` files

Loader functions skip lines starting with `#`, `;`, or `//`. Each `.mas` file may
begin with comment lines documenting provenance and JSON metadata.
