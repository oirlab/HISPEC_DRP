# Packaged CCF line masks (SPIROU `.mas` format)

## Origin and attribution

The **`.mas`** files in this directory use the **SPIROU / OPERA-style** three-column
format (wavelength start, wavelength end, weight in **air nanometers** for the wavelength
columns—see `load_mask_spirou_mas` in `hispecdrp.primitives.utils.ccf_utils`).

They are **redistributed for convenience** with HISPEC DRP so that
`algorithm="ccf_mask_kpf"` works out of the box. The **scientific content** of the line
masks comes from the SPIROU/CCF mask ecosystem, not from HISPEC.

### Upstream mask repository

- **spirou-ccf** (Eder Martioli): collection of CCF masks and related tooling.  
  **Directory:** <https://github.com/edermartioli/spirou-ccf/tree/main/ccf_masks>  
  **Library (context for SPIRou data handling):**  
  <https://github.com/edermartioli/spirou-ccf/blob/main/spiroulib.py>

When publishing results, cite the **SPIROU/APERO** pipeline and/or the **spirou-ccf**
repository as appropriate for the mask you use, in addition to HISPEC DRP if you use
this package.

### Files shipped here (examples)

| File | Role |
|------|------|
| `M2_filt_allRV.mas` | Default mask when `mask_path` is unset (`ccf_mask_kpf`). |
| `K2_filt_allRV.mas`, `K2_all.mas` | Alternate SPIROU-style masks for tests or custom runs. |

### Segment widths (empirical)

For each line, ``width_nm = λ_end − λ_start`` and ``mean_nm = (λ_start + λ_end)/2``.
Across the bundled ``*.mas`` files, **width in nm increases roughly linearly with
mean wavelength**, and **width / mean λ is nearly constant** (relative spread
≪ 1%): segments behave like a **fixed fractional width in wavelength**, i.e. close
to **constant Δln λ** per mask line. Regenerate the diagnostic figure with
``python hispecdrp/data/masks/plot_mask_widths.py`` (writes ``mask_width_vs_wavelength.png``).

Names and content are chosen to align with common **K2 / M2** mask naming in the
spirou-ccf `ccf_masks` tree; verify against upstream if you need exact equivalence.

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
it is **not** a copy of KPF proprietary source. KPF-style **two-column** text masks
(center wavelength + weight) are also supported via `load_mask_kpf_two_column` with an
explicit `mask_width_um`.

---

## Comment lines in `.mas` files

Loader functions skip lines starting with `#`, `;`, or `//`. Each bundled `.mas` file
may begin with comment lines documenting this README’s pointers.
