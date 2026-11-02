from pathlib import Path

import hispecdrp

print(f"hispecdrp {hispecdrp.__version__} @ {hispecdrp.__file__}")

OUT = Path("/Users/cale/Desktop/HISPEC_END2END_SIMS/RV_SIMS")
OUT.mkdir(parents=True, exist_ok=True)
print("output:", OUT.resolve())

import numpy as np

from hispecdrp.utils.line_mask_utils import paired_line_mask_path_for_template

FIBER = 1
BULK_RV_KM_S = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
#BULK_RV_KM_S = np.arange(-50, 51, 1)
TEMPLATE = "lte040.0-4.5-0.0a+0.0.BT-Settl.spec.7.h5"
APPLY_LSF = True

# MASK_STEM = Path(TEMPLATE).stem + ".delta.mas"
# MASK_LOCAL = OUT / MASK_STEM
# if MASK_LOCAL.is_file():
#     MASK_PATH = str(MASK_LOCAL)
# else:
#     MASK_PATH = paired_line_mask_path_for_template(
#         TEMPLATE, mask_style="delta"
#     )

MASK_STEM = Path(TEMPLATE).stem + ".lsf.ccfopt.delta.mas"
MASK_LOCAL = OUT / MASK_STEM
if MASK_LOCAL.is_file():
    MASK_PATH = str(MASK_LOCAL)
else:
    MASK_PATH = paired_line_mask_path_for_template(
        TEMPLATE, mask_style="lsf.ccfopt.delta"
    )

print(f"mask: {MASK_PATH}")


import numpy as np

from hispecdrp.primitives import StellarRVPrimitive
from hispecdrp.simulations.spectra import build_l2, quiet_pipeline_logging

hdr = (
    f"{'bulk_km/s':>10}  {'injected_m/s':>14}  {'recovered_m/s':>16}  "
    f"{'rv_err_m/s':>12}  {'bias_m/s':>12}  {'bias_mm/s':>10}"
)
print(hdr)
print("-" * len(hdr))

inj_m_s = []
rec_m_s = []
err_m_s = []

# This suppresses logging from the pipeline so the output is cleaner
with quiet_pipeline_logging():
    for bulk in BULK_RV_KM_S:
        l2 = build_l2(
            bulk,
            template=TEMPLATE,
            fiber=FIBER,
            apply_lsf=APPLY_LSF,
        )
        inj = float(bulk) * 1000.0
        out = StellarRVPrimitive.call(
            l2,
            mask_path=str(MASK_PATH),
            ccf_input_velocity=inj,
            #outlier_sigma=5.0,
            #window_line_by_line=False,
        )
        avg = getattr(out, f"rvccfavg_fib{FIBER}")
        rec = float(avg["RV"][0])
        err = float(avg["RV_ERR"][0])
        bias = rec - inj
        inj_m_s.append(inj)
        rec_m_s.append(rec)
        err_m_s.append(err)
        print(
            f"{bulk:10.1f}  {inj:14.4f}  {rec:16.6f}  {err:12.6f}  "
            f"{bias:12.6e}  {bias * 1e3:10.4f}"
        )

inj_m_s = np.asarray(inj_m_s, dtype=float)
rec_m_s = np.asarray(rec_m_s, dtype=float)
err_m_s = np.asarray(err_m_s, dtype=float)
bias_m_s = rec_m_s - inj_m_s
max_abs = float(np.max(np.abs(bias_m_s)))
rms = float(np.sqrt(np.mean(bias_m_s**2)))
d_bias = bias_m_s - bias_m_s[0]
max_d = float(np.max(np.abs(d_bias[1:])))
print()
print(f"max|bias| = {max_abs:.6e} m/s ({max_abs * 1e3:.4f} mm/s)")
print(f"rms bias  = {rms:.6e} m/s ({rms * 1e3:.4f} mm/s)")
print(f"max|differential bias| (vs RV=0) = {max_d:.6e} m/s ({max_d * 1e3:.4f} mm/s)")


from hispecdrp.simulations.plotting import plot_ideal_spectrum_rv_recovery

summary = plot_ideal_spectrum_rv_recovery(
    inj_m_s,
    rec_m_s,
    err_m_s,
    title="BT-Settl lte040 (LSF-matched): injected vs recovered RV",
    out_path=OUT / "lte040_lsf_simulated_vs_recovered_rv.png",
)
if summary.get("out_path"):
    print("wrote", summary["out_path"])
