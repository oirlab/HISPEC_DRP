RV Recovery
===========

Inject a known radial velocity into a BT-Settl template and recover it on three
paths. A simulated LFC wavelength solution and identity fiber flat are built
first so every path uses the same calibrations.

1. **Native L2** — paint the shifted spectrum onto the echelle WAVE grid and
   recover RV directly.
2. **Rate map → Stages 1–2** — paint a rate map, extract, blaze, and recover RV.
3. **UTR → Stages 0–2** — inject into a noiseless up-the-ramp exposure and run
   Stages 0–2 before recovering RV.

.. raw:: html

   <p><span style="color:red">The HISPEC DRP is in active development. Please submit issues and suggestions on the <a href="https://github.com/oirlab/HISPEC_DRP">GitHub</a>.</span></p>

.. raw:: html

   <p class="download-notebook">
     <a class="download-notebook-btn"
        href="../../_static/notebooks/rv_recovery.ipynb"
        data-notebook-download="rv_recovery.ipynb"
        download="rv_recovery.ipynb">Download as notebook</a>
   </p>

See the notebook for the full walkthrough.
