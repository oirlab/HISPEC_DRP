Simulated LFC Spectrum (2D Gaussian fits)
=========================================

Same LFC paint as :doc:`simulated_lfc_spectrum`, but every spot is fit in the
rate map with an :class:`astropy.modeling.models.Gaussian2D` (centroid,
semimajor / semiminor, rotation). The wavelength solution is derived from those
2D spectral-row centroids rather than from a 1D extract.

.. raw:: html

   <p><span style="color:red">The HISPEC DRP is in active development. Please submit issues and suggestions on the <a href="https://github.com/oirlab/HISPEC_DRP">GitHub</a>.</span></p>

.. raw:: html

   <p class="download-notebook">
     <a class="download-notebook-btn"
        href="../../_static/notebooks/simulated_lfc_2d_fit_example.ipynb"
        data-notebook-download="simulated_lfc_2d_fit_example.ipynb"
        download="simulated_lfc_2d_fit_example.ipynb">Download as notebook</a>
   </p>

Imports and output directory
----------------------------

.. code-block:: python

   from pathlib import Path

   import numpy as np

   OUT = Path("/path/to/hispec_simulated_lfc_2d_fit_example")
   OUT.mkdir(parents=True, exist_ok=True)

   SPECTROGRAPH, FIBER = "BSPEC", 2

   print(f"OUT = {OUT}")
   print(f"spectrograph={SPECTROGRAPH}  fiber={FIBER}")

See the notebook for the full walkthrough.
