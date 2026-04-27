"""Physical constants (SI) shared across HISPEC DRP."""

from astropy.constants import c

# Speed of light in m/s (CODATA via Astropy); Doppler factors and log-λ RV shifts.
C_M_S: float = float(c.si.value)
