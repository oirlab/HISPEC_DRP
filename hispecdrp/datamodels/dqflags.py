# Pixel-specific flags adapted from stdatamodels.jwst.datamodels.dqflags
# See HISPEC documentation
import numpy as np
DQ_FLAGS = {
    'GOOD':             np.uint32(0),      # No bits set, all is good
    'DO_NOT_USE':       np.uint32(2**0),   # Bad pixel. Do not use.
    'SATURATED':        np.uint32(2**1),   # Pixel saturated during exposure
    'JUMP_DET':         np.uint32(2**2),   # Jump detected during exposure
    'OUTLIER_MID_TIME': np.uint32(2**3),   # Mid exposure time is outlier in spectral order
    'OUTLIER':          np.uint32(2**4),   # Flagged by outlier detection
    'PERSISTENCE':      np.uint32(2**5),   # High persistence
    'NON_LINEAR': np.uint32(2**6),   # Non linear. None linearity correction might be worse than 1%.
    'RESERVED_1':       np.uint32(2**7),   # RESERVED_1/TBD
    'RESERVED_2':       np.uint32(2**8),   # RESERVED_2/TBD
    # 'UNRELIABLE_ERROR': np.uint32(2**8),   # Uncertainty exceeds quoted error
    'NON_SCIENCE':      np.uint32(2**9),   # Pixel not on science portion of detector
    'DEAD':             np.uint32(2**10),  # Dead pixel
    'HOT':              np.uint32(2**11),  # Hot pixel
    'WARM':             np.uint32(2**12),  # Warm pixel
    'LOW_QE':           np.uint32(2**13),  # Low quantum efficiency
    'RESERVED_3':       np.uint32(2**14),  # RESERVED_3/TBD
    'TELEGRAPH':        np.uint32(2**15),  # Telegraph pixel
    # 'NONLINEAR':        np.uint32(2**16),  # Pixel highly nonlinear
    'RESERVED_6':        np.uint32(2**16),  #
    'BAD_REF_PIXEL':    np.uint32(2**17),  # Reference pixel cannot be used
    'CROSS_BAD':        np.uint32(2**18),  # Center of cross-shaped bad pixel
    'CROSS_BAD_NEIGH':  np.uint32(2**19),  # Neighbor of cross-shaped bad pixel
    'RESERVED_4':       np.uint32(2**20),  # RESERVED_4/TBD
    'RESERVED_5':       np.uint32(2**21),  # RESERVED_5/TBD
    'UNRELIABLE_BIAS':  np.uint32(2**22),  # Bias variance large
    'UNRELIABLE_DARK':  np.uint32(2**23),  # Dark variance large
    'UNRELIABLE_SLOPE': np.uint32(2**24),  # Slope variance large (i.e., noisy pixel)
    'UNRELIABLE_FLAT':  np.uint32(2**25),  # Flat variance large
    'UNRELIABLE_WS':    np.uint32(2**26),  # Unreliable wavelength solution
    'LOW_INST_TRANS':   np.uint32(2**27),  # Low instrument transmission (blaze < ~90%)
    'TELLURIC':         np.uint32(2**28),  # Telluric absorption line
    'OH_LINE':          np.uint32(2**29),  # OH emission line
    'OTHER_BAD_PIXEL':  np.uint32(2**30),  # A catch-all flag
    'REFERENCE_PIXEL':  np.uint32(2**31),  # Pixel is a reference pixel
}
