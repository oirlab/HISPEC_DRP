# Pixel-specific flags adapted from stdatamodels.jwst.datamodels.dqflags
# See HISPEC documentation
DQ_FLAGS = {
         'GOOD':             0,      # No bits set, all is good
         'DO_NOT_USE':       2**0,   # Bad pixel. Do not use.
         'SATURATED':        2**1,   # Pixel saturated during exposure
         'JUMP_DET':         2**2,   # Jump detected during exposure
         'OUTLIER_MID_TIME': 2**3,   # Mid exposure time is outlier in spectral order
         'OUTLIER':          2**4,   # Flagged by outlier detection
         'PERSISTENCE':      2**5,   # High persistence
         'RESERVED_1':       2**6,   # RESERVED_1/TBD
         'RESERVED_2':       2**7,   # RESERVED_2/TBD
         'UNRELIABLE_ERROR': 2**8,   # Uncertainty exceeds quoted error
         'NON_SCIENCE':      2**9,   # Pixel not on science portion of detector
         'DEAD':             2**10,  # Dead pixel
         'HOT':              2**11,  # Hot pixel
         'WARM':             2**12,  # Warm pixel
         'LOW_QE':           2**13,  # Low quantum efficiency
         'RESERVED_3':       2**14,  # RESERVED_3/TBD
         'TELEGRAPH':        2**15,  # Telegraph pixel
         'NONLINEAR':        2**16,  # Pixel highly nonlinear
         'BAD_REF_PIXEL':    2**17,  # Reference pixel cannot be used
         'CROSS_BAD ':       2**18,  # Center of cross-shaped bad pixel
         'CROSS_BAD_NEIGH':  2**19,  # Neighbor of cross-shaped bad pixel
         'RESERVED_4':       2**20,  # RESERVED_4/TBD
         'RESERVED_5':       2**21,  # RESERVED_5/TBD
         'UNRELIABLE_BIAS':  2**22,  # Bias variance large
         'UNRELIABLE_DARK':  2**23,  # Dark variance large
         'UNRELIABLE_SLOPE': 2**24,  # Slope variance large (i.e., noisy pixel)
         'UNRELIABLE_FLAT':  2**25,  # Flat variance large
         'UNRELIABLE_WS ':   2**26,  # Unreliable wavelength solution
         'LOW_INST_TRANS ':  2**27,  # Low instrument transmission (blaze < ~90%)
         'TELLURIC ':        2**28,  # Telluric absorption line
         'OH_LINE ':         2**29,  # OH emission line
         'OTHER_BAD_PIXEL':  2**30,  # A catch-all flag
         'REFERENCE_PIXEL':  2**31,  # Pixel is a reference pixel
         }
