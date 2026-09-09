import numpy as np

from .base_model import CalibrationModel

from .extensions import IMAGE_HDU, DQ_L0

__all__ = ['NonlinearCorrectionModel']

class NonlinearCorrectionModel(CalibrationModel):
    """
    Specifies the nonlinearity of the detector readout response with a unique polynomial for each pixel.
    """
    EXTENSIONS = {
        # Ncoeff is a per-instance fit result with no symbol (ANY); the default
        # placeholder built below always uses 2 coefficients (an identity
        # linear fit: offset=0, slope=1), regardless of what a real fit uses.
        'COEFFS': IMAGE_HDU(np.float32, shape=('DETECTOR_SIZE_Y', 'DETECTOR_SIZE_X', 'ANY'), title='Polynomial coeffs describing the nonlinearity.'),
        'NONLIN_THRESH' : IMAGE_HDU(np.uint16, shape=('DETECTOR_SIZE_Y', 'DETECTOR_SIZE_X'), default=0.9 * np.iinfo(np.uint16).max, title='TBD, from JWST.'),
        'NONLIN_MAX' : IMAGE_HDU(np.uint16, shape=('DETECTOR_SIZE_Y', 'DETECTOR_SIZE_X'), default=np.iinfo(np.uint16).max, title='TBD, from JWST.'),
        **DQ_L0
    }
    CAL_TYPE = 'nonlin'

    def on_init(self):
        super().on_init()
        self.meta.set_default('exposure_type', 'DETFLAT')
        self.meta.set_default('target_name', 'WHITELIGHT')

    def make_default_array(self, extname : str, info : dict):
        if extname == 'COEFFS':
            # COEFFS' declared shape is (DETECTOR_SIZE_Y, DETECTOR_SIZE_X, ANY)
            # -- Ncoeff (ANY) is a per-instance fit result with no symbol, so
            # only the leading, resolvable axes are used here.
            shape = self._resolve_shape(tuple(axis for axis in info['shape'] if axis.upper() != 'ANY'))
            if shape is None:
                return None
            coeffs = np.zeros((*shape, 2), dtype=np.float32)
            coeffs[:, :, 1] = 1.0
            return coeffs
        return super().make_default_array(extname, info)