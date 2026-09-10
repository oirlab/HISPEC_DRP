import numpy as np

from .base_model import CalibrationModel

from .extensions import IMAGE_HDU, DQ_L0

__all__ = ['DetectorFlatModel']


class DetectorFlatModel(CalibrationModel):
    """
    The HISPEC detector flat is a 2D image that corrects the intrinsic pixel-to-pixel sensitivity of the detector.
    """
    EXTENSIONS = {
        f'FLAT': IMAGE_HDU(
            dtype=np.float32,
            shape=('DETECTOR_SIZE_Y', 'DETECTOR_SIZE_X'),
            default=0,
            title='Flat field data.',
            units='DN'
        ),
        f'FLAT_ERR': IMAGE_HDU(
            dtype=np.float32,
            shape=('DETECTOR_SIZE_Y', 'DETECTOR_SIZE_X'),
            default=0,
            title='Flat field error.',
            units='DN'
        ),
        f'FLAT_VAR_RNOISE': IMAGE_HDU(
            dtype=np.float32,
            shape=('DETECTOR_SIZE_Y', 'DETECTOR_SIZE_X'),
            default=0,
            title='Flat field variance (read noise).',
            units='DN^2'
        ),
        f'FLAT_VAR_POISSON': IMAGE_HDU(
            dtype=np.float32,
            shape=('DETECTOR_SIZE_Y', 'DETECTOR_SIZE_X'),
            default=0,
            title='Flat field variance (Poisson).',
            units='DN^2'
        ),
        **DQ_L0
    }
    CAL_TYPE = 'detflat'

    def on_init(self):
        super().on_init()
        self.meta.set_default('exposure_type', 'DETFLAT')
        self.meta.set_default('target_name', self.meta.exposure_type)
        for fiber in range(1, self.num_fibers+1):
            self.meta.set_default(f'object_fiber{fiber}', 'BLOCKED')