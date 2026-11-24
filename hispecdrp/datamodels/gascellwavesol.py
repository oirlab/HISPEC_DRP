import numpy as np

from .base_model import CalibrationModel
from .mixins import FiberAttributeProxyMixin
from .extensions import IMAGE_HDU, TABLE_HDU, create_dataspec_extensions, create_specdq_extensions, create_wave_extensions, create_chebyshev_extensions

__all__ = ['GasCellWavesolModel']


class GasCellWavesolModel(FiberAttributeProxyMixin, CalibrationModel):
    """
    The gas cell wavelength solution.
    """

    EXTENSIONS = {
        **create_dataspec_extensions(prefix='SPEC', title='Gas cell spectrum.', units='e-/s'),
        **create_specdq_extensions(title='Gas cell spectrum data quality.'),
        **create_wave_extensions(title='Gas cell spectrum wavelength grid.'),
        'MODEL_RESID': IMAGE_HDU(
            dtype=np.float32,
            shape=('N_ORDERS', 'DETECTOR_SIZE_Y'), enumerate='FIB',
            title='Gas cell spectrum minus the best-fit forward model.',
            units='e-/s',
        ),
        'MODEL': IMAGE_HDU(
            dtype=np.float32,
            shape=('N_ORDERS', 'DETECTOR_SIZE_Y'), enumerate='FIB',
            title='Best-fit forward model: continuum times the LSF-convolved FTS transmission (NaN on orders not fit).',
            units='e-/s',
        ),
        'CONT': IMAGE_HDU(
            dtype=np.float32,
            shape=('N_ORDERS', 'DETECTOR_SIZE_Y'), enumerate='FIB',
            title='Best-fit continuum (NaN on orders not fit; ones if no continuum is fit).',
            units='e-/s',
        ),
        # Real shape is (N_ORDERS, continuum_deg_intra_order + 1), NaN rows for orders not fit, with the
        # wavelength solution's pixel normalization -- a per-instance fit result, hence ANY and no default.
        'CONT_COEFFS': IMAGE_HDU(
            dtype=np.float64,
            shape=('ANY', 'ANY'), enumerate='FIB',
            title='Chebyshev coefficients of the continuum (see CONT_INFO for the layout).',
        ),
        'CONT_INFO': TABLE_HDU({
            'MODEL': str,  # '1d', or 'none' if no continuum is fit
        }, enumerate='FIB', title="Continuum model: '1d' (per order, in pixel) or 'none'."),
        **create_chebyshev_extensions(),
    }

    CAL_TYPE = 'gascell_wavesol'

    def on_init(self):
        super().on_init()
        self.meta.set_default('exposure_type', 'SPEC')
        self.meta.set_default('target_name', 'GASCELL')

    @property
    def primary_extension_name(self):
        for fiber in range(1, self.config.num_fibers + 1):
            if self.has_extension_value(f'spec_fib{fiber}'):
                return f'spec_fib{fiber}'
            if self.has_extension_value(f'wave_fib{fiber}'):
                return f'wave_fib{fiber}'
        return None
