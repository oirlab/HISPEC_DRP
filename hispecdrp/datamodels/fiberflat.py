from .base_model import CalibrationModel
from .extensions import create_dataspec_extensions, create_specdq_extensions, create_wave_extensions, create_midtime_extensions, create_trans_extensions

__all__ = ['FiberflatModel']


def _create_fiberflat_extensions(num_fibers: int):
    ext = {}
    for fiber in range(1, num_fibers + 1):
        ext.update(create_dataspec_extensions(fiber=fiber, prefix='SPEC'))
        ext.update(create_specdq_extensions(fiber=fiber))
        ext.update(create_wave_extensions(fiber=fiber))
        ext.update(create_midtime_extensions(fiber=fiber))
        ext.update(create_trans_extensions(fiber=fiber))
    return ext


class FiberflatModel(CalibrationModel):
    """
    The fiber flat model contains the relative transmission of each fiber as a function of wavelength.

    Extensions
    ----------

    HDU Name                 HDU Type   Data Type   Dimensions     Units     Description
    --------                 --------   ---------   ----------     -----     -----------
    SPEC_FIB#                Image      Float32     Nord x 4096    e-/s      Extracted fiber spectrum.
    SPEC_ERR_FIB#            Image      Float32     Nord x 4096    e-/s      Extracted fiber spectrum errors (all sources).
    SPEC_VAR_POISSON_FIB#    Image      Float32     Nord x 4096    (e-/s)^2  Extracted fiber spectrum variance (Poisson).
    SPEC_VAR_RNOISE_FIB#     Image      Float32     Nord x 4096    (e-/s)^2  Extracted fiber spectrum variance (read noise).
    SPEC_DQ_FIB#             Image      UInt32      Nord x 4096    None      Data quality.
    WAVE_FIB#                Image      Float64     Nord x 4096    micron    Wavelength solution.
    MID_TIME_FIB#            Image      Float64     Nord x 4096    MJD       Exposure mid-point times.
    TRANS_FIB#               Image      Float32     Nord x 4096    None      Normalized transmission.
    TRANS_ERR_FIB#           Image      Float32     Nord x 4096    None      Transmission error.
    """
    CAL_TYPE = 'fiberflat'
    
    def __init__(self, *args, **kwargs):
        config = kwargs.get('config', None)
        if config is None:
            config = self._get_instrument_config(
                meta=kwargs.get('meta', None),
                filepath=kwargs.get('filepath', None),
                instrument_name=kwargs.get('instrument_name', None),
            )
        self.EXTENSIONS = self.EXTENSIONS.copy()
        self.EXTENSIONS.update(_create_fiberflat_extensions(config.num_fibers))
        super().__init__(*args, **kwargs)
