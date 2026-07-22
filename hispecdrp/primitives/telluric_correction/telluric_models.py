import numpy as np
import importlib

from scipy import interpolate

def airmass_scale_singlemodel(
    input_model,
    output_model,
    lsf_model,
    fiber_number,
    spec_key,
    order_vec,
    do_orders,
    basemodel_filepath=None
    ):
    
    atmosphere_directory = importlib.resources.files('hispecdrp.data.atmosphere')
    if basemodel_filepath is not None:
        telluric_basemodel_filepath = basemodel_filepath
    else:
        telluric_basemodel_filepath = atmosphere_directory / 'mktrans_zm_10_10.dat'
    
    telluric_data = np.loadtxt(telluric_basemodel_filepath)
    telluric_wavelength = telluric_data[:, 0] # units assumed in microns
    telluric_trans = telluric_data[:, 1]

    # scale for airmass
    airmass = np.nanmean([input_model.meta.airmass_start, input_model.meta.airmass_end])
    airmass_scaled_telluric_trans = telluric_trans ** airmass

    # get observed spectrum from fiber
    fiber_spec      = getattr(input_model, f'{spec_key}FIB{fiber_number}')
    fiber_spec_err  = getattr(input_model, f'{spec_key}ERR_FIB{fiber_number}')
    fiber_wave      = getattr(input_model, f"WAVE_FIB{fiber_number}")
    fiber_dq        = getattr(input_model, f'SPEC_DQ_FIB{fiber_number}')

    # get lsf data for fiber
    fiber_lsf_params = getattr(lsf_model, f"LSF_PARAMS_FIB{fiber_number}")

    corrected_spec = np.empty_like(fiber_spec)
    corrected_spec_err = np.empty_like(fiber_spec_err)
    convolved_telluric_trans = np.empty_like(fiber_spec)

    for o, order in enumerate(order_vec):
        if order not in do_orders:
            continue

        convolved_telluric_trans[o] = _convolve_and_resample_telluric(
            telluric_wavelength,
            airmass_scaled_telluric_trans,
            fiber_wave[o],
            fiber_lsf_params[o],
                )

        # some cosntraint to add dq flag for low telluric transmission regions, e.g. < 0.5
        fiber_dq[o, np.where(convolved_telluric_trans[o] < 0.5)] = 1

        corrected_spec[o] = fiber_spec[o] / convolved_telluric_trans[o]
        corrected_spec_err[o] = fiber_spec_err[o] / convolved_telluric_trans[o]

    setattr(output_model, f'TEL_FIB{fiber_number}', convolved_telluric_trans)
    setattr(output_model, f'SPEC_FIB{fiber_number}', corrected_spec)
    setattr(output_model, f'SPEC_ERR_FIB{fiber_number}', corrected_spec_err)
    setattr(output_model, f'SPEC_DQ_FIB{fiber_number}', fiber_dq)

    return output_model

def _convolve_and_resample_telluric(telluric_wavelength, telluric_trans, fiber_wave, fiber_lsf_params):
    lsf_mu = fiber_lsf_params[:,0]
    lsf_sigma = fiber_lsf_params[:,1]

    convolved_telluric_trans = _convolve_and_resample(
        telluric_wavelength,
        telluric_trans,
        fiber_wave,
        lsf_sigma,
        mu_array=lsf_mu,
    )
    return convolved_telluric_trans

def _convolve_and_resample(
    model_wavelength, model_fluxes, observed_wavelength, sigma_array, num_sigma=5, mu_array=None, offset_array=None,
):
    """
    Convolve a model spectrum with a Gaussian LSF and sample it at the given observed wavelengths.

    Parameters
    ----------
    observed_wavelength : array of observed wavelengths to sample at
    sigma_array      : array of LSF sigmas for each channel
    model_wavelength : model spectrum wavelengths (sorted, increasing)
    model_fluxes : model spectrum fluxes
    num_sigma    : half-width of the convolution kernel, in sigmas
    mu_array       : array of LSF means for each channel (optional, default=None)
    offset_array   : array of LSF offsets for each channel (optional, default=None)

    Returns
    -------
    Convolved & resampled flux at each wv_channel, or an empty array
    if the required kernel resolution is impractically fine.
    """
    observed_wavelength = np.asarray(observed_wavelength, dtype=float)
    model_wavelength = np.asarray(model_wavelength, dtype=float)
    model_fluxes = np.asarray(model_fluxes, dtype=float)
    sigma_array = np.asarray(sigma_array, dtype=float)
    mu_array = np.asarray(mu_array, dtype=float) if mu_array is not None else np.zeros_like(sigma_array)
    offset_array = np.asarray(offset_array, dtype=float) if offset_array is not None else np.zeros_like(sigma_array)

    # Smallest model sampling within the observed wavelength range, to size the kernel
    in_range = (model_wavelength >= observed_wavelength.min()) & (model_wavelength < observed_wavelength.max())
    min_dwv = np.gradient(model_wavelength[in_range]).min()

    # Number of kernel points needed so the widest kernel is still
    # sampled at least as finely as the model
    filter_size = int(np.ceil((2 * num_sigma * sigma_array.max()) / min_dwv))

    # Kernel coordinates (in sigma units), broadcast to wavelength space
    filter_coords = np.linspace(-num_sigma, num_sigma, filter_size)
    filter_wv_coords = (
        filter_coords[None, :] * sigma_array[:, None]
        + (observed_wavelength + mu_array)[:, None]
    )

    # Gaussian LSF weights (same shape for every channel, in sigma units)
    lsf = np.exp(-filter_coords**2 / 2) / np.sqrt(2 * np.pi)

    # Interpolate model onto kernel wavelengths and take weighted average
    model_interp = interpolate.interp1d(
        model_wavelength, model_fluxes, kind="cubic", bounds_error=False
    )
    filter_model = model_interp(filter_wv_coords)

    convolved = np.nansum(filter_model * lsf, axis=1) / np.sum(lsf)
    return convolved + offset_array
