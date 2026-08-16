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
    basemodel_filepath=None,
    dq_threshold=0.5
    ):
    """
    Apply an airmass-scaled telluric correction to fiber's spectrum.

    Loads an airmass=1, PWV=1mm telluric transmission template (or a user-supplied template),
    scales it to the observation's airmass, convolves it with the fiber's
    wavelength-dependent LSF and resamples it onto the observed wavelength grid
    for each requested order, then divides the observed spectrum (and its
    error) by the resulting transmission curve. Pixels where the convolved
    telluric transmission falls below a threshold are flagged in the data quality array.

    Parameters
    ----------
    input_model : object
        Input data model containing the observed fiber spectra, errors,
        wavelengths, data quality arrays, and airmass metadata
        (`meta.airmass_start`, `meta.airmass_end`).
    output_model : object
        Output data model to populate with the telluric-corrected spectrum,
        error, data quality, and convolved telluric transmission arrays.
    lsf_model : object
        Model containing per-order LSF parameters (mu, sigma, offset) for the fiber.
    fiber_number : int or str
        Identifier of the fiber to process, used to build attribute names
        (e.g. `SPEC_FIB{fiber_number}`).
    spec_key : str
        Prefix used to look up the spectrum and error attributes on
        `input_model` (e.g. `SPEC` or `SPEC_UNCAL`).
    order_vec : sequence
        Sequence of order identifiers, indexed in parallel with the spectral
        arrays.
    do_orders : sequence
        Subset of `order_vec` entries to actually process; orders not in this
        set are skipped.
    basemodel_filepath : str or Path, optional
        Path to an alternate telluric basemodel file. If None, the default
        packaged model (`mktrans_zm_10_10.dat`) is used.
    dq_threshold : float, optional
        Transmission threshold below which pixels are flagged in the data quality array.

    Returns
    -------
    output_model : object
        The output model, updated in place with attributes
        `TEL_FIB{fiber_number}`, `SPEC_FIB{fiber_number}`,
        `SPEC_ERR_FIB{fiber_number}`, and `SPEC_DQ_FIB{fiber_number}`.
    """
    
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

        # some cosntraint to add dq flag for low telluric transmission regions, e.g. < threshold
        fiber_dq[o, np.where(convolved_telluric_trans[o] < dq_threshold)] = 1

        corrected_spec[o] = fiber_spec[o] / convolved_telluric_trans[o]
        corrected_spec_err[o] = fiber_spec_err[o] / convolved_telluric_trans[o]

    setattr(output_model, f'TEL_FIB{fiber_number}', convolved_telluric_trans)
    setattr(output_model, f'SPEC_FIB{fiber_number}', corrected_spec)
    setattr(output_model, f'SPEC_ERR_FIB{fiber_number}', corrected_spec_err)
    setattr(output_model, f'SPEC_DQ_FIB{fiber_number}', fiber_dq)

    return output_model

def _convolve_and_resample_telluric(telluric_wavelength, telluric_trans, fiber_wave, fiber_lsf_params):
    """
    Convolve a telluric transmission spectrum with a fiber's LSF and resample it.

    Thin wrapper around `_convolve_and_resample` that unpacks per-order LSF
    mean and sigma values from `fiber_lsf_params` before delegating to the
    general convolution/resampling routine.

    Parameters
    ----------
    telluric_wavelength : array
        Wavelengths of the telluric model spectrum, in microns.
    telluric_trans : array
        Telluric transmission values corresponding to
        `telluric_wavelength`.
    fiber_wave : array
        Observed wavelengths for this order to resample the telluric
        transmission onto.
    fiber_lsf_params : array, shape (n_channels, 2)
        Per-channel LSF parameters, where column 0 is the LSF mean (mu) and
        column 1 is the LSF sigma.

    Returns
    -------
    convolved_telluric_trans : array
        Telluric transmission convolved with the LSF and sampled at
        `fiber_wave`.
    """
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
    Convolve a model spectrum with a Gaussian LSF and sample it at given wavelengths.

    For each observed wavelength channel, builds a Gaussian kernel scaled by
    that channel's LSF sigma (and optionally shifted by a per-channel mu),
    interpolates the model spectrum onto the kernel's wavelength support via
    cubic interpolation, and computes the weighted average to produce the
    convolved, resampled flux.

    Parameters
    ----------
    model_wavelength : array
        Model spectrum wavelengths, sorted in increasing order.
    model_fluxes : array
        Model spectrum flux values corresponding to `model_wavelength`.
    observed_wavelength : array
        Observed wavelengths at which to sample the convolved model.
    sigma_array : array
        LSF sigma (in wavelength units) for each observed channel.
    num_sigma : int, optional
        Half-width of the convolution kernel, in units of sigma. Default is 5.
    mu_array : array, optional
        LSF mean offset for each observed channel. Defaults to zeros if not
        provided.
    offset_array : array, optional
        Constant additive offset applied to each channel's result after
        convolution. Defaults to zeros if not provided.

    Returns
    -------
    convolved : array
        Convolved and resampled flux at each observed wavelength channel.

    Notes
    -----
    The kernel resolution is set by the finest sampling of `model_wavelength`
    within the observed wavelength range, so an impractically fine model
    grid combined with a wide kernel (large `sigma_array.max()`) can result
    in a very large `filter_size`.
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
