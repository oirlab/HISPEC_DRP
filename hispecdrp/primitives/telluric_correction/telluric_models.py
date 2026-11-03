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
    lsf_model : GaussianLSFModel
        LFC-derived Gaussian LSF with Chebyshev FWHM coefficients. Kernels are
        built with :meth:`GaussianLSFModel.make_kernels`.
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
    fiber_spec = output_model[fiber_number].spec
    fiber_spec_err = output_model[fiber_number].spec_err
    fiber_wave = output_model[fiber_number].wave
    fiber_dq = output_model[fiber_number].spec_dq

    corrected_spec = np.empty_like(fiber_spec)
    corrected_spec_err = np.empty_like(fiber_spec_err)
    convolved_telluric_trans = np.empty_like(fiber_spec)

    for o, order in enumerate(order_vec):
        if order not in do_orders:
            continue

        convolved_telluric_trans[o] = _convolve_telluric_with_gaussian_lsf(
            telluric_wavelength,
            airmass_scaled_telluric_trans,
            fiber_wave[o],
            lsf_model,
            fiber_number=int(fiber_number),
            order=int(order),
        )

        # some cosntraint to add dq flag for low telluric transmission regions, e.g. < threshold
        fiber_dq[o, np.where(convolved_telluric_trans[o] < dq_threshold)] = 1

        corrected_spec[o] = fiber_spec[o] / convolved_telluric_trans[o]
        corrected_spec_err[o] = fiber_spec_err[o] / convolved_telluric_trans[o]

    output_model[fiber_number].tel = convolved_telluric_trans
    output_model[fiber_number].spec = corrected_spec
    output_model[fiber_number].spec_err = corrected_spec_err
    output_model[fiber_number].spec_dq = fiber_dq

    return output_model

def _convolve_telluric_with_gaussian_lsf(
    telluric_wavelength,
    telluric_trans,
    fiber_wave,
    lsf_model,
    *,
    fiber_number,
    order,
    kernel_size_sigma=5.0,
):
    """
    Convolve telluric transmission with the LFC Gaussian LSF and sample at ``fiber_wave``.

    Interpolates the telluric model onto a uniform wavelength grid, builds
    chromatic kernels with :meth:`GaussianLSFModel.make_kernels`, and applies
    :func:`~hispecdrp.primitives.lsf.lsf_convolve.convolve_numba_1d_chromatic`.
    """
    from ..lsf.lsf_convolve import convolve_numba_1d_chromatic

    w_obs = np.asarray(fiber_wave, dtype=float)
    w_tel = np.asarray(telluric_wavelength, dtype=float)
    t_tel = np.asarray(telluric_trans, dtype=float)
    finite = np.isfinite(w_obs)
    if int(finite.sum()) < 2:
        return np.full_like(w_obs, np.nan)
    wmin = float(np.nanmin(w_obs))
    wmax = float(np.nanmax(w_obs))
    dw = float(np.median(np.abs(np.diff(np.sort(w_obs[finite])))))
    if not np.isfinite(dw) or dw <= 0.0:
        raise ValueError("fiber_wave must have nonzero spacing")
    n = int(np.ceil((wmax - wmin) / dw)) + 1
    w_uni = np.linspace(wmin, wmax, n)
    if w_tel[0] > w_tel[-1]:
        tel_uni = np.interp(w_uni, w_tel[::-1], t_tel[::-1])
    else:
        tel_uni = np.interp(w_uni, w_tel, t_tel)
    _grid, kernels = lsf_model.make_kernels(
        wave=w_uni,
        fiber=int(fiber_number),
        order=int(order),
        kernel_size_sigma=float(kernel_size_sigma),
    )
    tel_conv = convolve_numba_1d_chromatic(tel_uni, kernels)
    if w_obs[0] > w_obs[-1]:
        return np.interp(w_obs, w_uni[::-1], tel_conv[::-1])
    return np.interp(w_obs, w_uni, tel_conv)

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
