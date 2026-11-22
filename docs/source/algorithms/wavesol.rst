===================
Wavelength Solution
===================

The wavelength solution is a single 2D Chebyshev polynomial in detector pixel and echelle order. It is used for the laser frequency comb (LFC), the hollow cathode lamp (HCL) and the gas cell. The LFC and HCL fit it to measured line positions, while the gas cell fits it by forward modeling the whole spectrum.

Model
-----

The polynomial describes the product of the echelle order :math:`m` and the wavelength :math:`\lambda`, since the grating equation makes :math:`m\lambda` a smooth function of the diffraction angle. At pixel :math:`x` in order :math:`m`,

.. math::

   \lambda(x, m) = \frac{1}{m} \sum_{i=0}^{N_x} \sum_{j=0}^{N_m} c_{ij}\, T_i(\tilde{x})\, T_j(\tilde{m}),

where :math:`T_k` is the Chebyshev polynomial of the first kind of degree :math:`k`, and :math:`N_x` and :math:`N_m` are the degrees within and between orders. The pixel and order are mapped to :math:`[-1, 1]` with

.. math::

   \tilde{v} = \frac{2v - (v_\mathrm{min} + v_\mathrm{max})}{v_\mathrm{max} - v_\mathrm{min}},

where the bounds are fixed by the detector and the spectrograph rather than by the fitted data. The pixel bounds are :math:`(-10,\ N_\mathrm{cols} - 1 + 10)`, and the order bounds are the full order range of the spectrograph padded by 1 on each side. Because the bounds are the same for every fit, coefficients (``cheb_coeffs_fib#``) from different fits can be compared directly, and the bounds are not stored (see :func:`~hispecdrp.primitives.wavesol.wavesol_cheb2d_primitive.wavesol_norm_bounds`).

The default :math:`N_x` is 6. By default, :math:`N_m` is set from the number of fitted orders :math:`N_\mathrm{orders}`. For 1 to 5 orders it is 0, 1, 2, 3 and 3. For more orders it is

.. math::

   N_m = \min\left(\operatorname{round}\left(6\, \frac{\ln N_\mathrm{orders}}{\ln 10}\right),\ 10\right).

Fitting to Lines (LFC and HCL)
------------------------------

Each fitted line :math:`k` gives a centroid :math:`x_k` in order :math:`m_k`, a known wavelength :math:`\lambda_k` and a centroid uncertainty in wavelength :math:`\sigma_k`. The coefficients minimize

.. math::

   \chi^2 = \sum_k w_k \left( m_k \lambda_k - \sum_{ij} c_{ij}\, T_i(\tilde{x}_k)\, T_j(\tilde{m}_k) \right)^2,

with :math:`w_k = 1/\sigma_k^2` (or uniform weights). This is linear in :math:`c_{ij}` and is solved with weighted linear least squares. The fit is then iterated with outlier rejection. With residuals :math:`\Delta\lambda_k = \lambda_k - \lambda(x_k, m_k)` and :math:`\Delta v_k = c\,\Delta\lambda_k / \lambda_k`, a line is kept if

.. math::

   |\Delta v_k| < v_\mathrm{max} \quad \text{and} \quad |\Delta\lambda_k| < \kappa\, \sigma_{\Delta\lambda},

where :math:`\sigma_{\Delta\lambda}` is the standard deviation of the residuals, :math:`v_\mathrm{max} = 1000` m/s and :math:`\kappa = 4` by default. The iteration stops when the number of kept lines no longer changes, or after 100 iterations.

Forward Modeling (Gas Cell)
---------------------------

The gas cell spectrum is fit with a forward model of the FTS transmission :math:`F_\mathrm{FTS}(\lambda)`, the product of the transmissions of all gas cells. The model is built on a pixel grid oversampled by a factor :math:`O`, with samples :math:`s` at pixel coordinates :math:`x_s`. The FTS spectrum is an instantaneous flux density, so it is evaluated at :math:`\lambda(x_s, m)` by linear interpolation. It is then convolved with the Gaussian LSF :math:`K` and sampled at the pixel centers, and multiplied by a per-order continuum:

.. math::

   M(x, m) = C_m(x)\, \Big[ K * F_\mathrm{FTS}\big(\lambda(x_s, m)\big) \Big](x), \qquad C_m(x) = \sum_{q=0}^{N_c} a_{mq}\, T_q(\tilde{x}).

The continuum absorbs the residual continuum of the blaze-corrected spectrum, so it is a low-degree polynomial (:math:`N_c = 2` by default).

The FTS spectrum is already broadened by the FTS resolution :math:`\Delta\sigma`, which is fixed in wavenumber (:math:`0.0056` cm\ :sup:`-1` for the bundled scans). It is treated as the FWHM of a Gaussian, so the LSF is narrowed in quadrature:

.. math::

   \mathrm{FWHM}_K^2 = \mathrm{FWHM}_\mathrm{LSF}^2 - \mathrm{FWHM}_\mathrm{FTS}^2, \qquad \mathrm{FWHM}_\mathrm{FTS} = 10^{-4}\, \lambda^2\, \Delta\sigma,

with :math:`\lambda` in microns and :math:`\Delta\sigma` in cm\ :sup:`-1`. Within each order, :math:`\mathrm{FWHM}_\mathrm{FTS}` is approximated as linear in pixel.

All orders are fit at once by minimizing

.. math::

   \chi^2 = \sum_{m, x} \left( \frac{D(x, m) - M(x, m)}{\sigma(x, m)} \right)^2

over :math:`c_{ij}` and :math:`a_{mq}` with Levenberg-Marquardt, where :math:`D` is the data and :math:`\sigma` its uncertainty. The fit starts from a least-squares fit of the polynomial to the input wavelength solution. The convolution is linear and its kernels do not depend on the parameters, so the Jacobian is analytic:

.. math::

   \frac{\partial M(x, m)}{\partial c_{ij}} = C_m(x)\, \frac{T_j(\tilde{m})}{m} \Big[ K * \big( F_\mathrm{FTS}'(\lambda(x_s, m))\, T_i(\tilde{x}_s) \big) \Big](x),

where :math:`F_\mathrm{FTS}'` is the slope of the linear interpolant.

Documentation
-------------

.. autoclass:: hispecdrp.primitives.wavesol.wavesol_cheb2d_primitive.CalculateChebyshev2DWLSPrimitive
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: hispecdrp.primitives.wavesol.gascell_wavesol_primitive.FitGasCellWavesolPrimitive
   :members:
   :undoc-members:
   :no-index:
