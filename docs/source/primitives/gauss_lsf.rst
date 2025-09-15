Line Spread Function (LSF)
==========================


Overview
--------

Determine the line spread function (LSF) of the instrument using Gaussian fits to the LFC lines.

**Class**: :py:class:`~hispecdrp.primitives.lsf.lsf_gauss_primitive.CalculateGaussianLSFPrimitive`


Algorithm
---------

The unstructured set of points :math:`\{x_i,\ m_i, \sigma_i\}` is fit with a 2D Chebyshev polynomial and then evaluated at each detecor pixel to determine the appropriate Gaussian LSF width at that pixel and wavelength. The LSF width is never updated according to any drifts. The LSF can then be calculated at any sub-pixel or wavelength by interpolating the resulting sigmas at each pixel.

Arguments
---------

- **input** : ``str`` | :py:class:`~hispecdrp.datamodels.level2.L2Model`
    The input file or data model to process. Should be a level 2 data product.
- **suffix** : ``str`` | ``None``
    Suffix to append to the output filename (default: None).
- **output_model** : ``str`` | type[:py:class:`~hispecdrp.datamodels.base_model.DataModel`] | ``None``
    The DataModel class or instance to return. Defaults to the string `'GaussianLSFModel'`. See method :py:meth:`~hispecdrp.primitives.base_primitive.BasePrimitive._get_output_model`.
- **deg_intra_order** : ``int``
    Degree within an echelle order.
- **deg_inter_order** : ``int`` | ``str``
    Degree of the inter-order fit polynomial. Set to ``'auto'`` to automatically determine the degree based on the number of orders available.
- **max_iters** : ``int``
    Maximum number of iterations for the fitting algorithm.
- **sigma_cut** : ``float``
    Sigma threshold for outlier rejection during fitting.
- **opt_weights** : ``bool``
    Whether to optimize weights based on the error in each :math:`\sigma`.
- **fit_orders** : ``list[int]`` | ``None``
    Specific orders to include in the fit. If ``None``, fit all available orders.
- **do_fibers** : ``list[int]`` | ``None``
    Which fiber(s) to fit.