LFC Wavelength Solution
=======================


Overview
--------

Determine the wavelength solution from the laser frequency comb (LFC) line positions.


**Class**: :py:class:`~hispecdrp.primitives.wavesol.wavesol_cheb2d_primitive.CalculateChebyshev2DWLSPrimitive`


Algorithm
---------

The unstructured set of points :math:`\{x_i,\ m_i, \lambda_i\}` is fit with a 2D Chebyshev polynomial to obtain the wavelength solution. Specifically, the data fit is :math:`\{x_i,\ m_i, m_i \lambda_i\}` where the 2D polynomial represents an expansion of the grating equation.

Arguments
---------

- **input** : ``str`` | :py:class:`~hispecdrp.datamodels.level2.L2Model`
    The input file or data model of the LFC. Should be a level 2 data product.
- **suffix** : ``str`` | ``None``
    Suffix to append to the output filename (default: None).
- **output_model** : ``str`` | type[:py:class:`~hispecdrp.datamodels.base_model.DataModel`] | ``None``
    The DataModel class or instance to return. Defaults to the string `'LFCChebyshevWavesolModel'`. See method :py:meth:`~hispecdrp.primitives.base_primitive.BasePrimitive._get_output_model`.
- **deg_intra_order** : ``int``
    Degree within an echelle order.
- **deg_inter_order** : ``int`` | ``str``
    Degree of the inter-order fit polynomial. Set to ``'auto'`` to automatically determine the degree based on the number of orders available.
- **max_iters** : ``int``
    Maximum number of iterations for the fitting algorithm.
- **sigma_cut** : ``float``
    Sigma threshold for outlier rejection during fitting.
- **max_vel_cut** : ``float``
    Maximum allowed velocity deviation (m/s) for line rejection.
- **opt_weights** : ``bool``
    Whether to optimize weights based on the error in :math:`x_i`.
- **fit_orders** : ``list[int]`` | ``None``
    Specific orders to include in the fit. If ``None``, fit all available orders.
- **do_fibers** : ``list[int]`` | ``None``
    Which fiber(s) to fit.