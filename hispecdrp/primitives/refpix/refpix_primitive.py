from ..base_primitive import BasePrimitive
from ...calibrations import NonlinearCorrectionSelector
from ... import datamodels
import numpy as np
import logging

import hispecdrp.primitives.refpix.reference_SCALES_DRP as scales_drp

logger = logging.getLogger(__name__)

__all__ = ['RefPixelCorrectionPrimitive']


class RefPixelCorrectionPrimitive(BasePrimitive):
    """
    This step subtracts the correlated read noise (i.e., 1/f noise) using reference pixels.
    Reference pixels are insensitive to light, but they track drifts in the readout electronics.

    For now: This primitive applies reference pixel correction to the input data
    using the SCALES-DRP algorithm (Author: Athira Unni).
    https://github.com/scalessim/SCALES-DRP/blob/main/scalesdrp/primitives/reference.py

    TODO: Eventually for HISPEC, the goal is to use 4 reference pixels on each edge of the detector as well as a 65th reference channel that is read concurrently with the 64 science channels.

    TODO: explore best 1/f noise correction algorithm.
    We might even be able to optionally fit and subtract the 1/f noise during the spectral extraction step in stage 1.

    TODO: eveluate benefits of implementing a IRS2 readout pattern for HISPEC, which has more reference pixels interleaved with science pixels.

    References:
    Roman Roman-STScI-000673: Application and Characterization of IRRC on Roman/WFI TVAC1 Data
    https://www.stsci.edu/files/live/sites/www/files/home/roman/documentation/technical-documentation/_documents/Roman-STScI-000673.pdf

    JWST:
    https://jwst-pipeline.readthedocs.io/en/stable/jwst/refpix/description.html

    Parameters
    ----------
    dataset : str or RampModel
        Input file path or data model to process. Currently, this is expected
        to be a raw (ramp) data product.
    suffix : str or None, optional
        Suffix to append to the output filename. Default is ``'refpix'``.
    method : str, optional
        The reference pixel correction method to use. Currently, only
        ``'scales_drp'`` is supported. Default is ``'scales_drp'``.
    n_channels : int or None, optional
        Number of readout channels in the detector. If ``None``, the value is
        taken from the input model metadata. Default is ``None``.
    n_ref_left : int or None, optional
        Number of reference pixels on the left side of each channel. If
        ``None``, the value is taken from the input model metadata. Default is ``None``.
    n_ref_right : int or None, optional
        Number of reference pixels on the right side of each channel. If
        ``None``, the value is taken from the input model metadata. Default is ``None``.
    vertical : bool or None, optional
        Whether the readout channels are oriented vertically. If ``None``, the
        value is taken from the input model metadata. Default is ``None``.

    Returns
    -------
    output : RampModel
        Reference pixel-corrected data model.

    References
    ----------
    Unni, A., Sallum, S., Benac, P., Fitzgerald, M. P., et al., "SCALES-DRP : A Data Reduction Pipeline for an Upcoming Keck Thermal Infrared Spectrograph", arXiv, arXiv:2509.15489 (2025)
    https://ui.adsabs.harvard.edu/abs/2025arXiv250915489U/abstract
    """

    PARAMETERS = {
        'run' : False,
        'suffix' : 'refpix',
        'method' : 'scales_drp',
        'max_cores' : 1,
        'n_channels' : None,
        'n_ref_left' : None,
        'n_ref_right' : None,
        'vertical' : None,
    }

    def _perform(self):

        input_model = datamodels.open(self.dataset)

        if self.n_channels is None:
            _n_channels =  input_model.meta.n_channels
        else:
            _n_channels = self.n_channels
        if self.vertical is None:
            _vertical = input_model.meta.channels_are_vertical
        else:
            _vertical = self.vertical
        if self.n_ref_left is None:
            _n_ref_left = input_model.meta.n_refpixs
        else:
            _n_ref_left = self.n_ref_left
        if self.n_ref_right is None:
            _n_ref_right = input_model.meta.n_refpixs
        else:
            _n_ref_right = self.n_ref_right

        result = self.get_output_model(input_model)
        logger.info(f"Applying reference pixel correction to {input_model}")

        if result.data.dtype != np.float32:
            logger.info(f"Converting UTR data from {result.data.dtype} to float32")
            result.data = result.data.astype(np.float32)

        if not _vertical:
            _data = np.transpose(result.data, (0,2,1))
        else:
            _data = result.data

        if self.method.lower() == 'scales_drp':
            scales_drp.ref_filter(_data, nchans=_n_channels, in_place=True, avg_type='frame', perint=False,
                   edge_wrap=False, left_ref=True, right_ref=True, nleft=_n_ref_left, nright=_n_ref_right)
        else:
            raise ValueError(f"Unknown reference pixel correction method: {self.method.lower()}")

        if not _vertical:
            result.data = np.transpose(_data, (0,2,1))
        else:
            result.data = _data


        return dict(output=result)