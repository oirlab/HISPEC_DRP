from ..base_primitive import BasePrimitive
from ...calibrations import NonlinearCorrectionSelector
from ... import datamodels
import numpy as np

import hispecdrp.primitives.refpix.reference_SCALES_DRP as scales_drp

__all__ = ['RefPixelCorrectionPrimitive']


class RefPixelCorrectionPrimitive(BasePrimitive):

    ARGUMENTS = {
        'run' : False,
        'suffix' : 'refpix',
        'ouput_model' : 'input',
        'method' : 'scales_drp',
        'max_cores' : 1,
        'n_channels' : None,
        'n_ref_left' : None,
        'n_ref_right' : None,
        'vertical' : None,
    }

    CALIBRATIONS = {}

    def _perform(self):

        input_model = datamodels.open(self.input)

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
        self.log.info(f"Applying reference pixel correction to {input_model}")

        if result.data.dtype != np.float32:
            self.log.info(f"Converting UTR data from {result.data.dtype} to float32")
            result.data = result.data.astype(np.float32)

        if not _vertical:
            _data = np.transpose(result.data, (0,2,1))
        else:
            _data = result.data

        if self.method.lower() == 'scales_drp':
            scales_drp.ref_filter(_data, nchans=_n_channels, in_place=True, avg_type='frame', perint=False,
                   edge_wrap=False, left_ref=True, right_ref=True, nleft=_n_ref_left, nright=_n_ref_right)
        else:
            raise ValueError(f"Unknown reference pixel correction method: {method}")

        if not _vertical:
            result.data = np.transpose(_data, (0,2,1))
        else:
            result.data = _data


        return dict(output=result)