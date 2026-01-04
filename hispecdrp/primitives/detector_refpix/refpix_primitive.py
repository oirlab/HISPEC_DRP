from ..base_primitive import BasePrimitive
from ...calibrations import NonlinearCorrectionSelector
from ... import datamodels
import numpy as np

import hispecdrp.primitives.detector_refpix.reference_SCALES_DRP as scales_drp

__all__ = ['RefPixelCorrectionPrimitive']


class RefPixelCorrectionPrimitive(BasePrimitive):

    ARGUMENTS = {
        'run' : False,
        'suffix' : 'refpix',
        'ouput_model' : 'input',
        'method' : 'scales_drp',
        'max_cores' : 1,
        'n_channels' : 1,
        'n_ref_left' : 4,
        'n_ref_right' : 4
    }

    CALIBRATIONS = {}

    def _perform(self):

        input_model = datamodels.open(self.input)

        if "TBMODIFIED" in input_model.meta: # todo: fix
            n_channels =  input_model.meta["TBMODIFIED"]
        else:
            n_channels = self.n_channels
        if "TBMODIFIED" in input_model.meta: # todo: fix
            n_ref_left =  input_model.meta["TBMODIFIED"]
        else:
            n_ref_left = self.n_ref_left
        if "TBMODIFIED" in input_model.meta: # todo: fix
            n_ref_right =  input_model.meta["TBMODIFIED"]
        else:
            n_ref_right = self.n_ref_right

        result = self.get_output_model(input_model)
        self.log.info(f"Applying reference pixel correction to {input_model}")

        if result.data.dtype != np.float32:
            self.log.info(f"Converting UTR data from {result.data.dtype} to float32")
            result.data = result.data.astype(np.float32)

        if self.method.lower() == 'scales_drp':
            scales_drp.ref_filter(result.data, nchans=n_channels, in_place=True, avg_type='frame', perint=False,
                   edge_wrap=False, left_ref=True, right_ref=True, nleft=n_ref_left, nright=n_ref_right)
        else:
            raise ValueError(f"Unknown reference pixel correction method: {method}")

        return dict(output=result)