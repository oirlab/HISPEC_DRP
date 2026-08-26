from ..base_primitive import BasePrimitive
from ... import datamodels
from typing import Sequence
import numpy as np

__all__ = ['TraceProfilePrimitive']


class TraceProfilePrimitive(BasePrimitive):

    """
    Build the 2-D trace profile from fiberflat rate maps.

    When an LFC rate map (and trace positions) are supplied, also measure
    spectroperfection moments: iso-λ shear and spectral/spatial 2nd moments.

    Parameters
    ----------
    lfc : DataModel or path or None
        Simulated or observed LFC L0 (2-D rate). Required for shear / LSF
        moments used by ``method='spectroperfection'``.
    tracepos : TracePosModel or path or None
        Trace positions for moment windows. Required with ``lfc``.
    extraction_aperture : float
        Cross-dispersion window for moment measurement.
    orders : list[int] or None
        Restrict moment measurement to these echelle orders. ``None`` uses all.
    """

    PARAMETERS = {
        'output_model' : 'TraceProfModel',
        'lfc': None,
        'tracepos': None,
        'extraction_aperture': 8.0,
        'orders': None,
    }

    def _perform(self):
        
        # Associate inputs by fiber number
        if isinstance(self.dataset, (datamodels.DataModel, str)):
            input = datamodels.group_by_illuminated_fiber([self.dataset])
        elif isinstance(self.dataset, Sequence):
            input = datamodels.group_by_illuminated_fiber(self.dataset)
        elif isinstance(self.dataset, dict):
            input = self.dataset
        else:
            raise ValueError(f"Unsupported input type: {type(self.dataset)}")
        
        first_key = next(iter(input))

        # Current keys: ('WHITELIGHT', 'BLOCKED', 'BLOCKED') : file, etc.
        # Want: ('1' : file, '2' : file, '3' : file)
        if not str(first_key).isdigit():
            input = {
                str(t.index(next(x for x in t if x.upper() != "BLOCKED")) + 1) : v
                for t, v in input.items()
            }
        
        # Prepare output model
        output_model = datamodels.TraceProfModel.from_datamodels(input, primitive=self)

        # Loop over fibers and copy profile data
        for fiber in input.keys():
            input_model = datamodels.open(input[fiber])
            setattr(output_model, f'PROF_FIB{fiber}', input_model.rate)
            setattr(output_model, f'PROF_ERR_FIB{fiber}', input_model.rate_err)

        lfc = getattr(self, "lfc", None)
        tracepos = getattr(self, "tracepos", None)
        if lfc is not None and tracepos is not None:
            from .traceprof_moments import fill_traceprof_spectroperfection_moments

            lfc_model = datamodels.open(lfc)
            tracepos_model = datamodels.open(tracepos)
            spectrograph = str(
                getattr(lfc_model.meta, "spectrograph", None)
                or getattr(output_model.meta, "spectrograph", None)
                or "BSPEC"
            ).upper()
            lfc_rate = np.asarray(lfc_model.rate, dtype=np.float64)
            for fiber in input.keys():
                ff_model = datamodels.open(input[fiber])
                fill_traceprof_spectroperfection_moments(
                    output_model,
                    fiberflat_rate=np.asarray(ff_model.rate, dtype=np.float64),
                    lfc_rate=lfc_rate,
                    tracepos=tracepos_model,
                    spectrograph=spectrograph,
                    fiber=int(fiber),
                    extraction_aperture=float(self.extraction_aperture),
                    orders=getattr(self, "orders", None),
                )

        return dict(output=output_model)