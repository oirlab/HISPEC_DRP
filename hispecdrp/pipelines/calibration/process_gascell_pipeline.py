from hispecdrp import datamodels
from ..calibration_pipeline_base import CalibrationPipeline
from ...primitives import FitGasCellWavesolPrimitive, RegisterCalibrationPrimitive, CoaddSpectraPrimitive, BlazeCorrectionPrimitive

from typing import Sequence

__all__ = ['ProcessGasCellPipeline']


class ProcessGasCellPipeline(CalibrationPipeline):

    """
    Pipeline to process 1D gas cell data to generate a wavelength calibration model.

    This pipeline performs the following steps:

    - :class:`~hispecdrp.primitives.coadd.CoaddSpectraPrimitive`
    - :class:`~hispecdrp.primitives.blaze_correction.BlazeCorrectionPrimitive`
    - :class:`~hispecdrp.primitives.wavesol.FitGasCellWavesolPrimitive`
    - :class:`~hispecdrp.primitives.register_calibration.RegisterCalibrationPrimitive` (for the wavelength solution)

    """

    PRIMITIVES = {
        'coadd_spectra': CoaddSpectraPrimitive,
        'blaze_corr' : BlazeCorrectionPrimitive,
        'gascell_wavesol': FitGasCellWavesolPrimitive,
        'register_wave_cal' : RegisterCalibrationPrimitive,
    }

    def _perform(self):

        # Load input models
        if not isinstance(self.dataset, Sequence):
            input_models = [datamodels.open(self.dataset)]
        else:
            input_models = [datamodels.open(m) for m in self.dataset]

        # Group by like fiber
        gascell_groups = self.group_gascell_spectra(input_models)

        # Coadd models
        gascell_models_coadded = {}
        for fiber_group in gascell_groups:
            gascell_models_coadded[fiber_group] = self.coadd_spectra.apply(
                gascell_groups[fiber_group],
                do_fibers=_fiber_list(fiber_group)
            )

        # Apply blaze correction to coadded models
        for fiber_group in gascell_groups:
            gascell_models_coadded[fiber_group] = self.blaze_corr.apply(
                gascell_models_coadded[fiber_group],
                do_fibers=_fiber_list(fiber_group)
            )

        # Compute wavelength solution
        wavesols = {}
        for fiber_group in gascell_groups:
            wavesol_result = self.gascell_wavesol.apply(
                gascell_models_coadded[fiber_group],
                do_fibers=_fiber_list(fiber_group)
            )
            self.register_wave_cal.apply(wavesol_result)
            wavesols[fiber_group] = wavesol_result

        wavesols = list(wavesols.values())

        if not isinstance(self.dataset, Sequence):
            wavesols = wavesols[0]

        # Return output
        return dict(output=wavesols)

    @staticmethod
    def group_gascell_spectra(models: list[datamodels.DataModel]):
        """
        Determine which fibers have a gas cell spectrum and group accordingly.

        Assumes all models share the same object_fiber configuration, so the
        set of gas cell fibers is determined once from the first model and applied
        to the entire list. If a single fiber carries the gas cell, the group is
        keyed by that fiber number directly; if multiple fibers carry it, it is
        keyed by the tuple of those fiber numbers.

        Parameters
        ----------
        models : list[datamodels.DataModel]
            List of input data models to group. All models are assumed to
            share the same object_fiber configuration.

        Returns
        -------
        gascell_groups : dict[int | tuple[int, ...], list[datamodels.DataModel]]
            Dictionary with a single key (fiber number, or tuple of fiber
            numbers) mapping to the full list of input models.

        Raises
        ------
        ValueError
            If no fiber carries a gas cell object.
        """
        m0 = models[0]
        gascell_fibers = []
        for fiber in range(1, m0.num_fibers + 1):
            fiber_obj = m0.meta.get(f"object_fiber{fiber}")
            if fiber_obj is None:
                continue
            # 'GASCELL' (HISPEC), 'GASCELL-QUAD' / 'GASCELL-HF4' (PARVI)
            if fiber_obj.lower().startswith('gascell'):
                gascell_fibers.append(fiber)

        if not gascell_fibers:
            raise ValueError("No gas cell spectra found in any fiber.")

        key = gascell_fibers[0] if len(gascell_fibers) == 1 else tuple(gascell_fibers)
        return {key: list(models)}


def _fiber_list(fiber_group : int | tuple[int, ...]) -> list[int]:
    return list(fiber_group) if isinstance(fiber_group, tuple) else [fiber_group]
