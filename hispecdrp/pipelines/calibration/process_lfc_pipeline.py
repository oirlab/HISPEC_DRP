from hispecdrp import datamodels
from ..calibration_pipeline_base import CalibrationPipeline
from ...primitives import FitLFCLinesPrimitive, CalculateChebyshev2DWLSPrimitive, CalculateGaussianLSFPrimitive, RegisterCalibrationPrimitive, CoaddSpectraPrimitive, BlazeCorrectionPrimitive

from typing import Sequence

__all__ = ['ProcessLFCPipeline']


class ProcessLFCPipeline(CalibrationPipeline):

    """
    Pipeline to process 1D laser frequency comb (LFC) data to generate a wavelength calibration model and line spread function (LSF) model.

    This pipeline performs the following steps:

    - :class:`~hispecdrp.primitives.lfc_line_fitting.FitLFCLinesPrimitive`
    - :class:`~hispecdrp.primitives.lfc_wavesol.CalculateChebyshev2DWLSPrimitive`
    - :class:`~hispecdrp.primitives.gauss_lsf.CalculateGaussianLSFPrimitive`
    - :class:`~hispecdrp.primitives.register_cal.RegisterCalibrationPrimitive` (for wavelength solution and LSF)
    
    """

    PRIMITIVES = {
        'coadd_spectra': CoaddSpectraPrimitive,
        'blaze_corr' : BlazeCorrectionPrimitive,
        'fit_lfc_lines': FitLFCLinesPrimitive,
        'lfc_wavesol': CalculateChebyshev2DWLSPrimitive,
        'register_wave_cal' : RegisterCalibrationPrimitive,
        'lsf': CalculateGaussianLSFPrimitive,
        'register_lsf_cal' : RegisterCalibrationPrimitive,
    }

    def _perform(self):

        # Load input models
        if not isinstance(self.dataset, Sequence):
            input_models = [datamodels.open(self.dataset)]
        else:
            input_models = [datamodels.open(m) for m in self.dataset]

        # Group by like fiber
        lfc_groups = self.group_lfc_spectra(input_models)

        # Coadd models
        lfc_models_L1_coadded = {}
        for fiber_group in lfc_groups:
            lfc_models_L1_coadded[fiber_group] = self.coadd_spectra.apply(
                lfc_groups[fiber_group],
                do_fibers=fiber_group
            )

        # Apply blaze correction to coadded models
        for fiber_group in lfc_groups:
            lfc_models_L1_coadded[fiber_group] = self.blaze_corr.apply(
                lfc_models_L1_coadded[fiber_group],
                do_fibers=fiber_group
            )

        # Fit LFC lines for all models
        line_fit_models = {}
        for fiber_group in lfc_groups:
            line_fit_models[fiber_group] = self.fit_lfc_lines.apply(
                lfc_models_L1_coadded[fiber_group],
                do_fibers=fiber_group
            )

        # Compute wavelength solution
        wavesols = {}
        for fiber_group in lfc_groups:
            wavesol_result = self.lfc_wavesol.apply(
                line_fit_models[fiber_group],
                do_fibers=fiber_group
            )
            self.register_wave_cal.apply(wavesol_result)
            wavesols[fiber_group] = wavesol_result

        # Compute LSF
        lsfs = {}
        for fiber_group in lfc_groups:
            lsf_result = self.lsf.apply(
                wavesols[fiber_group],
                do_fibers=fiber_group
            )
            self.register_lsf_cal.apply(lsf_result)
            lsfs[fiber_group] = lsf_result

        wavesols = list(wavesols.values())
        lsfs = list(lsfs.values())

        if not isinstance(self.dataset, Sequence):
            wavesols = wavesols[0]
            lsfs = lsfs[0]

        # Return output
        return dict(output=dict(wavesol=wavesols, lsf=lsfs))

    @staticmethod
    def group_lfc_spectra(models: list[datamodels.DataModel]):
        """
        Determine which fibers have an LFC spectrum and group accordingly.

        Assumes all models share the same object_fiber configuration, so the
        set of LFC fibers is determined once from the first model and applied
        to the entire list. If a single fiber carries the LFC, the group is
        keyed by that fiber number directly; if multiple fibers share the
        same LFC, it is keyed by the tuple of those fiber numbers.

        Parameters
        ----------
        models : list[datamodels.DataModel]
            List of input data models to group. All models are assumed to
            share the same object_fiber configuration.

        Returns
        -------
        lfc_groups : dict[int | tuple[int, ...], list[datamodels.DataModel]]
            Dictionary with a single key (fiber number, or tuple of fiber
            numbers) mapping to the full list of input models.

        Raises
        ------
        ValueError
            If no fiber carries an LFC object.
        """
        valid_lfc_objects = ('lfc', 'lfc-b', 'lfc-r')
        m0 = models[0]
        lfc_fibers = []
        for fiber in range(1, m0.num_fibers + 1):
            fiber_obj = m0.meta.get(f"object_fiber{fiber}")
            if fiber_obj is None:
                continue
            fiber_obj = fiber_obj.lower()
            if fiber_obj in valid_lfc_objects:
                lfc_fibers.append(fiber)

        if not lfc_fibers:
            raise ValueError("No LFC spectra found in any fiber.")

        key = lfc_fibers[0] if len(lfc_fibers) == 1 else tuple(lfc_fibers)
        return {key: list(models)}