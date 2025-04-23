=====================
Calibration Selectors
=====================

Selectors are Python classes that specify the logic to select a calibration file from the local calibration cache given an input datamodel or filepath. Each primitive uses a default selector (appropriate for that primitive) with corresponding arguments to override default behavior. Any selector can implement up to three methods to fully specify its behavior.

.. code-block:: python
    :caption: Generic methods a selector class may implement

    def get_candidates(self, input : datamodels.HISPECDataModel, db : CalibrationDB) -> List[Calibration]:
        """
        Get a list of candidate calibrations from the database.
        This method is mandatory for any Selector.
        """

    def select_best(self, input, candidates : List[Calibration], input : datamodels.HISPECDataModel) -> Calibration:
        """
        Select the best calibration from a list of candidates.
        By default, the first candidate is returned.
        """

    def select_fallback(self, input : datamodels.HISPECDataModel, db : CalibrationDB) -> Calibration:
        """
        Select a fallback calibration if no candidates are found.
        TBD how this is implemented and utilized and is only called if no valid calibration is found.
        """


.. code-block:: python
    :caption: Example of a selector class for a dark calibration

    def get_candidates(self, input : datamodels.HISPECDataModel, db : CalibrationDB):
        with db.session_manager() as session:
            return session.query(Calibration).filter(
                Calibration.cal_type == 'dark',
                Calibration.instera == input.meta.instrument.era,
                Calibration.spectrograph == input.meta.instrument.spectrograph,
                Calibration.master == True
            ).order_by(func.abs(Calibration.jd_start - input.meta.exposure.jd_start)).all()