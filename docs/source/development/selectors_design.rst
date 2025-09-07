=====================
Calibration Selectors
=====================

Selectors are Python classes that specify the logic to select a calibration file from the local calibration cache given an input datamodel or filepath. Each primitive uses a default selector (appropriate for each given calibration to retrieve). Any selector can implement the following three methods to fully specify its behavior.

.. code-block:: python
    :caption: Generic methods a selector class may implement

    def get_candidates(self, input : datamodels.HISPECDataModel, db : CalibrationDB) -> list[CalibrationORM]:
        """
        Get a list of candidate calibrations from the database.
        This method is mandatory for any Selector.
        """

    def select_best(self, input, candidates : list[CalibrationORM], input : datamodels.HISPECDataModel) -> CalibrationORM:
        """
        Select the best calibration from a list of candidates.
        By default, the first candidate is returned.
        """

    def select_fallback(self, input : datamodels.HISPECDataModel, db : CalibrationDB) -> CalibrationORM:
        """
        Select a fallback calibration if no candidates are found.
        TBD how this is implemented and utilized and is only called if no valid calibration is found.
        """


.. code-block:: python
    :caption: Example of a selector class for a dark calibration

    class HISPECDarkSelector(CalibrationSelector):
        def get_candidates(self, input : datamodels.HISPECDataModel, db : CalibrationDB):
            with db.session_manager() as session:
                return session.query(db.orm_class).filter(
                    db.orm_class.cal_type == 'dark',
                    db.orm_class.instrument_era == input.meta.instrument.era,
                    db.orm_class.spectrograph == input.meta.spectrograph,
                    db.orm_class.master_cal == True
                ).order_by(func.abs(db.orm_class.mjd_start - input.meta.mjd_start)).all()