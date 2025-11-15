=====================
Calibration Selectors
=====================

Selectors are Python classes that specify and run the logic to select the most appropriate calibration file from the local calibration cache. Each primitive uses a particular selector to retrieve a calibration that it needs. For example, the DarkSubtractionPrimitive uses the DarkSelector to associate the 'dark' calibration file with the input to DarkSubtractionPrimitive.

A selector can implement the following three methods to fully specify its behavior.

.. code-block:: python
    :caption: Generic methods a selector class may implement

    def get_candidates(self, input : DataModel, db : CalibrationDB) -> list[CalibrationORM]:
        """
        Get a list of candidate calibrations from the database.
        This method is mandatory for any Selector.
        """

    def select_best(self, input, candidates : list[CalibrationORM], input : datamodels.HISPECDataModel) -> CalibrationORM:
        """
        Select the best calibration from a list of candidates.
        By default, the first candidate is returned.
        """

    def select_fallback(self, input : DataModel, db : CalibrationDB) -> CalibrationORM:
        """
        Select a fallback calibration if no candidates are found.
        TBD how this is implemented and utilized and is only called if no valid calibration is found.
        """


.. code-block:: python
    :caption: Example of a selector class for a dark calibration

    class DarkSelector(CalibrationSelector):

        def get_candidates(self, meta : dict, db : CalibrationDB):
            with db.session_manager() as session:
                if meta['instrument_name'].lower() == 'hispec':
                    return session.query(db.orm_class).filter(
                        db.orm_class.cal_type == 'dark',
                        db.orm_class.instrument_era == meta['instrument_era'],
                        db.orm_class.spectrograph == meta['spectrograph'],
                        db.orm_class.master_cal == True
                    ).order_by(func.abs(db.orm_class.mjd_start - meta['mjd_start'])).all()
                elif meta['instrument_name'].lower() == 'parvi':
                    return session.query(db.orm_class).filter(
                        db.orm_class.cal_type == 'dark',
                        db.orm_class.instrument_era == meta['instrument_era'],
                    ).order_by(func.abs(db.orm_class.mjd_start - meta['mjd_start'])).all()