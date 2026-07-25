=====================
Calibration Selectors
=====================

Calibration Selectors are Python classes that specify and run the logic to select the most appropriate calibration file from the local calibration cache. All selectors should inherit from :py:class:`~hispecdrp.calibrations.selectors.HISPECCalibrationSelector`.

See the `KOA Middleware documentation  <https://oirlab.github.io/KOA_Middleware/selectors.html>`_ for more information on selectors and how to specify a new selector class.

For example, the :py:class:`~hispecdrp.calibrations.selectors.DarkSelector` class is implemented as follows:

.. code-block:: python

    class DarkSelector(HISPECCalibrationSelector):
        """
        Selector for dark calibration files.

        **Calibration Type:** dark

        **Selection Logic:**

        - Match instrument era
        - Match spectrograph
        - Closest in observation time
        """

        def get_candidates(self, meta : MetaModel, db : LocalCalibrationDB):
            rows = db.rows_where(
                """
                cal_type = :cal_type AND
                instrument_era = :era AND
                spectrograph = :spec
                """,
                {
                    "cal_type": "dark",
                    "era": meta["instrument_era"],
                    "spec": meta["spectrograph"],
                    "mjd_start": meta["mjd_start"],
                },
                order_by=f"ABS(mjd_start - :mjd_start)"
            )

            return list(rows)

Typically, ``get_candidates`` should **not** filter based on ``master_cal``, ``origin`` (i.e, 'KECK' or 'LOCAL'), or ``cal_version``.

After ``get_candidates`` is called, ``HISPECCalibrationSelectors`` automatically filters on the calibration ``origin``, whether or not the calibration is a master calibration, and finally the calibration version based on environment variables or inputs to the selector class contructor.