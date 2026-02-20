====================
Calibration Database
====================

Overview
--------

The HISPEC DRP uses a local calibration database to manage calibration files and metadata. This database is implemented as a local SQLite database that interacts with a cache of calibration files stored on disk.

HISPECCalibrationStore
----------------------

The :py:class:`~hispecdrp.calibrations.hispec_store.HISPECCalibrationStore` class provides the main interface for working with the calibration database.

Calibration Versioning
----------------------

Each calibration file is associated with a version, which is determined by the metadata of the calibration file. Version numbers start at "001" and increment for each "new version of that same calibration" up to "999".

Version Family
++++++++++++++

From a versioning context, calibration files are determined to be otherwise "identical" if they have the same values in a subset of metadata called the "version family".

The method :py:meth:`~hispecdrp.calibrations.hispec_store.HISPECCalibrationStore.get_version_family_values` returns a list of metadata keys that are used to determine the version family. By default, the version family includes the following metadata keys:

- **Date time of observation**: ``meta.datetime_obs``
- **Spectrograph**: ``meta.spectrograph``
- **Calibration Type**: ``meta.cal_type``
- **Master Calibration**: ``meta.master_cal``

**If two calibration files have the same values for all of the version family metadata keys andhave the same origin (See below), then they are considered to be in the same version family and must have a unique version number.**

Calibration Origin
------------------

**!! NOTE !! The behavior of *origin* is still being finalized subject to change. When creating new calibrations, populating the local/remote database, explicitly set the origin as needed.**

The origin defines the source of where the calibration came from, and provides an independent local-only namespace for assigning versions to calibration files.

- **KECK**: Calibrations generated at Keck which go through KOA.
- **LOCAL**: Calibrations that are generated outside of the main DRP processing at Keck and registered to the local cache.

It is stored in the FITS metadata and used during version generation and file registration. One can set this value globally with the ``KOA_CALBRATION_ORIGIN`` environment variable.

Important methods that interact with the origin include:


.. autoclass:: hispecdrp.primitives.register_calibration.register_cal_primitive.RegisterCalibrationPrimitive
    :noindex:
    :members:
    :undoc-members:

.. automethod:: hispecdrp.calibrations.hispec_store.HISPECCalibrationStore.register_calibration
    :noindex:

.. automethod:: hispecdrp.calibrations.hispec_store.HISPECCalibrationStore.sync_local_db_from_local_files
    :noindex:

.. automethod:: hispecdrp.calibrations.hispec_store.HISPECCalibrationStore.generate_calibration_version
    :noindex: