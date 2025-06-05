====================
Calibration Database
====================

Calibration metadata is stored in a SQL based database (DB) to allow for faster parsing of metadata to select an appropriate calibration file. The DB is implemented in `SQLAlchemy ORM <https://www.sqlalchemy.org/>`_ and supports SQLite and PostgreSQL.

The database columns are declared in a standard Python class with additional methods for initialization. Calibrations are stored in a single table, defined by the class :py:class:`~hispecdrp.calibrations.metadata_db.Calibration`, which includes any metadata used for matching science products with calibrations.

Schema
------

The `Calibration` table includes the following fields:

.. csv-table::
   :header: name, datatype, unit, description, required
   :widths: 20, 10, 10, 50, 10

   id,str,None,UUID mechanism TBD,Yes
   filename,str,None,Base filename,Yes
   koa_id,str,None,KOA-assigned identifier (if available) and base filename on KOA,No
   koa_filepath,str,None,Full path to the file on KOA,No
   cal_type,str,None,Calibration type (e.g. dark), Yes
   master,bool,None,Whether this is a master calibration (default: Yes), Yes
   instera,str,None,Instrument era/version, Yes
   spectrograph,str,None,Spectrograph name (BSPEC or RSPEC), Yes
   jd_start,float,None,Julian date of observation start, Yes
   datetime_obs,str,ISO 8601,ISO datetime of observation start, Yes
   read_mode,str,None,Detector readout mode, Yes
   target_name,str,None,Name of target object, Yes
   object_fiber1,str,None,Object in fiber 1, Yes
   object_fiber2,str,None,Object in fiber 2, Yes
   object_fiber3,str,None,Object in fiber 3, Yes
   drp_version,str,None,DRP software version, Yes
   note,str,None,Optional user-defined notes, Yes
   last_updated,str,ISO 8601,Timestamp of last update (auto-managed), Yes


CalibrationDB
-------------

The class :py:class:`~hispecdrp.calibrations.metadata_db.CalibrationDB` wraps the SQLAlchemy interface to the database and defines nearly all of the core logic for interfacing with a local (SQLite) or remote (POSTGRESQL) DB.

Key methods:

- :py:meth:`~hispecdrp.calibrations.metadata_db.CalibrationDB.add`

   Add one or many calibrations to the database.

- :py:meth:`~hispecdrp.calibrations.metadata_db.CalibrationDB.query`

   Higher level query method to retrieve calibrations based on a specified calibration type and datetime start/end. The utility of this method will be revisited as the DRP is developed.

- :py:meth:`~hispecdrp.calibrations.metadata_db.CalibrationDB.get_deltas`

   Based on a provided list of calibrations, returns entries which are not present in the DB.

- :py:meth:`~hispecdrp.calibrations.metadata_db.CalibrationDB.calibration_in_db`

   Check if a calibration is in the database.


Remote Database
---------------

Subclass of :py:class:`~hispecdrp.calibrations.metadata_db.CalibrationDB` for remote databases (e.g., PostgreSQL).
No custom logic yet beyond initialization.


Local Database
--------------

Subclass of :py:class:`~hispecdrp.calibrations.metadata_db.CalibrationDB` for local SQLite databases.
The local DB can also be instantiated from a local cache of calibration files without needing to be connected to a remote DB.