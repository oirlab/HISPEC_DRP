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

The class :py:class:`~hispecdrp.calibrations.metadata_db.CalibrationDB` wraps the SQLAlchemy interface to the database.

Key methods:

- :py:meth:`~hispecdrp.calibrations.metadata_db.CalibrationDB.add`
- :py:meth:`~hispecdrp.calibrations.metadata_db.CalibrationDB.query`
- :py:meth:`~hispecdrp.calibrations.metadata_db.CalibrationDB.get_deltas`
- :py:meth:`~hispecdrp.calibrations.metadata_db.CalibrationDB.calibration_in_db`

Remote Database
---------------

Subclass of `CalibrationDB` for remote databases (e.g., PostgreSQL).
No custom logic yet beyond initialization.


Local Database
--------------

Subclass of `CalibrationDB` for local SQLite databases.
No custom logic yet beyond initialization.