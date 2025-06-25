====================
Calibration Database
====================

Calibration metadata is stored in an SQL based database (DB) to allow for faster parsing of metadata to select an appropriate calibration file. The DB is implemented in `SQLAlchemy ORM <https://www.sqlalchemy.org/>`_ and supports SQLite and PostgreSQL.

ORM
---

The database columns are declared in a standard Python class with additional methods for initialization. The HISPEC Calibrations contain the following fields:

.. csv-table::
   :header: name, datatype, unit, description, required
   :widths: 20, 10, 10, 50, 10

   id,string,None,UUID mechanism TBD,Yes
   filename,string,None,Base filename,Yes
   koa_id,string,None,KOA-assigned identifier (if available) and base filename on KOA,No
   koa_filepath,string,None,Full path to the file on KOA,No
   cal_type,string,None,Calibration type (e.g. dark), Yes
   master,bool,None,Whether this is a master calibration (default: Yes), Yes
   instera,string,None,Instrument era/version, Yes
   spectrograph,string,None,Spectrograph name (BSPEC or RSPEC), Yes
   mjd_start,float,None,Modified Julian date of observation start, Yes
   datetime_obs,string,ISO 8601,ISO datetime of observation start, Yes
   read_mode,string,None,Detector readout mode, Yes
   target_name,string,None,Name of target object, Yes
   object_fiber1,string,None,Object in fiber 1, Yes
   object_fiber2,string,None,Object in fiber 2, Yes
   object_fiber3,string,None,Object in fiber 3, Yes
   drp_version,string,None,DRP software version, Yes
   note,string,None,Optional user-defined notes, Yes
   last_updated,string,ISO 8601,Timestamp of last update (auto-managed), Yes


CalibrationDB
-------------

The base class :py:class:`~hispecdrp.calibrations.metadata_db.CalibrationDB` wraps the SQLAlchemy interface to a database (i.e. the SQLAlchemy engine and session). It defines the core logic for interfacing with a local (SQLite) or remote (PostgreSQL) DB. Two subclasses are provided for managing the local and remote databases, respectively.

Key methods:

- :py:meth:`~hispecdrp.calibrations.metadata_database.CalibrationDB.add`

   Add one or many ``CalibrationORM``'s to the database.

- :py:meth:`~hispecdrp.calibrations.metadata_database.CalibrationDB.query`

   Higher level query method to retrieve ``CalibrationORM``'s based on a specified calibration type and datetime start/end. The utility of this method will be revisited as the DRP is developed.

- :py:meth:`~hispecdrp.calibrations.metadata_database.CalibrationDB.get_last_updated`

   Retrieve the most recent last_updated timestamp.


Remote Database
---------------

Subclass of :py:class:`~hispecdrp.calibrations.metadata_db.CalibrationDB` for remote databases (e.g., PostgreSQL).
No custom logic yet beyond initialization.


Local Database
--------------

Subclass of :py:class:`~hispecdrp.calibrations.metadata_db.CalibrationDB` for local SQLite databases.
The local DB can also be instantiated from a local cache of calibration files without needing to be connected to a remote DB.