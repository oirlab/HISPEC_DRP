====================
Calibration Database
====================

Calibration metadata is stored in an SQL-based database (DB) to allow for faster parsing of metadata to select an appropriate calibration file. The DB is implemented in `SQLAlchemy ORM <https://www.sqlalchemy.org/>`_ and supports SQLite and PostgreSQL.

See <link to koa_middleware> for more information on the calibration metadata database.

HISPEC ORM
----------

The HISPEC calibration Object Relational Mapping (ORM) contain the following fields:

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


PARVI ORM
---------

The PARVI calibration Object Relational Mapping (ORM) contain the following fields:

.. csv-table::
   :header: name, datatype, unit, description, required
   :widths: 20, 10, 10, 50, 10

   id,string,None,UUID mechanism TBD,Yes
   filename,string,None,Base filename,Yes
   cal_type,string,None,Calibration type (e.g. dark), Yes
   instera,string,None,Instrument era/version, Yes
   spectrograph,string,None,Spectrograph name (BSPEC or RSPEC), Yes
   mjd_start,float,None,Modified Julian date of observation start, Yes
   datetime_obs,string,ISO 8601,ISO datetime of observation start, Yes
   read_mode,string,None,Detector readout mode, Yes
   target_name,string,None,Name of target object, Yes
   object_fiber1,string,None,Object in fiber 1, Yes
   object_fiber2,string,None,Object in fiber 2, Yes
   object_fiber3,string,None,Object in fiber 3, Yes
   object_fiber4,string,None,Object in fiber 4, Yes
   drp_version,string,None,DRP software version, Yes
   note,string,None,Optional user-defined notes, Yes
   last_updated,string,ISO 8601,Timestamp of last update (auto-managed), Yes