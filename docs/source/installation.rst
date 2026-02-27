Installation
============

Python Environment
------------------

It is highly recommended users create a new Python environment with either `venv <https://docs.python.org/3/library/venv.html>`_, `Anaconda <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_, or `uv <https://docs.astral.sh/uv/pip/environments/>`_. The pipeline supports Python versions 3.12 or newer.

To create a **uv** environment specifically for the latest stable release of ``hispecdrp`` (in this example, called **hispec**)::

    uv venv hispec --python 3.13

This will create a fresh Python 3.13 environment.

Once you have created your **uv** environment, activate it::

    source hispec/bin/activate


Install from GitHub
-------------------

1. Clone the repository:

.. code-block:: bash

    git clone https://github.com/oirlab/HISPEC_DRP

2. Install live version with pip:

.. code-block:: bash

   cd HISPEC_DRP
   pip install -e .

The ``-e`` (editable) flag uses the specified package directory as the "live" source code, and does not copy it to the usual site-packages directory.

Configure Calibration Cache
---------------------------

The HISPEC DRP uses the `KOA Middleware <https://oirlab.github.io/KOA_Middleware/>`_ package to manage calibration files.

Required for Remote Database Access
+++++++++++++++++++++++++++++++++++

See the `Calibration configuration <calibrations/configure.html>`_ documentation for instructions on setting up the calibration cache and authenticating with the remote database at Keck. This requires having a Keck Observer account and API token autheticated through the browser.

For most users, once authenticated, it is sufficient to:

1. Set the **KOA_CALIBRATION_CACHE** environment variable to point to a local directory where calibration files will be stored.

2. Install the package `python-dotenv <https://github.com/theskumar/python-dotenv>`_ to manage passwords through environemnt variables.

.. code-block:: bash

   uv pip install python-dotenv

3. Define a file called ``.env`` in the same directory as your project.

If you have **existing calibration files** that you would like to work with, you can use the method :py:meth:`~hispecdrp.calibrations.hispec_store.HISPECCalibrationStore.sync_local_db_from_local_files` to populate the local database with entries for those files.

Also see :doc:`Calibration Examples <quickstart/calibration_examples>`.

See the `Calibration database documentation <calibrations/calibration_database.html>`_ for more details on how the calibration database.


Simulatinon Dependencies
------------------------

It is possible to run simulations without any additional packages, but if you would like to run the pyechelle-based simulations,
please install the `pyechelle`_ package as well. Currently, HISPEC_DRP simulations depend
on a modified `fork`_ of pyechelle, which should be cloned and installed directly. This fork is not guaranteed to work as expected for other spectrographs. 
The ``available_models.txt`` in the ``pyechelle/models`` directory of pyechelle must contain the names of any .hdf files used for HISPEC simulations 
(included in the fork).

To install pyechelle from the fork, run:

.. code-block:: bash

   git clone https://gitlab.com/abgibbs/pyechelle.git
   cd pyechelle
   pip install -e .

.. _fork: https://gitlab.com/abgibbs/pyechelle
.. _pyechelle: https://gitlab.com/Stuermer/pyechelle


Run the unit tests (optional)
-----------------------------

First install the testing dependencies. Quotes are needeed for some shells:

.. code-block:: bash

  pip install -e ".[test]"


By default, testing data is downloaded to `~/.astropy/cache/download/url/`. To specify a directory for the testing data, one can run:

Run the tests with:

.. code-block:: bash

  pytest -s --datadir=/path/to/cache/dir/

One can also set the ``HISPECDRP_TEST_DATA_DIR`` environment variable to specify a persistent location for the testing data.

In both cases, the testing data is persistent and **not deleted** after tests are finished.


Environment Variables
---------------------

The following environment variables can be set to configure the calibration data management:


HISPEC DRP
++++++++++

- **HISPECDRP_INSTRUMENT_NAME** (Optional)
  Instrument name for calibration database. Default: "HISPEC".

- **HISPECDRP_LOG_DIR** (Optional)
  Directory for HISPEC DRP log files. Default: current working directory.

- **HISPECDRP_LOG_LEVEL** (Optional)
  Logging level for HISPEC DRP log files. Default: "INFO". Options are: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".

- **HISPECDRP_TEST_DATA_DIR** (Optional)
  Directory for HISPEC DRP testing data. Default: ``~/.astropy/cache/download/url/`` or as specified by pytest ``--datadir`` option when running tests with pytest. This is a persistent directory and testing data is not deleted after tests are finished.

- **HISPECDRP_LOG_ENABLE_STDOUT** (Optional)
  Enable logging to standard output. Options are: 'true', 'false'.

- **HISPECDRP_LOG_ENABLE_FILE** (Optional)
  Enable logging to file. Options are: 'true', 'false'.


Calibrations
++++++++++++

- **KOA_CALIBRATION_CACHE** (Required)
  Path to cached calibrations directory.

- **KOA_LOCAL_DATABASE_FILENAME** (Optional)
  Local SQLite database filename. Default: ``<instrument_name>_calibrations.db``. Full path is determined by this and the ``KOA_CALIBRATION_CACHE`` environment variable.

- **KOA_USE_CACHED_CALIBRATIONS** (Optional)
  Use cached files ('true' or 'false'). Default: 'true'.

- **KOA_LOCAL_DATABASE_TABLE_NAME** (Optional)
  Local database table name. Default: ``<instrument_name>``

- **KOA_CALIBRATIONS_URL** (Optional)
  Remote database URL. Default: Keck Observer API URL. Default is "https://www3.keck.hawaii.edu/api/calibrations", and will be replaced with the appropriate KOA URL in the future.

- **KOA_CALIBRATION_ORIGIN** (Optional)
  **BEHAVIOR SUBJECT TO CHANGE**
  Behavior for selecting calibrations for processing, and for registering new calibrations to the local database. Options are: "KECK", "LOCAL".

- **KOA_SKIP_CALIBRATION_VERSION** (Optional)
  Skip generation of calibration versions when generating filenames even if KOA_CALIBRATION_CACHE is set. If this is not set, the behavior is determined by KOA_CALIBRATION_CACHE.