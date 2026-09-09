Data Handling
=============

Open Model
++++++++++

The method :py:meth:`datamodels.open <hispecdrp.datamodels.utils.open>` reads in a file and constructs the appropriate data model class for that data product as specified by the header key `DATAMODL`.

**Data arrays** are accessed with ``model.<attribute>`` where the attribute name is the extension name (case insensitive).

**Metadata** is accessed with ``model.meta.<attribute>`` where the attribute name is the property name in the metadata schema.

.. code-block:: python

    from hispecdrp import datamodels
    from hispecdrp.utils import download_gdrive_file
    
    # Download and open raw science data
    sci_path = download_gdrive_file('HB.20240924.00000.00_raw.fits')
    model = datamodels.open(sci_path)

    print(model.meta.datetime_obs) # -> '2024-09-24T00:00:00.000'
    print(model.meta.model_type) # -> 'RampModel'
    print(model.meta.exposure_type) # -> 'SPEC'
    print(model.meta.spectrograph) # -> 'BSPEC'
    print(model.meta.instrument_mode) # -> 'YJHK'
    print(model.data.shape) # -> (10, 4096, 4160) = (n_reads, ny, nx)



A file can also be passed to the :py:meth:`constructor <hispecdrp.datamodels.DataModel.__init__>` for the appropriate datamodel class:

.. code-block:: python

    from hispecdrp import datamodels
    model = datamodels.L0Model(sci_path)

Save Model
++++++++++

Saving a datamodel is done with :py:meth:`DataModel.save <hispecdrp.datamodels.DataModel.save>`:


.. code-block:: python

    from hispecdrp import datamodels
    model = datamodels.open(sci_path)
    model.save(sci_path.replace('.fits', '_copy.fits'))


Instantiate DataModel from Scratch
++++++++++++++++++++++++++++++++++

Instead of reading in a datamodel from a file, one can instantiate a datamodel from scratch.

.. code-block:: python

    from hispecdrp import datamodels
    model = datamodels.L0Model(
        rate=np.ones((4096, 4160), dtype=np.float32),
        meta={
            'datetime_obs': '2024-09-24T00:00:00.000',
            'exposure_type': 'DARK'
        }
    )


**NOTE**: When instantiating new datamodels without a file, one should at minimum pass the ``datetime_obs`` keyword to the metadata, or it will adopt the current time as the observation time.


Auto Initialization
-------------------

Datamodels containing array extensions can be auto initialized to a given shape with a default fill value. The array is auto initialized upon the first attribute access.

.. code-block:: python

    from hispecdrp import datamodels
    model = datamodels.L0Model(
        rate=np.ones((4096, 4160), dtype=np.float32),
    )
    print(model.has_extension('rate_err')) # -> True
    print(model.has_extension_value('rate_err')) # -> False
    print(model.rate_err) # -> array of shape (4096, 4160) filled with zeros


Fiber Views
-----------

Many models have extensions enumerated by fiber (e.g, ``model.spec_fib1, model.spec_fib2, model.spec_fib3``). 

Instead of accessing the fiber extensions directly, one can use

.. code-block:: python

    from hispecdrp import datamodels
    model = datamodels.L2Model()
    
    model[1].spec # => model.spec_fib1
    model[2].spec # => model.spec_fib2

    # Alternatively:
    model.spec[1] # => model.spec_fib1

    # Other exts:
    model[1].wave # => model.wave_fib1
    model[1].spec_err # => model.spec_err_fib1


Processing History
++++++++++++++++++

Each datamodel has a **receipt** attribute which is a Table that contains a history of all primitives that have been applied to the datamodel.

* **CLASS** — Primitive class
* **PROC_START_TIME** — Processing start time YYYY-MM-DDTHH:MM:SS.sss
* **PROC_END_TIME** — Processing end time YYYY-MM-DDTHH:MM:SS.sss
* **FILENAME** — Base filename
* **PARAMETERS** — JSON representation of parameters used for processing.
* **DRP_VERSION** — DRP semver
* **DRP_GIT_COMMIT_ID** — Full SHA1 commit ID
* **STATUS** — COMPLETED, FAILED, SKIPPED
* **CONFIG_FILE** — Name of the configuration file used for processing.


Calibration History
+++++++++++++++++++

Each datamodel has a **calibrations** attribute which is a Table that contains a history of all calibrations that have been used within the context of this datamodel. Typically this means the calibration was 'applied' to the datamodel in some way, but it can also be used to track calibrations that were used in the creation of this datamodel, for example using an existing calibration as an initial guess to create a new calibration.

* **CAL_ROLE** — Role of calibration
* **CAL_TYPE** — Type of calibration
* **FILENAME** — Base filename of the calibration
* **ID** — UUID
* **DATETIME_OBS** — Observation date/time of the calibration.
* **LAST_PROCESSED** — When calibration was last processed.
* **TIME_APPLIED** — Time calibration was applied YYYY-MM-DDTHH:MM:SS.sss
* **CLASS** — Primitive class that used the calibration.
* **DRP_VERSION** — DRP semver calibration was processed with


Data Ancestors
++++++++++++++

Each datamodel has a **data_ancestors** attribute which is a Table that contains the datamodels which were used to create this datamodel in contexts such as coadding.

The default columns are:

* **FILENAME** - Base filename of the ancestor datamodel
* **CLASS** - The class of the ancestor
* **TIME_APPLIED** - The time this ancestor was used.
* ... plus other metadata fields as configured by ``include_ancestors`` in the metadata schema.

Additional columns are controlled through the metadata schema attributes ``include_ancestors=true/false``.


Switching between HISPEC & PARVI
++++++++++++++++++++++++++++++++

The HISPEC DRP supports both HISPEC and PARVI data products. By default, the DRP assumes all data products are from HISPEC. Changing to PARVI data products can be done by setting the environment variable `HISPECDRP_INSTRUMENT_NAME` to `PARVI`.

.. code-block:: bash

    export HISPECDRP_INSTRUMENT_NAME=PARVI

or in Python:

.. code-block:: python

    import os
    os.environ['HISPECDRP_INSTRUMENT_NAME'] = 'PARVI'

Each datamodel instance has an attribute ``config`` which is a Python class (not instance) that contains any instrument specific configuration and additional defaults not specified in the metadata schemas.