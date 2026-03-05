Data Handling
-------------

Open Model
++++++++++

The method :py:meth:`~hispecdrp.datamodels.utils.open` reads in a file and constructs the appropriate data model class for that data product as specified by the header key `DATAMODL`.

**Data arrays** are accessed with ``model.<attribute>`` where the attribute name is the extension name (case insensitive).

**Metadata** is accessed with ``model.meta.<attribute>`` where the attribute name is the property name in the metadata schema.

.. code-block:: python

    from hispecdrp import datamodels
    from hispecdrp.utils import download_gdrive_file
    sci_path = download_gdrive_file('HB.20231006.00000.00_rate-TELLURICS-FIB1.fits')
    model = datamodels.open(sci_path)
    print(model.meta.datetime_obs) # -> '2023-10-06T00:00:00.000'
    print(model.meta.model_type) # -> 'L0Model'
    print(model.meta.exposure_type) # -> 'SPEC'
    print(model.meta.spectrograph) # -> 'BSPEC'
    print(model.meta.instrument_mode) # -> 'YJHK'
    print(model.data.shape) # -> (4096, 4160)

.. autofunction:: hispecdrp.datamodels.open


A file can alose be passed to the constructor for the appropriate datamodel class:

.. code-block:: python

    from hispecdrp import datamodels
    model = datamodels.HISPECL0Model(sci_path)
    ...

.. autofunction:: hispecdrp.datamodels.DataModel.__init__

Save Model
++++++++++

Saving a datamodel is done with `DataModel.save`:


.. code-block:: python

    from hispecdrp import datamodels
    model = datamodels.open(sci_path)
    model.save(sci_path.replace('.fits', '_copy.fits'))

.. autofunction:: hispecdrp.datamodels.DataModel.save


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


**NOTE**: When instantiating new datamodels without a file, one should at minimum pass the ``datetime_obs`` keyword to the metadata.

Processing History
++++++++++++++++++

Each datamodel has a **receipt** attribute which is a Table that contains a history of all primitives that have been applied to the datamodel.

* **CLASS** — Primitive class
* **PROC_START_TIME** — Processing start time YYYY-MM-DDTHH:MM:SS.sss
* **PROC_END_TIME** — Processing end time YYYY-MM-DDTHH:MM:SS.sss
* **FILENAME** — Base filename
* **ARGUMENTS** — JSON representation of arguments
* **DRP_VERSION** — DRP semver
* **DRP_GIT_COMMIT_ID** — Full SHA1 commit ID
* **STATUS** — COMPLETED, FAILED, SKIPPED


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