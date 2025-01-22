Data Models
===========

Data models handle the reading, writing, manipulation, and validation of data and metadata from FITS files. The base class for all data models is defined in `hispecdrp.datamodels.DataModel`.


Specification
-------------

**Extensions**: The `EXTENSIONS` class attribute specify the different HDUs (Header Data Units) contained in the FITS files. The `EXTENSIONS` attribute is a dictionary where the keys are the HDU names and the values are the HDU types. Data models do not need to specifiy entries specified in a parent class.

For example, the `EXTENSIONS` attribute for the base data model class is specified as follows:

.. code-block:: python

    EXTENSIONS = {
        'PRIMARY': {
            'hdu_type' : fits.PrimaryHDU,
            'dtype' : None
        },
        'RECEIPT': {
            'hdu_type' : fits.BinTableHDU,
            'columns' : {
                'Time' : str,
                'Code_Release' : str,
                'Branch_Name' : str,
                'Commit_Hash' : str,
                'Class' : str,
                'Datamodel' : str,
                'Arguments' : str,
                'Status' : str,
            },
        }
    }


**Metadata**: Metadata is stored in the ``meta`` attribute and accessed with attribute-style access. Metadata is validated against a list of schema files specified by the class attribute `HEADER_SCHEMAS`, which are also inherited by default by any subclass. See :doc:`Metadata schema <metadata_schema>` for more information.

.. toctree::

   metadata_schema

The `HEADER_SCHEMAS` attribute for the base data model class is specified as follows:

.. code-block:: python

    HEADER_SCHEMAS = [
        'core.yaml',
        'instrument.yaml',
        'exposure.yaml',
        'target.yaml',
        'drp_steps.yaml',
        'calibration_files.yaml'
    ]

View all YAML schemas `here. <https://github.com/oirlab/HISPEC_DRP/blob/main/hispecdrp/datamodels/schemas/>`_


Data Handling
-------------

The `datamodels.open` method reads in a file and constructs the appropriate data model class for that file as specified by the header key `model_type`.

.. autofunction:: hispecdrp.datamodels.open


Alternatively, a file can be passed to the correct datamodel class:

.. autofunction:: hispecdrp.datamodels.DataModel.__init__


Saving a datamodel can be done with `DataModel.save`:

.. autofunction:: hispecdrp.datamodels.DataModel.save


Examples
--------

Metadata example:

.. code-block:: python

    from hispecdrp import datamodels
    file_path = 'HISPEC/L0/HS_BSPEC_20240924T000000_LVL0_RATE-TELLURICS-FIB1.fits'
    model = datamodels.HISPECL0(file_path)
    print(model.meta.date) # -> '2024-09-24T00:00:00.000'
    print(model.meta.model_type) # -> 'HISPECL0'
    print(model.meta.exposure.exposure_type) # -> 'SCI'
    print(model.meta.instrument.spectrograph )# -> 'BSPEC'
    print(model.meta.instrument.mode) # -> 'YJHK'
    print(model.data.shape) # -> (4096, 4096)

    # Valid
    model.meta.exposure.integration_time = 10.0

    # Invalid
    model.meta.exposure.integration_time = '10.0'