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
        'calibration_files.yaml',
        'keck_program.yaml',
    ]

View all YAML schemas `here. <https://github.com/oirlab/HISPEC_DRP/blob/main/hispecdrp/datamodels/schemas/>`_


Data Handling
-------------

The `datamodels.open` method reads in a file and constructs the appropriate data model class for that file as specified by the header key `model_type`.

.. autofunction:: hispecdrp.datamodels.open


Alternatively, a file can be passed to the constructor for the appropriate datamodel class:

.. autofunction:: hispecdrp.datamodels.DataModel.__init__


Saving a datamodel can be done with `DataModel.save`:

.. autofunction:: hispecdrp.datamodels.DataModel.save


Examples
--------

Metadata example:

.. code-block:: python

    from hispecdrp import datamodels
    file_path = 'HISPEC/L0/HS_BSPEC_20240924T000000_LVL0_RATE-TELLURICS-FIB1.fits'
    model = datamodels.open(file_path)
    print(model.meta.date_obs) # -> '20240924'
    print(model.meta.utc) # -> '000000' (HHMMSS)
    print(model.meta.model_type) # -> 'HISPECL0Model'
    print(model.meta.exposure.exposure_type) # -> 'SCI'
    print(model.meta.instrument.spectrograph )# -> 'BSPEC'
    print(model.meta.instrument.mode) # -> 'YJHK'
    print(model.data.shape) # -> (4096, 4096)

    # Valid
    model.meta.exposure.integration_time = 10.0

    # Invalid, raises an error!
    model.meta.exposure.integration_time = '10.0'



Implement New DataModel
-----------------------

To implement a new data model, create a new class that inherits from the appropriate base type:

1. :py:class:`~hispecdrp.datamodels.base_model.DataModel`  - Base class for all data models.
2. :py:class:`~hispecdrp.datamodels.hispec_base.HISPECDataModel` - Base class for HISPEC data models, subtype of ``DataModel``.
3. :py:class:`~hispecdrp.datamodels.hispec_calibration_base.HISPECCalibrationModel` - Base class for HISPEC calibration data models, subtype of ``HISPECDataModel``.

All models should specify the two attributes:

1. ``EXTENSIONS`` - A dictionary specifying the HDUs in the FITS file. By default, the class inherits the ``EXTENSIONS`` attribute from the parent class.
2. ``HEADER_SCHEMAS`` - A list of metadata schema files specifying the metadata format. By default, the class inherits the ``HEADER_SCHEMAS`` attribute from the parent class.

Calibration models should also specify the ``CAL_TYPE`` attribute.

The new class should specify the ``EXTENSIONS`` attribute and the ``HEADER_SCHEMAS`` attribute. The ``EXTENSIONS`` attribute should specify the HDUs in the FITS file and the ``HEADER_SCHEMAS`` attribute should specify the metadata schema files.

Lastly, the new class should implement the optional ``level`` property if appropriate for that class, which returns the level of the data model. The level is a string that specifies the level of the data model. If not specified, no level is returned. Currently, this is used to construct filenames (see :py:meth:`~hispecdrp.datamodels.HISPECDataModel._generate_filename`).

For example, a flat model is implemented as follows:

.. code-block:: python

    from astropy.io import fits
    from .hispec_calibration_base import HISPECCalibrationModel
    import numpy as np

    __all__ = ['HISPECFlatModel']

    class HISPECFlatModel(HISPECCalibrationModel):

        EXTENSIONS = {
            'DATA': {
                'hdu_type' : fits.ImageHDU,
                'dtype' : np.float32
            },
            'ERR_ALL': {
                'hdu_type' : fits.ImageHDU,
                'dtype' : np.float32
            },
            'ERR_FLOOR': {
                'hdu_type' : fits.ImageHDU,
                'dtype' : np.float32
            },
            'DQ': {
                'hdu_type' : fits.ImageHDU,
                'dtype' : np.uint32
            }
        }

        CAL_TYPE = 'flat'

        @property
        def level(self):
            return '0'