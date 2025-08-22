Data Models
===========

Data models handle the reading, writing, manipulation, and validation of data and metadata from FITS files. The base class for all data models is :py:class:`~hispecdrp.datamodels.base_model.DataModel`.


Specification
-------------

Extensions
++++++++++

The ``EXTENSIONS`` class attribute specifies the different HDUs (Header Data Units) contained in the FITS files. It is a dictionary where the keys are the HDU names and the values are the HDU types. Data models do not need to specifiy entries specified in a parent class.

For example, the default ``DataModel.EXTENSIONS`` attribute is specified as follows:

.. code-block:: python

    EXTENSIONS = {
        'PRIMARY': {
            'hdu_type' : fits.PrimaryHDU,
        },
        'RECEIPT': {
            'hdu_type' : fits.BinTableHDU,
            'columns' : {
                'PROC_TIME' : str,
                'DRP_VERSION' : str,
                'CLASS' : str,
                'FILENAME' : str,
                'ARGUMENTS' : str,
                'STATUS' : str,
            }
        }
    }


Metadata
++++++++

Metadata is stored in the ``meta`` attribute and accessed with attribute-style access. Metadata is validated against a list of schema files specified by the class attribute ``HEADER_SCHEMAS``, which are also inherited by default by any subclass. See :doc:`Metadata schema <metadata_schema>`.

The ``HEADER_SCHEMAS`` attribute for the base HISPEC data model :py:class:`~hispecdrp.datamodels.hispec.hispec_base.HISPECDataModel` class is simply specified as follows:

.. code-block:: python

    HEADER_SCHEMAS = [
        'hispec_core.yaml',
    ]

**All schemas:**

* `HISPEC. <https://github.com/oirlab/HISPEC_DRP/tree/main/hispecdrp/datamodels/hispec/schemas>`_
* `PARVI. <https://github.com/oirlab/HISPEC_DRP/tree/main/hispecdrp/datamodels/parvi/schemas>`_


Data Handling
-------------

Open Model
++++++++++

The method :py:meth:`~hispecdrp.datamodels.utils.open` reads in a file and constructs the appropriate data model class for that file as specified by the header key `model_type`.


.. code-block:: python

    from hispecdrp import datamodels
    file_path = 'HISPEC/L0/HB.20240924.0000.00_RATE-TELLURICS-FIB1.fits'
    model = datamodels.open(file_path)
    print(model.meta.exposure.datetime_obs) # -> '2024-09-24T00:00:00.000'
    print(model.meta.model_type) # -> 'HISPECL0Model'
    print(model.meta.exposure.exposure_type) # -> 'SCI'
    print(model.meta.instrument.spectrograph )# -> 'BSPEC'
    print(model.meta.instrument.mode) # -> 'YJHK'
    print(model.data.shape) # -> (4096, 4096)

.. autofunction:: hispecdrp.datamodels.open


Alternatively, a file can be passed to the constructor for the appropriate datamodel class:


.. code-block:: python

    from hispecdrp import datamodels
    file_path = 'HISPEC/L0/HB.20240924.0000.00_RATE-TELLURICS-FIB1.fits'
    model = datamodels.HISPECL0Model(file_path)
    print(model.meta.exposure.datetime_obs) # -> '2024-09-24T00:00:00.000'
    print(model.meta.model_type) # -> 'HISPECL0Model'
    print(model.meta.exposure.exposure_type) # -> 'SCI'
    print(model.meta.instrument.spectrograph )# -> 'BSPEC'
    print(model.meta.instrument.mode) # -> 'YJHK'
    print(model.data.shape) # -> (4096, 4096)

.. autofunction:: hispecdrp.datamodels.DataModel.__init__


Save Model
++++++++++

Saving a datamodel can be done with `DataModel.save`:


.. code-block:: python

    from hispecdrp import datamodels
    file_path = 'HISPEC/L0/HB.20240924.0000.00_RATE-TELLURICS-FIB1.fits'
    model = datamodels.open(file_path)
    model.save(file_path.replace('.fits', '_copy.fits'))

.. autofunction:: hispecdrp.datamodels.DataModel.save


Implement New Data Model
------------------------

To implement a new data model, create a new class that inherits from the appropriate base type:

1. :py:class:`~hispecdrp.datamodels.base_model.DataModel` - Base class for all data models. Inherit from if implementing a new instrument.

2. :py:class:`~hispecdrp.datamodels.hispec.hispec_base.HISPECDataModel` - Base class for HISPEC data models, subtype of :py:class:`~hispecdrp.datamodels.base_model.DataModel`. Use for new science based HISPEC models or new abstract HISPEC models.

3. :py:class:`~hispecdrp.datamodels.parvi.parvi_base.PARVIDataModel` - Base class for PARVI data models, subtype of :py:class:`~hispecdrp.datamodels.base_model.DataModel`. Use for new science based PARVI models or new abstract PARVI models.

4. :py:class:`~hispecdrp.datamodels.hispec.hispec_base.HISPECCalibrationModel` - Base class for HISPEC calibration data models, subtype of :py:class:`~hispecdrp.datamodels.hispec.hispec_base.HISPECDataModel`.

5. :py:class:`~hispecdrp.datamodels.parvi.parvi_base.PARVICalibrationModel` - Base class for PARVI calibration data models, subtype of :py:class:`~hispecdrp.datamodels.parvi.parvi_base.PARVIDataModel`.


**Specify additional extensions** in ``EXTENSIONS`` and metadata in ``HEADER_SCHEMAS``.

**Calibration models must also specify the** ``CAL_TYPE`` **attribute.**: ``CAL_TYPE`` (string) specifies the type of calibration for calibration lookups. The string should be lowercase, and use underscores to weparate words if necessary.

**Implement the ``data_level_suffix`` property**, which returns the suffix used in filenames to clearly identify this datamodel type when viewing files. See :py:meth:`~hispecdrp.datamodels.hispec.hispec_base.HISPECDataModel._generate_filename`.

For example, the HISPEC LFC fit parameters model is defined as follows:

.. literalinclude:: ../../../hispecdrp/datamodels/hispec/lfcfitparams.py
   :language: python


Metadata
--------

.. toctree::

   metadata_schema