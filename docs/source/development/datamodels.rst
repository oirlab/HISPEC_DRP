Data Models
===========

Data models handle the reading, writing, manipulation, and validation of data and metadata from FITS files. The base class for all data models is :py:class:`~hispecdrp.datamodels.base_model.DataModel`.


Specification
-------------

Extensions
++++++++++

The ``EXTENSIONS`` class attribute specifies the different HDUs (Header Data Units) contained in the FITS files. It is a dictionary where the keys are the HDU names and the values contain basic information like the HDU type and column names for tables. Data models do not need to specifiy entries specified in a parent class.

For example, the default ``DataModel.EXTENSIONS`` attribute is specified as follows and is automatically inherited for each subtype.

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


Specifying an image HDU:

.. code-block:: python

    EXTENSIONS = {
        'MYDATA': {
            'hdu_type' : fits.ImageHDU,
        }
    }

**In practice, there are many helpful methods for creating these extensions in :py:mod:`~hispecdrp.datamodels.extensions`.**


Metadata
++++++++

Metadata for each datamodel is specified with a `pydantic <https://docs.pydantic.dev/latest/>`_ model. Each entry in the FITS file header (e.g., each FITS header `Card`) corresponds to a pydantic `Field <https://docs.pydantic.dev/latest/concepts/fields/>`_. A special constructor is provided to create pydantic Fields tailored for FITS cards:

.. automethod:: hispecdrp.datamodels.meta_data.FitsField
   :noindex:

A collection of like-fields are specified within a subtype of :py:class:`~hispecdrp.datamodels.meta_data.MetaModel`, which itself is a pydantic `Model <https://docs.pydantic.dev/latest/concepts/models/>`_. Each datamodel must specify the class attribute ``META_MODEL`` (if not specified by a parent type) which is the Python class specifying the final collection of fields for that datamodel. Fields from parent types are automatically adopted, and the final structure is always flat.

The **live** metadata resides in the datamodel's ``meta`` attribute, which supports attribute-style access: ``model.meta.<attr>``. Trying to set an attribute to an invalid value will raise a `pydantic.ValidationError <https://docs.pydantic.dev/latest/usage/models/#model-validation>`_.

.. code-block:: python
    :caption: Example metadata attribute access

    from hispecdrp import datamodels
    file_path = 'HISPEC/L0/HB.20240924.0000.00_RATE-TELLURICS-FIB1.fits'
    model = datamodels.open(file_path)
    print(model.meta.datetime_obs) # -> '2024-09-24T00:00:00.000'

    model.meta.datetime_obs = 1 # Raises ValidationError!

    # How to get information on a field
    print(model.meta.__class__.model_fields['datetime_obs']) # -> <Summary of FieldInfo...>


Declare new MetaModel
^^^^^^^^^^^^^^^^^^^^^

Instrument agnostic MetaModels live in :py:mod:`~hispecdrp.datamodels.meta_data` and provide nearly all metadata fields. Below are the special models declared for HISPEC.

.. code-block:: python

    class HISPECTargetMeta(TargetMeta):
        object_fiber1: str | None = FitsField(
            fits_keyword="OBJFIB1",
            title="Object name in fiber 1"
        )
        object_fiber2: str | None = FitsField(
            fits_keyword="OBJFIB2",
            title="Object name in fiber 2"
        )
        object_fiber3: str | None = FitsField(
            fits_keyword="OBJFIB3",
            title="Object name in fiber 3"
        )

    class HISPECCoreMeta(HISPECTargetMeta, CoreMeta):
        ORM_TABLE_NAME = 'HISPEC'

        koa_id: str | None = FitsField(
            fits_keyword="KOAID",
            title="Keck Observatory Archive (KOA) ID"
        )
        koa_filepath: str | None = FitsField(
            fits_keyword="KOAFPATH",
            title="KOA file path"
        )

    class HISPECCalibrationMeta(HISPECCoreMeta, CalibrationCoreMeta):
        pass


Data Handling
-------------

Open Model
++++++++++

The method :py:meth:`~hispecdrp.datamodels.utils.open` reads in a file and constructs the appropriate data model class for that data product as specified by the header key `DATAMODL`.


.. code-block:: python

    from hispecdrp import datamodels
    file_path = 'HISPEC/L0/HB.20240924.0000.00_RATE-TELLURICS-FIB1.fits'
    model = datamodels.open(file_path)
    print(model.meta.datetime_obs) # -> '2024-09-24T00:00:00.000'
    print(model.meta.model_type) # -> 'HISPECL0Model'
    print(model.meta.exposure_type) # -> 'SCI'
    print(model.meta.spectrograph) # -> 'BSPEC'
    print(model.meta.instrument_mode) # -> 'YJHK'
    print(model.data.shape) # -> (4096, 4096)

.. autofunction:: hispecdrp.datamodels.open


Alternatively, a file can be passed to the constructor for the appropriate datamodel class:


.. code-block:: python

    from hispecdrp import datamodels
    file_path = 'HISPEC/L0/HB.20240924.0000.00_RATE-TELLURICS-FIB1.fits'
    model = datamodels.HISPECL0Model(file_path)
    ...


.. autofunction:: hispecdrp.datamodels.DataModel.__init__


Save Model
++++++++++

Saving a datamodel is done with `DataModel.save`:


.. code-block:: python

    from hispecdrp import datamodels
    file_path = 'HISPEC/L0/HB.20240924.0000.00_RATE-TELLURICS-FIB1.fits'
    model = datamodels.open(file_path)
    model.save(file_path.replace('.fits', '_copy.fits'))

.. autofunction:: hispecdrp.datamodels.DataModel.save


Implement New Data Model
------------------------

1. To implement a new data model, create a new class that inherits from the appropriate base type:

- :py:class:`~hispecdrp.datamodels.base_model.DataModel` - Base class for all data models. Inherit from if implementing a new instrument.

- :py:class:`~hispecdrp.datamodels.base_model.CalibrationModel` - Base class for all calibration data models. Inherit from if implementing a new calibration model.

- :py:class:`~hispecdrp.datamodels.hispec.hispec_base.HISPECDataModel` - Base class for HISPEC data models, subtype of :py:class:`~hispecdrp.datamodels.base_model.DataModel`. Use for new science based HISPEC models or new abstract HISPEC models.

- :py:class:`~hispecdrp.datamodels.parvi.parvi_base.PARVIDataModel` - Base class for PARVI data models, subtype of :py:class:`~hispecdrp.datamodels.base_model.DataModel`. Use for new science based PARVI models or new abstract PARVI models.

- :py:class:`~hispecdrp.datamodels.hispec.hispec_base.HISPECCalibrationModel` - Base class for HISPEC calibration data models, subtype of :py:class:`~hispecdrp.datamodels.hispec.hispec_base.HISPECDataModel` and :py:class:`~hispecdrp.datamodels.base_model.CalibrationModel`.

- :py:class:`~hispecdrp.datamodels.parvi.parvi_base.PARVICalibrationModel` - Base class for PARVI calibration data models, subtype of :py:class:`~hispecdrp.datamodels.parvi.parvi_base.PARVIDataModel` and :py:class:`~hispecdrp.datamodels.base_model.CalibrationModel`.


2. Specify the pydantic metadata model at the attribute ``META_MODEL`` if different from the parent datamodel's ``META_MODEL``.

3. Calibration models must also specify the ``CAL_TYPE`` attribute: ``CAL_TYPE`` (string) specifies the type of calibration for calibration lookups. The string should be lowercase, and use underscores to separate words if necessary (e.g. ``lfc_wavesol``).

4. Define the ``data_level_suffix`` *property*, which returns the suffix used in filenames to clearly identify this datamodel type when viewing files. For example, see :py:meth:`~hispecdrp.datamodels.hispec.hispec_base.HISPECDataModel._generate_filename`.

For example, the HISPEC LFC fit parameters model is defined as follows:

.. literalinclude:: ../../../hispecdrp/datamodels/lfcfitparams.py
   :language: python