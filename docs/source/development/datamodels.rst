Data Models
===========

Data models handle the reading, writing, manipulation, and validation of data and metadata from FITS files. The base class for all HISPEC and PARVI data models is :py:class:`~hispecdrp.datamodels.base_model.DataModel`.


Extensions
++++++++++

The ``EXTENSIONS`` class attribute specifies the different HDUs (Header Data Units) contained in the FITS files. It is a dictionary where the keys are the HDU names and the values contain basic information like the HDU type and column names for tables. Data models do not need to specifiy entries specified in a parent class.

Extensions are automatically inherited by child classes; new models must only specify new extensions. The base class :py:class:`~hispecdrp.datamodels.base_model.DataModel` specifies 4 extensions that are common to all data models:

1. **PRIMARY** - The primary HDU, which contains all of the metadata but no data.
2. **RECEIPT** - A binary table HDU that contains a record of the primitives that have been applied to this data product, and the parameters used for each primitive.
3. **CALIBRATIONS** - A binary table HDU that contains a record of the calibration files that have been applied to this data product.
4. **DATA_ANCESTORS** - A binary table HDU that contains a record of the input data products that were used to create this data product (for example, when coadding data products), and a subset of the metadata for each input data product.

For a given datamodel, extensions can be accessed using standard Python attribute access: ``model.<extension_name>`` and is case-insensitive.

Metadata
++++++++

Metadata for each datamodel is specified with `JSON Schema <https://json-schema.org/>`_ using YAML syntax.

The **live** metadata values reside in the datamodel's ``meta`` attribute, which also supports attribute-style access: ``model.meta.<attr>``. Trying to set an attribute to an invalid value will raise a ``ValueError``.


Implement New Data Model
------------------------

1. Subclass DataModel
~~~~~~~~~~~~~~~~~~~~~

To implement a new data model, create a new class that inherits from the appropriate base type:

- :py:class:`~hispecdrp.datamodels.base_model.DataModel` - Base class for all data models. Inherit from for most data non-calibration models.

- :py:class:`~hispecdrp.datamodels.base_model.CalibrationModel` - Base class for all calibration data models. Inherit from if implementing a new calibration model.

2. EXTENSIONS
~~~~~~~~~~~~~

Specify the ``EXTENSIONS`` class attribute for any new extensions that are not already specified in the parent class. Each extension must specify the HDU type (e.g. ``fits.ImageHDU`` or ``fits.BinTableHDU``) and for tables, the column names and types.

Example:

.. code-block:: python

    class MyDataModel(DataModel):
        """A data model for my data products."""

        EXTENSIONS = {
            'MYDATA': {
                'hdu_type' : fits.ImageHDU,
                'dtype' : np.float32,
            }
            'MYDATA_ERR': {
                'hdu_type' : fits.ImageHDU,
                'dtype' : np.float32,
            }
            'MYDATA_TABLE': {
                'hdu_type' : fits.BinTableHDU,
                'columns' : {
                    'col_name1': np.float32,
                    'col_name2': np.int16,
                    'col_name3': str,
                }
            }
        }

**Note**: There are several helpful methods and pre-defined extensions within :py:mod:`~hispecdrp.datamodels.extensions`.

**Note**: If the datamodel's extensions are dependent on the number of fibers, then the extensions can be generated dynamically within the datamodel's ``_get_extensions`` method, which is called during initialization after the instrument configuration is determined. By default, this method just returns the extensions at ``cls.EXTENSIONS``. If the extensions depend on the number of fibers, then this method can be overridden to generate the extensions dynamically based on the instrument configuration. Access ``self.config.num_fibers`` to get the number of fibers from the instrument configuration:

For example, the :py:class:`~hispecdrp.datamodels.level2.L2Model` data extensions are specified as:

.. code-block:: python

    class MyDataModel(DataModel):
        ...

        def _get_extensions(self) -> dict:
            ext = self.EXTENSIONS.copy()
            ext.update(_create_l2_extensions(self.config.num_fibers))
            return ext

where ``_create_l2_extensions`` is a helper function that generates the extensions based on the number of fibers.


3. Data Level Suffix
~~~~~~~~~~~~~~~~~~~~

For **non-calibration models**, define the ``data_level_suffix`` *property*, which returns the suffix used in filenames to clearly identify this datamodel type when viewing files. See :py:meth:`~hispecdrp.datamodels.base_model.DataModel._generate_filename` for more details. For example, level 0 data products use the suffix `rate`:

.. code-block:: python

    class MyDataModel(DataModel):
        ...

        @property
        def data_level_suffix(self) -> str:
            return 'rate'

4. Calibration Models
~~~~~~~~~~~~~~~~~~~~~

**Calibration models** must also specify the ``CAL_TYPE`` attribute: ``CAL_TYPE`` (string) specifies the type of calibration for calibration lookups. The string should be lowercase, and use underscores to separate words if necessary (e.g. ``dark``, ``lfc_wavesol``).


5. New Metadata
~~~~~~~~~~~~~~~

If the model necessitates adding new metadata fields, add an entry to the appropriate metadata schema file. If a new metadata schema file is needed, add a new YAML file to the ``hispecdrp/datamodels/metadata_schemas/`` directory, and update the appropriate file that merges them. Currently, this is ``hispecdrp/datamodels/metadata_schemas/hispec_metadata_schema.yaml``.

Each entry can contain the following schema fields:

- **title** (string) - Human-readable title for this metadata field, used as a comment in the FITS header.

- **fits_hdu** (string) - Name of the FITS HDU where this metadata should be stored. Default is 'PRIMARY'.

- **type** (string) - The JSON schema type for this metadata field (string, number, boolean, integer). See `JSON Schema Validation <https://json-schema.org/understanding-json-schema/reference/type.html>`_ for more details.

- **unit** (string) - The physical unit for this metadata field, if applicable. This is appended to the FITS header comment when saving.

- **cal_db** (string or boolean) - If this metadata field should be included in the calibration database, specify the column name to use in the database. If True, use the same name as the property name. If it is a string, include it and use the provided string instead. If False, do not include this metadata field in the calibration database. Default is False.

- **merge_rule** (string) - The rule to use when merging this metadata field from multiple data models. This is used when coadding data products and determining the metadata for the output data product. The options are:
    - 'drop' - Drop the value (i.e. set to None)
    - 'first' - Use the first value in time
    - 'last' - Use the last value in time
    - 'sum' - Sum the values
    - 'max' - Use the max of the values
    - 'min' - Use the min of the values
    - 'mean' - Use the mean of the values
    - 'fiber_state' - Special rule for merging fiber state data such as ``meta.object_fiber#``.

- **include_ancestors** (boolean) - Whether to include this metadata field in the ancestors table when merging data models (e.g. when coadding). Default is False.


    