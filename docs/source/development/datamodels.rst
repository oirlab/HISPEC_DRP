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

Specify the ``EXTENSIONS`` class attribute for any new extensions that are not already specified in the parent class. Build each entry with :func:`~hispecdrp.datamodels.extensions.IMAGE_HDU` or :func:`~hispecdrp.datamodels.extensions.TABLE_HDU` rather than a raw dict -- both return the same extension-info dict shape, but validate their arguments and keep the model's docstring in sync automatically (see below).

Example:

.. code-block:: python

    from .extensions import IMAGE_HDU, TABLE_HDU

    class MyDataModel(DataModel):
        """A data model for my data products."""

        EXTENSIONS = {
            'MYDATA': IMAGE_HDU(
                np.float32,
                shape=('DETECTOR_SIZE_Y', 'DETECTOR_SIZE_X'),
                default=0,
                title='My data.',
                units='DN',
            ),
            'MYDATA_ERR': IMAGE_HDU(np.float32, default=0, title='My data error.', units='DN'),
            'MYDATA_TABLE': TABLE_HDU({
                'col_name1': np.float32,
                'col_name2': np.int16,
                'col_name3': str,
            }),
        }

Key arguments (see each function's docstring for the full list):

- **dtype** / **columns** -- the array dtype (``IMAGE_HDU``) or a ``{column: dtype}`` mapping (``TABLE_HDU``).
- **shape** (``IMAGE_HDU`` only) -- a tuple of literal ints and/or symbolic axis names, matched case-insensitively but written uppercase by convention. Omit it when this extension always matches the model's primary extension's shape (e.g. ``MYDATA_ERR`` above) -- it's filled in automatically at construction time. Valid symbolic names:

    - ``DETECTOR_SIZE_Y`` / ``DETECTOR_SIZE_X`` - from the constructor's ``shape=`` override, else :attr:`~hispecdrp.datamodels.base_model.DataModel.default_detector_size`.
    - ``N_ORDERS`` - from ``meta.order_start``/``meta.order_end``.
    - ``N_FIBERS`` - from ``config.num_fibers``.
    - ``N_READS`` - no independent source; only resolvable when inferred from another extension's already-set real array data using the same symbol.
    - ``ANY`` - never resolvable, by design; the axis is left unconstrained.
- **default** (``IMAGE_HDU`` only) -- fill value used to lazily auto-initialize the array the first time it's accessed, once its shape resolves.
- **enumerate** -- declare the extension once (e.g. ``enumerate='FIB'``) and it automatically expands into one ``<NAME>_FIB{n}`` entry per fiber; no per-model override needed for extensions that depend on the number of fibers.
- **title** / **units** -- documentation only, never read by I/O or auto-init logic. They fill the "Description"/"Units" columns of the ``Extensions`` table that's generated from ``EXTENSIONS`` and spliced into the model's own docstring (see :meth:`~hispecdrp.datamodels.base_model.DataModel.extensions_table`).

**Note**: :py:mod:`~hispecdrp.datamodels.extensions` also has higher-level helpers (e.g. ``create_dataspec_extensions``, ``create_specdq_extensions``) that wrap ``IMAGE_HDU``/``TABLE_HDU`` for common per-fiber extension groups -- check there before writing a new extension by hand.


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