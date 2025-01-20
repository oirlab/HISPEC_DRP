Metadata Schema
===============

Metadata for each datamodel is specified in a list of YAML formatted files.

FITS Header Properties
----------------------

.. code-block:: yaml

  property_name:
    comment: # Optional FITS comment.
    fits_keyword:  # FITS keyword. Required.
    type: # `string`, `None`, `boolean`, `number`, `integer`. JSON schema type. Defaults to `string`.
    default: # Optional default value.
    fits_hdu: # Optional name of HDU for this entry. Defaults to `PRIMARY`.

Examples
--------

.. literalinclude:: ../../hispecdrp/datamodels/schemas/instrument.yaml