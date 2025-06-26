Metadata Schema
===============

Metadata for each datamodel is specified in a list of YAML formatted files.

FITS Header Properties
----------------------

.. code-block:: yaml
  
   property_name:
     keyword:  # FITS keyword. Required.
     comment: # Optional FITS comment. Defaults to an empty string.
     fits_hdu: # Optional name of HDU for this entry. Defaults to `PRIMARY`.
     datatype: # `string`, `None`, `boolean`, `number`, `integer`. JSON schema type. Defaults to `string`.
     unit: # Optional units for this entry. Defaults to `None`.
     default: # Optional default value.

Examples
--------

Below is the HISPEC

**hispec_core.yaml**

.. literalinclude:: ../../../hispecdrp/datamodels/hispec/schemas/hispec_core.yaml
