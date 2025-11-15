Instrument Configuration
========================

Datamodels are not specific to HISPEC or PARVI. All HISPEC and PARVI specific configuration is handled through the DataModel attribute 'config' which is a Python class (not instance) defining any required instrument specific parameters. Users should not need to interact with these classes directly, but they are documented here for reference.

**Classes:**

* :py:class:`~hispecdrp.datamodels.hispec_config.HISPECConfig`
* :py:class:`~hispecdrp.datamodels.parvi_config.PARVIConfig`