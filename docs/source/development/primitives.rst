Primitives
==========

Overview
--------

Primitives are the algorithmic building blocks of the HISPEC_DRP which specify the processesing for a single processing step. The base class :py:class:`~hispecdrp.primitives.base_primitive.BasePrimitive` specifies a common interface for all primitives. Note that some functionality is inherited from :py:class:`~hispecdrp.processing_component.ProcessingComponent`.


Running a Primitive
-------------------

First instantiate the primitive object and then call its :py:meth:`~hispecdrp.primitives.base_primitive.BasePrimitive.apply` method. This method wraps the main processing of a primitive, catches any exceptions that occur during processing, and manages settings arguments and outputs.

The primary **input** argument to ``apply()`` is called ``input``, which can be anything the primitive expects for input, but must be one object. For most primitives, this is a single DataModel or filepath to a DataModel. Additional keyword arguments can be passed via ``**kwargs`` to ``apply()`` or the primtive's constructor when instantiated. These keyword arguments will override any default or configuration file settings for the primitive.

A primitive must **return** a single object, which can be a single DataModel, a list of DataModels, or a dictionary of DataModels. If additional outputs are needed (i.e., the dark calibration model or file that was used), they can be found with either ``primitive._output[<key>]`` or simply ``primitive.<key>`` if there is no collision with the explicit attributes.

.. code-block:: python
    :caption: Example of running a dark subtraction primitive

    from hispecdrp.primitives import DarkSubtractionPrimitive

    # Instantiate and call the primitive with optional keyword arguments
    prim = DarkSubtractionPrimitive() # Specify additional kwargs here
    dark_subtracted_model = prim.apply('/path/to/dark.fits') # Or here.

    # Access the dark calibration model used (filepath or DataModel depending on configuration)
    dark = prim.dark  # or prim._output['dark']


Primitive Arguments
-------------------

Primitives are configured with arguments that control their behavior. These arguments can be set in three ways, with the following **priority**:

1. **Class-Level Default Arguments**: ``primitive.ARGUMENTS``
   - Arguments defined in the class attribute ``ARGUMENTS`` are the starting values and have the **lowest priority.** The base class :py:class:`~hispecdrp.processing_component.ProcessingComponent` defines a set of default arguments common to all processing components, and :py:class:`~hispecdrp.primitives.base_primitive.BasePrimitive` adds a few additional arguments specific to only primitives.

2. **Configuration Arguments**: ``primitive.config['arguments']``
   - Arguments loaded from a configuration file or a pre-defined configuration dictionary with key ``arguments``. These will override the default args. Configuration files and dictionaries can only be passed to the primitive's constructor.

3. **Keyword Arguments**: ``kwargs``
   - Keyword arguments passed directly to Primitive's constructor or ``apply`` method have the **highest priority.**


Primitive Methods
-----------------

Every primitive must implement the following method:


* .. automethod:: hispecdrp.primitives.base_primitive.BasePrimitive._perform
    :noindex:


Several additional methods are defined in :py:class:`~hispecdrp.processing_component.ProcessingComponent` and :py:class:`~hispecdrp.primitives.base_primitive.BasePrimitive` that can be optionally overridden by subclasses to customize behavior. All have default implementations, and should be overriden with care.


Default Arguments
^^^^^^^^^^^^^^^^^

The base class :py:class:`~hispecdrp.processing_component.ProcessingComponent` defines default arguments in the class attribute ``ARGUMENTS`` along with default values. :py:class:`~hispecdrp.primitives.base_primitive.BasePrimitive` defines a few additional arguments specific to only primitives.


Primitive Calibrations
----------------------

If a primitive needs a calibration file, it should be added to the class attribute ``CALIBRATIONS`` dictionary:

.. code-block:: python

    CALIBRATIONS = {
        '<cal_role>': {
            'selector': DarkSelector,
            'selector_kwargs': {}
        }
    }

In most cases, '<cal_role>' should be identical with the calibration type if it's clear there is exactly only one calibration of that type associated with the science data. If multiple calibrations of the same type are possible, the role should be more descriptive, e.g. 'etalon_before', 'etalon_after'.

Once a primitive is instantiated, an attribute <cal_role> is added to the primitive with value depending on the contents of the optional configuration file.

Logging
-------

Each primitive creates a unique logger instance which is accessible via ``self.log``. The logger will log messages at the level specified ``level``. The log messages can be directed to a file if log_path is provided, and always logs to the console for now.

Primitive Configuration Files
-----------------------------

Primitives can also be configured with config files, but only serializable arguments can be set here. An example configuration for the dark primitive is below. Some values are redundant with the default ARGUMENTS, but are included here to be explicit.

Configuration files can optionally specify the primitive class to run with the ``class`` key. ``class`` can be a:

1. Module + class path name:

    - ``hispecdrp.primitives.dark_subtraction.dark_primitive.DarkSubtractionPrimitive``
    - ``hispecdrp.primitives.DarkSubtractionPrimitive``

2. Class name (e.g. ``DarkSubtractionPrimitive``), in which case it will try to resolve the class using the top level module (``hispecdrp``).

3. Not provided: If the class name is not provided, the desired object must be explicitly constructed with that class's `__init__`.

The utility method :py:meth:`~hispecdrp.utils.config.initialize_from_config` can be used to create an instance of the desired primitive.


.. code-block:: yaml
    :caption: Example primitive configuration file
    
    %YAML 1.2
    ---

    logging:
      level: info
      log_path: None
  
    arguments:
      run: True
      save_result: True
      output_dir: None
      suffix: 'examplesuffix'

    calibrations:
      dark: '/path/to/dark_calibration.fits'


Implementing a Primitive
------------------------

The dark subtraction primitive represents a typical approach for implementing a primitive. It can be found on the `GitHub repository <https://github.com/oirlab/HISPEC_DRP/blob/main/hispecdrp/primitives/dark_subtraction/dark_primitive.py>`_ and is included below for reference.

.. literalinclude:: ../../../hispecdrp/primitives/dark_subtraction/dark_primitive.py

Supporting Calibration(s)
^^^^^^^^^^^^^^^^^^^^^^^^^

If the primitive needs to use a new type of calibration model (as input or output), a new DataModel class should be implemented inside the module :py:mod:`~hispecdrp.datamodels` and added to the model register. See :doc:`Data Models Development <datamodels>`.

If this primitive needs a new type of calibration selector, a new Selector class should be implemented inside the module :py:mod:`~hispecdrp.calibrations.selectors` and added to the selectors register. See :doc:`Calibration Selectors Development <selectors_design>`.