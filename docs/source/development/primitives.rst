==========
Primitives
==========

Primitives are the core processing components of the HISPEC DRP. Each primitive performs a specific data processing task and can be configured with different parameters and calibrations to fully specify the data processing behavior.

Implement New Primitive
-----------------------

1. Subclass BasePrimitive
~~~~~~~~~~~~~~~~~~~~~~~~~

To implement a new primitive (``NewPrimitive``), first create a new Python class that inherits from :py:class:`~hispecdrp.primitives.base_primitive.BasePrimitive`. If formally part of the HISPEC DRP, this class should be added to the `primitives` module in the code base, following the structure of existing primitives.

2. Arguments
~~~~~~~~~~~~

By default, ``NewPrimitive`` will inherit all arguments from :py:class:`~hispecdrp.primitives.base_primitive.BasePrimitive`. It must specify any additional arguments in the class attribute ``ARGUMENTS``.

.. code-block:: python

    class NewPrimitive(BasePrimitive):

        ARGUMENTS = {
            'arg_name': ... # <default_value>
        }

3. Calibrations
~~~~~~~~~~~~~~~

Any required calibrations must be specified in the class attribute ``CALIBRATIONS``. The keys are the *calibration type* (``meta.cal_type``). The values specify how to retrieve the calibration, and must include a 'selector' key which specifies the Selector class to use to retrieve the calibration file. See :doc:`Calibration Selectors Development <selectors>` for information on Calibration selectors. Here, we use the dark calibration as an example.

.. code-block:: python

    class NewPrimitive(BasePrimitive):
    
        CALIBRATIONS = {
            'dark' : {
                
                # The selector class to use to
                # retrieve the calibration file.
                'selector': DarkSelector,

                # Optional kwargs to pass to the
                # selector's constructor
                'selector_kwargs': {}
            }
        }

4. Perform Method
~~~~~~~~~~~~~~~~~

The primitive must implement the required method :py:meth:`~hispecdrp.primitives.base_primitive.BasePrimitive._perform`. This method contains the main processing code for the primitive. It takes in no arguments and returns a dictionary, which must contain at least one key - ``output``. If additional / ancillary outputs are needed (i.e., the dark calibration model or file that was used), they can be found with either ``primitive._output[<key>]`` or simply ``primitive.<key>`` if there is no collision with the explicit attributes.

.. code-block:: python

    class NewPrimitive(BasePrimitive):

        def _perform(self):
            # Main processing code here
            # Primary input is accessed with self.input
            # Access arguments with ``self.<arg_name>``
            ...
            return dict(output=result, extra1=<extra_output1>, extra2=<extra_output2>)

Logging
+++++++

Logging messages is done through the standard Python logging module.

.. code-block:: python

    # Always put this at the top of the file!
    import logging
    logger = logging.getLogger(__name__)

    class NewPrimitive(BasePrimitive):

        def _perform(self):
            logger.info("This is an info message.")
            ...


5. Previews
~~~~~~~~~~~

(Optional) The primitive can optionally implement the method :py:meth:`~hispecdrp.primitives.base_primitive.BasePrimitive._create_preview` to generate a preview of the results if either of the arguments ``save_preview`` or ``show_preview`` is True. Figures should be generated with the Plotly library.

See the :py:class`~hispecdrp.primitives.dark_subtraction.make_dark_primitive.MakeDarkPrimitive` for a current example.

6. Unit tests
~~~~~~~~~~~~~

After implementing a primitive, please also create a corresponding unit test in the testing suite `tests <https://github.com/oirlab/HISPEC_DRP/blob/main/tests>`_ to at minimum validate the basic processing steps within the primitive.

7. Documentation
~~~~~~~~~~~~~~~~

Implement appropriate docstrings following the structure for other primitives.