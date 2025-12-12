Installation
============

1. Clone the repository:

.. code-block:: bash

    git clone https://github.com/oirlab/HISPEC_DRP

2. Create and activate a virtual environment. Python 3.12 is required.

.. code-block:: bash

    conda create -n "hispec" python=3.12
    conda activate hispec

3. Install live version with pip:

.. code-block:: bash

   cd HISPEC_DRP
   pip install -e .

Specifically, the ``-e`` (editable) flag uses the specified directory as the "live" source code, and does not copy it to the usual site-packages directory.

Optionally one can install the test dependencies and run the tests:

.. code-block:: bash

   pip install -e ".[test]"
   pytest -s


By default, testing data is downloaded to `~/.astropy/cache/download/url/`. To specify a directory for the testing data, one can run:

.. code-block:: bash

   pytest -s --datadir=/path/to/cache/dir/

In both cases, the testing data is persistent and **not deleted** after tests are finished.

If you would like to run the pyechelle-based simulations, please install the `pyechelle`_ package as well. 
If using pyechelle, the ``available_models.txt`` file in the ``simulations/hdf/`` directory should be copied to the ``pyechelle`` models directory.
Modifying pyechelle may be necessary to ensure accurate simulations for HISPEC so it is recommended to clone the pyechelle repository directly from Gitlab:

.. code-block:: bash

   git clone https://gitlab.com/Stuermer/pyechelle.git
   cd pyechelle
   pip install -e .

As of December 2025, the following modifications to pyechelle are necessary for HISPEC simulations:

* In ``raytracing.py``, in the ``run()`` function, after ``slit_fun = self._get_slit_function(f)``, set ``slit_fun = None``. This will trigger raytracing for single mode fibers.
* In ``raytracing.py``, in the ``run()`` function, after ``e = SystemEfficiency(...)``, set ``e = None``. This will disable the grating efficiency calculation which is not currently compatible with DRP simulations.

.. _pyechelle: https://gitlab.com/Stuermer/pyechelle