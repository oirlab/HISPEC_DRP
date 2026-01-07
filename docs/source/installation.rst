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

It is possible to run simulations without any additional packages, but if you would like to run the pyechelle-based simulations,
please install the `pyechelle`_ package as well. Currently, HISPEC_DRP simulations depend
on a modified `fork`_ of pyechelle, which should be cloned and installed directly. This fork is not guaranteed to work as expected for other spectrographs. 
The ``available_models.txt`` in the ``pyechelle/models`` directory of pyechelle must contain the names of any .hdf files used for HISPEC simulations 
(included in the fork).

To install pyechelle from the fork, run:

.. code-block:: bash

   git clone https://gitlab.com/abgibbs/pyechelle.git
   cd pyechelle
   pip install -e .

.. _fork: https://gitlab.com/abgibbs/pyechelle
.. _pyechelle: https://gitlab.com/Stuermer/pyechelle