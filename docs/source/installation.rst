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