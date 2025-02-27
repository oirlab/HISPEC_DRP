Installation
============

1. Clone the repository:

.. code-block:: bash

    git clone https://github.com/oirlab/HISPEC_DRP

2. Create and activate a virtual environment. Anaconda and Python 3.12 are recommended, but not required. The minimum tested version is *currently* Python 3.11.

.. code-block:: bash

    conda create -n "hispec_drp_env" python=3.12
    conda activate hispec_drp_env

3. Install live version with pip:

.. code-block:: bash

   cd HISPEC_DRP
   pip install -e .

Specifically, the ``-e`` (editable) uses the specified directory as the "live" source code, and does not copy it to the usual site-packages directory.