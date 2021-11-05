======
cxiapi
======


.. image:: https://img.shields.io/pypi/v/cxiapi.svg
        :target: https://pypi.python.org/pypi/cxiapi

.. image:: https://img.shields.io/travis/JunCEEE/cxiapi.svg
        :target: https://travis-ci.com/JunCEEE/cxiapi

.. image:: https://readthedocs.org/projects/cxiapi/badge/?version=latest
        :target: https://cxiapi.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status




An API and analysis toolkit for CXI dataset, including hitfinding, calibration and size distribution.


* Free software: MIT license
* Documentation: https://cxiapi.readthedocs.io.


Installation for EuXFEL
------------------------
1. Load exfel module.

.. code-block:: bash

    $ module load exfel exfel_anaconda3

2. Git clone this repository and install cxiapi.

.. code-block:: bash

    $ git clone https://github.com/JunCEEE/cxiapi.git
    $ cd cxiapi
    $ pip install --user ./
    
Create cxi files for analysis
-----------------------------
Take run 0372 as an example:

.. code-block:: bash

    $ extra-data-make-virtual-cxi /gpfs/exfel/exp/SPB/202130/p900201/raw/r0372 -o r0372.cxi
    
Calculate hit scores and save these frame scores in hits.h5 files 
-----------------------------------------------------------------
After installing the newest cxiapi, get the hit scores from a run (e.g. 372):

.. code-block:: bash

    $ getFrameScores  372

Analyze the hit scores 
------------------------------------------------------------------
Example hit analysis jupyter notebook: https://github.com/JunCEEE/cxiapi/blob/main/examples/hitAnalyzer.ipynb

Some basic API usage examples: https://github.com/JunCEEE/cxiapi/blob/main/examples/newCXI.ipynb

 
Credits
-------
Juncheng E at European XFEL writes the codes.

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
