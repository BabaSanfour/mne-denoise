Development workflow
====================

Contributions are welcome! The guidelines below mirror the wider MNE ecosystem.

Environment
-----------

.. code-block:: console

   git clone https://github.com/snesmaeili/mne-denoise.git
   cd mne-denoise
   pip install -e .[dev,mne,docs]
   pre-commit install

Quality checks
--------------

.. code-block:: console

   pre-commit run --all-files
   pytest
   make -C docs html

Communication
-------------

- File issues and pull requests on GitHub.
- Review the `contributing guide <../CONTRIBUTING.md>`_ for coding standards and governance.
- Adhere to the `Code of Conduct <../CODE_OF_CONDUCT.md>`_.
