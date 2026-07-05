from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Package metadata lives in pyproject.toml. This file only carries the build
# logic that PEP 621 metadata cannot express: the Cython extension and the
# cffi extension modules.

# Define the Cython extension module
extensions = [
    Extension(
        name="Allohubpy.Allohubpy_cython",       # Module name
        sources=["Allohubpy/Allohubpy_cython.pyx"],
        include_dirs=[np.get_include()],
        language="c"
    )
]

setup(
    cffi_modules=["Allohubpy/src/kabsch_extension_build.py:ffibuilder",
                  "Allohubpy/src/encodeframe_extension_build.py:ffibuilder"],
    ext_modules=cythonize(extensions, language_level=3),
)
