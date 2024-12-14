from setuptools import setup, Extension
from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy as np 
import os

# Define the Cython extension module
extensions = [
    Extension(
        name="Allohubpy.Allohubpy_cython",       # Module name
        sources=["Allohubpy/Allohubpy_cython.pyx"],  
        include_dirs=[np.get_include()],    
        language="c",
        language_level=3,
    )
]

# Setup function
setup(
    name="Allohubpy",
    version="1.0",    
    author="Oriol",
    author_email="o.carmona@ucl.ac.uk",
    python_requires=">=3.10",
    cffi_modules=["Allohubpy/src/kabsch_extension_build.py:ffibuilder","Allohubpy/src/encodeframe_extension_build.py:ffibuilder"],
    description="Allostery signal detection analysis using a information theory framework.",
    packages=find_packages(),
    ext_modules=cythonize(extensions),
    install_requires=["numpy", "pandas", "seaborn", "matplotlib", "scipy", "statsmodels", "networkx", "mdtraj", "fair-esm", "torch", "scipy", "cffi", "mini3di"],           # Other dependencies
    zip_safe=False,
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Fraternalilab/AlloHubPy"
)
