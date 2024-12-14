# Allohubpy
Allohubpy is a python package for the detection and charectarization of allosteric signals using a information theoric approach. 

The method captures local conformational changes associated with global motions from molecular dynamics simulations through the use of a Structural Alphabet, which simplifies the complexity of the Cartesian space by reducing the dimensionality down to a string of encoded fragments. These encoded fragments can then be used to compute the shanon entropy, mutual information between positions and build networks of correlated motions.

The folder notebooks contains examples for how to run the package with some sample data.

## Installation

The package can be installed through pip with pip install Allohubpy.

Alternatively, one can compile the required code by running python setup.py build_ext --inplace and manualy adding the package to the PYTHONPATH.