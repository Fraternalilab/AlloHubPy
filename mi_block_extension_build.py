
from cffi import FFI

ffibuilder = FFI()

ffibuilder.cdef("void wrap_mi_block(int length, int depth, int *Xarray, double *MIarray) ;")

ffibuilder.set_source("_mi_block", """ #include "src/mi_block.h" """, sources=["src/mi_block.c"],
                      include_dirs=["./AlloHubPy", "./AlloHubPy/src"], libraries=["m"])

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)