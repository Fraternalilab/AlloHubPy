from cffi import FFI
import os

ffibuilder = FFI()

ffibuilder.cdef("double wrmsd_kabsch(unsigned int size,  float (*Xarray)[3], float (*Yarray)[3]);")

#src_dir = os.path.abspath("Allohubpy/src")
src_dir = os.path.join("Allohubpy", "src")
ffibuilder.set_source(
    "Allohubpy._kabsch",
    """#include "kabsch.h" """,
    sources=[os.path.join(src_dir, "kabsch.c")],
    include_dirs=[src_dir],
    libraries=["m", "gsl"],
)


if __name__ == "__main__":
    ffibuilder.compile(verbose=True)