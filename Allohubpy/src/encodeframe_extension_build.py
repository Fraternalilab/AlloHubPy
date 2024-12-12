from cffi import FFI
import os

ffibuilder = FFI()

ffibuilder.cdef("void encode_frame(unsigned int n_windows, unsigned int n_fragments, unsigned int f_size, float (*MDframe)[3], float (*Fragments)[3], int *Encoding);")

src_dir = os.path.abspath("Allohubpy/src")
ffibuilder.set_source(
    "Allohubpy._encodeframe",
    """ #include "encodeframe.h" """,
    sources=[os.path.join(src_dir, "kabsch.c"), os.path.join(src_dir, "encodeframe.c")],
    include_dirs=[src_dir],
    libraries=["m", "gsl"],
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)