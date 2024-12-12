/*
* Encodes a matrix of atoms into SA fragments using the kabsch algorithm.
* The protein is encoded using an sliding window against a library of fragments.
* The algorithm returns an index of the fragment that best fits each window.
*/

#ifndef ENCODEFRAME_H
#define ENCODEFRAME_H


#if defined(__cplusplus)
extern "C" {
#endif

void encode_frame(unsigned int n_windows, unsigned int n_fragments, unsigned int f_size, float (*MDframe)[3],
 float (*Fragments)[3], int *Encoding);


 #if defined(__cplusplus)
}
#endif

#endif
