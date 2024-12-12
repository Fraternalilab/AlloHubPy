/*
* Encodes a matrix of atoms into SA fragments using the kabsch algorithm.
* The protein is encoded using an sliding window against a library of fragments.
* The algorithm returns an index of the fragment that best fits each window.
*/

#include "encodeframe.h"
#include "kabsch.h"
#include <stdio.h>

void encode_frame(unsigned int n_windows, unsigned int n_fragments, unsigned int f_size, float (*MDframe)[3],
 float (*Fragments)[3], int *Encoding)
{
  // allocating memory
  float MD_fragment[f_size][3];
  float fragment[f_size][3];
  double rmsd[n_fragments];
  // Iterate over the MD frame using an sliding windows of size = f_size
  unsigned int i,j,k,l,m;
  unsigned int f_count = 0;
  for (i = 0; i < n_windows; i++){
     // Initialize the MD fragment to process
     for (j = 0; j < f_size; j++){
         MD_fragment[j][0] = MDframe[j + f_count][0];
         MD_fragment[j][1] = MDframe[j + f_count][1];
         MD_fragment[j][2] = MDframe[j + f_count][2];
     }
     f_count += 1;
     // Iterate over the fragment library
     for (k = 0; k < n_fragments; k++){
         // Initialize fragment
         for (l = 0; l < f_size; l++){
             fragment[l][0] = Fragments[l + k * f_size][0];
             fragment[l][1] = Fragments[l + k * f_size][1];
             fragment[l][2] = Fragments[l + k * f_size][2];
         }
         // Stored rmsd between the MD fragment and library fragments
         rmsd[k] = wrmsd_kabsch(f_size, MD_fragment, fragment);
     }

     // Find the index of the lowest rmsd fragment
     double min_rmsd = rmsd[0];
     unsigned int min_index = 0;
     for (m = 1; m < n_fragments; m++){
         if (rmsd[m] < min_rmsd){
             min_rmsd = rmsd[m];
             min_index = m;
         }
     }

     // Store the index of the lowest rmsd fragment

     Encoding[i] = min_index;
  }
};
