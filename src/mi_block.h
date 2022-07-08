/*==============================================================================
mi_block : Mutual Information in Trajectory Block
Input is character matrix: alignment of structural strings from a block
(time-slice) in a trajectory.
Output is the vector of MI values of all column pair combinations in the block.
(C) 2021 Jens Kleinjung, 2022 Oriol Gracia i Carmona

==============================================================================*/

#ifndef MIBLOCK_H
#define MIBLOCK_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(__cplusplus)
extern "C" {
#endif

void wrap_mi_block(int length, int depth, int *Xarray, double *MIarray);
void mi_block(int **X, int depth, int length, double *MIarray);
void colcharfreq(int **X, int colpos, int l_incol, unsigned int *charfreq_i);
double colpairmi(int **X, int colpos1, int colpos2, int l_col,
					unsigned int *charfreq1, unsigned int *charfreq2);

#if defined(__cplusplus)
}
#endif

#endif