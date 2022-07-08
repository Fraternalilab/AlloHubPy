/*==============================================================================
mi_block : Mutual Information in Trajectory Block
Input is a character matrix: alignment of structural strings from a block
(time-slice) in a trajectory.
Output is the vector of MI values of all column pair combinations in the block.
The code is hard-wired for ASCII values 0-24, representing our alphabet
with characters A-Y.
(C) 2021 Jens Kleinjung, Oriol Gracia i Carmona
==============================================================================*/

#include "mi_block.h"

/*____________________________________________________________________________*/
/* 'Xarray' is a character array of 'size' alignment (characters). */
/* They are cast into a matrix format for the mi_block routine below. */

void wrap_mi_block(int length, int depth, int *Xarray, double *MIarray) {
	unsigned int i, j, rowix, colix;
	/* integer cast of alignment block length */
	const int ilength = length;
	/* integer cast of alignment block depth */
	const int idepth = depth;
	/* size of alignment */
	const int isize = ilength * idepth;
	/* pointer to array of alignment depth (= column) */
	int **X = 0;
	/* alignment matrix */
    X = (int **)malloc(idepth * sizeof(int *));
    for (i = 0; i < idepth; ++ i) {
        X[i] = (int *)malloc(ilength * sizeof(int));
	}
	/* cast R coordinate arrays into X matrix */
	/* expected format: column1, column2, column3, ... */
	for (i = 0; i < isize; ++ i) {
		rowix = i % idepth;
		colix = floor(i / idepth);
		/* X */
		X[rowix][colix] = Xarray[i];
	}
	/* debugging only: print input matrix */
	/*
	for (i = 0; i < idepth; ++ i) {
	  for (j = 0; j < ilength; ++ j) {
	    printf("%d ", X[i][j]);
	  }
	  printf("\n");
	}*/


	/* call MI routine for the alignment block 'X' */
	/* assign resulting vector of pairwise MI values to MIarray */
  mi_block(X, idepth, ilength, MIarray);
  for (i = 0; i < idepth; ++ i) {
    free(X[i]);
	}
	free(X);
}

/*____________________________________________________________________________*/
/* compute MI values for all column pairs in given trajectory block */
void mi_block(int **X, int depth, int length, double *MIarray) {
	unsigned int i, j, k;

	/* pointer to array of alphabet size */
	unsigned int (*charfreq)[25] = 0;
	/* matrix of per-column character frequencies*/
	charfreq = malloc(length * sizeof(unsigned int [25]));

	/* initialise single column character frequencies */
	for (i = 0; i < length; ++ i) {
		for (j = 0; j < 25; ++ j) {
			charfreq[i][j] = 0;
		}
	}
	/* character frequencies for all columns */
	for (i = 0; i < length; ++ i) {
		colcharfreq(X, i, depth, charfreq[i]);
		/* debugging only: print character frequencies */
		/*for (j = 0; j < 25; ++ j) {
			printf("%f ", charfreq[i][j]);
		}
		printf("\n");*/
	}
        
	/* MI values for all column pairs */
	for (i = 0, k = 0; i < length - 1; ++ i) {
		for (j = i + 1; j < length; ++ j) {
			MIarray[k] =
				colpairmi(X, i, j, depth, &(charfreq[i])[0], &(charfreq[j])[0]);
			k++;
		}
	}
	free(charfreq);
}

/*____________________________________________________________________________*/
/* character frequency of input column */
/* this is used as input for MI computation and is performed */
/*   separately here for each column only once */
void colcharfreq(int **X, int colpos, int l_col, unsigned int *charfreq_i) {
	unsigned int i;

	for (i = 0; i < l_col; ++ i) {
		++ charfreq_i[X[i][colpos]];
	}
}


/*____________________________________________________________________________*/
/* MI value between a given pair of input columns */
double colpairmi(int **X, int colpos1, int colpos2, int l_col,
				unsigned int *charfreq1, unsigned int *charfreq2) {
	unsigned int i, j;
	unsigned int allocated = 64;
	const float freq_inc = 1. / (float)l_col;
	float mi_colpair = 0.;
    float entropy = 0.;

	/* character pair counter of dimension [25][25] :
	   AA, AB, AC, ..., YW, YX, YY */
	unsigned int (*charfreq12)[25] = 0;
	/* hash table for counted character pairs */
	unsigned int (*countedpair)[2] = 0;
	unsigned int n_countedpair = 0;

	/* allocate memory */
    charfreq12 = malloc(25 * sizeof(unsigned int[25]));
    countedpair = malloc(allocated * sizeof(unsigned int[2]));

	/* initialise column pair character frequencies */
	for (i = 0; i < 25; ++ i) {
		for (j = 0; j < 25; ++ j) {
			charfreq12[i][j] = 0;
		}
	}

	/* count character pairs; characters are integers from 0 to 24 */
	/* keep a hash table to minimise access time to counted pairs */
	for (i = 0; i < l_col; ++ i) {
		++ charfreq12[X[i][colpos1]][X[i][colpos2]];
		/* register only new character pairs as entry in hash table */
		if (charfreq12[X[i][colpos1]][X[i][colpos2]] == 1) {
			countedpair[n_countedpair][0] = X[i][colpos1];
			countedpair[n_countedpair][1] = X[i][colpos2];
			++ n_countedpair;
			/* allocate additional memory */
			if (n_countedpair == allocated) {
				allocated += 64;
				countedpair = realloc(countedpair, allocated * sizeof(unsigned int[2]));
			}
		}
	}

	/*printf("cols %d:%d, counted pairs %d\n", colpos1, colpos2, n_countedpair);*/

	/*float test_colpair = 0.;*/
	/* compute MI using the single and pair character frequencies computed above */
	for (i = 0, mi_colpair = 0; i < n_countedpair; ++ i) {
		/* the count frequencies are normalised here by multiplication with the
			inverse of the column length, which is a frequency increment per count */
		/*printf("%d %d %u\n", charfreq1[countedpair[i][0]], charfreq1[countedpair[i][0]],
							charfreq12[countedpair[i][0]][countedpair[i][1]]);*/
		/*float p1 = charfreq1[countedpair[i][0]] * freq_inc;
		float p2 = charfreq2[countedpair[i][1]] * freq_inc;
		float p12 = charfreq12[countedpair[i][0]][countedpair[i][1]] * freq_inc;
		printf("%f %f %f\n", p1, p2, p12);*/
		/*test_colpair += p12 * log( p12 / (p1 * p2)) ;*/

		mi_colpair += (charfreq12[countedpair[i][0]][countedpair[i][1]] * freq_inc) *
						log( charfreq12[countedpair[i][0]][countedpair[i][1]] /
						(charfreq1[countedpair[i][0]] *
						 charfreq2[countedpair[i][1]] *
						 freq_inc));
       entropy -= (charfreq12[countedpair[i][0]][countedpair[i][1]] * freq_inc) * 
                   log(charfreq12[countedpair[i][0]][countedpair[i][1]] * freq_inc);
	}

    /* Compute finite size error */
    int bxy = 0;
    int bx = 0;
    int by = 0;
    float error = 0.;
    for (i = 0; i < 25; ++ i) {
        if (charfreq1[i] > 0){++bx;}
        if (charfreq2[i] > 0){++by;}
		for (j = 0; j < 25; ++ j) {
            if (charfreq12[i][j] > 0){++bxy;}
        }
    }
    error = ((bxy - bx - by + 1)/(2*l_col));
    
	/* in bits */
	/*mi_colpair /= log(2);*/

	/*printf("\tMI = %f bits\n", mi_colpair);*/

	/* free memory */
	free(charfreq12);
	free(countedpair);
    mi_colpair = ((mi_colpair - error)/entropy);
    return (mi_colpair > 0. ? mi_colpair : 0.);
}

/*============================================================================*/
