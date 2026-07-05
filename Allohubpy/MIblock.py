import numpy as np
from Allohubpy.MI_residue_projection import window_mi_to_residue_mi


class MIBlock:

    
    def __init__(self, mi_array):
        """
        Initializes the handler for handling mutual information matrices.

        Args:
            mi_array (numpy matrix): matrix with shape (fragments, fragments) with the mutual information per fragment.
            length (int): number of fragments.

        """

        self.mi_matrix = mi_array
        self.length = mi_array.shape[0]
        self.eigenvectors = []
        self.eigenvalues = []
        # dictionary that maps the eigenvalues to their eigenvectors
        self.eigenvalues_map = []


    def __add__(self, other):
        """
        Magic method to add MI matrices.

        Args:
            other (MIBlock object or float): MI matrix or number to add to the mi matrix.

        Returns:
            MIblock object with a mi matrix that is the sum of the two provided mi matrices or numbers.

        Raises:
            TypeError if the provided objects can't be added together

        """

        if isinstance(other, MIBlock):
            # Add the matrices
            new_matrix = self.mi_matrix + other.mi_matrix
            # Return a new instance with the resulting matrix
        else:
            try:
                new_matrix = self.mi_matrix + other
            except:
                raise TypeError("Both operands can't be added together")

        return MIBlock(new_matrix)
    

    def __mul__(self, other):
        """
        Magic method to multiply MI matrices.

        Args:
            other (MIBlock object or float): MI matrix to multiply with the mi matrix.

        Returns:
            MIblock object with a mi matrix that is the Element-wise Multiplication of the two provided mi matrices.

        Raises:
            TypeError if the provided objects can't be multiplied together

        """

        try:
            new_matrix = self.mi_matrix * other
        except:
            raise TypeError("Both operands can't be multiplied together")
        return MIBlock(new_matrix)


    def __truediv__(self, other):
        """
        Magic method to divide MI matrices.

        Args:
            other (MIBlock object or float): MI matrix to divide with the mi matrix.

        Returns:
            MIblock object with a mi matrix that is the Element-wise Division of the two provided mi matrices.

        Raises:
            TypeError if the provided objects can't be divided together

        """

        try:
            new_matrix = self.mi_matrix / other
        except:
            raise TypeError("Both operands can't be divided together")
        return MIBlock(new_matrix)


    def get_mi_matrix(self):
        """
        Extracts the mi matrix.

        Returns:
            numpy matrix: matrix of shape (fragments, fragments) with the mutual information
        """

        return self.mi_matrix


    def project_to_residues(self, fragment_size=4, **kwargs):
        """
        Projects the fragment-space MI matrix back to residue space.

        The MI matrix is computed in SA fragment space, which is (fragment_size - 1)
        positions shorter than the residue sequence. This maps the (M x M) fragment
        matrix onto an (N_res x N_res) residue-level matrix, where
        N_res = M + fragment_size - 1.

        Args:
            fragment_size (int): Number of residues per SA fragment (4 for M32K25).
            **kwargs: Extra options forwarded to window_mi_to_residue_mi, e.g.
                      mask_overlapping_windows, coverage_normalize, overlap_mask_distance.

        Returns:
            MIBlock object holding the residue-level MI matrix of shape (N_res, N_res).
        """

        mi_res = window_mi_to_residue_mi(self.mi_matrix, window_len=fragment_size, **kwargs)
        return MIBlock(mi_res)


    def remove_adjacent_mi(self, wide):
        """
        Removes MI of fragments that are contiguous to remove bonded effects from the analysis

        Args:
            wide (int): number of fragments around each fragment that need to be set to 0.
        """

        for i in range(len(self.mi_matrix)):
            for j in range(wide):
                up_val = i + j
                down_val = i - j
                if down_val >= 0:
                    self.mi_matrix[i][down_val] = 0.0
                    self.mi_matrix[down_val][i] = 0.0
                if up_val < len(self.mi_matrix):
                    self.mi_matrix[i][up_val] = 0.0
                    self.mi_matrix[up_val][i] = 0.0


    def compute_eigensystem(self):
        """
        Computes the eigensystem for a given MI matrix
        """

        # reset old values
        self.eigenvectors = []
        self.eigenvalues = []
        self.eigenvalues_map = []

        #eigenvalues, self.eigenvectors = scipy.sparse.linalg.eigs(self.mi_matrix)
        eigenvalues, self.eigenvectors = np.linalg.eigh(self.mi_matrix)
        self.eigenvalues_map = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = np.sort(eigenvalues)[::-1]


    def search_highest(self, thresh):
        """
        Prints pairs of fragments whose signal exceed a predefined threshold
        """

        tri_upper_index = np.triu_indices(len(self.mi_matrix), k=1)
        for i in range(len(tri_upper_index[0])):
                j = tri_upper_index[0][i]
                k = tri_upper_index[1][i]
                if self.mi_matrix[j,k] > thresh:
                    print("%s   %s   is   %s" % (j,k,self.mi_matrix[j,k]))


    def remove_low(self, point):
        """
        Removes Mi signal of all the points bellow a given threshold.
        This is used to remove the noise from pairs of fragments whose correlation is not significant

        Args:
            point (float): minimum number to keep in the mi matrix.
        """

        for i in range(self.length):
            for j in range(self.length):
                if self.mi_matrix[i][j] < point:
                    self.mi_matrix[i][j] = 0














