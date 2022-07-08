import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class MIBlock:
    def __init__(self, mi_array, length):
        self.length = length
        self.mi_matrix = np.zeros(shape=(length, length))
        self.fill_matrix(mi_array)
        self.eigenvectors = []
        self.eigenvalues = []
        # dictionary that maps the eigenvalues to their eigenvectors
        self.eigenvalues_map = []

    def fill_matrix(self, mi_array):
        """
        Fromats the MI array into a matrix
        :return:
        """
        tri_upper_index = np.triu_indices(self.length, k=1)
        for i in range(len(tri_upper_index[0])):
            self.mi_matrix[tri_upper_index[0][i]][tri_upper_index[1][i]] = mi_array[i]
        self.mi_matrix = self.mi_matrix + self.mi_matrix.T - np.diag(np.diag(self.mi_matrix))
        for i in range(len(self.mi_matrix)):
            self.mi_matrix[i][i] = 1.0

    def remove_adjacent_mi(self, wide):
        """
        Removes MI of fragments that are contiguos to remove bonded effects from the analysis
        :return:
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
        :return:
        """
        # reset old values
        self.eigenvectors = []
        self.eigenvalues = []
        self.eigenvalues_map = []

        eigenvalues, self.eigenvectors = np.linalg.eigh(self.mi_matrix)
        self.eigenvalues_map = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = np.sort(eigenvalues)[::-1]

    def explained_variance(self):
        # To do
        pass

    def average_fragments(self):
        new_mi = np.zeros(shape=(self.length-8, self.length-8))
        # Assumes 4 length fragments
        for i in range(len(new_mi)):
            for j in range(len(new_mi)):
                if i != j:
                    new_sum = np.sum(self.mi_matrix[i:i+8, j:j+8])
                    # count how many squares see itself
                    count = 0
                    for ii in range(i, i+8):
                        for jj in range(j, j+8):
                            if ii == jj:
                                count += 1
                    # susbtract those squares
                    new_sum -= count
                    new_mi[i][j] = new_sum/(64 - count)
                else:
                    new_mi[i][j] = 1.0
        self.mi_matrix = new_mi
        self.length = len(new_mi)


    def plot_mi(self):
        """
        Plots the MI matrix as a hetmap
        :return:
        """
        ax = sns.heatmap(self.mi_matrix,  vmin=0, vmax=1) #, linewidth=0.5, cmap='coolwarm')
        plt.title("2-D Heat Map")
        plt.show()

    def search_highest(self, thresh):
        """
        Prints pairs of fragments whose signal exceed a predefined threshold
        :return:
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
        This is used to remove the noise from pairs of fragments whose correlation is not significative
        :return:
        """
        for i in range(self.length):
            for j in range(self.length):
                if self.mi_matrix[i][j] < point:
                    self.mi_matrix[i][j] = 0














