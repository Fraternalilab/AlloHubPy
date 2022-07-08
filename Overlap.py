import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class Overlap:

    def __init__(self, traj_list, ergodicity=False, ev_list=[0, 1]):
        self.traj_list = traj_list
        self.overlap_matrix = []
        self.ev_list = ev_list
        if not ergodicity:
            self.init_overlap_matrix()

    def init_overlap_matrix(self):
        """
        Initializes a matrix to hold the overlap data
        :return:
        """
        length = sum([len(x.mi_traj) for x in self.traj_list])
        self.overlap_matrix = np.zeros(shape=(length, length))

    def fill_overlap_matrix(self):
        """
        Fills the overlap matrix with the eigenvector overlap between to eigenspaces
        :return:
        """
        mi_list = []
        for tr in self.traj_list:
            mi_list += tr.mi_traj

        for i, mi1 in enumerate(mi_list):
            for j, mi2 in enumerate(mi_list):
                self.overlap_matrix[i][j] = self.eigen_overlap(mi1, mi2, self.ev_list)

    def plot_overlap(self):
        """
        Plots the overlap matrix as a heatmap
        :return:
        """
        ax = sns.heatmap(self.overlap_matrix,  vmin=0, vmax=1) #, linewidth=0.5, cmap='coolwarm')
        plt.title("2-D Heat Map")
        plt.show()

    def compute_similarities(self):
        """
        Computes the similarity between pairs of simulations
        :return:
        """
        # Computes similarities of all the cuadrants of the overlap matrix
        len_0 = 0  # counts the position of the matrix of the current trajectorie
        len_1 = 0  # counts the position of the matrix of the second trajectorie
        for i, tr in enumerate(self.traj_list):
            len_1 = len_0 + len(tr.mi_traj)
            tr0_end = len_1
            # now compute the similarities block1 with itself
            block1_m = self.overlap_matrix[list(range(len_0, tr0_end)), :][:, list(range(len_0, tr0_end))]
            block1 = np.sum(block1_m)/(len(tr.mi_traj)**2)
            for j in range(i+1, len(self.traj_list)):
                tr1_end = len_1 + len(self.traj_list[j].mi_traj)
                # now compute the similarities of block2 with itself
                block2_m = self.overlap_matrix[list(range(len_1, tr1_end)), :][:, list(range(len_1, tr1_end))]
                block2 = np.sum(block2_m) / (len(self.traj_list[j].mi_traj) ** 2)
                # now compute the similarities between blocks
                block12_m = self.overlap_matrix[list(range(len_0, tr0_end)), :][:, list(range(len_1, tr1_end))]
                block12 = np.sum(block12_m) / (len(self.traj_list[j].mi_traj) * len(tr.mi_traj))
                similarity = 1 - ((block1 + block2)/2 - block12)
                print("SIMILARITIES BETWEEN TRAJECTORY %s  and %s" % (i, j))
                print(block12)
                print(block2)
                print(block1)
                print(similarity)
                #now update the length of traj 2 start
                len_1 += len(self.traj_list[j].mi_traj)
            # update the length of the traj 1 start
            len_0 += len(tr.mi_traj)

    @staticmethod
    def eigen_overlap(mi_obj1, mi_obj2, ev_list):
        """
        Calculates the overlap between two eigensystems
        :param mi_obj1: first MIblock object
        :param mi_obj2: second MIblock object
        :param ev_list: list of indexes of eigenvectors to use
        :return:
        """
        # create list of where the top eigenvectors indexes map to
        evc_list1 = mi_obj1.eigenvalues_map[ev_list]
        evc_list2 = mi_obj2.eigenvalues_map[ev_list]
        # sum of eigen values
        ev_sum = np.sum([mi_obj1.eigenvalues[ev_list], mi_obj2.eigenvalues[ev_list]])
        term_2 = 0
        for i in range(len(ev_list)):
            squared_part = np.sqrt(mi_obj1.eigenvalues[i] * mi_obj2.eigenvalues[ev_list])
            d_part = np.einsum('i,ij->j', mi_obj1.eigenvectors[:, evc_list1[i]], mi_obj2.eigenvectors[:, evc_list2])
            d_part = np.power(d_part, 2)
            term_2 += np.sum(squared_part * d_part)
        """
        # calculation of the second term of the overlap traced
        squared_part = np.sqrt(mi_obj1.eigenvalues[ev_list] * mi_obj2.eigenvalues[ev_list])
        d_part = np.power(np.einsum('ij,ij->j', mi_obj1.eigenvectors[:, evc_list1], mi_obj2.eigenvectors[:, evc_list2]), 2)
        """
        # calculation of overlap
        overlap = (ev_sum - 2 * term_2)/ev_sum
        if overlap > 0:
            return 1 - np.sqrt(overlap)
        else:
            return 1.0





