import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

class Overlap:

    def __init__(self, traj_list, ev_list=[0,1,2]):
        """
        Initializes the handler for handling fold changes and overlap of mutual information matrices.

        Args:
            traj_list (list of lists of MI objects): Lists of list of mutual information matrices. Each list will be treated as a separated set.
            ev_list (list of ints): Top eigenvectors to use for the overlap calculation.

        """
        self.traj_list = traj_list
        self.overlap_matrix = []
        self.ev_list = ev_list
        self.cluster_labels = []
        self._init_overlap_matrix()


    def _init_overlap_matrix(self):
        """
        Initializes a matrix to hold the overlap data
        """

        length = sum([len(x) for x in self.traj_list])
        self.overlap_matrix = np.zeros(shape=(length, length))


    def fill_overlap_matrix(self, use_clusters=False):
        """
        Fills the overlap matrix with the eigenvector overlap between to eigenspaces
        """
        
        mi_list = []
        for tr in self.traj_list:
            mi_list += tr

        if use_clusters:
            clust_dict = {}
            new_mi_list = []
            # This assumes clusters have been computed
            for i, mi in enumerate(mi_list):
                clust_num = self.cluster_labels[i]
                clust_dict.setdefault(clust_num, [])
                clust_dict[clust_num].append(mi)
            # order clusters based on size
            labels = clust_dict.keys()
            labels = sorted(labels, key=lambda x: len(clust_dict[x]), reverse=True)
            # save new order
            for l in labels:
                new_mi_list += clust_dict[l]
            mi_list = new_mi_list

        for i, mi1 in enumerate(mi_list):
            for j, mi2 in enumerate(mi_list):
                self.overlap_matrix[i][j] = self.eigen_overlap(mi1, mi2, self.ev_list)

    def get_overlap_matrix(self):
        return self.overlap_matrix

    def cluster_overlap(self, distance_threshold=0.99):
        """
        Compute the clusters base don the overlap of the eigenvectors
        Returns the labels of the clusters
        """
        model = AgglomerativeClustering(metric='precomputed', n_clusters=10, distance_threshold=None, linkage='complete').fit(1.0 - self.overlap_matrix)
        self.cluster_labels = model.labels_


    def extract_biggest_clusters(self, num=10):
        # get sizes of clusters
        counts_list = []
        labels_set = set(self.cluster_labels)
        print(labels_set)
        for i in range(len(labels_set)):
            count = len(self.cluster_labels[self.cluster_labels == i])
            counts_list.append((count,i))
            print("Cluster %s has %s memberes" % (i, count))
        # sort them and get top num
        counts_list = sorted(counts_list, key=lambda x: x[0], reverse=True)
        keep_indexes = [x[1] for x in counts_list[:num]]
        # separate clusters by their trajectory
        cluster_dict = {}
        pos_count = 0
        for i,tr in enumerate(self.traj_list):
            cluster_dict.setdefault(i, [])
            for j in range(len(tr)):
                c_i = j + pos_count
                cluster_num = self.cluster_labels[c_i]
                if cluster_num in keep_indexes:
                    cluster_dict[i].append(keep_indexes.index(cluster_num))
                else:
                    cluster_dict[i].append(num)
            pos_count += len(tr)
        # return dict
        return cluster_dict

    def compute_similarities(self):
        """
        Computes the similarity between pairs of simulations
        :return:
        """
        # Computes similarities of all the cuadrants of the overlap matrix
        len_0 = 0  # counts the position of the matrix of the current trajectory
        len_1 = 0  # counts the position of the matrix of the second trajectory
        result_matrix = np.zeros(shape=(len(self.traj_list), len(self.traj_list)))

        for i, tr in enumerate(self.traj_list):
            len_1 = len_0 + len(tr)
            tr0_end = len_1
            # now compute the similarities block1 with itself
            block1_m = self.overlap_matrix[list(range(len_0, tr0_end)), :][:, list(range(len_0, tr0_end))]
            block1 = np.sum(block1_m)/(len(tr)**2)
            for j in range(i+1, len(self.traj_list)):
                tr1_end = len_1 + len(self.traj_list[j])
                # now compute the similarities of block2 with itself
                block2_m = self.overlap_matrix[list(range(len_1, tr1_end)), :][:, list(range(len_1, tr1_end))]
                block2 = np.sum(block2_m) / (len(self.traj_list[j]) ** 2)
                # now compute the similarities between blocks
                block12_m = self.overlap_matrix[list(range(len_0, tr0_end)), :][:, list(range(len_1, tr1_end))]
                block12 = np.sum(block12_m) / (len(self.traj_list[j]) * len(tr))
                similarity = 1 - ((block1 + block2)/2 - block12)
                print("SIMILARITIES BETWEEN TRAJECTORY %s  and %s" % (i, j))
                print(f"\tOverlap between trajectories: {np.round(block12,4)}")
                print(f"\tOverlap of trajectory 1 with itself: {np.round(block1,4)}")
                print(f"\tOverlap of trajectory 2 with itself: {np.round(block2,4)}\n Similary {np.round(similarity,4)}")
                
                # save results in matrix
                result_matrix[i][j] = np.round(block12,4)
                result_matrix[j][j] = np.round(block12,4)
                result_matrix[i][i] = np.round(block1,4)
                result_matrix[j][j] = np.round(block2,4)
                #now update the length of traj 2 start
                len_1 += len(self.traj_list[j])
            # update the length of the traj 1 start
            len_0 += len(tr)

        return result_matrix

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
        
    def separate_groups(self, traj_mapping, splitting):
        # split them by groups
        mapping_dict = {}
        for i,traj in enumerate(self.traj_list):
            if splitting:
                for mi_matrix in traj:
                    mapping_dict.setdefault(traj_mapping[i], []).append(mi_matrix)
            else:
                average_mi = np.sum(traj)/len(traj)
                mapping_dict.setdefault(traj_mapping[i], []).append(average_mi)

        return mapping_dict


    def updown_regulation(self, traj_mapping, splitting=True):

        # Splitt trajectories and MI blocks into their conditions based on traj_mapping
        results = {}
        mapping = self.separate_groups(traj_mapping, splitting)
        trajectory_groups = sorted(mapping.keys())
        # For each condition par fetch trajectories of each group, compute avg and fold change
        for i in range(len(trajectory_groups)):
            for j in range(i+1, len(trajectory_groups)):
                g1 = trajectory_groups[i]
                g2 = trajectory_groups[j]
                # Compute an average for the whole condition 1 
                m1 = np.sum(mapping[g1])/len(mapping[g1])
                # Compute an average for the whole condition 2 
                m2 = np.sum(mapping[g2])/len(mapping[g2])
                # flatten the matrices
                flat_m1_avg = m1.mi_matrix.flatten()
                flat_m2_avg = m2.mi_matrix.flatten()
                flat_m1 = [m.mi_matrix.flatten() for m in mapping[g1]]
                flat_m2 = [m.mi_matrix.flatten() for m in mapping[g2]]
                flat_m1 = np.array(flat_m1)
                flat_m2 = np.array(flat_m2)
                # compute log2 fold change for all fragments pairs
                log2_fold_change = np.log2(flat_m1_avg/flat_m2_avg)
                index_matching = [(ii, jj) for ii in range(m1.mi_matrix.shape[0]) for jj in range(m1.mi_matrix.shape[1])]
                # Filter out symetric cases and fragments with themselves
                indexes_to_remove = []
                indexes_to_keep = []
                for idx, idx_pair in enumerate(index_matching):
                    if idx_pair[0] >= idx_pair[1]:
                        indexes_to_remove.append(idx)
                    else:
                        indexes_to_keep.append(idx)
                index_matching = [index_matching[item] for item in indexes_to_keep]
                log2_fold_change = np.delete(log2_fold_change, indexes_to_remove)
                flat_m1_temp = []
                for m in flat_m1:
                    flat_m1_temp.append(np.delete(m, indexes_to_remove))
                flat_m2_temp = []
                for m in flat_m2:
                    flat_m2_temp.append(np.delete(m, indexes_to_remove))
                flat_m1 = np.array(flat_m1_temp)
                flat_m2 = np.array(flat_m2_temp)
                #Compute statistics and pvalue adjustment
                p_values = []
                # Loop through each fragment
                for f_index in range(len(index_matching)):
                    pair_c1 = flat_m1[:, f_index]
                    pair_c2 = flat_m2[:, f_index]
                    # Perform an independent two-sample t-test
                    _, p_value = ttest_ind(pair_c1, pair_c2)
                    p_values.append(p_value)
                # Create a temporal holder for adjusted p values
                adj_p_values = np.full_like(p_values, 0.0)
                # Dataframe to hold the information
                df = pd.DataFrame({
                    "FragmentPairs":index_matching,
                    "log2FoldChange": log2_fold_change,
                    "PValues": p_values,
                    "AdjustedPValues": adj_p_values
                })
                # remove NaNs
                df = df.dropna()
                
                # pvalue adjustment
                _, adj_p_values, _, _ = multipletests(df["PValues"], method='fdr_bh')
                df["AdjustedPValues"] = adj_p_values
                # Remove infinite differences
                df = df[np.isfinite(df["log2FoldChange"])]
                print(df)
                
                results[(i,j)] = df
        return results
            






                






