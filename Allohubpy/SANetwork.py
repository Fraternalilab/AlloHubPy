import numpy as np
import networkx as nx
import copy
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests
import pandas as pd

class SANetWork:

    def __init__(self, traj_list, distance_limit):
        """
        Initializes the handler for the fragment network and computes the mean MI matrix.

        Args:
            traj_list (array): List of MIblock objects.
            distance_limit (float): Maximum distance at which two fragments are considered in contact. 
                                    Fragment distances are estimated based on the C alpha of their first residue

        """

        self.traj_list = traj_list
        self.distance_limit = distance_limit
        self.distance = []
        self.mean_mi = np.mean(self.traj_list)
        self.graph = None


    def _process_distance(self, fragment_size):
        """
        Adjustes the distance matrix to match the size reduction caused by the encoding into fragments.

        Args:
            fragment_size (int): Number of amino acids that make one fragment. The final length of the sequence is N - (fragment_size - 1)
        """

        elements = len(self.distance)
        target = elements - fragment_size + 1
        self.distance = self.distance[:target, :target]


    def read_distance_files(self, distance_file, fragment_size):
        """
        Reads a txt file containing a matrix of distances.

        Args:
            distance_file (str): Path to the file with the distance matrix.
            fragment_size (int): Number of amino acids that make one fragment.
        """

        self.distance = np.loadtxt(distance_file, dtype=float)
        self._process_distance(fragment_size)   


    def set_distance_matrix(self, matrix, fragment_size):
        """
        Sets and processes a distance matrix

        Args:
            matrix (np.array): Numpy matrix with the distances between C alphas
            fragment_size (int): Number of amino acids that make one fragment.
        """
        self.distance = matrix
        self._process_distance(fragment_size)


    def create_graph(self, threshold):
        """
        Creates a graph with the nodes = the fragments ids and the edges = 1 - mutual information.
        Connections further than self.distance_limit or that are not significant given the selected alpha are removed.

        Args:
            threshold (float): top % of mi signal to keep
        """

        weights = 1 - self.mean_mi.mi_matrix
        self.graph = nx.Graph()
        node_ids = [i for i in range(self.mean_mi.length)]
        # Add nodes
        self.graph.add_nodes_from(node_ids)

        # Create indexes for each pair interaction
        index_matching = [(ii, jj) for ii in range(self.mean_mi.mi_matrix.shape[0]) 
                          for jj in range(self.mean_mi.mi_matrix.shape[1])]

        mi_distribution = []

        # Filter out symetric cases and fragments with themselves
        indexes_to_keep = []
        for idx, idx_pair in enumerate(index_matching):
            if idx_pair[0] < idx_pair[1]:
                indexes_to_keep.append(idx)
                mi_distribution.append(self.mean_mi.mi_matrix[idx_pair[0]][idx_pair[1]])

        index_matching = [index_matching[item] for item in indexes_to_keep]
        index_matching = np.array(index_matching)

        # create a numpy array 
        mi_distribution = np.array(mi_distribution)

        # Compute the Nth percentil
        threshold_p = np.percentile(mi_distribution, 100 - threshold)  # 80th percentile

        # Find indices of points in the top percentile chosen
        indices_f = np.where(mi_distribution >= threshold_p)[0]  # Indices of elements meeting the condition

        # Add edges with weights only for nodes that are in contact
        for i, j in index_matching[indices_f]:
            if self.distance[i][j] < self.distance_limit:
                self.graph.add_edge(i, j, weight=weights[i,j])


    def get_graph(self):
        """
        Extracts the graph.

        Returns:
            graph (network x object): Graph of the fragments
        """

        return self.graph
    
    
    def set_graph(self, graph):
        """
        Sets the graph to an existing network x graph.

        Args:
            graph (network x object): Graph of the fragments
        """   

        self.graph = graph


    def compute_centrality(self):
        """
        Compute the eigenvector centrality of the graph.

        Returns:
            df (pandas DataFrame): Data frame with columns ['fragments', 'Centrality']
        """

        eigen_centrality = nx.eigenvector_centrality(self.graph, max_iter=500)
        df = pd.DataFrame(list(eigen_centrality.items()), columns=['fragments', 'Centrality'])
        return df
    
    
    def identify_preferential_connections(self, start_fragments, end_fragments):
        """
        Compute shortest paths from selected start fragments to the target end fragments.

        Args:
            start_fragments (list ints): List of fragments indexes for the starting points.
            end_fragments (list ints): List of fragments indexes for the ending points.

        Returns:
            subgraph (networkx graph): Graph containing only the selected start and end fragments and the nodes belonging to the shortest paths.
            shortest_paths (list of list): List of shortest path for each starting fragment to each ending fragment.
            shortest_distances (list of floats): List of the shortest path distances for each starting fragment to each ending fragment. 
            z_scores (list of floats): List of the z scores for each of the computed shortest paths.
        """

        shortest_paths = {}
        shortest_distances = {}
        involved_nodes = set() # keep track of nodes important

        for fragment in start_fragments:
            for fragment2 in end_fragments:
                try:
                    shortest_paths.setdefault(fragment, {})
                    shortest_distances.setdefault(fragment, {})
                    shortest_path = nx.shortest_path(self.graph, source=fragment, target=fragment2, weight='weight')
                    shortest_distance = nx.shortest_path_length(self.graph, source=fragment, target=fragment2, weight='weight')
                    shortest_paths[fragment][fragment2] = shortest_path
                    shortest_distances[fragment][fragment2] = shortest_distance
                    involved_nodes.update(shortest_path)
                except:
                    # No connections
                    shortest_paths[fragment][fragment2] = []
                    shortest_distances[fragment][fragment2] = None

        # Create a graph of the found nodes
        subgraph = self.graph.subgraph(involved_nodes).copy()

        # Compute the Z cores
        z_scores = {}
        for fragment in shortest_distances:
            distances = []
            for fragment2 in shortest_distances[fragment]:
                if shortest_distances[fragment][fragment2]:
                    distances.append(shortest_distances[fragment][fragment2])
            mean_d = np.mean(distances)
            std_d = np.std(distances)
            z_scores[fragment] = (distances - mean_d) / std_d

        return subgraph, shortest_paths, shortest_distances, z_scores



        



        


