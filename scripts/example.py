from Allohubpy import SAtraj
from Allohubpy import Overlap
from Allohubpy import SANetwork
from Allohubpy.plotter import Allohub_plots
import numpy as np

def save_array_to_txt(array, filename, delimiter=',', fmt='%.18e'):
    """
    Saves a NumPy array to a text file.

    Parameters:
        array (numpy.ndarray): The NumPy array to save.
        filename (str): The path to the output text file.
        delimiter (str): The string used to separate values (default is ',').
        fmt (str): Format for each element in the array (default is '%.18e' for scientific notation).
    """
    try:
        np.savetxt(filename, array, delimiter=delimiter, fmt=fmt)
        print(f"Array saved successfully to {filename}")
    except Exception as e:
        print(f"Error saving array: {e}")

# Encode Trajectory if necessary

# Initialize Structural Alphabet trajectory handler
print("Initialize Structural Alphabet trajectory handler")
sa_traj1 = SAtraj.SAtraj(50, SAtraj.ALPHABETS["M32K25"])
sa_traj2 = SAtraj.SAtraj(50, SAtraj.ALPHABETS["M32K25"])
# Load encoded data into the model
print("Load encoded data into the model")
sa_traj1.load_data("mock.txt")
sa_traj2.load_data("mock2.txt")

# Plot the evolution of time of the Structural Alphabet trajectory
Allohub_plots.plot_SA_traj(sa_traj1.get_int_traj(), SAtraj.ALPHABETS["M32K25"], action="save", name="mock1_SA_traj.png")
Allohub_plots.plot_SA_traj(sa_traj2.get_int_traj(), SAtraj.ALPHABETS["M32K25"], action="save", name="mock2_SA_traj.png")

# Compute the shanon entropy
print("Compute the shanon entropy")
entropy1 = sa_traj1.compute_entropy()
entropy2 = sa_traj2.compute_entropy()
# Save entropy values
save_array_to_txt(entropy1, "mock_SA_shanon_entropy.txt")
save_array_to_txt(entropy2, "mock_SA_shanon_entropy.txt")
# Plot Shanon entropy
Allohub_plots.plot_shanon_entropy(entropy1, action="save", name="mock1_SA_shanon_entropy.png")
Allohub_plots.plot_shanon_entropy(entropy2, action="save", name="mock2_SA_shanon_entropy.png")

# Compute and plot the structural profile (abundance of each fragment conformations)
print("Compute and plot the structural profile (abundance of each fragment conformations)")
Allohub_plots.plot_fragment_probabilities(probability_matrix=sa_traj1.get_probabilities(), vocabulary=SAtraj.ALPHABETS["M32K25"],
                                          action="save", name="mock1_fragment_probabilities.png")
Allohub_plots.plot_fragment_probabilities(probability_matrix=sa_traj2.get_probabilities(), vocabulary=SAtraj.ALPHABETS["M32K25"],
                                          action="save", name="mock2_fragment_probabilities.png")

# Compute and plot the transition probability matrix of probabilities > 0.01
print("Compute and plot the transition probability matrix of probabilities > 0.01")
transition_matrix1 = sa_traj1.compute_transitions()
transition_matrix2 = sa_traj2.compute_transitions()
save_array_to_txt(transition_matrix1, "mock1_SA_transition_matrix.txt")
save_array_to_txt(transition_matrix2, "mock2_SA_transition_matrix.txt")
Allohub_plots.plot_transition_probabilities(trans_matrix=transition_matrix1, vocabulary=SAtraj.ALPHABETS["M32K25"], 
                                            action="save", name="mock_transition_probailities.png")
Allohub_plots.plot_transition_probabilities(trans_matrix=transition_matrix2, vocabulary=SAtraj.ALPHABETS["M32K25"], 
                                            action="save", name="mock_transition_probailities.png")
# Calculate the MI information
print("Calculate the MI information")
mi_traj1 = sa_traj1.compute_mis(max_workers=5)
mi_traj2 = sa_traj2.compute_mis(max_workers=5)

# Plot the MI matrix for the first Block
Allohub_plots.plot_mi_matrix(mi_traj1[0].get_mi_matrix(), action="save", name="mock1_mi.png")
Allohub_plots.plot_mi_matrix(mi_traj2[0].get_mi_matrix(), action="save", name="mock2_mi.png")

# Do an eigenvector decomposition of the matrices
print("Do an eigenvector decomposition of the matrices")
for mi_tr in mi_traj1:
    mi_tr.compute_eigensystem()
for mi_tr in mi_traj2:
    mi_tr.compute_eigensystem()

# Create the overlap handler to compute similarities between the trajectories
overlap = Overlap.Overlap([mi_traj1, mi_traj2], ev_list=[0, 1])
# Compute the eigenoverlap between trajectories
overlap.fill_overlap_matrix()
# plot the overlap matrix
Allohub_plots.plot_overlap(overlap.get_overlap_matrix())
# Compute similarities between overlap matrices
overlap.compute_similarities()

# Cluster and see performance with clustering (to do)

# Find upregulated and downregulated fragments
print("Find upregulated and downregulated fragments")
updown_regulated_fragments = overlap.updown_regulation(traj_mapping=[0,1],splitting=True)
t12_updown = updown_regulated_fragments[(0,1)]
# Print top 5 upregulated and down regulated fragments
pval_threshold = 0.001
significant_fragments = t12_updown[t12_updown['AdjustedPValues'] < pval_threshold]
top_upregulated = significant_fragments[significant_fragments['log2FoldChange'] > 0].sort_values('log2FoldChange', ascending=False).head(5)
top_downregulated = significant_fragments[significant_fragments['log2FoldChange'] < 0].sort_values('log2FoldChange').head(5)
print(top_downregulated)
print(top_upregulated)
Allohub_plots.plot_updownregulation(t12_updown,  fold_threshold=2.5, pvalue_threshold=0.001, action="save", name="volcano.png")

# Create graph representations for all states based on the defined mapping
SAgraph1 = SANetwork.SANetWork(mi_traj1, neighbour_threshold=1, distance_limit=12)
SAgraph2 = SANetwork.SANetWork(mi_traj1, neighbour_threshold=1, distance_limit=12)

SAgraph1.read_distance_files("distance_matrix.txt", fragment_size=4)
SAgraph2.read_distance_files("distance_matrix.txt", fragment_size=4)

SAgraph1.create_graph(pval_threshold=0.05)
SAgraph2.create_graph(pval_threshold=0.05)

# Compute eigenvector centrality to find rellevant nodes
centrality_1_df = SAgraph1.compute_centrality()
centrality_2_df = SAgraph2.compute_centrality()

top_centr1 = centrality_1_df.sort_values("Centrality", ascending=False).head(10)
top_centr2 = centrality_2_df.sort_values("Centrality", ascending=False).head(10)

print(top_centr1)
print(top_centr2)

# Compute shortestpath form regulatory to effector nodes
start_fragments = [20, 21, 22, 23, 24,]
end_fragments = [372, 373, 374, 375, 376]
subgraph1, shortest_paths1, shortest_distances1, z_score1 = SAgraph1.identify_preferential_connections(start_fragments=start_fragments, end_fragments=end_fragments)
subgraph2, shortest_paths2, shortest_distances2, z_score2 = SAgraph2.identify_preferential_connections(start_fragments=start_fragments, end_fragments=end_fragments)
print("Shortest Paths")
print(shortest_paths1)
print(shortest_paths2)
print("Shortest Distances")
print(shortest_distances1)
print(shortest_distances2)
print("Z scores")
print(z_score1)
print(z_score2)
# Plot
Allohub_plots.plot_SA_graph(subgraph1, start_fragments, end_fragments, "save", "SA_graph1.png")
Allohub_plots.plot_SA_graph(subgraph2, start_fragments, end_fragments, "save", "SA_graph2.png")




