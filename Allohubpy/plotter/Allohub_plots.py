import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx

# Example plots to use in analysis or to use as inspiration for personalized plots for Allohubpy

def plot_shanon_entropy(entropy, action="save", ylim=(0,4), name="SA_shanon_entropy.png"):
    """
    Plots the shanon entropies

    Args:
        entropy (np.array or list): Shanon entropies for each fragment. 
        action (str): What to do with the plot, 'save' for saving it, 'show' for displaying it
        ylim (tuple of floats): Maximum and minimum value for the y axis.
        name (str): name to use to save the plot, including format.
    """

    plt.figure(figsize=(16, 5))
    plt.plot(entropy, marker='o', color='b', linestyle='-', linewidth=0.5, markersize=3)

    # Customize x and y labels and title
    plt.xlabel("Fragment Index")
    plt.ylabel("H/bits")

    # Customize x and y labels and title
    plt.ylim(ylim[0],ylim[1])
    plt.xlabel("Fragment Index")
    plt.ylabel("H/bits")

    # Display the plot
    if action == "save":
        plt.savefig(name)
    else:
        plt.show()

    # clean up
    plt.close()


def plot_shanon_entropy_sd(entropy_arrays, ylim = (0,4), action="save", name="SA_shanon_entropy.png"):
    """
    Plots the Shannon entropies for multiple entropy arrays, with error shade representing the 
    standard deviation of the mean at each fragment index.
    
    Args:
        entropy_arrays (list of numpy arrays): list of numpy arrays containing Shannon entropies for each fragment.
        action (str): What to do with the plot, 'save' for saving it, 'show' for displaying it
        ylim (tuple of floats): Maximum and minimum value for the y axis.
        name (str): name to use to save the plot, including format.
    """

    # Convert the list of entropy arrays into a numpy array (shape: num_arrays x num_fragments)
    entropy_arrays = np.array(entropy_arrays)
    
    # Calculate the mean and standard deviation along the arrays axis (axis=0 for fragment-wise)
    mean_entropy = np.mean(entropy_arrays, axis=0)
    std_entropy = np.std(entropy_arrays, axis=0)

    # Plot the mean entropy
    plt.figure(figsize=(16, 5))
    plt.plot(mean_entropy, marker='o', color='b', linestyle='-', linewidth=0.5, markersize=3, label="Mean Entropy")
    
    # Plot the shaded area for the error (mean ± standard deviation)
    plt.fill_between(np.arange(len(mean_entropy)), mean_entropy - std_entropy, mean_entropy + std_entropy, 
                     color='blue', alpha=0.2, label="Error (Std Dev)")

    # Customize x and y labels and title
    plt.ylim(ylim[0], ylim[1])
    plt.xlabel("Fragment Index")
    plt.ylabel("H/bits")

    # Display or save the plot
    if action == "save":
        plt.savefig(name)
    else:
        plt.show()

    # Clean up
    plt.close()


def plot_network_centrality(df, action="save", name="SA_netwrok_centrality.png"):
    """
    Plots the network centrality values.

    Args:
        df (pandas DataFrame with columns 'fragments' and 'Centrality'): Dataframe with each fragment id and their centrality in the network. 
        action (str): What to do with the plot, 'save' for saving it, 'show' for displaying it
        name (str): name to use to save the plot, including format.
    """

    # Create the line plot with points
    plt.figure(figsize=(16, 5))
    sns.lineplot(data=df, x="fragments", y="Centrality", marker="o", markersize=3, linewidth=0.5)

    # Customize the plot
    plt.title(f'Centrality of the nodes in the network')
    plt.xlabel("Fragment")
    plt.ylabel("Eigenvector Centrality")

    # Display or save the plot
    if action == "save":
        plt.savefig(name)
    else:
        plt.show()

    # Clean up
    plt.close()


def plot_fragment_probabilities(probability_matrix, vocabulary, action="save", name="SA_probabilities.png"):
    """
    Plots the probabilities for each possible state of each fragment on the SA trajectory.

    Args:
        probability_matrix (np.array): matrix with the probabilities of each fragment to each possible fragment token. shape (fragments, vocab_size)
        vocabulary (list of str): list of all possible tokens (states) in the used vocabulary
        action (str): What to do with the plot, 'save' for saving it, 'show' for displaying it
        name (str): name to use to save the plot, including format.
    """

    # Create a dictionary that maps each integer state to a character
    state_labels = {i:key for i,key in enumerate(vocabulary)}
    r_probability_matrix = np.round(probability_matrix,2)

    # Plotting the prob matrix as a heatmap in grayscale

    plt.figure(figsize=(10, 6))
    sns.heatmap(r_probability_matrix.T, cmap="gray_r", cbar_kws={'label': 'Probability'})

    # Customize x and y labels
    plt.xlabel("Fragment Index")
    plt.ylabel("States")
    plt.title("Probability Distribution")

    # Set y-tick labels to the mapped characters from the dictionary
    plt.yticks(np.arange(len(vocabulary)) + 0.5, [state_labels[i] for i in range(len(vocabulary))], rotation=0)

    # Display the plot
    if action == "save":
        plt.savefig(name)
    else:
        plt.show()

    # clean up
    plt.close()


def plot_SA_traj(satraj, vocabulary, action="save", name="SA_traj.png"):
    """
    Plots the SA aligment.

    Args:
        satraj (np.array): Array with encoded trajectory using ints shape (num_frames, num_fragments)
        vocabulary (list of strs): list of all possible tokens (states) in the used vocabulary
        action (str): What to do with the plot, 'save' for saving it, 'show' for displaying it
        name (str): name to use to save the plot, including format.
    """

    # Define a dictionary that maps specific integers to letters
    state_labels = {i:key for i,key in enumerate(vocabulary)}
    
    # Get unique values in the data that are in the dictionary keys
    unique_values = sorted(set(satraj.flatten()) & set(state_labels.keys()))

    # Create the plot
    plt.figure(figsize=(10, 8))
    im = plt.imshow(satraj, aspect='auto', cmap='rainbow')

    # Create color bar with custom ticks based on the dictionary
    cbar = plt.colorbar(im, ticks=unique_values)
    cbar.set_ticklabels([state_labels[val] for val in unique_values])
    cbar.set_label('Mapped Fragments')
    
    # Label the axes
    plt.xlabel('Fragment index')
    plt.ylabel('Simulation Frame')

    # Display the plot
    if action == "save":
        plt.savefig(name)
    else:
        plt.show()
    # clean up
    plt.close()


def plot_transition_probabilities(trans_matrix, vocabulary, action="save", name="SA_transition_probabilities.png"):
    """
    Plots the transitions probabilities for each fragment.

    Args:
        trans_matrix (np.array): Array with the transition probabilties of each fragment to each fragment, shape (fragment, fragment)
        vocabulary (list of strs): list of all possible tokens (states) in the used vocabulary
        action (str): What to do with the plot, 'save' for saving it, 'show' for displaying it
        name (str): name to use to save the plot, including format.
    """

    # Define threshold for displaying values
    threshold = 0.01
    # Define a dictionary that maps specific integers to letters
    state_labels = {i:key for i,key in enumerate(vocabulary)}

    plt.figure(figsize=(16, 12))
    trans_matrix = np.round(trans_matrix, 2)
    sns.heatmap(trans_matrix, annot=np.where(trans_matrix > threshold, trans_matrix, ""), 
                cmap="gray_r", fmt='', cbar_kws={'label': 'Probability'}, linewidths=0.5)
    
    # Set custom tick labels using the index-to-letter mapping
    plt.xticks(ticks=np.arange(len(state_labels)) + 0.5, labels=[state_labels[i] for i in range(len(state_labels))])
    plt.yticks(ticks=np.arange(len(state_labels)) + 0.5, labels=[state_labels[i] for i in range(len(state_labels))])

    # Customize axis labels and title
    plt.xlabel("Starting fragment")
    plt.ylabel("Ending fragment")

    plt.title("Transition Probability Matrix Heatmap (Threshold: >0.01)")
    # Display the plot
    if action == "save":
        plt.savefig(name)
    else:
        plt.show()
    # clean up
    plt.close()


def plot_mi_matrix(mi_matrix, action="save", name="Mi_matrix.png"):
    """
    Plots the MI matrix as a hetmap

    Args:
        mi_matrix (np.array): Array with the mi signal, shape (fragment, fragment).
        action (str): What to do with the plot, 'save' for saving it, 'show' for displaying it.
        name (str): name to use to save the plot, including format.
    """

    white_to_blue = LinearSegmentedColormap.from_list("white_to_blue", ["white", "darkblue"])
    sns.heatmap(mi_matrix,  vmin=0, vmax=0.4, cmap=white_to_blue)
    plt.title("2-D Heat Map")
    # Display the plot
    if action == "save":
        plt.savefig(name)
    else:
        plt.show()
    # clean up
    plt.close()


def plot_overlap(overlap_matrix, action="save", vmax=0.4, name="Mi_matrix.png"):
    """
    Plots the overlap matrix as a heatmap

    Args:
        overlap_matrix (np.array): Array with the eigenvector overlap signal, shape (mi blocks, mi blocks).
        vmax (float): Maximum value for the color scale.
        action (str): What to do with the plot, 'save' for saving it, 'show' for displaying it.
        name (str): name to use to save the plot, including format.
    """

    sns.heatmap(overlap_matrix,  vmin=0.0, vmax=vmax, cmap='coolwarm')
    plt.title("2-D Overlap Heat Map")
    # Display the plot
    if action == "save":
        plt.savefig(name)
    else:
        plt.show()
    # clean up
    plt.close()


def plot_updownregulation(updown_regulation_df, fold_threshold, pvalue_threshold, ylim = 8, action="save", name="updown_regulation.png"):
    """
    Plots the volcano plot for the upregulated and down regulated fragment pairs

    Args:
        updown_regulation_df (pandas DataFrame with columns 'log2FoldChange', 'AdjustedPValues', 'FragmentPairs'): 
                            Dataframe with the p values and foldchanges for each fragment pair.
        fold_threshold (float): Maximum absolute fold change to consider the signal as relevant.
        pvalue_threshold (float): Minimum significance level to consider a fragment pair coupling as significant.
        ylim (float): Highest value for the y axis ( -log10 pvalue).
        action (str): What to do with the plot, 'save' for saving it, 'show' for displaying it.
        name (str): name to use to save the plot, including format.
    """
    
    # shorten name
    df = updown_regulation_df
    plt.figure(figsize=(10, 6))

    # Apply both fold change and p-value thresholds for coloring
    upregulated = df[(df['log2FoldChange'] > fold_threshold) & (df['AdjustedPValues'] < pvalue_threshold)]
    downregulated = df[(df['log2FoldChange'] < -fold_threshold) & (df['AdjustedPValues'] < pvalue_threshold)]
    not_significant = df[~((df['log2FoldChange'] > fold_threshold) | (df['log2FoldChange'] < -fold_threshold)) | (df['AdjustedPValues'] >= pvalue_threshold)]
    
    # Scatter plots for each category with smaller point sizes (adjust 's' value)
    plt.scatter(not_significant['log2FoldChange'], -np.log10(not_significant['AdjustedPValues']), color='grey', label='Not significant', s=30)
    plt.scatter(upregulated['log2FoldChange'], -np.log10(upregulated['AdjustedPValues']), color='red', label='Upregulated', s=30)
    plt.scatter(downregulated['log2FoldChange'], -np.log10(downregulated['AdjustedPValues']), color='blue', label='Downregulated', s=30)

    # Label axes
    plt.xlabel('Log2 Fold Change')
    plt.ylabel('-Log10 Adjusted P-value')
    plt.ylim(0,ylim)
    plt.title('Volcano Plot of Fragment Coupling Changes')
    plt.legend()
    plt.grid(True)

    # Display or save the plot
    if action == "save":
        plt.savefig(name)
    else:
        plt.show()

    # clean up
    plt.close()


def plot_SA_graph(graph, start_nodes, end_nodes, action="save", name="SA_graph.png"):
    """
    Plots the fragment graph with colored nodes, red for ending nodes, blue for start nodes, green the rest. 
    The edges thickness is proportional to their weight. 

    Args:
        graph (natworkx graph): networkx graph to plot.
        start_nodes (list ints): Starting nodes. That is list of nodes to color differently (blue).
        end_nodes (list ints): Ending nodes. That is list of nodes to color differently (red).
        action (str): What to do with the plot, 'save' for saving it, 'show' for displaying it.
        name (str): name to use to save the plot, including format.
    """
    
    node_colors = []
    for node in graph.nodes():
        if node in end_nodes:
            node_colors.append('red')
        elif node in start_nodes:
            node_colors.append('blue')
        else:
            node_colors.append('green')
            
    # Define edge thickness based on weight
    plt.figure(dpi=1000)
    
    edge_weights = nx.get_edge_attributes(graph, 'weight')
    edge_thickness = [weight for _, _, weight in graph.edges(data='weight')]
   
    # Create the graph layout
    pos = nx.spring_layout(graph)

    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=250)
    
    # Draw edges with thickness proportional to weight
    nx.draw_networkx_edges(graph, pos, width=edge_thickness, edge_color='black')
    
    # Draw labels for nodes
    nx.draw_networkx_labels(graph, pos, font_color='white', font_weight='bold', font_size=6)

    # Draw edge labels (e.g., weights)
    #nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_weights, font_size=6, font_color='black')

    # Display or save the plot
    if action == "save":
        plt.savefig(name, dpi=1000)
    else:
        plt.show()

    # clean up
    plt.close()