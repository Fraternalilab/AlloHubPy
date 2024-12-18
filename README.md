# Allohubpy
Allohubpy is a python package for the detection and charectarization of allosteric signals using a information theoric approach. 

The method captures local conformational changes associated with global motions from molecular dynamics simulations through the use of a Structural Alphabet, which simplifies the complexity of the Cartesian space by reducing the dimensionality down to a string of encoded fragments. These encoded fragments can then be used to compute the shannon entropy, mutual information between positions and build networks of correlated motions.

The folder notebooks contains examples for how to run the package with some sample data.

## Installation

Note: The user needs to have installed the GSL library
In ubuntu one can use: sudo apt-get install libgsl-dev


The package can be installed through pip with pip install Allohubpy.
To install all the required packages: pip install -r requirements.txt

Alternatively, one can compile the required code by running python setup.py such as build_ext --inplace and manualy adding the package to the PYTHONPATH.
One can also clone the repository and run pip install .

## Examples

Examples on how to run the code can be found on the notebooks folder.
All necessary data is provided in the respctive data folders under notebook.

## Usage

### Plotting

The package comes with premade plotting functions that can be used directly or as a template.
All plotting functions are found under allohubpy/plotter/Allohub_plots.py

## TrajProcessor

The TrajProcessor module offers encoder for 3di and M32K25. 
One can choose how many frames to skip as equilibration and the frequency (Stride) of the frames to be used.



```python
from Allohubpy import TrajProcessor

enc_3di = TrajProcessor.Encoder3DI("outputname_3di.sa")

# Encoder for M32K25
enc_mk = TrajProcessor.SAEncoder("outputname_mk.sa")

# Trajectory is saved every 10 ps. with stride = 10 only the 1-th frames will be processed producing an spacing of 100 ps between frames
# We skip 100 frames -> 1ns as extra equilibration

## Load repl1 of condition 1
enc_3di.load_trajectory(topology="topo.pdb", mdtraj="mdtraj.xtc", skip=100, stride=10)
enc_3di.encode()

enc_mk.load_trajectory(topology="topo.pdb", mdtraj="mdtraj.xtc", skip=100, stride=10)
enc_mk.encode()
```


### SA handler

The SA handler for a SA trajectory can be initialized as follows:
Block_size is the number of frames that will be used for each mutual information estimation and alphabet is the list of possible tokens for the selected alphabet. M32K25 and 3DI alphabets are provided by default.

The SA trajectory can be loaded with .load_data. Each encoded frame should be one line.

```python
from Allohubpy import SAtraj
sahandler = SAtraj.SAtraj(block_size=100, alphabet=SAtraj.ALPHABETS["M32K25"])
sahandler.load_data("safile")
```

After loading the data one can compute:
The probabilities of each fragment with: .get_probabilities()
The transition matrix between fragments with: .compute_transitions()
And the Shannon entropy: .compute_entropy()

Plots can be created using the provided plotting functions.

```python
from Allohubpy.plotter import Allohub_plots

entropy = sahandler.compute_entropy()
Allohub_plots.plot_shannon_entropy_sd(entropy, ylim=(0,4), action="show")

fragment_probs = sahandler.get_probabilities()
Allohub_plots.plot_fragment_probabilities(probability_matrix=fragment_probs, vocabulary=SAtraj.ALPHABETS["M32K25"], action="show")

transition_matrix = sahandler.compute_transitions()
Allohub_plots.plot_transition_probabilities(trans_matrix=transition_matrix, vocabulary=SAtraj.ALPHABETS["M32K25"], action="show")
```

Finaly, the mutual information matrices can be obtained by running:

```python
mi_array = sahandler.compute_mis(max_workers=8)
```

### Mutual information object

The computed mutual information matrices are stored in a MI object. One can retrive the matrix by doing .get_mi_matrix()
The eigenvector decomposition can be done by calling .compute_eigensystem()

Mi matrices can also be added together using addition and substraction.

### Overlap object

The obtained mi matrices with their eigenvectors and eienvalues computed can then be passed to the overlap handler which can be used to assess convergence and find up and down regulated fragments.

```python
from Allohubpy import Overlap

overlap = Overlap.Overlap([mi_array1, mi_array2, ....], ev_list=[0,1,2])
overlap.fill_overlap_matrix()

# plot overlap
Allohub_plots.plot_overlap(overlap.get_overlap_matrix(), vmax=0.4, action="show")

# compute similarities
similarity_matrix = overlap.compute_similarities()
```

For the up and down regulated fragments one needs to provide a mapping of the mi_arrays to the condition they belong.
The method will return a dictionary of pandas dataframes for each combination of conditions.

Each dataframe has the following columns:
FragmentPairs, log2FoldChange, AdjustedPValues and PValues

```python
pdown_regulated_fragments = overlap.updown_regulation(traj_mapping=[0,0,0,1,1,1],splitting=True)
t12_updown = updown_regulated_fragments[(0,1)]

Allohub_plots.plot_updownregulation(t12_updown,  fold_threshold=2, ylim=10, pvalue_threshold=0.01, action="show")
```

### SA Network

One can also create graph representations based on the mutual information.
For that a matrix of distances from all c alphas to all c alphas is needed.

The selected top pairs with higher signal at a given distance will be selected to build the network.

```python
from Allohubpy import SANetwork

SAgraph = SANetwork.SANetWork(traj_list= mi_array1 +  mi_array2 + ..., distance_limit=7)

SAgraph.set_distance_matrix(matrix=distance_matrix, fragment_size=4)

SAgraph.create_graph(threshold=90)
```

The graph can be extracted with .get_graph() and analyzed with .compute_centrality() or by extracting the shortest path from a set of selected residues.

```python
centrality_df = SAgraph_fbp.compute_centrality()
Allohub_plots.plot_network_centrality(centrality_df, action="show")

site1_fragments = [476, 509] 
site2_fragments = [260, 281]

# Subgraph is a networkx object with the nodes and edges of the shortest paths connecting those residues
# Shortest_pathd ans shortest_distances are list of the shortest paths and their distances respectively.
# z_score provides an estimate of how statistically coupled the two sites are
subgraph, shortest_paths, shortest_distances, z_score = SAgraph_fbp.identify_preferential_connections(start_fragments=site1_fragments,
                                                                                                       end_fragments=site2_fragments)

Allohub_plots.plot_SA_graph(subgraph, site1_fragments, site2_fragments, action="show")

```

## Incorporating own alphabets

The package also provides base classes to be used to create new alphabet encodings.

```python
from Allohubpy.TrajProcessor import AbsEncoder

class MyEncoder(AbsEncoder):


    # Atoms to keep should be the list of atom names that your encoding needs from the md trajectory
    def __init__(self, output_file_name):
        super().__init__(atoms_to_keep=["CA", "CB", "N", "C"], output_file_name=output_file_name)



    def process_frame(self, frame_dict):
        # Process frame is called on every MD frame when one call .encode()
        # frame_dict is a dictionary with the following keys:
        # "residues" containing the list of residues present
        # One key for each atom name in atoms to keep. for example "CA", "CB", etc.
        # The function needs to return the encoded string for that frame.
        # Under each key there is a list for all the elements under that group,f or example:
        # "CA" will have all c alphas and "CB" all c betas. 
        # If one residue does not contain that atom name, then it will not be present in the array, so len(CA) != len(CB)
        # One can use the residues list (Three letter code) to deal with it.
        
        return encoded_string

```
