import mdtraj as md
import os
import numpy as np
from Allohubpy._encodeframe.lib import encode_frame
from cffi import FFI
from Allohubpy import SAtraj
import mini3di


class AbsEncoder:

    def __init__(self, atoms_to_keep, output_file_name):
        """
        Initializes the abstract class for encoders

        Args:
            atoms_to_keep (list of strings): Atom names to extract.
            output_file_name (str): Name of the output file.

        Example:
            Use on a newly created encoder:
            super().__init__(atoms_to_keep=["CA", "CB", "N", "C"], output_file_name="output.txt")
        """

        self.atoms_to_keep = atoms_to_keep
        self.output = output_file_name
        self.traj = []


    def load_trajectory(self, topology, mdtraj, skip=0, stride=None):
        """
        Loads the MD trajectory into memory and creates the iterable object

        Args:
            topology (str): name of the file to use as topology.
            mdtrajectory (str): name of the MD trajectory file
            skip (int): Number of frames to skip from the trajectory
            stride (int or None) : Frames to stride, that is the frames divisible by stride will be kept
        """

        self.traj = TrajLoader(topology, mdtraj, self.atoms_to_keep,  skip, stride)


    def process_frame(self, frame_dict):
        """
        Abstract method to over-ride to process a given dictionary with the x,y,z coordinates for the selected atoms.

        Args:
            frame_dcit (dict {atom_name: np.array}): Dictionary containing the selected atom names to extract and their XYZ coordinates 
                                                     in a numpy array of shape (num residues, 3)

        Returns:
            encoded string (str)
        """

        raise NotImplementedError


    def encode(self):
        """
        Encodes the whole trajectory and saves it to a file
        """
        
        if not isinstance(self.traj, TrajLoader):
            raise AttributeError("Encoder has no trajectory loaded use .load_trajectory first")
        
        with open(self.output, "w") as out:

            for results_dict in self.traj:
                encoded_string = self.process_frame(results_dict)
                out.write(encoded_string)
                out.write("\n")

    def set_output_file(self, output_file):
        """
        Sets the output file to save the encoded trajectories.

        Args:
            output_file (str): file to save the results to.
        """

        self.output = output_file




class Encoder3DI(AbsEncoder):


    def __init__(self, output_file_name):
        super().__init__(atoms_to_keep=["CA", "CB", "N", "C"], output_file_name=output_file_name)
        self.encoder = mini3di.Encoder()


    def process_frame(self, frame_dict):
        """
        Method to process a given dictionary with the x,y,z coordinates for the selected atoms.

        Args:
            frame_dcit (dict {atom_name: np.array}): Dictionary containing the selected atom names to extract and their XYZ coordinates 
                                                     in a numpy array of shape (num residues, 3)

        Returns:
            encoded string (str)
        """

        # Add Nans to the atoms without CB
        c_beta_fixed = []
        residue_idx = 0 # keep track of the shifted residues

        # iterate over residues present in the sequence
        for residue in frame_dict["residues"]:
            # For non special residues just add the xyz of the CB
            if residue not in ["GLY", "PRO"]:
                c_beta_fixed.append(frame_dict["CB"][residue_idx])
                residue_idx += 1 # update counter

            # For special residues add nans for the xyz coordinates
            else:
                c_beta_fixed.append(np.array([np.nan, np.nan, np.nan]))

        # Convert to numpy
        c_beta_fixed = np.array(c_beta_fixed)

        # load data into mini3di encoder
        states = self.encoder.encode_atoms(ca=frame_dict["CA"],
                                           cb=c_beta_fixed,
                                           n=frame_dict["N"],
                                           c=frame_dict["C"])
        
        # process the data into a string sequence
        sequence = self.encoder.build_sequence(states)

        return sequence
    


class SAEncoder(AbsEncoder):

    def __init__(self, output_file_name):
        super().__init__(atoms_to_keep=["CA"], output_file_name=output_file_name)

        self.sadict = {"A": [ 2.630,  11.087,-12.054,  2.357,  13.026,-15.290,   1.365,  16.691,-15.389,   0.262,  18.241,-18.694],
                       "B": [ 9.284,  15.264, 44.980, 12.933,  14.193, 44.880,  14.898,  12.077, 47.307,  18.502,  10.955, 47.619],
                       "C": [25.311,  23.402, 33.999, 23.168,  25.490, 36.333,  23.449,  24.762, 40.062,  23.266,  27.976, 42.095],
                       "D": [23.078,   3.265, -6.609, 21.369,   6.342, -4.176,  20.292,   6.283, -0.487,  17.232,   7.962,  1.027],
                       "E": [72.856,  22.785, 26.895, 70.161,  25.403, 27.115,  70.776,  28.306, 29.539,  69.276,  31.709, 30.364],
                       "F": [41.080,  47.709, 33.614, 39.271,  44.390, 33.864,  36.049,  44.118, 31.865,  32.984,  43.527, 34.064],
                       "G": [59.399,  59.100, 40.375, 57.041,  57.165, 38.105,  54.802,  54.093, 38.498,  54.237,  51.873, 35.502],
                       "H": [-1.297,  14.123,  7.733,  1.518,  14.786,  5.230,   1.301,  17.718,  2.871,  -0.363,  16.930, -0.466],
                       "I": [40.106,  24.098, 63.681, 40.195,  25.872, 60.382,  37.528,  27.160, 58.053,  37.489,  25.753, 54.503],
                       "J": [25.589,   1.334, 11.216, 27.604,   1.905, 14.443,  30.853,  -0.042, 14.738,  30.051,  -1.020, 18.330],
                       "K": [17.239,  71.346, 65.430, 16.722,  74.180, 67.850,  18.184,  77.576, 67.092,  20.897,  77.030, 69.754],
                       "L": [82.032,  25.615,  4.316, 81.133,  23.686,  7.493,  83.903,  21.200,  8.341,  81.485,  19.142, 10.443],
                       "M": [28.972,  -1.893, -7.013, 28.574,  -5.153, -5.103,  30.790,  -7.852, -6.647,  30.144, -10.746, -4.275],
                       "N": [-4.676,  72.183, 52.250, -2.345,  71.237, 55.105,   0.626,  71.396, 52.744,   1.491,  72.929, 49.374],
                       "O": [ 0.593,  -3.290,  6.669,  2.032,  -2.882,  3.163,   4.148,  -6.042,  3.493,   7.276,  -4.148,  2.496],
                       "P": [29.683,  47.318, 25.490, 26.781,  47.533, 27.949,  26.068,  51.138, 26.975,  27.539,  52.739, 30.088],
                       "Q": [34.652,  36.550, 18.964, 33.617,  37.112, 15.311,  32.821,  40.823, 15.695,  34.062,  43.193, 12.979],
                       "R": [ 8.082,  44.667, 15.947,  5.161,  46.576, 17.520,   5.855,  49.813, 15.603,   3.022,  50.724, 13.161],
                       "S": [64.114, 65.465, 28.862,  63.773,  68.407, 26.422,  67.481,  69.227, 26.232,  67.851,  68.149, 22.610],
                       "T": [18.708,-123.580,-46.136, 18.724,-126.113,-48.977,  18.606,-123.406,-51.661,  14.829,-123.053,-51.400],
                       "U": [61.732,  49.657, 35.675, 62.601,  46.569, 33.613,  65.943,  46.199, 35.408,  64.205,  46.488, 38.806],
                       "V": [88.350,  40.204, 52.963, 86.971,  39.540, 49.439,  85.732,  36.159, 50.328,  83.085,  37.796, 52.614],
                       "W": [23.791,  23.069,  3.102, 26.051,  22.698,  6.166,  23.278,  21.203,  8.349,  21.071,  19.248,  5.952],
                       "X": [ 1.199,   3.328, 36.383,  1.782,   3.032, 32.641,   1.158,   6.286, 30.903,   1.656,   8.424, 34.067],
                       "Y": [33.001,  12.054,  8.400, 35.837,  11.159, 10.749,  38.009,  10.428,  7.736,  35.586,   7.969,  6.163]}


        self.fragment_size = int(len(self.sadict[list(self.sadict.keys())[0]]) / 3)
        # Convert the sa library to a format compatible with the C wrapper
        self.sa_library = np.ndarray(shape=(len(self.sadict) * self.fragment_size, 3),
                                        buffer=np.array([self.sadict[key] for key in sorted(self.sadict.keys())], dtype=np.float32),
                                        dtype=np.float32)
        self.sa_library /= 10
        # Generate a mapping of the SA fragment name to its index
        self.sa_code_map = self._generate_samap()
        self.output_file = {}
        self.output_file_name = ""
        self.ffi = FFI()


    def _generate_samap(self):
        """
        Generates a map of SA fragment names to their respective indexes
        
        Returns:
            mapping dictionary of the ints to M32k25 letters
        """

        samap = {}
        for i, key in enumerate(self.sadict.keys()):
            samap[i] = key
        return samap


    def _c_encode(self, frame):
        """
        Encodes the given frame, into a string of SA fragments
        
        Args:
            frame (np.ndarray, shape = (number of residues,3)): trajectory frame.
        
        Returns:
            string of the encoded frame
        """

        # Variables needed for the C function
        protein_length = int(frame.shape[0])
        n_windows = protein_length - self.fragment_size + 1
        n_fragments = int(len(self.sadict))
        encoding = np.zeros(n_windows, dtype=np.int32)
        mdframe = self.ffi.cast("float(*)[3]", frame.ctypes.data)
        fragments = self.ffi.cast("float(*)[3]", self.sa_library.ctypes.data)
        c_encoding = self.ffi.cast("int *", self.ffi.from_buffer(encoding, require_writable=True))
        # Call to the C function that does the encoding
        encode_frame(n_windows, n_fragments, self.fragment_size, mdframe, fragments, c_encoding)
        # Convert the list of indexes to SA fragment names
        econded_string = self._map_encoding(c_encoding, n_windows)
        return econded_string


    def _map_encoding(self, encoded_prot, n_windows):
        encoded_string = ""
        for i in range(n_windows):
            encoded_string += self.sa_code_map[encoded_prot[i]]
        return encoded_string


    def process_frame(self, frame_dict):
        """
        Method to process a given dictionary with the x,y,z coordinates for the selected atoms.

        Args:
            frame_dcit (dict {atom_name: np.array}): Dictionary containing the selected atom names to extract and their XYZ coordinates 
                                                     in a numpy array of shape (num residues, 3)

        Returns:
            encoded string (str)
        """

        # Encode the protein frame
        frame = frame_dict["CA"] # M32k25 only uses C alphas
        encoded_string = self._c_encode(frame)
        return encoded_string



class TrajLoader:

    def __init__(self, topology, mdtrajectory, to_extract = ["CA"], skip=0, stride=None):
        """
        Holds the trajectory and extracts the relevant atoms and chains from it

        Args:
            topology (str): name of the file to use as topology.
            mdtrajectory (str): name of the MD trajectory file
            to_extract (list of str): Names of the atoms for which the indexes should be extracted.
            skip (int): Number of frames to skip from the trajectory
            stride (int or None) : Frames to stride, that is the frames divisible by stride will be kept
        """

        self.topology_file = topology
        self.md_file = mdtrajectory
        self.to_extract = to_extract
        self.skip = skip
        self.stride = stride
        self.to_keep_dict = {}
        self.topology = None
        self.md_traj = None
        self.index = 0
        # Now load data
        self.read_topology()
        self.load_traj()


    def read_topology(self):
        """
        Reads the topology and extracts the relevant indexes
        """

        self.topology = md.load(self.topology_file)

        for group in self.to_extract:
            indexes = self.topology.topology.select(f"name {group} and protein")
            self.to_keep_dict.setdefault(group, indexes)


    def load_traj(self):
        """
        Loads the trajectory into memory
        """

        # Check if topology file = md trajectory file. This means that the input is a pdb
        if self.topology_file == self.md_file:
            self.md_traj = self.topology
        else:
            self.md_traj = md.load(self.md_file, top=self.topology_file)
            self.md_traj = self.md_traj[self.skip::self.stride]

    
    def __iter__(self):
        """
        Makes the object iterable
        """

        self.index = 0  # Initialize the index for iteration
        return self

    def __next__(self):
        """
        Goes to the requested frame and extract the values of interest for the selected atom names.

        Returns:
            dictionary with the atom names as keys with their xyz coordinates (shape (residues, xyz)) 
            and the residues of the sequence in three letter format under the key "residues".
        """

        results = {}

        if self.index < len(self.md_traj):
            results["residues"] = [res.name for res in self.md_traj.topology.residues]
            for group in self.to_keep_dict:
                indexes = self.to_keep_dict[group]
                current_frame = self.md_traj.xyz[self.index, indexes, :]
                results.setdefault(group, current_frame)

            self.index += 1  # Move to the next index
            return results
        else:
            raise StopIteration  # End of iteration
