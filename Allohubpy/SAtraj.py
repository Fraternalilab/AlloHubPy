import numpy as np
from math import floor
from Allohubpy.Allohubpy_cython import calculate_transition_probabilities, calculate_mutual_information
from Allohubpy.MIblock import MIBlock
from scipy.stats import entropy
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

ALPHABETS = {"M32K25": ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y"],
             "3DI": ["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y","X"]}

class SAtraj:


    def __init__(self, block_size, alphabet):
        """
        Holds a SA trajectory and splices them in blocks and calculates the MI

        Args:
            block_size (int): Number of frames for each block.
            alphabet (array of str): list of possible letters.
        """

        self.b_size = block_size # number of frames the each block contains
        # Holds the trajectory in text format
        self.text_traj = []
        # Holds the trajectory in int format
        self.int_traj = []
        # Holds MI matrices objects
        self.mi_traj = []
        # Holds the alphabet mapping
        self.char2int_map = {}
        self.alphabet = alphabet
        self._process_alphabet(alphabet)


    def _reset(self):
        """
        Resets the values in memory to provide a clean state to load new data.
        """
        self.text_traj = []
        self.int_traj = []
        self.mi_traj = []



    def _process_alphabet(self, alphabet):
        """
        Processes the used alphabet and creates a mapping of chr to ints
        Args:
            sa_file: Path to file with the encoded trajectory.
        """
        for i,char in enumerate(alphabet):
            self.char2int_map[char] = i


    def _read_sa(self, sa_file, randomize):
        """
        Loads the SA trajectory (sasta format) into memory and encodes it into ints

        Args:
            sa_file (str): Path to file with the encoded trajectory.
            randomize (bool): Whether or not the frames should be randomized
        """

        int_traj = []
        n_lines = 0
        length = 0
        with open(sa_file, "r") as inn:
            for line in inn:
                line = line.rstrip()
                n_lines += 1
                if not length:
                    length = len(line)
                self.text_traj.append(line)
                for char in line:
                    int_traj.append(self.char2int_map[char])

        self.int_traj = np.ndarray(buffer=np.array(int_traj, dtype=np.int32), shape=(n_lines, length), dtype=np.int32)

        if randomize:
            np.random.shuffle(self.int_traj)


    def load_data(self, sa_file, randomize=True):
        """
        Loads all data into memory

        Args:
            sa_file (str): Path to file with the encoded trajectory.
            randomize (bool): Whether or not the frames should be randomized.
        """

        self._reset()
        self._read_sa(sa_file, randomize)

    
    def compute_mis(self, max_workers=None, disable_tqdm=False):
        """
        Encodes all the blocks and creates MI objects

        Args:
            max_workers (int or None): Maximum number of workers to use for multiprocessing. None eqauls all possible.
            disable_tqdm (bool): whether to show progress bar or not.

        Returns:
            numpy array with the mutual information for each fragment shape (fragment, fragment)
        """
        
        print("ENCODING TRAJECTORY")
        rango = int(floor(len(self.int_traj)/self.b_size))
        if rango == 0:
            rango = 1
        mi_traj = [None] * rango

        # Start a ProcessPoolExecutor
        tt = [self.int_traj[idx * self.b_size : (idx+1) * self.b_size] for idx in range(rango)]

        # Create a tqdm progress bar

        progress_bar = tqdm(total=rango, desc="Computing MI", unit="block", disable=disable_tqdm)

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks to the executor and keep track of the futures
            futures = {executor.submit(calculate_mutual_information, self.int_traj[idx * self.b_size : (idx+1) * self.b_size,], len(self.alphabet)): 
                       idx for idx in range(rango)}
            
            # Collect results as they complete
            for future in as_completed(futures):
                idx = futures[future]  # Get the index for this future
                res = future.result()
                progress_bar.update(1)
                mi_block = MIBlock(res, len(self.int_traj[0]))
                mi_traj[idx] = mi_block  # Store the result at the correct index
        """
        non parallelized approach (kept in since its easier to read)
        mi_traj = []
        for i in range(rango):
            # If block size is not multiple of the number of frames the last frames will be discarded
            b = i * self.b_size
            f = (i+1) * self.b_size
            block = self.int_traj[b:f,]
            mi_array = calculate_mutual_information(block, len(self.alphabet))
            mi_block = MIBlock(mi_array, len(self.text_traj[0]))

            mi_traj.append(mi_block)
        """
        print("TRAJECTORY COMPLETED")
        return mi_traj
    
    def combine(self, other):
        """
        Combines two SAtraj objects

        Args:
            other(SAtraj object): SAtraj object to combine with.

        Returns:
            combined SAtraj object.
        """
        combined = SAtraj(self.b_size ,self.alphabet)
        combined.int_traj = np.vstack((self.int_traj, other.int_traj))
        return combined


    def compute_entropy(self, bootstrap=0):
        """
        Computes the shanon entropy for each alphabet fragment

        Args:
            bootstrap (int): Number of samples to create to estimate statistics

        Returns:
            List containing the shanon entropies for each position or list of list if bootstrap !=0
        """
        results = []
        iterations = 1
        if bootstrap:
            iterations = bootstrap

        for sampl in range(iterations):
            s_entropy = []
            for i in range(self.int_traj.shape[1]):
                column = self.int_traj[:, i]
                if bootstrap:
                    column = np.random.choice(column, size=len(column), replace=True)
                probabilities = self.fragments_probabilities(column)
                s_entropy.append(entropy(probabilities, base=2))
            if bootstrap:
                results.append(s_entropy)

        if bootstrap:
            return results
        else:
            return s_entropy


    def fragments_probabilities(self, column):
        """
        Computes the probabilities for each letter given an array

        Args:
            column (np.array): data array for which the shanon entropy should be computed.

        Returns:
            np.array with the probabilities for each possible fragment
        """

        value_counts = np.bincount(column)
        value_counts = np.pad(value_counts, (0, len(self.alphabet) - len(value_counts)), 'constant', constant_values=0)
        #t_probs = value_counts[value_counts > 0] / len(column)  # Normalize to get probabilities
        t_probs = value_counts / len(column)
        # the probability array may be shorter than the alphabet if the higher integers are never seen
        probabilities = np.zeros(len(self.alphabet))
        probabilities += t_probs # This ensures that the length is correct

        return probabilities
    

    def compute_transitions(self):
        """
        Computes the transition probabilities for each fragment to each fragment using cython module.

        Returns:
            np.array of transition probabilities of shape (alphabet, alphabet)
        """

        # check that self.int_traj is the correct format
        return calculate_transition_probabilities(self.int_traj, len(self.alphabet))


    def get_probabilities(self):
        """
        Calculates the probabilities for each possible state for each SA fragment

        Returns:
            np.array with the probabilities of each fragment with shape (num fragments, vocab_size)

        """

        # First compute the probabilities for each column
        prob_array = []
        for i in range(self.int_traj.shape[1]):
            column = self.int_traj[:, i]
            prob_array.append(self.fragments_probabilities(column))
        
        return np.array(prob_array)
  
    
    def get_int_traj(self):
        """
        Returns the encoded trajectory into ints.

        Returns:
            np.array with the encoded trajectory into ints with shape (num_frames, num_fragments)
        """

        return self.int_traj
        
           



