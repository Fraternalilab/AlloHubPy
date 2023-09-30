import numpy as np
from math import floor
from MIblock import MIBlock
from _mi_block.lib import wrap_mi_block
from cffi import FFI
"""
Holds a SA trajectory and splices them in blocks and calculates the MI
"""


class SAtraj:
    char2int_map = {"A":0, "B":1, "C":2, "D":3, "E":4, "F":5, "G":6, "H":7, "I":8, "J":9,
                    "K":10, "L":11, "M":12, "N":13, "O":14, "P":15, "Q":16, "R":17, "S":18,
                    "T":19, "U":20, "V":21, "W":22, "X":23, "Y":24}

    def __init__(self, block_size):
        self.b_size = block_size
        self.length = 0
        # Holds the trajectory in text format
        self.text_traj = []
        # Holds the trajectory in int format
        self.int_traj = np.array([])
        # Holds MI matrices objects
        self.mi_traj = []


    def compute_average(self):
        return np.mean(np.array([x.mi_matrix for x in self.mi_traj]), axis=0)
    
    def split(self, num):
        trajs = []
        splited_text = np.array_split(self.text_traj, num)
        splited_int_traj = np.array_split(self.int_traj, num)
        for i in range(num):
            new_traj = SAtraj(self.b_size)
            new_traj.int_traj = splited_int_traj[i]
            new_traj.text_traj = splited_text[i]
            trajs.append(new_traj)
        return trajs


    def read_sa(self, sa_file):
        """
        Reads SA trajectory file and encodes it into ints
        :return:
        """
        int_traj = []
        n_lines = 0
        with open(sa_file, "r") as inn:
            for line in inn:
                line = line.rstrip()
                if not line.startswith(">"):
                    n_lines += 1
                    if not self.length:
                         self.length = len(line)
                    self.text_traj.append(line)
                    for char in line:
                        int_traj.append(self.char2int_map[char])
        self.int_traj = np.ndarray(buffer=np.array(int_traj), shape=(n_lines, self.length), dtype=int)


    @staticmethod
    def encode_block(block, length, depth):
        """
        Encodes one block of trajectory frames
        :param block: block of trajectory frames in np array of shape (1, length * depth) of ints
        :param length: number of SA in one frame
        :param depth: number of frames in one block
        :return: list of MI values of upper part of matrix without diagonal
        """
        # Initialize ffi
        ffi = FFI()
        # Initialize memory holders
        mi_array = np.zeros(int((length * (length - 1))/2), dtype=np.float64)
        md_block = np.zeros(int(length * depth), dtype=int)
        # Cast initialized arrays into memory
        c_md_block = ffi.cast("int *", ffi.from_buffer(md_block))
        c_mi_array = ffi.cast("double *", ffi.from_buffer(mi_array, require_writable=True))
        # Load C array with the data
        count = 0
        for col in range(length):
            for element in block[:,col]:
                c_md_block[count] = element
                count += 1
        # Call appropiate function
        wrap_mi_block(length, depth, c_md_block, c_mi_array)

        return mi_array

    def encode_all(self):
        """
        Encodes all the blocks and creates MI objects
        :return:
        """
        print("ENCODING TRAJECTORY")
        rango = int(floor(len(self.text_traj)/self.b_size))
        if rango == 0:
            rango = 1
        for i in range(rango):
            # If block size is not multiple of the number of frames the last frames will be discarded
            b = i * self.b_size
            f = (i+1) * self.b_size
            block = self.int_traj[b:f,]
            mi_block = MIBlock(self.encode_block(block, len(self.text_traj[0]),
                                                 self.b_size), len(self.text_traj[0]))

            self.mi_traj.append(mi_block)
        print("TRAJECTORY COMPLETED")







