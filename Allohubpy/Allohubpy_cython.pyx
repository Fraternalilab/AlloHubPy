# Allohub_cython.pyx
import numpy as np
cimport numpy as np
from libc.math cimport log2
cimport cython

def calculate_transition_probabilities(np.ndarray[np.int32_t, ndim=2] matrix, int num_states):
    cdef int n_rows = matrix.shape[0]
    cdef int n_cols = matrix.shape[1]
    
    # Initialize a 2D array to store transition counts
    cdef np.ndarray[np.int32_t, ndim=2] transition_matrix = np.zeros((num_states, num_states), dtype=np.int32)
    cdef int i, j, current_state, next_state
    
    # Count transitions in each column
    for j in range(n_cols):
        for i in range(n_rows - 1):
            current_state = matrix[i, j]
            next_state = matrix[i + 1, j]
            transition_matrix[current_state, next_state] += 1

    # Convert counts to probabilities
    cdef np.ndarray[np.float64_t, ndim=2] transition_probabilities = np.zeros((num_states, num_states), dtype=np.float64)
    for i in range(num_states):
        row_sum = np.sum(transition_matrix[i, :])
        if row_sum > 0:
            for j in range(num_states):
                transition_probabilities[i, j] = transition_matrix[i, j] / row_sum
    
    return transition_probabilities

def calculate_mutual_information(np.ndarray[int, ndim=2] matrix, int num_states):
    cdef int n_rows = matrix.shape[0]
    cdef int n_cols = matrix.shape[1]
    cdef int col_a, col_b, i, x, y
    cdef int b_x, b_y, b_xy
    cdef double p_xy, p_x, p_y, mi, total_count, entropy, f_error, mi_co

    # Initialize mutual information matrix
    cdef np.ndarray[double, ndim=2] mi_matrix = np.zeros((n_cols, n_cols), dtype=np.float64)
    
    # Loop over each pair of columns
    for col_a in range(n_cols):
        for col_b in range(col_a, n_cols):
            # Dynamically allocate count arrays as NumPy arrays
            joint_counts = np.zeros((num_states, num_states), dtype=np.int32)
            count_x = np.zeros(num_states, dtype=np.int32)
            count_y = np.zeros(num_states, dtype=np.int32)
            
            # Populate counts for column pairs
            for i in range(n_rows):
                x = matrix[i, col_a]
                y = matrix[i, col_b]
                joint_counts[x, y] += 1
                count_x[x] += 1
                count_y[y] += 1

            # Initialize variables for error estimation
            b_x = 0
            b_y = 0
            b_xy = 0
            f_error = 0.0

            # Calculate mutual information
            mi = 0.0
            entropy = 0.0
            total_count = float(n_rows)
            
            for x in range(num_states):
                # count non zero states of p_x and p_y
                if count_x[x] != 0:
                    b_x += 1
                if count_y[x] != 0:
                    b_y += 1
                for y in range(num_states):    
                    # Mutual information
                    if joint_counts[x, y] > 0:
                        p_xy = joint_counts[x, y] / total_count
                        p_x = count_x[x] / total_count
                        p_y = count_y[y] / total_count
                        mi += p_xy * log2(p_xy / (p_x * p_y))
                        # Counts of non zero joint counts for error estimate
                        b_xy += 1
                        # Entropy
                        entropy -= p_xy * log2(p_xy)

            # Calculate error and corrected mi
            #print(f"b_x: {b_x}, b_y:{b_y}. b_xy: {b_xy}") # Debug
            f_error = (b_xy - b_x - b_y + 1)/(2*total_count)
            #print(f"error: {f_error}, entropy: {entropy}, mi: {mi}") # Debug
            if entropy > 0:
                mi_co = (mi - f_error)/entropy
            else:
                mi_co = 1.0

            # Store mutual information value in the matrix (symmetrically)
            mi_matrix[col_a, col_b] = mi_co
            mi_matrix[col_b, col_a] = mi_co  # Symmetric matrix assignment
    
    return mi_matrix

