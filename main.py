from SAtraj import SAtraj
from Overlap import Overlap
import numpy as np
from multiprocessing import Pool

def compute_mi(obj):
    obj.encode_all()
    return obj


def main(in_files, block_size, processes):
    # READ TRAJS
    trajectories = []
    for in_file in in_files:
        f = SAtraj(in_file, block_size)
        trajectories.append(f)
        np.random.shuffle(f.int_traj)

    print("Reading trajs finished")

    # Compute MI
    with Pool(processes=processes) as pool:
        results = pool.map(compute_mi, trajectories)
    print("MI computed")

    # Compute Eigensystems
    for tr in results:
        print(tr.SA_file)
        for i,mi_tr in enumerate(tr.mi_traj):
            mi_tr.remove_low(0.0025)
            mi_tr.remove_adjacent_mi(6)
            #mi_tr.search_highest(0.3)
            mi_tr.compute_eigensystem()
            #mi_tr.plot_mi()

    print("Eigen systems computed")
    ov = Overlap(results, ergodicity=False, ev_list=[0,1,2])
    ov.fill_overlap_matrix()
    ov.compute_similarities()
    ov.plot_overlap()

    # Compute overlap of top eigenvectors


if __name__ == "__main__":
    main([], 500, 10)

