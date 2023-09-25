from SAtraj import SAtraj
from Overlap import Overlap
import numpy as np
from multiprocessing import Pool
import scipy.stats as stats

def compute_mi(obj):
    obj.encode_all()
    return obj

def main(in_files, block_size, processes, split, mapping):
    # READ TRAJS
    trajectories = []
    for in_file in in_files:
        f = SAtraj(block_size)
        f.read_sa(in_file)
        if split > 1:
            sub_traj = f.split(split)
            for traj in sub_traj:
                trajectories.append(traj)
        else:
            trajectories.append(f)
    
    for f in trajectories:
        np.random.shuffle(f.int_traj)

    print("Reading trajs finished")

    # update mapping
    if split > 1:
        new_map = []
        for e in mapping:
            for i in range(split):
                new_map.append(e)
        mapping = new_map

    # Compute MI
    with Pool(processes=processes) as pool:
        results = pool.map(compute_mi, trajectories)
    print("MI computed")

    # Compute Eigensystems
    for tr in results:
        print(tr.SA_file)
        for i,mi_tr in enumerate(tr.mi_traj):
            #mi_tr.remove_low(0.0025)
            #mi_tr.remove_adjacent_mi(6)
            #mi_tr.search_highest(0.3)
            mi_tr.compute_eigensystem()
            #mi_tr.plot_mi()

    # Coompute overlap of top eigenvectors
    print("Eigen systems computed")
    ov = Overlap(results, ergodicity=False, ev_list=[0,1,2])
    ov.fill_overlap_matrix()
    ov.compute_similarities()
    ov.plot_overlap()

    # Hub detection
    # Compute average MI per trajectory
    avg_mi = [tr.compute_average() for tr in results]
    # Compute difference between conditions
    mi_holder = {}
    for i,mi in enumerate(avg_mi):
        mi_holder.setdefault(mapping[i], []).append(mi)
    # Check there are enough replicates for each condition
    for key in mi_holder:
        if len(mi_holder[key]) < 2:
            print("Not enough replicates or splitting to compute statistics")
            exit(0)
    # Compute differences between conditions
    keys = mi_holder.keys()
    for i in range(len(keys)):
        for j in range(i,len(keys)):
            if i == j:
                continue
            else:
                diff = np.abs(np.mean(mi_holder[keys[i]]) - np.mean(mi_holder[keys[j]]))
                # extract number of pairs whose difference is bigger than 0.2
                hits = []
                magnitude = []
                for ii in range(diff.shape[0]):
                    for jj in range(diff.shape[1]):
                        if diff[ii][jj] >= 0.2:
                            hits.append([ii,jj])
                            magnitude.append(diff[ii][jj])
                hits = np.array(hits)
                magnitude = np.array(magnitude)

                # Compute p-value
                pvalues = []
                for hit in hits:
                    condition1 = [x[hit] for x in mi_holder[keys[i]]]
                    condition2 = [x[hit] for x in mi_holder[keys[j]]]
                    pvalues.append(stats.ttest_ind(a=condition1, b=condition2).pvalue)
                pvalues = np.array(pvalues)

                # Apply Benjamini-Hochberg correction to p-value
                # sort p values
                sort_indx = np.argsort(pvalues)
                sort_indx = sort_indx[::-1]
                hits = hits[sort_indx]
                pvalues = pvalues[sort_indx]
                magnitude = magnitude[sort_indx]
                counts = len(pvalues)
                adjusted = []
                for k,p in enumerate(pvalues):
                    adjusted.append(min(1, p * (counts/counts-k)))
                adjusted = np.array(adjusted)
                
                # revert lists and save them to a file
                hits = hits[::-1]
                magnitude = magnitude[::-1]
                pvalues = pvalues[::-1]
                adjusted = adjusted[::-1]

                # save to file
                with open("difference_condition_%s_and_condition_%s.dat" % (i,j), "w") as out:
                    out.write("Hub1;Hub2;diff;pvalue;adjusted-pvalue\n")
                    for ndx in range(len(hits)):
                        out.write("%s;%s;%s;%s;%s\n" % (hits[ndx][0], hits[ndx][1], magnitude[ndx], pvalues[ndx], adjusted[ndx]))
 

if __name__ == "__main__":
    main([], 500, 4, 4, [0,0,1,1])

