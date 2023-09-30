from SAtraj import SAtraj
from Overlap import Overlap
import numpy as np
from multiprocessing import Pool
import scipy.stats as stats

def compute_mi(obj):
    obj.encode_all()
    return obj

def main(in_files, overlap, hubs, block_size, processes, split, thresh, false_discovery, mapping):
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
    if overlap:
        # Compute Eigensystems
        for tr in results:
            print("Computing Eigenvalues")
            for i,mi_tr in enumerate(tr.mi_traj):
                mi_tr.compute_eigensystem()
                #mi_tr.plot_mi()

        # Coompute overlap of top eigenvectors
        print("Eigen systems computed")
        ov = Overlap(results, ergodicity=False, ev_list=[0,1,2])
        ov.fill_overlap_matrix()
        ov.compute_similarities()
        ov.plot_overlap()

    if not hubs:
        exit(0)
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
    keys = sorted(list(keys))
    for i in range(len(keys)):
        for j in range(i,len(keys)):
            if i == j:
                continue
            else:
                diff = np.mean(mi_holder[keys[i]],axis=0)/np.mean(mi_holder[keys[j]], axis=0)
                diff = np.log2(diff)
                flat_diff = np.sort(np.abs(diff.flatten()))
                cutoff = flat_diff[-thresh]
                # extract number of pairs whose difference is bigger than 0.2
                hits = []
                magnitude = []
                for ii in range(diff.shape[0]):
                    for jj in range(diff.shape[1]):
                        if diff[ii][jj] >= cutoff or diff[ii][jj] <= -cutoff:
                            if [jj,ii] in hits or ii == jj:
                                continue
                            hits.append([ii,jj])
                            magnitude.append(diff[ii][jj])
                hits = np.array(hits)
                magnitude = np.array(magnitude)
                # Compute p-value
                pvalues = []
                for hit in hits:
                    condition1 = [x[hit[0]][hit[1]] for x in mi_holder[keys[i]]]
                    condition2 = [x[hit[0]][hit[1]] for x in mi_holder[keys[j]]]

                    pvalues.append(stats.ttest_ind(a=condition1, b=condition2).pvalue)
                pvalues = np.array(pvalues)
                # Apply Benjamini-Hochberg correction to p-value
                # sort p values
                sort_indx = np.argsort(pvalues)
                hits = hits[sort_indx]
                pvalues = pvalues[sort_indx]
                magnitude = magnitude[sort_indx]
                counts = len(pvalues)
                adjusted = []
                for k in range(len(pvalues)):
                    adjusted.append(((k+1)/counts) * false_discovery) 
                

                # save to file
                with open("difference_condition_%s_and_condition_%s.dat" % (i,j), "w") as out:
                    out.write("Hub1;Hub2;diff;pvalue;adjusted-pvalue\n")
                    for ndx in range(len(hits)):
                        out.write("%s;%s;%s;%s;%s\n" % (hits[ndx][0], hits[ndx][1], magnitude[ndx], pvalues[ndx], adjusted[ndx]))
 

if __name__ == "__main__":
    # Variables
    """in_files: alphabet encoded trajectories,
       overlap: run the convergence calculation,
       hubs: find possible allosteric hubs,
       block_size: size of the blocks to compute MI from, default 500,
       processes: Number of processes to use to compute the MI matrices,
       split: Number of chunks in which the trajecotry will be splitted,
       thresh: cutoff on how may top hits one wants (-1 for all),
       false_discovery: false discovery rate for the p value adjustement,
       mapping: list of length in_files that tells to which condition each trajectory belongs to
    """
    main(in_files=["../encoded_md/apo_repl1_tc1.sasta",
        "../encoded_md/apo_repl2_tc1.sasta",
        "../encoded_md/apo_repl3_tc1.sasta",
        "../encoded_md/apo_repl4_tc1.sasta",
        "../encoded_md/apo_repl5_tc1.sasta",
        "../encoded_md/fbp_repl1_tc1.sasta",
        "../encoded_md/fbp_repl2_tc1.sasta",
        "../encoded_md/fbp_repl3_tc1.sasta",
        "../encoded_md/fbp_repl4_tc1.sasta",
        "../encoded_md/fbp_repl5_tc1.sasta"
        ], 
        overlap=False,
        hubs=True,
        block_size=500,
        processes=8,
        split=1, 
        thresh=-1, 
        false_discovery=0.25,
        mapping=[0,0,0,0,0,1,1,1,1,1])

