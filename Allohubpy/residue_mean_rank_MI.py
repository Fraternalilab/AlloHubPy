import numpy as np

def mean_rank_and_mi(sorted_results):
    # collect all residue ids present
    residue_ids = sorted(set(int(row[0]) for arr in sorted_results for row in arr))

    mean_rank = []
    mean_mi = []

    for resid in residue_ids:
        ranks = []
        mis = []

        for arr in sorted_results:
            residues = arr[:, 0].astype(int)
            values = arr[:, 1].astype(float)

            matches = np.where(residues == resid)[0]
            if len(matches) > 0:
                rank = matches[0] + 1   # rank starts at 1
                ranks.append(rank)
                mis.append(values[matches[0]])

        mean_rank.append(np.mean(ranks))
        mean_mi.append(np.mean(mis))

    mean_rank = np.array(mean_rank)
    mean_mi = np.array(mean_mi)
    residue_ids = np.array(residue_ids)

    # sort by decreasing mean MI and increasing mean rank
    sorted_idx = np.lexsort((-mean_mi, mean_rank))

    result = np.column_stack((
        residue_ids[sorted_idx],
        mean_rank[sorted_idx],
        mean_mi[sorted_idx]
    ))

    return result


def mean_rank_and_mi_with_sd(sorted_results):
    residue_ids = sorted(set(int(row[0]) for arr in sorted_results for row in arr))

    mean_rank = []
    sd_rank = []
    mean_mi = []
    sd_mi = []

    for resid in residue_ids:
        ranks = []
        mis = []

        for arr in sorted_results:
            residues = arr[:, 0].astype(int)
            values = arr[:, 1].astype(float)

            matches = np.where(residues == resid)[0]
            if len(matches) > 0:
                rank = matches[0] + 1   # rank starts at 1
                ranks.append(rank)
                mis.append(values[matches[0]])

        ranks = np.array(ranks, dtype=float)
        mis = np.array(mis, dtype=float)

        mean_rank.append(np.mean(ranks))
        sd_rank.append(np.std(ranks, ddof=1) if len(ranks) > 1 else 0.0)
        mean_mi.append(np.mean(mis))
        sd_mi.append(np.std(mis, ddof=1) if len(mis) > 1 else 0.0)

    residue_ids = np.array(residue_ids)
    mean_rank = np.array(mean_rank)
    sd_rank = np.array(sd_rank)
    mean_mi = np.array(mean_mi)
    sd_mi = np.array(sd_mi)

    # Standard ranking convention:
    # lower mean rank = better
    # higher mean MI = better
    # lower SD rank = better
    sorted_idx = np.lexsort((sd_rank, -mean_mi, mean_rank))

    result = np.column_stack((
        residue_ids[sorted_idx],
        mean_rank[sorted_idx],
        sd_rank[sorted_idx],
        mean_mi[sorted_idx],
        sd_mi[sorted_idx]
    ))

    return result

