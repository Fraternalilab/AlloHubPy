"""
Reconcile significant fragment-pair dMI with per-residue MI importance
  by mapping dMI to residues and scoring.

This script implements three steps:

1) Map fragment-pair differential MI (dMI) to a residue-residue matrix
2) Residue-pair scoring
3) Residue scoring

Assumptions
-----------
- Fragments are defined by the residues they cover.
- Significant fragment pairs are provided as:
      (frag_i, frag_j, dmi_value)
  where dmi_value can be signed or unsigned.
- Per-residue importance can be given either as:
      - an "importance" score where larger = better
      - a "rank" where smaller = better
- Residue indices can be 0-based or 1-based, but be consistent.

Main outputs
------------
- dmi_res: residue-residue matrix obtained from fragment-pair dMI
- pair_scores: ranked residue pairs combining dMI and residue importance/rank
- residue_scores: ranked residues combining aggregated dMI and residue importance/rank
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Literal, Sequence, Tuple, Union, Optional
import numpy as np


# ============================================================
# Types
# ============================================================

FragmentID = int
ResidueID = int
FragmentPair = Tuple[FragmentID, FragmentID, float]


# ============================================================
# Helpers
# ============================================================

def _validate_square_matrix(mat: np.ndarray, name: str) -> None:
    if not isinstance(mat, np.ndarray):
        raise TypeError(f"{name} must be a numpy array.")
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"{name} must be a square matrix.")


def _normalize_vector(
    x: np.ndarray,
    higher_is_better: bool = True,
    mode: Literal["minmax", "zscore", "rank_fraction"] = "minmax",
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Normalize a 1D vector into a comparable scale.

    Parameters
    ----------
    x : np.ndarray
        Input vector.
    higher_is_better : bool
        If False, the scale is inverted after normalization.
    mode : {"minmax", "zscore", "rank_fraction"}
        Normalization mode.
    eps : float
        Small constant to avoid divide-by-zero.

    Returns
    -------
    np.ndarray
        Normalized vector.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be a 1D array.")

    if mode == "minmax":
        xmin = np.min(x)
        xmax = np.max(x)
        out = (x - xmin) / (xmax - xmin + eps)

    elif mode == "zscore":
        mu = np.mean(x)
        sd = np.std(x)
        out = (x - mu) / (sd + eps)

        # Shift to non-negative for easier multiplicative scoring
        out = out - np.min(out)
        out = out / (np.max(out) + eps)

    elif mode == "rank_fraction":
        # Larger x -> larger normalized value
        order = np.argsort(x)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(x) + 1)
        out = ranks / len(x)

    else:
        raise ValueError("mode must be one of: 'minmax', 'zscore', 'rank_fraction'.")

    if not higher_is_better:
        out = 1.0 - out

    return out


def ranks_to_importance(
    ranks: Sequence[float],
    mode: Literal["inverse", "inverse_minmax", "rank_fraction"] = "inverse_minmax",
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Convert ranks into an importance-like score where larger = better.

    Assumes rank 1 is best.

    Parameters
    ----------
    ranks : sequence of float
        Residue ranks, smaller rank = better.
    mode : {"inverse", "inverse_minmax", "rank_fraction"}
        Conversion mode.
    eps : float
        Small constant to avoid divide-by-zero.

    Returns
    -------
    np.ndarray
        Importance scores, larger = better.
    """
    ranks = np.asarray(ranks, dtype=float)
    if ranks.ndim != 1:
        raise ValueError("ranks must be a 1D array.")
    if np.any(ranks <= 0):
        raise ValueError("All ranks must be > 0.")

    if mode == "inverse":
        return 1.0 / (ranks + eps)

    elif mode == "inverse_minmax":
        inv = 1.0 / (ranks + eps)
        return _normalize_vector(inv, higher_is_better=True, mode="minmax")

    elif mode == "rank_fraction":
        # best rank gets value نزدیک 1, worst gets value near 0
        n = len(ranks)
        return (n - ranks + 1) / n

    else:
        raise ValueError("mode must be one of: 'inverse', 'inverse_minmax', 'rank_fraction'.")


def build_fragment_to_residues(
    fragment_starts: Sequence[int],
    window_size: int,
) -> Dict[int, List[int]]:
    """
    Build fragment -> residues mapping for sliding windows.

    Example:
    fragment_starts = [0,1,2], window_size = 4
    gives:
      0 -> [0,1,2,3]
      1 -> [1,2,3,4]
      2 -> [2,3,4,5]

    Parameters
    ----------
    fragment_starts : sequence of int
        Start residue index for each fragment.
    window_size : int
        Number of residues covered by each fragment.

    Returns
    -------
    dict
        Mapping fragment_id -> list of residue ids
    """
    if window_size < 1:
        raise ValueError("window_size must be >= 1")

    mapping = {}
    for frag_id, start in enumerate(fragment_starts):
        mapping[frag_id] = list(range(start, start + window_size))
    return mapping


# ============================================================
# 1) Fragment-pair dMI -> residue-residue dMI matrix
# ============================================================

# creates for example for fragment '5' a residue index array '5,6,7,8'
def fragment_to_residue_indices(n_residues: int, window_size: int = 4):
    return {
        i: [i + j for j in range(window_size)]
        for i in range(n_residues - window_size + 1)
    }

# take the 'fragment_to_residues' from the output of the
#   'fragment_to_residue_indices' function above 
def fragment_pairs_to_residue_dmi(
    fragment_pairs: Sequence[FragmentPair],
    fragment_to_residues: Dict[FragmentID, Sequence[ResidueID]],
    n_residues: int,
    distribute: Literal["uniform", "mean", "sum"] = "uniform",
    symmetric: bool = True,
) -> np.ndarray:
    """
    Map significant fragment-pair dMI values to a residue-residue matrix.

    Parameters
    ----------
    fragment_pairs : sequence of (frag_i, frag_j, dmi_value)
        Significant fragment pairs with their dMI value.
    fragment_to_residues : dict
        Maps fragment index -> covered residues.
    n_residues : int
        Total number of residues.
    distribute : {"uniform", "mean", "sum"}
        How to distribute a fragment-pair dMI over residue pairs:
        - "uniform" / "mean": dMI divided by number of residue pairs
        - "sum": full dMI added to each residue pair
    symmetric : bool
        If True, mirror values into [j, i].

    Returns
    -------
    np.ndarray
        Residue-residue dMI matrix of shape (n_residues, n_residues)
    """
    if n_residues < 1:
        raise ValueError("n_residues must be >= 1")
    if distribute not in {"uniform", "mean", "sum"}:
        raise ValueError("distribute must be 'uniform', 'mean', or 'sum'")

    dmi_res = np.zeros((n_residues, n_residues), dtype=float)

    for frag_i, frag_j, dmi_value in fragment_pairs:
        if frag_i not in fragment_to_residues:
            raise KeyError(f"Fragment {frag_i} not found in fragment_to_residues.")
        if frag_j not in fragment_to_residues:
            raise KeyError(f"Fragment {frag_j} not found in fragment_to_residues.")

        res_i = list(fragment_to_residues[frag_i])
        res_j = list(fragment_to_residues[frag_j])

        if len(res_i) == 0 or len(res_j) == 0:
            continue

        if distribute in {"uniform", "mean"}:
            weight = dmi_value / (len(res_i) * len(res_j))
        else:  # "sum"
            weight = dmi_value

        for r in res_i:
            if r < 0 or r >= n_residues:
                raise IndexError(f"Residue index {r} out of bounds for n_residues={n_residues}")
            for s in res_j:
                if s < 0 or s >= n_residues:
                    raise IndexError(f"Residue index {s} out of bounds for n_residues={n_residues}")
                dmi_res[r, s] += weight
                if symmetric and r != s:
                    dmi_res[s, r] += weight

    return dmi_res


# ============================================================
# 2) Score residue pairs
# ============================================================

def score_residue_pairs(
    dmi_res: np.ndarray,
    residue_values: Sequence[float],
    residue_value_type: Literal["importance", "rank"] = "importance",
    pair_combine: Literal["mean", "product", "max", "min"] = "product",
    use_abs_dmi: bool = True,
    normalize_residue_values: bool = True,
    top_k: Optional[int] = None,
    exclude_diagonal: bool = True,
    upper_triangle_only: bool = True,
) -> List[Tuple[int, int, float, float, float, float]]:
    """
    Score residue pairs by combining residue-level dMI with residue importance/rank.

    Score formula:
        pair_score(i,j) = dMI_term(i,j) * residue_term(i,j)

    where residue_term is derived from residue importance or rank.

    Parameters
    ----------
    dmi_res : np.ndarray
        Residue-residue dMI matrix.
    residue_values : sequence of float
        Per-residue importance or rank.
    residue_value_type : {"importance", "rank"}
        Interpretation of residue_values.
    pair_combine : {"mean", "product", "max", "min"}
        How to combine residue scores for residues i and j.
    use_abs_dmi : bool
        If True, use abs(dmi_res). Recommended if direction is not the main point.
    normalize_residue_values : bool
        If True, normalize residue importance before pair scoring.
    top_k : int or None
        Number of top pairs to return. If None, return all.
    exclude_diagonal : bool
        Exclude i == j.
    upper_triangle_only : bool
        If True, return only i < j.

    Returns
    -------
    list of tuples
        Each entry:
        (i, j, pair_score, dmi_value_used, residue_score_i, residue_score_j)
    """
    _validate_square_matrix(dmi_res, "dmi_res")
    n = dmi_res.shape[0]

    residue_values = np.asarray(residue_values, dtype=float)
    if residue_values.ndim != 1 or len(residue_values) != n:
        raise ValueError("residue_values must be a 1D array with length matching dmi_res shape[0].")

    if residue_value_type == "importance":
        res_score = residue_values.copy()
        if normalize_residue_values:
            res_score = _normalize_vector(res_score, higher_is_better=True, mode="minmax")

    elif residue_value_type == "rank":
        res_score = ranks_to_importance(residue_values, mode="inverse_minmax")

    else:
        raise ValueError("residue_value_type must be 'importance' or 'rank'.")

    dmi_used = np.abs(dmi_res) if use_abs_dmi else dmi_res

    results = []
    for i in range(n):
        j_start = i + 1 if upper_triangle_only else 0
        for j in range(j_start, n):
            if exclude_diagonal and i == j:
                continue

            if pair_combine == "mean":
                residue_term = 0.5 * (res_score[i] + res_score[j])
            elif pair_combine == "product":
                residue_term = res_score[i] * res_score[j]
            elif pair_combine == "max":
                residue_term = max(res_score[i], res_score[j])
            elif pair_combine == "min":
                residue_term = min(res_score[i], res_score[j])
            else:
                raise ValueError("pair_combine must be one of: 'mean', 'product', 'max', 'min'.")

            pair_score = dmi_used[i, j] * residue_term
            results.append((i, j, float(pair_score), float(dmi_used[i, j]),
                            float(res_score[i]), float(res_score[j])))

    results.sort(key=lambda x: x[2], reverse=True)

    if top_k is not None:
        results = results[:top_k]

    return results


# ============================================================
# 3B) Score residues
# ============================================================

def aggregate_residue_dmi_strength(
    dmi_res: np.ndarray,
    mode: Literal["sum_abs", "sum_signed", "mean_abs", "mean_signed", "max_abs"] = "sum_abs",
    exclude_diagonal: bool = True,
) -> np.ndarray:
    """
    Aggregate residue-pair dMI into a per-residue dMI strength.

    Parameters
    ----------
    dmi_res : np.ndarray
        Residue-residue dMI matrix.
    mode : {"sum_abs", "sum_signed", "mean_abs", "mean_signed", "max_abs"}
        Aggregation mode.
    exclude_diagonal : bool
        Whether to ignore diagonal terms.

    Returns
    -------
    np.ndarray
        Per-residue dMI strength.
    """
    _validate_square_matrix(dmi_res, "dmi_res")
    mat = dmi_res.copy()

    if exclude_diagonal:
        np.fill_diagonal(mat, 0.0)

    if mode == "sum_abs":
        return np.sum(np.abs(mat), axis=1)

    elif mode == "sum_signed":
        return np.sum(mat, axis=1)

    elif mode == "mean_abs":
        return np.mean(np.abs(mat), axis=1)

    elif mode == "mean_signed":
        return np.mean(mat, axis=1)

    elif mode == "max_abs":
        return np.max(np.abs(mat), axis=1)

    else:
        raise ValueError("Invalid mode for residue dMI aggregation.")


def score_residues(
    dmi_res: np.ndarray,
    residue_values: Sequence[float],
    residue_value_type: Literal["importance", "rank"] = "importance",
    dmi_agg_mode: Literal["sum_abs", "sum_signed", "mean_abs", "mean_signed", "max_abs"] = "sum_abs",
    combine_mode: Literal["product", "mean", "weighted_sum"] = "product",
    alpha: float = 0.5,
    normalize_terms: bool = True,
    top_k: Optional[int] = None,
) -> List[Tuple[int, float, float, float]]:
    """
    Score residues by combining aggregated dMI strength with residue importance/rank.

    Parameters
    ----------
    dmi_res : np.ndarray
        Residue-residue dMI matrix.
    residue_values : sequence of float
        Per-residue importance or rank.
    residue_value_type : {"importance", "rank"}
        Interpretation of residue_values.
    dmi_agg_mode : {"sum_abs", "sum_signed", "mean_abs", "mean_signed", "max_abs"}
        How to aggregate residue dMI.
    combine_mode : {"product", "mean", "weighted_sum"}
        How to combine dMI strength and residue importance.
    alpha : float
        Weight for dMI term in weighted_sum:
            final = alpha * dmi_strength + (1-alpha) * residue_importance
    normalize_terms : bool
        Normalize both terms to [0,1] before combining.
    top_k : int or None
        Number of top residues to return.

    Returns
    -------
    list of tuples
        Each entry:
        (residue_index, final_score, dmi_strength, residue_importance)
    """
    _validate_square_matrix(dmi_res, "dmi_res")
    n = dmi_res.shape[0]

    residue_values = np.asarray(residue_values, dtype=float)
    if residue_values.ndim != 1 or len(residue_values) != n:
        raise ValueError("residue_values must be a 1D array with length matching dmi_res shape[0].")

    dmi_strength = aggregate_residue_dmi_strength(dmi_res, mode=dmi_agg_mode, exclude_diagonal=True)

    if residue_value_type == "importance":
        residue_importance = residue_values.copy()
    elif residue_value_type == "rank":
        residue_importance = ranks_to_importance(residue_values, mode="inverse_minmax")
    else:
        raise ValueError("residue_value_type must be 'importance' or 'rank'.")

    if normalize_terms:
        dmi_strength = _normalize_vector(dmi_strength, higher_is_better=True, mode="minmax")
        residue_importance = _normalize_vector(residue_importance, higher_is_better=True, mode="minmax")

    if combine_mode == "product":
        final = dmi_strength * residue_importance

    elif combine_mode == "mean":
        final = 0.5 * (dmi_strength + residue_importance)

    elif combine_mode == "weighted_sum":
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be between 0 and 1.")
        final = alpha * dmi_strength + (1.0 - alpha) * residue_importance

    else:
        raise ValueError("combine_mode must be one of: 'product', 'mean', 'weighted_sum'.")

    results = [(i, float(final[i]), float(dmi_strength[i]), float(residue_importance[i]))
               for i in range(n)]
    results.sort(key=lambda x: x[1], reverse=True)

    if top_k is not None:
        results = results[:top_k]

    return results


# ============================================================
# Convenience pipeline
# ============================================================

def reconcile_fragment_dmi_with_residue_scores(
    fragment_pairs: Sequence[FragmentPair],
    fragment_to_residues: Dict[FragmentID, Sequence[ResidueID]],
    n_residues: int,
    residue_values: Sequence[float],
    residue_value_type: Literal["importance", "rank"] = "importance",
    distribute: Literal["uniform", "mean", "sum"] = "uniform",
    pair_combine: Literal["mean", "product", "max", "min"] = "product",
    residue_dmi_agg_mode: Literal["sum_abs", "sum_signed", "mean_abs", "mean_signed", "max_abs"] = "sum_abs",
    residue_combine_mode: Literal["product", "mean", "weighted_sum"] = "product",
    top_k_pairs: int = 20,
    top_k_residues: int = 20,
) -> Tuple[np.ndarray, List[Tuple[int, int, float, float, float, float]], List[Tuple[int, float, float, float]]]:
    """
    Full pipeline:
    1) fragment dMI -> residue dMI matrix
    2) residue-pair scoring
    3) residue scoring

    Returns
    -------
    dmi_res, pair_scores, residue_scores
    """
    dmi_res = fragment_pairs_to_residue_dmi(
        fragment_pairs=fragment_pairs,
        fragment_to_residues=fragment_to_residues,
        n_residues=n_residues,
        distribute=distribute,
        symmetric=True,
    )

    pair_scores = score_residue_pairs(
        dmi_res=dmi_res,
        residue_values=residue_values,
        residue_value_type=residue_value_type,
        pair_combine=pair_combine,
        use_abs_dmi=True,
        normalize_residue_values=True,
        top_k=top_k_pairs,
        exclude_diagonal=True,
        upper_triangle_only=True,
    )

    residue_scores = score_residues(
        dmi_res=dmi_res,
        residue_values=residue_values,
        residue_value_type=residue_value_type,
        dmi_agg_mode=residue_dmi_agg_mode,
        combine_mode=residue_combine_mode,
        alpha=0.5,
        normalize_terms=True,
        top_k=top_k_residues,
    )

    return dmi_res, pair_scores, residue_scores


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    # Example protein with 10 residues
    n_residues = 10

    # Example sliding 4-residue fragments starting at residues 0..6
    fragment_to_residues = build_fragment_to_residues(
        fragment_starts=[0, 1, 2, 3, 4, 5, 6],
        window_size=4,
    )

    # Significant fragment-pair dMI values:
    # (fragment_i, fragment_j, dmi_value)
    fragment_pairs = [
        (0, 3, 1.8),
        (1, 4, -2.2),
        (2, 5, 1.1),
        (0, 6, 0.9),
    ]

    # Example per-residue mean MI importance
    residue_importance = np.array([0.2, 0.5, 0.7, 0.9, 0.8, 0.6, 0.4, 0.3, 0.1, 0.2])

    # Or example ranks (1 = best)
    residue_ranks = np.array([9, 6, 3, 1, 2, 4, 5, 7, 10, 8])

    # ---- run with importance ----
    dmi_res, pair_scores, residue_scores = reconcile_fragment_dmi_with_residue_scores(
        fragment_pairs=fragment_pairs,
        fragment_to_residues=fragment_to_residues,
        n_residues=n_residues,
        residue_values=residue_importance,
        residue_value_type="importance",
        distribute="uniform",
        pair_combine="product",
        residue_dmi_agg_mode="sum_abs",
        residue_combine_mode="product",
        top_k_pairs=10,
        top_k_residues=10,
    )

    print("\nResidue-residue dMI matrix:")
    print(np.round(dmi_res, 3))

    print("\nTop residue pairs:")
    print("(i, j, pair_score, abs_dmi, residue_score_i, residue_score_j)")
    for row in pair_scores:
        print(row)

    print("\nTop residues:")
    print("(residue, final_score, dmi_strength, residue_importance)")
    for row in residue_scores:
        print(row)

    # ---- run with ranks instead of importance ----
    dmi_res_rank, pair_scores_rank, residue_scores_rank = reconcile_fragment_dmi_with_residue_scores(
        fragment_pairs=fragment_pairs,
        fragment_to_residues=fragment_to_residues,
        n_residues=n_residues,
        residue_values=residue_ranks,
        residue_value_type="rank",
        distribute="uniform",
        pair_combine="product",
        residue_dmi_agg_mode="sum_abs",
        residue_combine_mode="product",
        top_k_pairs=10,
        top_k_residues=10,
    )

    print("\nTop residue pairs using ranks:")
    for row in pair_scores_rank:
        print(row)

    print("\nTop residues using ranks:")
    for row in residue_scores_rank:
        print(row)

