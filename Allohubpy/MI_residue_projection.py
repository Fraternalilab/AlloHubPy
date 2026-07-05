from __future__ import annotations

import numpy as np


def window_mi_to_residue_mi(
    mi_win: np.ndarray,
    window_len: int = 4,
    mask_overlapping_windows: bool = True,
    overlap_mask_distance: int | None = None,
    coverage_normalize: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Project an MI matrix between overlapping windows (fragments) to a residue-residue MI matrix.

    Assumptions:
        - Windows are consecutive and overlapping by (window_len - 1) residues (stride 1).
        - Window i covers residues [i, i+1, ..., i+window_len-1].
        - mi_win is (M x M) where M = N_res - window_len + 1.

    Args:
        mi_win (np.ndarray): Square (M x M) MI matrix in SA fragment/window space.
        window_len (int): Number of residues per SA fragment (4 for the M32K25 alphabet).
        mask_overlapping_windows (bool): Zero out MI between windows that share residues
                                         to remove trivial sequence-overlap signal.
        overlap_mask_distance (int or None): Windows with |i - j| <= this value are masked.
                                             Defaults to window_len - 1 (3 for 4-mers).
        coverage_normalize (bool): Normalize for the fact that terminal residues appear in
                                   fewer windows than central ones.
        eps (float): Small constant to avoid division by zero during normalization.

    Returns:
        mi_res (np.ndarray): (N_res x N_res) residue-level MI matrix.
    """

    mi_win = np.asarray(mi_win, dtype=float)
    if mi_win.ndim != 2 or mi_win.shape[0] != mi_win.shape[1]:
        raise ValueError("mi_win must be a square (M x M) matrix.")
    M = mi_win.shape[0]

    # Number of residues implied by M windows of length L with stride 1:
    N_res = M + window_len - 1

    # Optional: mask MI between windows that overlap strongly along sequence
    # For window_len=4, windows overlap if |i-j| <= 3.
    if mask_overlapping_windows:
        if overlap_mask_distance is None:
            overlap_mask_distance = window_len - 1  # 3 for 4-mers
        mi_win = mi_win.copy()
        idx = np.arange(M)
        dist = np.abs(idx[:, None] - idx[None, :])
        mi_win[dist <= overlap_mask_distance] = 0.0

    # Build W (M x N_res): each window distributes weight to its residues.
    # Uniform weights: 1/window_len for residues in the window.
    W = np.zeros((M, N_res), dtype=float)
    w = 1.0 / window_len
    for i in range(M):
        W[i, i:i + window_len] = w  # residues covered by window i

    # Project: mi_res = W^T * mi_win * W
    mi_res = W.T @ mi_win @ W

    # Optional: normalize for coverage differences (termini appear in fewer windows)
    if coverage_normalize:
        coverage = W.sum(axis=0)  # length N_res
        D = np.diag(1.0 / (coverage + eps))
        mi_res = D @ mi_res @ D

    # Enforce symmetry (numerical)
    mi_res = 0.5 * (mi_res + mi_res.T)
    return mi_res


def residue_importance_from_mi(mi_res: np.ndarray, mode: str = "abs") -> np.ndarray:
    """
    Summarize residue-residue MI into a per-residue score.

    mode:
      - "abs": sum of |MI| (hotspot regardless of sign; MI is usually >=0 anyway)
      - "sum": sum of MI
    """
    mi_res = np.asarray(mi_res, dtype=float)
    if mi_res.ndim != 2 or mi_res.shape[0] != mi_res.shape[1]:
        raise ValueError("mi_res must be square.")
    A = mi_res.copy()
    np.fill_diagonal(A, 0.0)

    if mode == "abs":
        return np.sum(np.abs(A), axis=1)
    if mode == "sum":
        return np.sum(A, axis=1)
    raise ValueError("mode must be 'abs' or 'sum'.")


# -------------------- Example --------------------
if __name__ == "__main__":
    # Example: M windows => implied residues N_res = M + 3 for 4-mers
    M = 10
    rng = np.random.default_rng(0)
    mi_win = rng.random((M, M))
    mi_win = 0.5 * (mi_win + mi_win.T)  # symmetric example

    mi_res = window_mi_to_residue_mi(mi_win, window_len=4,
                                     mask_overlapping_windows=True,
                                     coverage_normalize=True)

    score = residue_importance_from_mi(mi_res, mode="abs")
    print("mi_win shape:", mi_win.shape)
    print("mi_res shape:", mi_res.shape)  # (M+3, M+3)
    print("top residues:", np.argsort(score)[::-1][:5])
