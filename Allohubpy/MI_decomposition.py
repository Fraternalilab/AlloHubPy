import numpy as np
import pandas as pd
from scipy.stats import norm, t as t_dist
from statsmodels.stats.multitest import multipletests
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def pair_table(int_traj, frag1, frag2, num_states, pseudocount=1e-3):
    """
    Joint probability table of the structural-alphabet states of two fragments.

    Args:
        int_traj (numpy array): integer-encoded SA trajectory of shape (frames, fragments).
        frag1 (int): index of the first fragment (rows of the table).
        frag2 (int): index of the second fragment (columns of the table).
        num_states (int): number of letters in the alphabet.
        pseudocount (float): count added to every cell, so that all cells are positive
            and iterative proportional fitting always converges.

    Returns:
        numpy array of shape (num_states, num_states) with the joint probabilities.
    """

    return pair_tables(int_traj, [(frag1, frag2)], num_states, pseudocount)[0]


def pair_tables(int_traj, fragment_pairs, num_states, pseudocount=1e-3):
    """
    Joint probability tables of many fragment pairs at once.

    Args:
        int_traj (numpy array): integer-encoded SA trajectory of shape (frames, fragments).
        fragment_pairs (array-like of shape (pairs, 2)): fragment index pairs.
        num_states (int): number of letters in the alphabet.
        pseudocount (float): count added to every cell.

    Returns:
        numpy array of shape (pairs, num_states, num_states) with the joint probabilities.
    """

    pairs = np.asarray(fragment_pairs, dtype=np.int64).reshape(-1, 2)
    n_cells = num_states * num_states
    codes = int_traj[:, pairs[:, 0]].astype(np.int64) * num_states + int_traj[:, pairs[:, 1]]
    codes += np.arange(len(pairs), dtype=np.int64) * n_cells
    counts = np.bincount(codes.ravel(), minlength=len(pairs) * n_cells)
    counts = counts.reshape(len(pairs), num_states, num_states) + pseudocount
    return counts / counts.sum(axis=(1, 2), keepdims=True)


def ipf_fit(table, row_target, col_target, max_iter=1000, tol=1e-8):
    """
    Iterative proportional fitting: rescales joint tables to given marginals.

    The interaction term of a table (its log-probabilities after removal of all
    single-variable terms) is preserved; only the single-variable terms change.
    Works on a single table (states, states) or a stack of tables (pairs, states, states).

    Args:
        table (numpy array): joint probability table(s) with positive cells.
        row_target (numpy array): target marginal(s) of the rows.
        col_target (numpy array): target marginal(s) of the columns.
        max_iter (int): maximum number of row/column scaling sweeps.
        tol (float): convergence threshold on the largest marginal deviation.

    Returns:
        numpy array: the fitted joint probability table(s).
    """

    fitted = table.copy()
    if fitted.ndim == 2:
        fitted, row_target, col_target = fitted[None], row_target[None], col_target[None]
        single = True
    else:
        single = False
    # only tables that have not yet converged are rescaled
    active = np.arange(len(fitted))
    for _ in range(max_iter):
        sub = fitted[active]
        rows, cols = row_target[active], col_target[active]
        sub *= (rows / sub.sum(axis=-1))[..., :, None]
        sub *= (cols / sub.sum(axis=-2))[..., None, :]
        fitted[active] = sub
        deviation = np.abs(sub.sum(axis=-1) - rows).max(axis=-1)
        active = active[deviation >= tol]
        if not len(active):
            break
    return fitted[0] if single else fitted


def mi_from_table(table, normalise=True):
    """
    Plug-in mutual information of joint probability table(s), in bits.

    Args:
        table (numpy array): joint probability table (states, states) or stack (pairs, states, states).
        normalise (bool): divide the MI by the joint entropy, as in the AlloHubPy
            MI matrices (without their finite-sample error correction).

    Returns:
        float or numpy array: (normalised) mutual information per table.
    """

    p_x = table.sum(axis=-1, keepdims=True)
    p_y = table.sum(axis=-2, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        mi_terms = np.where(table > 0, table * np.log2(table / (p_x * p_y)), 0.0)
        h_terms = np.where(table > 0, -table * np.log2(table), 0.0)
    mi = mi_terms.sum(axis=(-2, -1))
    if normalise:
        joint_entropy = h_terms.sum(axis=(-2, -1))
        with np.errstate(divide="ignore", invalid="ignore"):
            mi = np.where(joint_entropy > 0, mi / joint_entropy, 0.0)
    return float(mi) if np.ndim(mi) == 0 else mi


def decompose_pair(table_a, table_b, normalise=True):
    """
    Splits the MI change between two conditions into an interaction part and a population part.

    The interaction part is the MI change caused by the change of the interaction term
    (interaction surprisal, or interaction free energy) at fixed marginals; the population
    part is the MI change caused by the change of the marginals at fixed interaction term.
    Both orders of the comparison are averaged (B fitted to A's marginals, and A fitted to
    B's marginals), so the decomposition is symmetric and the two parts sum exactly to the
    MI change. Works on single tables or on stacks of tables.

    Args:
        table_a (numpy array): joint table(s) of the reference condition A.
        table_b (numpy array): joint table(s) of condition B.
        normalise (bool): use MI normalised by the joint entropy.

    Returns:
        dict with MI_A, MI_B, dMI (= MI_B - MI_A), dMI_int and dMI_pop.
    """

    mi_a = mi_from_table(table_a, normalise)
    mi_b = mi_from_table(table_b, normalise)
    # interaction of B with the marginals of A, and interaction of A with the marginals of B
    mi_b_on_a = mi_from_table(ipf_fit(table_b, table_a.sum(axis=-1), table_a.sum(axis=-2)), normalise)
    mi_a_on_b = mi_from_table(ipf_fit(table_a, table_b.sum(axis=-1), table_b.sum(axis=-2)), normalise)

    d_int = 0.5 * ((mi_b_on_a - mi_a) + (mi_b - mi_a_on_b))
    d_pop = 0.5 * ((mi_b - mi_b_on_a) + (mi_a_on_b - mi_a))
    return {"MI_A": mi_a, "MI_B": mi_b, "dMI": mi_b - mi_a, "dMI_int": d_int, "dMI_pop": d_pop}


def _decompose_arrays(int_traj_a, int_traj_b, pairs, num_states, pseudocount, normalise, chunk):
    """Decomposition of all pairs in chunks; returns a dict of arrays."""

    parts = {key: [] for key in ("MI_A", "MI_B", "dMI", "dMI_int", "dMI_pop")}
    for start in range(0, len(pairs), chunk):
        sub = pairs[start:start + chunk]
        res = decompose_pair(pair_tables(int_traj_a, sub, num_states, pseudocount),
                             pair_tables(int_traj_b, sub, num_states, pseudocount), normalise)
        for key in parts:
            parts[key].append(np.atleast_1d(res[key]))
    return {key: np.concatenate(val) if val else np.array([]) for key, val in parts.items()}


def _fractions(df):
    with np.errstate(divide="ignore", invalid="ignore"):
        df["f_int"] = df["dMI_int"] / df["dMI"]
        df["f_pop"] = df["dMI_pop"] / df["dMI"]
    return df


def decompose_delta_mi(int_traj_a, int_traj_b, fragment_pairs, num_states,
                       pseudocount=1e-3, normalise=True, chunk=1000):
    """
    Interaction and population parts of the MI change for a list of fragment pairs.

    Frames of all replicas of a condition are pooled (stack the integer trajectories
    before calling), because single MI blocks are too sparse for fitting joint tables.
    The fractions f_int and f_pop sum to 1. They lie outside [0, 1] when the two parts
    have opposite signs, i.e. when interaction and population changes partly cancel.

    Args:
        int_traj_a (numpy array): pooled integer SA trajectory of the reference condition A,
            shape (frames, fragments).
        int_traj_b (numpy array): pooled integer SA trajectory of condition B.
        fragment_pairs (iterable of tuples): fragment index pairs (i, j).
        num_states (int): number of letters in the alphabet.
        pseudocount (float): count added to every cell of the joint tables.
        normalise (bool): use MI normalised by the joint entropy.
        chunk (int): number of pairs processed at once (memory control).

    Returns:
        pandas DataFrame with one row per pair: FragmentPairs, MI_A, MI_B, dMI,
        dMI_int, dMI_pop, f_int, f_pop.
    """

    pairs = [tuple(p) for p in fragment_pairs]
    res = _decompose_arrays(int_traj_a, int_traj_b, np.asarray(pairs, dtype=np.int64).reshape(-1, 2),
                            num_states, pseudocount, normalise, chunk)
    df = pd.DataFrame({"FragmentPairs": pairs, **res})
    return _fractions(df)


def _block_resample(int_traj, block_size, rng):
    """Moving-block bootstrap sample of a time-ordered trajectory, same length as the input."""

    n_frames = len(int_traj)
    size = min(block_size, n_frames)
    n_blocks = int(np.ceil(n_frames / size))
    starts = rng.integers(0, n_frames - size + 1, n_blocks)
    index = (starts[:, None] + np.arange(size)[None, :]).ravel()[:n_frames]
    return int_traj[index]


def _bootstrap_replicate(int_trajs_a, int_trajs_b, pairs, num_states, block_size,
                         pseudocount, normalise, chunk, seed):
    """One moving-block bootstrap replicate of the decomposition (worker function)."""

    rng = np.random.default_rng(seed)
    sample_a = np.vstack([_block_resample(t, block_size, rng) for t in int_trajs_a])
    sample_b = np.vstack([_block_resample(t, block_size, rng) for t in int_trajs_b])
    res = _decompose_arrays(sample_a, sample_b, pairs, num_states, pseudocount, normalise, chunk)
    return {key: res[key] for key in ("dMI", "dMI_int", "dMI_pop")}


def delta_mi_significance(int_trajs_a, int_trajs_b, fragment_pairs, num_states, n_boot=100,
                             block_size=100, pseudocount=1e-3, normalise=True, seed=0, chunk=1000,
                             max_workers=None, disable_tqdm=False):
    """
    Significance of the total, interaction and population parts of the MI change.

    The estimates come from the pooled frames of all replicas of each condition. Their
    standard errors come from a moving-block bootstrap: each replica's time-ordered
    trajectory is resampled in blocks of consecutive frames, which preserves the time
    correlation within blocks. P-values are two-sided normal approximations of
    estimate / standard error, adjusted per component over all pairs (Benjamini-Hochberg).

    Args:
        int_trajs_a (list of numpy arrays): time-ordered integer SA trajectories of the
            replicas of the reference condition A, each of shape (frames, fragments).
        int_trajs_b (list of numpy arrays): the same for condition B.
        fragment_pairs (iterable of tuples): fragment index pairs (i, j).
        num_states (int): number of letters in the alphabet.
        n_boot (int): number of bootstrap samples.
        block_size (int): number of consecutive frames per bootstrap block.
        pseudocount (float): count added to every cell of the joint tables.
        normalise (bool): use MI normalised by the joint entropy.
        seed (int): seed of the bootstrap random number generator; each replicate gets its
            own child seed, so results do not depend on the number of workers.
        chunk (int): number of pairs processed at once (memory control).
        max_workers (int or None): number of worker processes for the bootstrap replicates.
            None equals all possible; 1 runs in the calling process.
        disable_tqdm (bool): whether to show progress bar or not.

    Returns:
        pandas DataFrame with one row per pair: FragmentPairs, MI_A, MI_B, dMI, dMI_int,
        dMI_pop, f_int, f_pop, and per component (dMI, int, pop) the standard error SE_*,
        the p-value P_* and the adjusted p-value AdjP_*.
    """

    pairs_list = [tuple(p) for p in fragment_pairs]
    pairs = np.asarray(pairs_list, dtype=np.int64).reshape(-1, 2)
    est = _decompose_arrays(np.vstack(int_trajs_a), np.vstack(int_trajs_b), pairs,
                            num_states, pseudocount, normalise, chunk)

    seeds = np.random.SeedSequence(seed).spawn(n_boot)
    components = ("dMI", "dMI_int", "dMI_pop")
    boot = {key: np.empty((n_boot, len(pairs))) for key in components}
    args = (int_trajs_a, int_trajs_b, pairs, num_states, block_size, pseudocount, normalise, chunk)
    progress_bar = tqdm(total=n_boot, desc="Bootstrap", unit="sample", disable=disable_tqdm)
    if max_workers == 1:
        for b in range(n_boot):
            res = _bootstrap_replicate(*args, seeds[b])
            for key in components:
                boot[key][b] = res[key]
            progress_bar.update(1)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_bootstrap_replicate, *args, seeds[b]): b for b in range(n_boot)}
            for future in as_completed(futures):
                res = future.result()
                for key in components:
                    boot[key][futures[future]] = res[key]
                progress_bar.update(1)
    progress_bar.close()

    df = _fractions(pd.DataFrame({"FragmentPairs": pairs_list, **est}))
    for key, label in zip(components, ("dMI", "int", "pop")):
        se = boot[key].std(axis=0, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            p_values = 2 * norm.sf(np.abs(est[key]) / se)
        p_values = np.where(np.isfinite(p_values), p_values, 1.0)
        df[f"SE_{label}"] = se
        df[f"P_{label}"] = p_values
        df[f"AdjP_{label}"] = multipletests(p_values, method="fdr_bh")[1] if len(p_values) else p_values
    return df


def _jackknife_replicate(int_trajs_a, int_trajs_b, pairs, num_states, pseudocount, normalise, chunk):
    """Decomposition of one leave-one-replica-out sample (worker function)."""

    res = _decompose_arrays(np.vstack(int_trajs_a), np.vstack(int_trajs_b), pairs,
                            num_states, pseudocount, normalise, chunk)
    return {key: res[key] for key in ("dMI", "dMI_int", "dMI_pop")}


def _jackknife_runs(int_trajs_a, int_trajs_b, pairs, num_states, pseudocount, normalise, chunk, max_workers):
    """Pooled estimates and leave-one-replica-out decompositions (first the replicas of A, then of B)."""

    if len(int_trajs_a) < 2 or len(int_trajs_b) < 2:
        raise ValueError("the jackknife needs at least two replicas per condition")
    est = _decompose_arrays(np.vstack(int_trajs_a), np.vstack(int_trajs_b), pairs,
                            num_states, pseudocount, normalise, chunk)
    samples = [([t for k, t in enumerate(int_trajs_a) if k != i], int_trajs_b) for i in range(len(int_trajs_a))]
    samples += [(int_trajs_a, [t for k, t in enumerate(int_trajs_b) if k != i]) for i in range(len(int_trajs_b))]
    args = (pairs, num_states, pseudocount, normalise, chunk)
    if max_workers == 1:
        loo = [_jackknife_replicate(a, b, *args) for a, b in samples]
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            loo = list(executor.map(_jackknife_replicate, *zip(*samples), *[[x] * len(samples) for x in args]))
    return est, loo


def _jackknife_test(estimate, loo_a, loo_b):
    """Jackknife standard error, Welch-Satterthwaite degrees of freedom and two-sided t-test p-value."""

    n_a, n_b = len(loo_a), len(loo_b)
    var_a = (n_a - 1) / n_a * ((loo_a - loo_a.mean(axis=0)) ** 2).sum(axis=0)
    var_b = (n_b - 1) / n_b * ((loo_b - loo_b.mean(axis=0)) ** 2).sum(axis=0)
    se = np.sqrt(var_a + var_b)
    with np.errstate(divide="ignore", invalid="ignore"):
        dof = (var_a + var_b) ** 2 / (var_a ** 2 / (n_a - 1) + var_b ** 2 / (n_b - 1))
        p_values = 2 * t_dist.sf(np.abs(estimate) / se, dof)
    return se, dof, np.where(np.isfinite(p_values), p_values, 1.0)


def _add_jackknife_columns(df, estimates, loo, n_a):
    for key, label in zip(("dMI", "dMI_int", "dMI_pop"), ("dMI", "int", "pop")):
        loo_values = np.array([r[key] for r in loo])
        se, dof, p_values = _jackknife_test(estimates[key], loo_values[:n_a], loo_values[n_a:])
        df[f"SEjk_{label}"] = se
        df[f"DFjk_{label}"] = dof
        df[f"Pjk_{label}"] = p_values
        df[f"AdjPjk_{label}"] = multipletests(p_values, method="fdr_bh")[1] if len(p_values) else p_values
    return df


def delta_mi_jackknife(int_trajs_a, int_trajs_b, fragment_pairs, num_states, pseudocount=1e-3,
                       normalise=True, chunk=1000, max_workers=None):
    """
    Significance of the total, interaction and population parts of the MI change,
    with standard errors from the variation between replicas.

    The estimates come from the pooled frames of all replicas of each condition. Each
    replica is left out in turn (within its condition), and the delete-one jackknife
    variances of the two conditions are added. Unlike the block bootstrap of
    delta_mi_significance, this includes the differences between replicas, so it is the
    stricter test when replicas sample different regions. P-values are two-sided t-tests
    with Welch-Satterthwaite degrees of freedom (n_A - 1 and n_B - 1 per condition),
    adjusted per component over all pairs (Benjamini-Hochberg).

    Args:
        int_trajs_a (list of numpy arrays): integer SA trajectories of the replicas of the
            reference condition A (at least 2), each of shape (frames, fragments).
        int_trajs_b (list of numpy arrays): the same for condition B.
        fragment_pairs (iterable of tuples): fragment index pairs (i, j).
        num_states (int): number of letters in the alphabet.
        pseudocount (float): count added to every cell of the joint tables.
        normalise (bool): use MI normalised by the joint entropy.
        chunk (int): number of pairs processed at once (memory control).
        max_workers (int or None): number of worker processes for the leave-one-out samples.
            None equals all possible; 1 runs in the calling process.

    Returns:
        pandas DataFrame with one row per pair: FragmentPairs, MI_A, MI_B, dMI, dMI_int,
        dMI_pop, f_int, f_pop, and per component (dMI, int, pop) the jackknife standard
        error SEjk_*, degrees of freedom DFjk_*, p-value Pjk_* and adjusted p-value AdjPjk_*.
    """

    pairs_list = [tuple(p) for p in fragment_pairs]
    pairs = np.asarray(pairs_list, dtype=np.int64).reshape(-1, 2)
    est, loo = _jackknife_runs(int_trajs_a, int_trajs_b, pairs, num_states, pseudocount, normalise, chunk, max_workers)
    df = _fractions(pd.DataFrame({"FragmentPairs": pairs_list, **est}))
    return _add_jackknife_columns(df, est, loo, len(int_trajs_a))


def _fragment_sums(values, pairs, n_frag):
    """Sum of pair values over all pairs of each fragment."""

    return np.bincount(pairs[:, 0], values, n_frag) + np.bincount(pairs[:, 1], values, n_frag)


def fragment_delta_mi_jackknife(int_trajs_a, int_trajs_b, num_states, fragment_pairs=None, pseudocount=1e-3,
                                normalise=True, chunk=1000, max_workers=None):
    """
    Hub-level significance: total, interaction and population parts of the MI change summed
    over all pairs of each fragment, tested with the replica jackknife.

    For every fragment, the signed changes of all its pairs are summed (net gain or loss of
    coupling of that fragment). The sum is computed for the pooled estimate and for every
    leave-one-replica-out sample, and tested as in delta_mi_jackknife. Summing over pairs
    averages out noise that is independent between pairs, and there is one test per fragment
    instead of one per pair. Changes of opposite sign on different pairs of a fragment cancel.

    Args:
        int_trajs_a (list of numpy arrays): integer SA trajectories of the replicas of the
            reference condition A (at least 2), each of shape (frames, fragments).
        int_trajs_b (list of numpy arrays): the same for condition B.
        num_states (int): number of letters in the alphabet.
        fragment_pairs (iterable of tuples or None): pairs to sum over; None uses all pairs.
        pseudocount (float): count added to every cell of the joint tables.
        normalise (bool): use MI normalised by the joint entropy.
        chunk (int): number of pairs processed at once (memory control).
        max_workers (int or None): number of worker processes for the leave-one-out samples.

    Returns:
        pandas DataFrame with one row per fragment: Fragment, NPairs, dMI, dMI_int, dMI_pop,
        f_int, f_pop, and per component (dMI, int, pop) SEjk_*, DFjk_*, Pjk_* and AdjPjk_*
        (Benjamini-Hochberg over fragments).
    """

    n_frag = int_trajs_a[0].shape[1]
    if fragment_pairs is None:
        fragment_pairs = [(i, j) for i in range(n_frag) for j in range(i + 1, n_frag)]
    pairs = np.asarray([tuple(p) for p in fragment_pairs], dtype=np.int64).reshape(-1, 2)
    est, loo = _jackknife_runs(int_trajs_a, int_trajs_b, pairs, num_states, pseudocount, normalise, chunk, max_workers)

    keys = ("dMI", "dMI_int", "dMI_pop")
    est_frag = {key: _fragment_sums(est[key], pairs, n_frag) for key in keys}
    loo_frag = [{key: _fragment_sums(r[key], pairs, n_frag) for key in keys} for r in loo]
    df = _fractions(pd.DataFrame({"Fragment": np.arange(n_frag),
                                  "NPairs": _fragment_sums(np.ones(len(pairs)), pairs, n_frag).astype(int),
                                  **est_frag}))
    return _add_jackknife_columns(df, est_frag, loo_frag, len(int_trajs_a))


def fragment_fractions(pair_decomposition):
    """
    Aggregates the pair decomposition per fragment (node level).

    For each fragment, the MI changes of all its pairs are summed, and the fractions
    are those of the summed change.

    Args:
        pair_decomposition (pandas DataFrame): output of decompose_delta_mi.

    Returns:
        pandas DataFrame with one row per fragment: Fragment, NPairs, dMI, dMI_int,
        dMI_pop, f_int, f_pop.
    """

    rows = []
    for _, row in pair_decomposition.iterrows():
        for frag in row["FragmentPairs"]:
            rows.append({"Fragment": frag, "dMI": row["dMI"],
                         "dMI_int": row["dMI_int"], "dMI_pop": row["dMI_pop"]})
    if not rows:
        return pd.DataFrame(columns=["Fragment", "NPairs", "dMI", "dMI_int", "dMI_pop", "f_int", "f_pop"])

    df = pd.DataFrame(rows).groupby("Fragment").agg(
        NPairs=("dMI", "size"), dMI=("dMI", "sum"),
        dMI_int=("dMI_int", "sum"), dMI_pop=("dMI_pop", "sum")).reset_index()
    return _fractions(df)
