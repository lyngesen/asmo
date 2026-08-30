import numpy as np


def build_B_front_2d(B):
    """
    Preprocess B (minimization) into a monotone Pareto front usable for fast dominance checks.

    Parameters
    ----------
    B : (m, 2) ndarray
        Candidate dominating points, minimization objectives.

    Returns
    -------
    front_x : (k,) ndarray
        Sorted by x ascending (first objective).
    front_y : (k,) ndarray
        Corresponding y values of points on the front (strictly decreasing after prefix-min filter).
    pref_ymin : (k,) ndarray
        Prefix minima of front_y (pref_ymin[i] = min(front_y[:i+1])).
    pref_argmin : (k,) ndarray
        Indices into [0..k-1] giving the argmin position achieving pref_ymin at each prefix.

    Notes
    -----
    - Duplicates on x are collapsed by keeping the smallest y per x.
    - Points dominated within B are removed.
    """
    B = np.asarray(B)
    if B.ndim != 2 or B.shape[1] != 2:
        raise ValueError("B must be a 2D array of shape (m, 2)")

    # Sort by x ascending, then y ascending; keep minimal y per unique x
    order = np.lexsort((B[:, 1], B[:, 0]))
    B_sorted = B[order]
    # Collapse duplicates on x by taking minimal y per unique x
    x_vals, x_start_idx = np.unique(B_sorted[:, 0], return_index=True)
    # For each unique x, find minimal y within the block [start, next_start)
    # We can compute mins via splitting indices
    y_mins = np.minimum.reduceat(B_sorted[:, 1], x_start_idx)
    B_uniq = np.column_stack((x_vals, y_mins))

    # Now scan and keep points that set a new running minimum on y
    # (standard 2D Pareto front construction for minimization)
    x = B_uniq[:, 0]
    y = B_uniq[:, 1]
    keep = np.zeros_like(y, dtype=bool)
    running_min = np.inf
    for i in range(len(y)):
        if y[i] < running_min:
            keep[i] = True
            running_min = y[i]
    front_x = x[keep]
    front_y = y[keep]

    # Prefix minima of y and their argmin indices
    pref_ymin = np.empty_like(front_y)
    pref_argmin = np.empty_like(np.arange(front_y.size))
    cur_min = np.inf
    cur_arg = -1
    for i in range(front_y.size):
        if front_y[i] < cur_min:
            cur_min = front_y[i]
            cur_arg = i
        pref_ymin[i] = cur_min
        pref_argmin[i] = cur_arg

    return front_x, front_y, pref_ymin, pref_argmin


def check_non_dominated_A_by_B_2d(
    A, front_x, front_y, pref_ymin, pref_argmin, nonDomA, strictDom=True, eps=0.0
):
    """
    Update in-place which points of A are NOT dominated by B in 2D (minimization).

    Parameters
    ----------
    A : (n, 2) ndarray
        Points to check (minimization).
    front_x, front_y, pref_ymin, pref_argmin :
        Outputs of build_B_front_2d(B).
    nonDomA : (n,) boolean ndarray
        In/out marker array: True = currently considered non-dominated.
        This is updated in-place; previously False entries are left as-is.
    strictDom : bool
        If True, use strict dominance:
            b <= a in both coords AND (b_x < a_x OR b_y < a_y)
        If False, use non-strict dominance:
            b <= a in both coords
    eps : float
        Optional tolerance for strict-equality tests on y (default 0.0).
        If >0, comparisons use:
            y_min < a_y - eps  (strict)   and
            y_min <= a_y + eps (non-strict).
        For the strict tie-break (x < a_x when y ties), we still use exact <.

    Notes
    -----
    - Assumes minimization in both coordinates.
    - Vectorized over A; O(n log k), where k = |B_front|.
    - Only marks dominated points by flipping nonDomA[i] to False; non-dominated remain True.
    """
    A = np.asarray(A)
    if A.ndim != 2 or A.shape[1] != 2:
        raise ValueError("A must be a 2D array of shape (n, 2)")
    if nonDomA.shape[0] != A.shape[0]:
        raise ValueError("nonDomA length must match number of rows in A")

    # Only process those still considered non-dominated
    active = np.where(nonDomA)[0]
    if active.size == 0 or front_x.size == 0:
        return  # Nothing to update

    ax = A[active, 0]
    ay = A[active, 1]

    # For each a_x, find rightmost index i with front_x[i] <= a_x
    idx = np.searchsorted(front_x, ax, side="right") - 1
    valid = idx >= 0
    if not np.any(valid):
        return

    idx_v = idx[valid]
    # For the prefix up to idx_v, we use the argmin (k) where y is minimal
    k = pref_argmin[idx_v]
    y_min = pref_ymin[idx_v]
    x_at_k = front_x[k]

    if not strictDom:
        # Non-strict dominance: exists b with b_x <= a_x and b_y <= a_y
        dominated = y_min <= (ay[valid] + eps)
    else:
        # Strict dominance:
        # Case 1: y_min < a_y  => dominated
        # Case 2: y_min == a_y (within eps), dominated only if x_at_k < a_x
        less_y = y_min < (ay[valid] - eps)
        tie_y = ~less_y & (np.abs(y_min - ay[valid]) <= eps)
        dominated = less_y | (tie_y & (x_at_k < ax[valid]))

    # Write back: active[valid][dominated] are dominated -> set to False
    to_flip_idx = active[valid][dominated]
    nonDomA[to_flip_idx] = False


def check_non_dominated_A_by_B(
    A: np.ndarray,
    B: np.ndarray,
    strict: bool = False,
    return_indices: bool = False,
    eps: float = 0.0,
) -> np.ndarray:
    """
    Return the subset of points in A that are NOT dominated by any point in B (2D minimization).

    Parameters
    ----------
    A : (n, 2) ndarray
        Candidate points to keep if not dominated by B.
    B : (m, 2) ndarray
        Dominating candidate points.
    strict : bool, default False
        If False (non-strict): b dominates a if b_x <= a_x and b_y <= a_y.
        If True  (strict):     b dominates a if b_x <= a_x and b_y <= a_y
                               and at least one is strict (<). Implemented by:
                               b_y < a_y OR (b_y == a_y and b_x < a_x).
    return_indices : bool, default False
        If True, return the indices of A that are NOT dominated by B.
        If False, return the filtered A points.
    eps : float, default 0.0
        Numerical tolerance applied on the y-comparison. For strict:
            y_min < a_y - eps ⇒ dominated
            |y_min - a_y| <= eps ⇒ tie; dominated only if x_at_k < a_x
        For non-strict:
            y_min <= a_y + eps ⇒ dominated

    Returns
    -------
    A_nd : (k, 2) ndarray  or  idx : (k,) ndarray
        If return_indices=False, returns A[mask] where mask is True for non-dominated points.
        If return_indices=True, returns the indices of A that are non-dominated by B.

    Notes
    -----
    - Assumes minimization on both coordinates.
    - Time complexity: O(m log m) to preprocess B and O(n log k) for queries,
      where k is the size of B's Pareto front.
    """
    A = np.asarray(A)
    B = np.asarray(B)
    if A.ndim != 2 or A.shape[1] != 2:
        raise ValueError("A must be a 2D array with shape (n, 2).")
    if B.ndim != 2 or B.shape[1] != 2:
        raise ValueError("B must be a 2D array with shape (m, 2).")

    n = A.shape[0]
    if B.shape[0] == 0:
        # Nothing can dominate A
        mask_keep = np.ones(n, dtype=bool)
        return np.where(mask_keep)[0] if return_indices else A

    # ---- Build 2D Pareto front of B (minimization) and prefix minima of y ----
    # Sort by x asc, then y asc
    order = np.lexsort((B[:, 1], B[:, 0]))
    B_sorted = B[order]

    # Collapse duplicate x by keeping the minimal y per x
    x_vals, x_start = np.unique(B_sorted[:, 0], return_index=True)
    # compute minimal y per block via reduceat
    y_mins = np.minimum.reduceat(B_sorted[:, 1], x_start)
    B_uniq = np.column_stack((x_vals, y_mins))

    # Keep points that define a new running minimum on y
    x = B_uniq[:, 0]
    y = B_uniq[:, 1]
    keep = np.zeros_like(y, dtype=bool)
    run_min = np.inf
    for i in range(y.size):
        if y[i] < run_min:
            keep[i] = True
            run_min = y[i]
    front_x = x[keep]
    front_y = y[keep]

    # If B had no effect (e.g., NaNs filtered out), nothing dominates
    if front_x.size == 0:
        mask_keep = np.ones(n, dtype=bool)
        return np.where(mask_keep)[0] if return_indices else A

    # Prefix minima of front_y and argmin indices
    pref_ymin = np.empty_like(front_y)
    pref_argmin = np.empty(front_y.size, dtype=int)
    cur_min = np.inf
    cur_arg = -1
    for i in range(front_y.size):
        if front_y[i] < cur_min:
            cur_min = front_y[i]
            cur_arg = i
        pref_ymin[i] = cur_min
        pref_argmin[i] = cur_arg

    # ---- Vectorized dominance checks for all A ----
    ax = A[:, 0]
    ay = A[:, 1]

    # For each a_x, find rightmost index with front_x[i] <= a_x
    idx = np.searchsorted(front_x, ax, side="right") - 1
    valid = idx >= 0

    # Default: keep all; then mark dominated as False
    mask_keep = np.ones(n, dtype=bool)
    if not np.any(valid):
        # Nothing in front has x <= a_x for any a; all remain True
        return np.where(mask_keep)[0] if return_indices else A

    idx_v = idx[valid]
    k = pref_argmin[idx_v]
    y_min = pref_ymin[idx_v]
    x_at_k = front_x[k]

    if not strict:
        # Non-strict: dominated if ∃b with b_x <= a_x and b_y <= a_y
        dominated_valid = y_min <= (ay[valid] + eps)
    else:
        # Strict:
        # dominated if y_min < a_y - eps OR (|y_min - a_y| <= eps and x_at_k < a_x)
        less_y = y_min < (ay[valid] - eps)
        tie_y = ~less_y & (np.abs(y_min - ay[valid]) <= eps)
        dominated_valid = less_y | (tie_y & (x_at_k < ax[valid]))

    # Apply only to valid positions
    mask_keep[valid] &= ~dominated_valid

    return np.where(mask_keep)[0] if return_indices else A[mask_keep]


def check_dominated_A_by_B(
    A: np.ndarray,
    B: np.ndarray,
    strict: bool = False,
    return_indices: bool = False,
    eps: float = 0.0,
) -> np.ndarray:
    """
    Return the subset of points in A that ARE dominated by points in B (2D minimization).

    Parameters
    ----------
    A : (n, 2) ndarray
        Points to test for dominance.
    B : (m, 2) ndarray
        Dominating candidate points.
    strict : bool, default False
        If False: b dominates a if (b_x <= a_x and b_y <= a_y).
        If True:  one inequality must be strict.
    return_indices : bool, default False
        If True, return indices of dominated points, else return the dominated points.
    eps : float, default 0.0
        Numerical tolerance for dominance checks.

    Returns
    -------
    dominated_points : ndarray
        If return_indices=False:  A[dominated_mask]
        If return_indices=True:   indices of A that are dominated by B
    """
    A = np.asarray(A)
    B = np.asarray(B)
    if A.ndim != 2 or A.shape[1] != 2:
        raise ValueError("A must be a 2D array with shape (n, 2).")
    if B.ndim != 2 or B.shape[1] != 2:
        raise ValueError("B must be a 2D array with shape (m, 2).")

    n = A.shape[0]
    if B.shape[0] == 0:
        # No point can dominate A
        return np.empty((0, 2)) if not return_indices else np.array([], dtype=int)

    # ---- PREPROCESS B into its 2D Pareto front for minimization ----
    order = np.lexsort((B[:, 1], B[:, 0]))
    B_sorted = B[order]

    x_vals, x_start = np.unique(B_sorted[:, 0], return_index=True)
    y_mins = np.minimum.reduceat(B_sorted[:, 1], x_start)
    B_uniq = np.column_stack((x_vals, y_mins))

    x = B_uniq[:, 0]
    y = B_uniq[:, 1]
    keep = np.zeros_like(y, dtype=bool)
    run = np.inf
    for i in range(y.size):
        if y[i] < run:
            keep[i] = True
            run = y[i]
    front_x = x[keep]
    front_y = y[keep]

    if front_x.size == 0:
        return np.empty((0, 2)) if not return_indices else np.array([], dtype=int)

    # Prefix minima on y
    pref_ymin = np.empty_like(front_y)
    pref_argmin = np.empty(front_y.size, dtype=int)
    cur_min = np.inf
    cur_arg = -1
    for i in range(front_y.size):
        if front_y[i] < cur_min:
            cur_min = front_y[i]
            cur_arg = i
        pref_ymin[i] = cur_min
        pref_argmin[i] = cur_arg

    # ---- CHECK DOMINANCE ----
    ax = A[:, 0]
    ay = A[:, 1]

    idx = np.searchsorted(front_x, ax, side="right") - 1
    valid = idx >= 0

    dominated = np.zeros(n, dtype=bool)
    if not np.any(valid):
        return np.empty((0, 2)) if not return_indices else np.array([], dtype=int)

    idx_v = idx[valid]
    k = pref_argmin[idx_v]
    y_min = pref_ymin[idx_v]
    x_at_k = front_x[k]

    if not strict:
        dominated_valid = y_min <= (ay[valid] + eps)
    else:
        less_y = y_min < (ay[valid] - eps)
        tie_y = ~less_y & (np.abs(y_min - ay[valid]) <= eps)
        dominated_valid = less_y | (tie_y & (x_at_k < ax[valid]))

    dominated[valid] = dominated_valid

    if return_indices:
        return np.where(dominated)[0]
    else:
        return A[dominated]
