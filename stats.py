import numpy as np
from collections import Counter


def run_length_stats(a: np.ndarray):
    """
    Compute run-length statistics for a 1-D uint8 array.
    Returns:
        mean_run, max_run, run_hist (dict: run_length -> count)
    """
    assert a.dtype == np.uint8 and a.ndim == 1

    # Find indices where value changes
    diff = np.diff(a)
    change_points = np.nonzero(diff != 0)[0] + 1
    # Add start and end
    idx = np.concatenate(([0], change_points, [len(a)]))
    # Run lengths = differences of segment boundaries
    run_lengths = np.diff(idx)

    mean_run = run_lengths.mean()
    max_run = run_lengths.max()
    # Histogram of run lengths (as Python dict)
    unique, counts = np.unique(run_lengths, return_counts=True)
    run_hist = dict(zip(unique.tolist(), counts.tolist()))

    return mean_run, max_run, run_hist


def context_entropy(a: np.ndarray, order: int = 1) -> float:
    """
    Estimate context (conditional) entropy H(X | context) for a 1-D uint8 array.

    order = 1  -> H(X_t | X_{t-1})
    order = 2  -> H(X_t | X_{t-1}, X_{t-2}), etc.

    Returns entropy in bits per symbol.
    """
    assert a.dtype == np.uint8 and a.ndim == 1
    assert order >= 0

    n = len(a)
    if n <= order:
        return 0.0

    if order == 0:
        # ordinary zero-order entropy
        counts = np.bincount(a, minlength=256)
        probs = counts[counts > 0] / n
        return float(-np.sum(probs * np.log2(probs)))

    # Build conditional counts: context -> Counter(next_symbol)
    ctx_counts = {}
    for i in range(order, n):
        ctx = tuple(a[i - order:i])  # context as tuple of previous bytes
        x = int(a[i])
        if ctx not in ctx_counts:
            ctx_counts[ctx] = Counter()
        ctx_counts[ctx][x] += 1

    # Compute H(X | context) = sum_ctx P(ctx) * H(X | ctx)
    total = n - order
    H = 0.0
    for ctx, cnt in ctx_counts.items():
        ctx_total = sum(cnt.values())
        p_ctx = ctx_total / total
        probs = np.array(list(cnt.values()), dtype=float) / ctx_total
        H_ctx = -np.sum(probs * np.log2(probs))
        H += p_ctx * H_ctx

    return float(H)

def stat_report(data, predicted_data=None, previous_report=None):  
    if previous_report:
        report = previous_report  
    else:
        report = {}

    # -----------------------------
    # 1. Block-wise Entropy
    # -----------------------------
    def shannon_entropy(block):
        counts = np.array(list(Counter(block).values()))
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs + 1e-12))  # avoid log(0)
    
    if "int_entropy" not in report:
        report["int_entropy"] = shannon_entropy(data.view(np.uint32))
    if "entropy" not in report:
        report["entropy"] = shannon_entropy(data.view(np.uint8))
    if "cont_entr_1" not in report:
        report["cont_entr_1"] = context_entropy(data.view(np.uint8), 1)
    if "cont_entr_2" not in report:
        report["cont_entr_2"] = context_entropy(data.view(np.uint8), 2)
    if "cont_entr_4" not in report:
        report["cont_entr_4"] = context_entropy(data.view(np.uint8), 4)
    # if "cont_entr_8" not in report: 
    #     report["cont_entr_8"] = context_entropy(data.view(np.uint8), 8)
    if "avg_rl" not in report:
        report["avg_rl"] = run_length_stats(data.view(np.uint8))[0]

    return report