"""Gower-distance privacy metrics used in the thesis revision analysis.

The formulas in this module are intentionally kept equivalent to the thesis
revision implementation. The orchestration and file I/O live in
:mod:`src.privacy_evaluation`.
"""

import numpy as np
import pandas as pd


def _split_X(X, num_idx, cat_idx):
    X_num = X[:, num_idx] if len(num_idx) else None
    X_cat = X[:, cat_idx] if len(cat_idx) else None
    return X_num, X_cat


def _gower_num_block(X_block_num, Y_num, num_min, num_rng):
    """Return the summed normalized numeric contribution to Gower distance."""
    if X_block_num is None or Y_num is None or X_block_num.shape[1] == 0:
        return 0.0, 0

    safe_rng = np.where(num_rng == 0.0, 1.0, num_rng)
    Xn = np.clip((X_block_num - num_min) / safe_rng, 0.0, 1.0)
    Yn = np.clip((Y_num - num_min) / safe_rng, 0.0, 1.0)
    diff = np.abs(Xn[:, None, :] - Yn[None, :, :])
    return diff.sum(axis=2), X_block_num.shape[1]


def _gower_cat_block(X_block_cat, Y_cat):
    """Return the summed categorical contribution to Gower distance."""
    if X_block_cat is None or Y_cat is None or X_block_cat.shape[1] == 0:
        return 0.0, 0

    mismatches = (X_block_cat[:, None, :] != Y_cat[None, :, :]).astype(float)
    return mismatches.sum(axis=2), X_block_cat.shape[1]


def gower_min_distances(X, Y, num_idx, cat_idx, num_min, num_rng, block_size=1024):
    """Compute each row's minimum Gower distance to a row in ``Y``."""
    n_x = X.shape[0]
    X_num, X_cat = _split_X(X, num_idx, cat_idx)
    Y_num, Y_cat = _split_X(Y, num_idx, cat_idx)

    min_dist = np.full(n_x, np.inf, dtype=float)
    argmin_idx = np.full(n_x, -1, dtype=int)

    p_num = X_num.shape[1] if X_num is not None else 0
    p_cat = X_cat.shape[1] if X_cat is not None else 0
    denom = float(p_num + p_cat) if (p_num + p_cat) > 0 else 1.0

    for start in range(0, n_x, block_size):
        end = min(start + block_size, n_x)
        num_contrib, num_count = _gower_num_block(
            X_num[start:end] if p_num else None,
            Y_num if p_num else None,
            num_min,
            num_rng,
        )
        cat_contrib, cat_count = _gower_cat_block(
            X_cat[start:end] if p_cat else None,
            Y_cat if p_cat else None,
        )

        if num_count and cat_count:
            dist_block = (num_contrib + cat_contrib) / denom
        elif num_count:
            dist_block = num_contrib / denom
        elif cat_count:
            dist_block = cat_contrib / denom
        else:
            raise ValueError("No columns provided for Gower distance.")

        local_argmin = dist_block.argmin(axis=1)
        local_min = dist_block[np.arange(dist_block.shape[0]), local_argmin]
        update = local_min < min_dist[start:end]
        min_dist[start:end][update] = local_min[update]
        argmin_idx[start:end][update] = local_argmin[update]

    return min_dist, argmin_idx


def gower_min_distances_loo(X, num_idx, cat_idx, num_min, num_rng, block_size=1024):
    """Compute leave-one-out nearest-neighbor Gower distances within ``X``."""
    n = X.shape[0]
    d_rr = np.full(n, np.inf, dtype=float)
    idx_rr = np.full(n, -1, dtype=int)
    X_num, X_cat = _split_X(X, num_idx, cat_idx)
    p_num = X_num.shape[1] if X_num is not None else 0
    p_cat = X_cat.shape[1] if X_cat is not None else 0
    denom = float(p_num + p_cat) if (p_num + p_cat) > 0 else 1.0

    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        num_contrib, _ = _gower_num_block(
            X_num[start:end] if p_num else None,
            X_num if p_num else None,
            num_min,
            num_rng,
        )
        cat_contrib, _ = _gower_cat_block(
            X_cat[start:end] if p_cat else None,
            X_cat if p_cat else None,
        )

        if p_num and p_cat:
            dist = (num_contrib + cat_contrib) / denom
        elif p_num:
            dist = num_contrib / denom
        else:
            dist = cat_contrib / denom

        rows = np.arange(start, end)
        dist[np.arange(end - start), rows] = np.inf
        local_argmin = dist.argmin(axis=1)
        local_min = dist[np.arange(dist.shape[0]), local_argmin]
        d_rr[start:end] = local_min
        idx_rr[start:end] = local_argmin

    return d_rr, idx_rr


def authenticity_metric(real_train, synth, num_idx, cat_idx, epsilon=0.0, block_size=1024):
    """Compute the Gower-distance authenticity metric of Alaa et al."""
    if len(num_idx):
        num_min = real_train[:, num_idx].min(axis=0)
        num_max = real_train[:, num_idx].max(axis=0)
        num_rng = num_max - num_min
    else:
        num_min = np.array([])
        num_rng = np.array([])

    d_rr, idx_rr = gower_min_distances_loo(
        real_train,
        num_idx,
        cat_idx,
        num_min,
        num_rng,
        block_size=block_size,
    )
    d_sr, match_idx = gower_min_distances(
        synth,
        real_train,
        num_idx,
        cat_idx,
        num_min,
        num_rng,
        block_size=block_size,
    )

    d_rr_at_match = d_rr[match_idx]
    authentic_flags = d_sr > d_rr_at_match
    authenticity = float(authentic_flags.mean())

    return {
        "authenticity": authenticity,
        "unauthentic_rate": 1.0 - authenticity,
        "near_dups": int(np.sum(d_sr <= epsilon)),
        "d_sr": d_sr,
        "match_idx": match_idx,
        "d_rr_at_match": d_rr_at_match,
        "d_rr_all": d_rr,
        "idx_rr_all": idx_rr,
    }


def encode_categoricals_joint(
    df_train: pd.DataFrame,
    df_synth: pd.DataFrame,
    cat_cols,
    missing_token_prefix="__MISSING__",
):
    """Encode categoricals with a deterministic vocabulary shared by two frames."""
    train = df_train.copy()
    synth = df_synth.copy()
    mappings = {}

    for col in cat_cols:
        missing_token = f"{missing_token_prefix}:{col}"
        train_col = train[col].astype("object").where(train[col].notna(), other=missing_token)
        synth_col = synth[col].astype("object").where(synth[col].notna(), other=missing_token)

        vocabulary = pd.Index(train_col.unique()).union(pd.Index(synth_col.unique()))
        vocabulary = sorted(vocabulary, key=lambda value: str(value))
        mapping = {value: index for index, value in enumerate(vocabulary)}
        mappings[col] = mapping

        train[col] = train_col.map(mapping).astype("int64")
        synth[col] = synth_col.map(mapping).astype("int64")

    return train, synth, mappings


def summarize_authenticity_outputs(out):
    """Convert detailed authenticity output to JSON-ready reporting values."""
    d_rr_all = out["d_rr_all"]
    d_sr = out["d_sr"]
    d_rr_match = out["d_rr_at_match"]
    tau1, tau5 = np.quantile(d_rr_all, [0.01, 0.05])

    ratio = d_sr / d_rr_match
    return {
        "authenticity": float(out["authenticity"]),
        "too_close_1pct": float(np.mean(d_sr <= tau1)),
        "too_close_5pct": float(np.mean(d_sr <= tau5)),
        "r_median": float(np.median(ratio)),
        "r_q25": float(np.percentile(ratio, 25)),
        "r_q75": float(np.percentile(ratio, 75)),
        "near_dups": int(out["near_dups"]),
        "min_dsr": float(np.min(d_sr)),
    }


def compute_aa(train_np, test_np, synth_np, num_idx, cat_idx, block_size=1024):
    """Compute train/test nearest-neighbor adversarial accuracy and privacy loss."""
    if len(num_idx):
        num_min = train_np[:, num_idx].min(axis=0)
        num_max = train_np[:, num_idx].max(axis=0)
        num_rng = num_max - num_min
    else:
        num_min = np.array([])
        num_rng = np.array([])

    d_TT, _ = gower_min_distances_loo(
        train_np, num_idx, cat_idx, num_min, num_rng, block_size
    )
    d_EE, _ = gower_min_distances_loo(
        test_np, num_idx, cat_idx, num_min, num_rng, block_size
    )
    d_SS, _ = gower_min_distances_loo(
        synth_np, num_idx, cat_idx, num_min, num_rng, block_size
    )

    d_T_S, _ = gower_min_distances(
        train_np, synth_np, num_idx, cat_idx, num_min, num_rng, block_size
    )
    d_S_T, _ = gower_min_distances(
        synth_np, train_np, num_idx, cat_idx, num_min, num_rng, block_size
    )
    d_E_S, _ = gower_min_distances(
        test_np, synth_np, num_idx, cat_idx, num_min, num_rng, block_size
    )
    d_S_E, _ = gower_min_distances(
        synth_np, test_np, num_idx, cat_idx, num_min, num_rng, block_size
    )

    left_train = np.mean(d_T_S > d_TT)
    right_train = np.mean(d_S_T > d_SS)
    train_aa = 0.5 * (left_train + right_train)

    left_test = np.mean(d_E_S > d_EE)
    right_test = np.mean(d_S_E > d_SS)
    test_aa = 0.5 * (left_test + right_test)

    d_T_E, _ = gower_min_distances(
        train_np, test_np, num_idx, cat_idx, num_min, num_rng, block_size
    )
    d_E_T, _ = gower_min_distances(
        test_np, train_np, num_idx, cat_idx, num_min, num_rng, block_size
    )
    rr_aa_left = np.mean(d_T_E > d_TT)
    rr_aa_right = np.mean(d_E_T > d_EE)
    train_vs_test_aa = 0.5 * (rr_aa_left + rr_aa_right)

    return {
        "train_AA": float(train_aa),
        "test_AA": float(test_aa),
        "privacy_loss": float(test_aa - train_aa),
        "train_vs_test_AA": float(train_vs_test_aa),
    }
