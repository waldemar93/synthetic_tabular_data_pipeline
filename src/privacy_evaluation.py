import copy
import heapq
import json
import os
from dataclasses import dataclass, MISSING
from typing import Literal, Optional, List, Dict

import hydra
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.base import PROJECT_ROOT
from src.helpers import load_experiment_data, load_columns_json_for_experiment
from src.privacy_metrics import (
    authenticity_metric,
    compute_aa,
    encode_categoricals_joint,
    summarize_authenticity_outputs,
)


SUPPORTED_PRIVACY_METRICS = {
    'hamming_distance',
    'authenticity',
    'nearest_neighbor_adversarial_accuracy',
}


@dataclass
class PrivacyEvalConfig:
    """ Configuration for the privacy evaluation of synthetic data. """
    experiment_name: str = MISSING
    synthetic_data: Dict[str, List[str]]
    hamming_dist_num_bins: int = 10
    privacy_metrics: Optional[List[str]] = None
    authenticity_epsilon: float = 0.00001
    gower_block_size: int = 1024


@dataclass
class MainPrivacyEvalConfig:
    """Main configuration for the privacy evaluation synthetic data, containing a PrivacyEvalConfig instance."""
    privacy_eval_gen: PrivacyEvalConfig = MISSING


def _get_privacy_metric_names(privacy_metrics=None) -> List[str]:
    """Return configured metric names while preserving the legacy default."""
    if privacy_metrics is None:
        return ['hamming_distance']

    metric_names = OmegaConf.to_object(privacy_metrics)
    return list(metric_names)


def _validate_config(cfg: PrivacyEvalConfig) -> None:
    """
    Validate the provided configuration for the privacy evaluation.

    :param cfg: Configuration to validate.
    """

    exp_folder_path = os.path.join(PROJECT_ROOT, 'experiments', cfg.experiment_name)
    if not os.path.exists(exp_folder_path):
        raise AttributeError(f'The provided experiment: "{cfg.experiment_name}" does not exist.')

    for model in cfg.synthetic_data:
        model_folder_path = os.path.join(exp_folder_path, model.upper())
        if not os.path.exists(model_folder_path):
            raise AttributeError(f'The provided model folder "{model.upper()}" does not exist for the experiment '
                                 f'"{cfg.experiment_name}".')
        if not isinstance(OmegaConf.to_object(cfg.synthetic_data[model]), list):
            raise AttributeError(f'A list of filenames needs to be provided for each generative model. '
                                 f'For the model "{model.upper()}" no list was provided.')
        for path in cfg.synthetic_data[model]:
            path = path if path.endswith('.csv') else path + '.csv'
            file_path = os.path.join(model_folder_path, 'synthetic_data', path)
            if not os.path.exists(file_path):
                raise AttributeError(f'The provided filename "{path}" for the model "{model.upper()}" does not exist '
                                     f'for the experiment "{cfg.experiment_name}".')

    privacy_metrics = _get_privacy_metric_names(getattr(cfg, 'privacy_metrics', None))
    if not privacy_metrics:
        raise AttributeError('At least one privacy metric needs to be provided.')

    unsupported_metrics = set(privacy_metrics) - SUPPORTED_PRIVACY_METRICS
    if unsupported_metrics:
        raise AttributeError(
            f'Unknown privacy metric(s): {", ".join(sorted(unsupported_metrics))}. '
            f'Choose from: {", ".join(sorted(SUPPORTED_PRIVACY_METRICS))}.'
        )

    if 'hamming_distance' in privacy_metrics and not isinstance(cfg.hamming_dist_num_bins, int):
        raise AttributeError(f'The parameter "hamming_dist_num_bins" needs to be an Integer, '
                             f'but {cfg.hamming_dist_num_bins} was provided.')

    authenticity_epsilon = getattr(cfg, 'authenticity_epsilon', 0.00001)
    if not isinstance(authenticity_epsilon, (int, float)) or authenticity_epsilon < 0:
        raise AttributeError('The parameter "authenticity_epsilon" needs to be a non-negative number.')

    gower_block_size = getattr(cfg, 'gower_block_size', 1024)
    if not isinstance(gower_block_size, int) or gower_block_size <= 0:
        raise AttributeError('The parameter "gower_block_size" needs to be a positive Integer.')


def hamming_distance(df_from, df_to, categorical_cols, num_bins=10):
    """
    Calculate the Hamming distance for each datapoint in df_from to each datapoint to df_to.

    :param df_from: Source dataframe.
    :param df_to: Target dataframe.
    :param categorical_cols: List of columns that are considered categorical.
    :param num_bins: Number of bins for discretizing numerical columns.

    :return: A tuple containing: Average minimum hamming distance, Average minimum hamming distance difference for the
    closest two datapoints, List of percentile for the minimum hamming distances, Number of exact matches.
    """

    df_from = copy.deepcopy(df_from)
    df_to = copy.deepcopy(df_to)

    # all columns that are not categorical are considered numerical
    num_cols = [col for col in df_to.columns if col not in categorical_cols]

    for col in num_cols:
        # get the cutoff values for num_bins for each numerical columns
        cut_offs = pd.qcut(df_to[col], q=num_bins, retbins=True, duplicates='drop')[1]

        # Set the first bin to negative infinity and the last to positive infinity
        cut_offs[0] = -np.inf
        cut_offs[-1] = np.inf

        df_to[col] = pd.cut(df_to[col], bins=cut_offs, labels=list(range(len(cut_offs) - 1)))
        df_from[col] = pd.cut(df_from[col], bins=cut_offs, labels=list(range(len(cut_offs) - 1)))

    # calculate dist for every datapoint in synth_df to every point in orig_df
    min_dists = []
    min_dist_two_dif = []
    for i in range(len(df_from)):
        # calculate the Hamming distance to every point in orig_df
        dists = (df_to != df_from.iloc[i]).sum(axis=1)
        smallest_two = heapq.nsmallest(2, dists)
        min_dists.append(smallest_two[0])
        min_dist_two_dif.append(smallest_two[1] - smallest_two[0])

    # calculate the average of the minimum Hamming distances
    avg_min_dist = np.mean(min_dists)
    avg_min_dist_two_dif = np.mean(min_dist_two_dif)

    # min dist extra
    min_dists.sort()
    num_zeros = min_dists.count(0)
    percentiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    # Percentiles
    percentile_values = np.percentile(min_dists, [p * 100 for p in percentiles])

    return avg_min_dist, avg_min_dist_two_dif, percentile_values, num_zeros


def eval_hamming_distances(df_from: pd.DataFrame, df_to: pd.DataFrame,
                           categorical_cols: List[str],
                           mode: Literal['balanced', 'unbalanced'] = 'balanced',
                           balanced_count: Optional[int] = None,
                           df_from_df_to_same=False,
                           hamming_dist_num_bins=10):
    """
    Evaluate Hamming distances between two dataframes.

    :param df_from: Source dataframe.
    :param df_to: Target dataframe.
    :param categorical_cols: List of columns that are considered categorical.
    :param mode: Evaluation mode ('balanced' or 'unbalanced').
    :param balanced_count: Number of samples in each balanced split.
    :param df_from_df_to_same: Indicates if source and target dataframes are the same.
    :param hamming_dist_num_bins: Number of bins for discretizing numerical columns.

    :return: A tuple containing: Average minimum hamming distance, Average minimum hamming distance difference for the
    closest two datapoints, List of percentile for the minimum hamming distances, Number of exact matches.
    """

    if mode == 'balanced' and balanced_count is None:
        raise AttributeError(f'The chosen mode is "balanced", but "balanced_count" was not provided.')

    if mode == 'balanced':
        from_split_count = round(len(df_from) / balanced_count)
        to_split_count = round(len(df_to) / balanced_count)

        # shuffle the data
        df_from = df_from.sample(frac=1, random_state=1).reset_index(drop=True)
        df_to = df_to.sample(frac=1, random_state=1).reset_index(drop=True)

        # separate data into sets
        df_from_sets = np.array_split(df_from, from_split_count)
        df_to_sets = np.array_split(df_to, to_split_count)
    else:
        df_from_sets = [df_from]
        df_to_sets = [df_to]

    # for averaging
    avg_min_dist, avg_min_dist_two_dif, num_zeros = 0., 0., 0.
    avg_percentiles = {0.05: 0., 0.25: 0., 0.5: 0., 0.75: 0., 0.95: 0.}
    skip = 0

    for i in range(len(df_from_sets)):
        for j in range(len(df_to_sets)):
            # same random seeds used for shuffling of the same df -> same sets
            if df_from_df_to_same and j <= i:
                skip += 1
                continue

            df_from_set = df_from_sets[i]
            df_to_set = df_to_sets[j]

            min_dist, min_dist_two_dif, percentiles, zeros = hamming_distance(df_from_set, df_to_set,
                                                                              categorical_cols=categorical_cols,
                                                                              num_bins=hamming_dist_num_bins)

            avg_min_dist += min_dist
            avg_min_dist_two_dif += min_dist_two_dif
            num_zeros += zeros

            avg_percentiles[0.05] += percentiles[0]
            avg_percentiles[0.25] += percentiles[1]
            avg_percentiles[0.5] += percentiles[2]
            avg_percentiles[0.75] += percentiles[3]
            avg_percentiles[0.95] += percentiles[4]

    exp_count = len(df_from_sets) * len(df_to_sets) if not df_from_df_to_same \
        else len(df_from_sets) * len(df_to_sets) - skip

    avg_min_dist /= exp_count
    avg_min_dist_two_dif /= exp_count
    avg_percentiles[0.05] /= exp_count
    avg_percentiles[0.25] /= exp_count
    avg_percentiles[0.5] /= exp_count
    avg_percentiles[0.75] /= exp_count
    avg_percentiles[0.95] /= exp_count

    return avg_min_dist, avg_min_dist_two_dif, avg_percentiles, num_zeros


def _prepare_gower_data(df_train, df_test, df_synth, columns_json):
    """Prepare the arrays and column indices used by the revision metrics."""
    categorical_cols = columns_json['categorical']
    boolean_cols = columns_json['boolean']
    numerical_cols = columns_json['integer'] + columns_json['float']

    numerical_idx = [df_train.columns.get_loc(col) for col in numerical_cols]
    categorical_idx = [
        df_train.columns.get_loc(col) for col in (categorical_cols + boolean_cols)
    ]

    train_encoded, test_encoded, mappings = encode_categoricals_joint(
        df_train,
        df_test,
        categorical_cols,
    )

    synth_encoded = df_synth[df_train.columns].copy()
    for col, mapping in mappings.items():
        synth_encoded[col] = synth_encoded[col].replace(mapping)

    return (
        train_encoded.values.astype(float),
        test_encoded.values.astype(float),
        synth_encoded.values.astype(float),
        numerical_idx,
        categorical_idx,
    )


def evaluate_privacy(
    experiment_name: str,
    gen_model_names: Dict[str, List[str]],
    hamming_dist_num_bins=10,
    privacy_metrics=None,
    authenticity_epsilon=0.00001,
    gower_block_size=1024,
):
    """
    Evaluate the privacy of generated synthetic data against original data.

    :param experiment_name: Name of the experiment.
    :param gen_model_names: Mapping of generative model names to a list of synthetic data filenames.
    :param hamming_dist_num_bins: Number of bins for Hamming distance calculation.
    :param privacy_metrics: Metrics to compute. Defaults to the legacy Hamming-distance evaluation.
    :param authenticity_epsilon: Near-duplicate threshold in Gower space.
    :param gower_block_size: Row block size for Gower-distance calculations.
    """

    privacy_metrics = _get_privacy_metric_names(privacy_metrics)
    if not privacy_metrics:
        raise ValueError('At least one privacy metric needs to be provided.')
    unsupported_metrics = set(privacy_metrics) - SUPPORTED_PRIVACY_METRICS
    if unsupported_metrics:
        raise ValueError(
            f'Unknown privacy metric(s): {", ".join(sorted(unsupported_metrics))}. '
            f'Choose from: {", ".join(sorted(SUPPORTED_PRIVACY_METRICS))}.'
        )

    use_hamming = 'hamming_distance' in privacy_metrics
    use_authenticity = 'authenticity' in privacy_metrics
    use_nnaa = 'nearest_neighbor_adversarial_accuracy' in privacy_metrics
    if use_authenticity or use_nnaa:
        if not isinstance(authenticity_epsilon, (int, float)) or authenticity_epsilon < 0:
            raise ValueError('"authenticity_epsilon" needs to be a non-negative number.')
        if not isinstance(gower_block_size, int) or gower_block_size <= 0:
            raise ValueError('"gower_block_size" needs to be a positive Integer.')

    results = {}
    exp_folder_path = os.path.join(PROJECT_ROOT, 'experiments', experiment_name)
    _, df_train, df_test = load_experiment_data(experiment_name, for_training=False)
    columns_json = load_columns_json_for_experiment(experiment_name)
    categorical_cols = columns_json['boolean'] + columns_json['categorical']

    results['parameters'] = {'hamming_dist_num_bins': hamming_dist_num_bins,
                             'balanced_count': len(df_test),
                             'privacy_metrics': privacy_metrics}

    if use_authenticity or use_nnaa:
        results['parameters']['authenticity_epsilon'] = authenticity_epsilon
        results['parameters']['gower_block_size'] = gower_block_size

    if use_hamming:
        orig_train_avg_min_dist, orig_train_avg_min_dist_two_dif, orig_train_avg_percentiles, orig_train_num_zeros = eval_hamming_distances(
            df_train, df_train, categorical_cols=categorical_cols, balanced_count=len(df_test), df_from_df_to_same=True,
            hamming_dist_num_bins=hamming_dist_num_bins)

        orig_test_avg_min_dist, orig_test_avg_min_dist_two_dif, orig_test_avg_percentiles, orig_test_num_zeros = eval_hamming_distances(
            df_train, df_test, categorical_cols=categorical_cols, balanced_count=len(df_test),
            hamming_dist_num_bins=hamming_dist_num_bins)

        results['original train'] = {
            'train_avg_min_dist': orig_train_avg_min_dist,
            'test_avg_min_dist': orig_test_avg_min_dist,
            'train_avg_min_dist_two_dif': orig_train_avg_min_dist_two_dif,
            'test_avg_min_dist_two_dif': orig_test_avg_min_dist_two_dif,
            'train_avg_percentile_0.05': orig_train_avg_percentiles[0.05],
            'test_avg_percentile_0.05': orig_test_avg_percentiles[0.05],
            'train_avg_percentile_0.25': orig_train_avg_percentiles[0.25],
            'test_avg_percentile_0.25': orig_test_avg_percentiles[0.25],
            'train_avg_percentile_0.50': orig_train_avg_percentiles[0.5],
            'test_avg_percentile_0.50': orig_test_avg_percentiles[0.5],
            'train_avg_percentile_0.75': orig_train_avg_percentiles[0.75],
            'test_avg_percentile_0.75': orig_test_avg_percentiles[0.75],
            'train_avg_percentile_0.95': orig_train_avg_percentiles[0.95],
            'test_avg_percentile_0.95': orig_test_avg_percentiles[0.95],
            'train_num_zeros_sum': orig_train_num_zeros,
            'test_num_zeros_sum': orig_test_num_zeros,
            'dif_train_test': 1 - (orig_train_avg_min_dist / orig_test_avg_min_dist)
        }

    for gen_model in gen_model_names:
        for filename in gen_model_names[gen_model]:
            filename = filename if filename.endswith('.csv') else filename + '.csv'
            df_synth = pd.read_csv(os.path.join(exp_folder_path, gen_model.upper(), 'synthetic_data', filename))

            dataset_results = {}

            if use_hamming:
                for col in df_train:
                    if df_train[col].dtype != df_synth[col].dtype:
                        df_synth[col] = df_synth[col].astype(df_train[col].dtype)

                train_avg_min_dist, train_avg_min_dist_two_dif, train_avg_percentiles, train_num_zeros = eval_hamming_distances(df_synth, df_train, categorical_cols=categorical_cols, balanced_count=len(df_test), hamming_dist_num_bins=hamming_dist_num_bins)
                test_avg_min_dist, test_avg_min_dist_two_dif, test_avg_percentiles, test_num_zeros = eval_hamming_distances(df_synth, df_test, categorical_cols=categorical_cols, balanced_count=len(df_test), hamming_dist_num_bins=hamming_dist_num_bins)
                dataset_results.update({
                    'train_avg_min_dist': train_avg_min_dist,
                    'test_avg_min_dist': test_avg_min_dist,
                    'train_avg_min_dist_two_dif': train_avg_min_dist_two_dif,
                    'test_avg_min_dist_two_dif': test_avg_min_dist_two_dif,
                    'train_avg_percentile_0.05': train_avg_percentiles[0.05],
                    'test_avg_percentile_0.05': test_avg_percentiles[0.05],
                    'train_avg_percentile_0.25': train_avg_percentiles[0.25],
                    'test_avg_percentile_0.25': test_avg_percentiles[0.25],
                    'train_avg_percentile_0.50': train_avg_percentiles[0.5],
                    'test_avg_percentile_0.50': test_avg_percentiles[0.5],
                    'train_avg_percentile_0.75': train_avg_percentiles[0.75],
                    'test_avg_percentile_0.75': test_avg_percentiles[0.75],
                    'train_avg_percentile_0.95': train_avg_percentiles[0.95],
                    'test_avg_percentile_0.95': test_avg_percentiles[0.95],
                    'train_num_zeros_sum': train_num_zeros,
                    'test_num_zeros_sum': test_num_zeros,
                    'dif_train_test': 1 - (train_avg_min_dist / test_avg_min_dist),
                    'dif_syn_train_orig_train': (train_avg_min_dist / orig_train_avg_min_dist) - 1,
                    'dif_syn_test_orig_test': (test_avg_min_dist / orig_test_avg_min_dist) - 1
                })

            if use_authenticity or use_nnaa:
                train_np, test_np, synth_np, numerical_idx, categorical_idx = _prepare_gower_data(
                    df_train,
                    df_test,
                    df_synth,
                    columns_json,
                )

                if use_authenticity:
                    authenticity = authenticity_metric(
                        train_np,
                        synth_np,
                        numerical_idx,
                        categorical_idx,
                        epsilon=authenticity_epsilon,
                        block_size=gower_block_size,
                    )
                    dataset_results['authenticity'] = summarize_authenticity_outputs(authenticity)

                if use_nnaa:
                    dataset_results['nearest_neighbor_adversarial_accuracy'] = compute_aa(
                        train_np,
                        test_np,
                        synth_np,
                        numerical_idx,
                        categorical_idx,
                        block_size=gower_block_size,
                    )

            results[gen_model.upper()+'_'+filename] = dataset_results

    # save results
    with open(os.path.join(exp_folder_path, 'eval_privacy.json'), 'w') as fh:
        json.dump(results, fh, indent=2)

    return results


@hydra.main(config_path="../config", config_name="6_privacy_eval_gen", version_base="1.1")
def main_privacy_eval(cfg: MainPrivacyEvalConfig) -> None:
    """
    Main function to execute privacy evaluation.

    :param cfg: Configuration for privacy evaluation.
    """
    cfg: PrivacyEvalConfig = cfg.privacy_eval_gen
    print(OmegaConf.to_yaml(cfg))
    _validate_config(cfg)
    evaluate_privacy(cfg.experiment_name, gen_model_names=cfg.synthetic_data,
                     hamming_dist_num_bins=getattr(cfg, 'hamming_dist_num_bins', 10),
                     privacy_metrics=getattr(cfg, 'privacy_metrics', None),
                     authenticity_epsilon=getattr(cfg, 'authenticity_epsilon', 0.00001),
                     gower_block_size=getattr(cfg, 'gower_block_size', 1024))


if __name__ == '__main__':
    main_privacy_eval()
