from collections import defaultdict
from heapq import nsmallest
from typing import List
import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OrdinalEncoder
from src.models import GenerativeModel


def sample_according_to_constraint(df_synth: DataFrame, constraint: str, model: GenerativeModel,
                                   categorical_columns: List[str], ignore_cols: List[str] = None,
                                   euclid_median_max=None, max_dist_accepts=None, max_dist_accepts_percent=0.1,
                                   same_datapoints=1, scaling_max_dist_accept_after_zero_round_count=2,
                                   initial_random_seed=42, verbose=True):
    # defining distance functions
    def euclidean_distance(row1, row2):
        return distance.euclidean(row1, row2)

    def hamming_distance(row1, row2):
        return sum(el1 != el2 for el1, el2 in zip(row1, row2))

    # combined distance function
    def combined_distance(row1_cat, num_distances, df_cat):
        cat_distances = df_cat.apply(lambda x: hamming_distance(row1_cat, x), axis=1)
        total_distances = num_distances + cat_distances
        smallest = nsmallest(5, enumerate(total_distances), key=lambda x: x[1])
        return smallest

    if ignore_cols is None:
        ignore_cols = []
    random_seed = initial_random_seed
    count = len(df_synth)

    num_columns = [col for col in df_synth.columns if (col not in categorical_columns) and (col not in ignore_cols)]
    cat_columns = [col for col in categorical_columns if col not in ignore_cols]

    if euclid_median_max is None:
        euclid_median_max = len(num_columns) * 2

    if max_dist_accepts is None:
        # accepting a maximum difference
        max_dist_accepts = (len(cat_columns) + euclid_median_max) * max_dist_accepts_percent
        max_dist_accepts = round(max_dist_accepts)

    df_to_replace = df_synth.loc[df_synth.apply(lambda row: eval(constraint), axis=1)].reset_index(drop=True)
    df_synth = df_synth.loc[df_synth.apply(lambda row: not (eval(constraint)), axis=1)].reset_index(drop=True)

    if verbose:
        print('missing', count - len(df_synth))

    added_history = []
    smallest_dist = np.inf
    while True:
        calculate_small_dist = True
        # update max_dist_accepts
        if len(added_history) >= scaling_max_dist_accept_after_zero_round_count and sum(added_history[-scaling_max_dist_accept_after_zero_round_count:]) == 0:
            old_val = max_dist_accepts
            # increase by 10 percent, but at least by one if it includes the smallest dist of last batch, otherwise increase it to include the smallest batch
            if round(max_dist_accepts / 10) > 0:
                max_dist_accepts += round(max_dist_accepts / 10)
            else:
                max_dist_accepts = 1

            if smallest_dist > max_dist_accepts:
                max_dist_accepts = round(np.ceil(smallest_dist))

            if verbose:
                print(f'max_dist_accepts was increased from {old_val} to {max_dist_accepts}')

        random_seed += 1
        new_synth = model.sample(count, random_seed)
        new_synth = new_synth.loc[new_synth.apply(lambda row: not (eval(constraint)), axis=1)].reset_index(drop=True)

        # calculate all numerical distances
        num_distances_all = df_to_replace[num_columns].apply(lambda row: new_synth[num_columns].apply(lambda x: euclidean_distance(row, x), axis=1), axis=1)

        # calculate scale_factor based on 50th percentile
        scale_factor = 1
        percentile_50 = np.percentile(num_distances_all.values.flatten(), 50)
        if percentile_50 > euclid_median_max:
            scale_factor = euclid_median_max / percentile_50

        # scale numerical distances
        if scale_factor != 1:
            num_distances_all = num_distances_all * scale_factor

        distance_list = []
        # calculate and store the three smallest distances and their indices for each row in df1
        for index, row in df_to_replace.iterrows():
            num_distances_row = num_distances_all.loc[index]
            dist = combined_distance(row[cat_columns], num_distances_row, new_synth[cat_columns])

            if calculate_small_dist:
                smallest_dist_tmp = min(dist, key=lambda item: item[1])[1]
                if smallest_dist_tmp < max_dist_accepts:
                    calculate_small_dist = False
                    smallest_dist = np.inf
                elif smallest_dist_tmp < smallest_dist:
                    smallest_dist = smallest_dist_tmp

            dist = [(*t, index) for t in dist if t[1] <= max_dist_accepts]
            distance_list += dist

        # sorted by their distance
        distance_list = sorted(distance_list, key=lambda x: x[1])

        if same_datapoints > 1:
            cand2count = defaultdict(int)
            ind_replaced = set()
            for new_synth_ind, _, to_replace_ind in distance_list:
                if to_replace_ind not in ind_replaced and cand2count[new_synth_ind] < same_datapoints:
                    ind_replaced.add(to_replace_ind)
                    cand2count[new_synth_ind] += 1

            # Create an empty DataFrame to store the duplicated rows
            dup_df = pd.DataFrame(columns=new_synth.columns)

            for new_synth_ind, count in cand2count.items():
                # Duplicate the row 'count' times and append it to dup_df
                dup_df = dup_df.append([new_synth.iloc[new_synth_ind]] * count, ignore_index=True)
        else:
            cand_used = set()
            ind_replaced = set()
            for new_synth_ind, _, to_replace_ind in distance_list:
                if to_replace_ind not in ind_replaced and new_synth_ind not in cand_used:
                    ind_replaced.add(to_replace_ind)
                    cand_used.add(new_synth_ind)

            dup_df = new_synth.iloc[list(cand_used)]

        # Append dup_df to df_synth
        df_synth = df_synth.append(dup_df, ignore_index=True).reset_index(drop=True)

        df_to_replace = df_to_replace.drop(ind_replaced).reset_index(drop=True)

        if len(df_to_replace) == 0:
            break
        elif verbose:
            print(f'{len(dup_df)} added, {len(df_to_replace)} remaining')
        added_history.append(len(dup_df))
    return df_synth


def sample_according_to_constraint2(df_synth: DataFrame, constraint: str, model: GenerativeModel,
                                   categorical_columns: List[str], ignore_cols: List[str] = None, euclid_max_scale=20,
                                   max_dist=40, initial_random_seed=42):
    # Define your custom distance function
    def mixed_distance(x, y):
        num_distance = distance.euclidean(x[num_index_range], y[num_index_range]) #np.sqrt(np.sum((x[num_index_range] - y[num_index_range]) ** 2))
        cat_distance = sum(x[cat_index_range] != y[cat_index_range])#sum(el1 != el2 for el1, el2 in zip(x[cat_index_range], y[cat_index_range]))
        return num_distance + cat_distance

    if ignore_cols is None:
        ignore_cols = []
    random_seed = initial_random_seed
    count = len(df_synth)

    num_columns = [col for col in df_synth.columns if (col not in categorical_columns) and (col not in ignore_cols)]
    cat_columns = [col for col in categorical_columns if col not in ignore_cols]

    # Get the column indices for numerical and categorical data
    num_index_range = [df_synth.columns.get_loc(c) for c in num_columns]
    cat_index_range = [df_synth.columns.get_loc(c) for c in cat_columns]

    df_to_replace = df_synth.loc[df_synth.apply(lambda row: eval(constraint), axis=1)].reset_index(drop=True)
    df_synth = df_synth.loc[df_synth.apply(lambda row: not (eval(constraint)), axis=1)].reset_index(drop=True)

    encoder = OrdinalEncoder()
    df_to_replace[cat_columns] = encoder.fit_transform(df_to_replace[cat_columns])

    print('missing', count - len(df_synth))

    while True:
        random_seed += 1
        new_synth = model.sample(count, random_seed)
        new_synth = new_synth.loc[new_synth.apply(lambda row: not (eval(constraint)), axis=1)].reset_index(drop=True)
        new_synth[cat_columns] = encoder.transform(new_synth[cat_columns])

        # Initialize the model
        neigh = NearestNeighbors(n_neighbors=3, metric=mixed_distance)
        neigh.fit(new_synth)

        # Find the 3 nearest neighbors for each point in df1
        distances, indices = neigh.kneighbors(df_to_replace)
        break
