#put in results folder

import json
import pandas as pd
import os
import shutil
import collections
from pprint import pprint
from typing import List, Optional
import numpy as np
from scipy.stats import hmean
from util.globals import *
from pathlib import Path
from collections import defaultdict
import pickle


alg_name = "FLEXCL"
separate_based_on = ["end_token_index", "ds_name", "v_weight_decay"]
trace_subject = True

def semantic_token_location(file:str):
    with open(file, 'r') as f:
        file_dict = json.load(f)
    output =  {}
    if file_dict['fact_token'] < file_dict['first_subject_token']:
        location = 'before_subject'
    elif file_dict['first_subject_token'] <= file_dict['fact_token'] < file_dict['last_subject_token']:
        location = 'subject_not_last'
    elif file_dict['fact_token'] == file_dict['last_subject_token']:
        location = 'subject_last'
    elif file_dict['last_subject_token'] < file_dict['fact_token'] < file_dict['total_num_tokens'] - 4:
        location = 'after_subject_before_last'
    else:
        location = 'last'
    output["location"] = location
    output["overlap"] = file_dict['last_subject_token'] == file_dict['total_num_tokens'] - 1
    output["case_id"] = file_dict['case_id']
    output["token_ratio"]=(file_dict['fact_token'] + 1) / file_dict['total_num_tokens']
    output["layer"]=file_dict['layer']
    output['signed_distance_from_last_subject_token'] = file_dict['fact_token']-file_dict['last_subject_token']
    output['unsigned_distance_from_last_subject_token'] = abs(file_dict['fact_token']-file_dict['last_subject_token'])
    output['normalized_signed_distance_from_last_subject_token'] = (file_dict['fact_token']-file_dict['last_subject_token'])/file_dict['total_num_tokens']
    output['normalized_unsigned_distance_from_last_subject_token'] = abs(file_dict['fact_token']-file_dict['last_subject_token'])/file_dict['total_num_tokens']
    output['case_id'] = file_dict['case_id']
    return output

def all_semantic_token_locations(editing_locations_dir:str):
    output = {}
    for file in list(Path(editing_locations_dir).glob("editing_location_case_*.json")):
        local_results = semantic_token_location(file)
        output[local_results['case_id']] = local_results
    return output

def stats(
    dir_name,
    runs: Optional[List],
    first_n_cases=None,
    get_uncompressed=False,
    abs_path=False,
):  # runs = None -> all runs

    summaries = []
    uncompressed = []

    for run_dir in (RESULTS_DIR / dir_name if not abs_path else dir_name).iterdir():
        # Skip if we're not interested

        if runs is not None and all(run not in str(run_dir) for run in runs):
            continue

        # Iterate through all case files
        cur_sum = collections.defaultdict(lambda: [])
        files = list(run_dir.glob("case_*.json"))
        files.sort(key=lambda x: int(str(x).split("_")[-1].split(".")[0]))
        for case_file in files:
            try:
                with open(case_file, "r") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                print(f"Could not decode {case_file} due to format error; skipping.")

            case_id = data["case_id"]
            if first_n_cases is not None and case_id >= first_n_cases:
                break

            cur_sum["time"].append(data["time"])

            for prefix in ["pre", "post"]:
                # Probability metrics for which new should be lower (better) than true
                for key in ["rewrite_prompts_probs", "paraphrase_prompts_probs"]:
                    if prefix not in data or key not in data[prefix]:
                        continue

                    sum_key_discrete = f"{prefix}_{key.split('_')[0]}_success"
                    sum_key_cont = f"{prefix}_{key.split('_')[0]}_diff"
                    sum_key_newobj = f"{prefix}_{key.split('_')[0]}_newobj" #edit1

                    cur_sum[sum_key_discrete].append(
                        np.mean(
                            [
                                x["target_true"] > x["target_new"]
                                for x in data[prefix][key]
                            ]
                        )
                    )
                    cur_sum[sum_key_cont].append(
                        np.mean(
                            [
                                np.exp(-x["target_new"]) - np.exp(-x["target_true"])
                                for x in data[prefix][key]
                            ]
                        )
                    )
                    cur_sum[sum_key_newobj].append(
                        np.mean(
                            [
                                np.exp(-x["target_new"])
                                for x in data[prefix][key]
                            ]
                        )
                    ) #edit2


                # Probability metrics for which true should be lower (better) than new
                sum_key_discrete = f"{prefix}_neighborhood_success"
                sum_key_cont = f"{prefix}_neighborhood_diff"
                sum_key_newobj = f"{prefix}_neighborhood_newobj" #edit3
                key = "neighborhood_prompts_probs"
                if prefix in data and key in data[prefix]:
                    cur_sum[sum_key_discrete].append(
                        np.mean(
                            [
                                x["target_true"] < x["target_new"]
                                for x in data[prefix][key]
                            ]
                        )
                    )
                    cur_sum[sum_key_cont].append(
                        np.mean(
                            [
                                np.exp(-x["target_true"]) - np.exp(-x["target_new"])
                                for x in data[prefix][key]
                            ]
                        )
                    )
                    cur_sum[sum_key_newobj].append(
                        np.mean(
                            [
                                np.exp(-x["target_true"])
                                for x in data[prefix][key]
                            ]
                        )
                    ) #edit3

                # zsRE evaluation metrics
                for key in ["rewrite", "paraphrase", "neighborhood"]:
                    sum_key = f"{prefix}_{key}_acc"
                    key = f"{key}_prompts_correct"

                    if prefix not in data or key not in data[prefix]:
                        continue

                    cur_sum[sum_key].append(np.mean(data[prefix][key]))

                # Generation metrics that can be directly averaged
                for key in ["ngram_entropy", "reference_score", "essence_score"]:
                    if prefix in data and key in data[prefix]:
                        cur_sum[f"{prefix}_{key}"].append(data[prefix][key])

        if len(cur_sum) == 0:
            continue

        num_items = len(cur_sum[next(iter(cur_sum.keys()))])
        metadata = {
            "run_dir": str(run_dir),
            "num_cases": num_items,
        }

        uncompressed.append(dict(cur_sum, **metadata))

        cur_sum = {k: (np.mean(v), np.std(v)) for k, v in cur_sum.items()}
        for prefix in ["pre", "post"]:
            for k_efficacy, k_generalization, k_specificity in [
                (
                    f"{prefix}_rewrite_success",
                    f"{prefix}_paraphrase_success",
                    f"{prefix}_neighborhood_success",
                ),
                (
                    f"{prefix}_rewrite_acc",
                    f"{prefix}_paraphrase_acc",
                    f"{prefix}_neighborhood_acc",
                ),
            ]:
                if k_generalization in cur_sum and k_specificity in cur_sum:
                    cur_sum[f"{prefix}_score"] = (
                        hmean(
                            [
                                cur_sum[k_efficacy][0],
                                cur_sum[k_generalization][0],
                                cur_sum[k_specificity][0],
                            ]
                        ),
                        np.nan,
                    )
                    break

        for k, v in cur_sum.items():
            if all(exclude not in k for exclude in ["essence_score", "time"]):
                # Constant multiplication scales linearly with mean and stddev
                cur_sum[k] = tuple(np.around(z * 100, 2) for z in v)

        cur_sum.update(metadata)
        pprint(cur_sum)
        summaries.append(cur_sum)

    return uncompressed if get_uncompressed else summaries

def detailed_stats(
    dir_name,
    runs: Optional[List],
    first_n_cases=None,
    get_uncompressed=False,
    abs_path=False,
):  # runs = None -> all runs

    summaries_per_location = defaultdict(lambda: [])
    uncompressed_per_location = defaultdict(lambda: [])

    for run_dir in (RESULTS_DIR / dir_name if not abs_path else dir_name).iterdir():
        # Skip if we're not interested

        if runs is not None and all(run not in str(run_dir) for run in runs):
            continue

        broken_down = all_semantic_token_locations(Path(str(run_dir) + '/editing_locations'))

        # Iterate through all case files
        cur_sum_per_location = defaultdict(lambda: defaultdict(lambda: []))
        files = list(run_dir.glob("case_*.json"))
        files.sort(key=lambda x: int(str(x).split("_")[-1].split(".")[0]))
        for case_file in files:
            try:
                with open(case_file, "r") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                print(f"Could not decode {case_file} due to format error; skipping.")

            case_id = data["case_id"]
            location = broken_down[case_id]["location"]
            summaries = summaries_per_location[location]
            uncompressed = uncompressed_per_location[location]
            cur_sum = cur_sum_per_location[location]

            if first_n_cases is not None and case_id >= first_n_cases:
                break

            cur_sum["time"].append(data["time"])

            for prefix in ["pre", "post"]:
                # Probability metrics for which new should be lower (better) than true
                for key in ["rewrite_prompts_probs", "paraphrase_prompts_probs"]:
                    if prefix not in data or key not in data[prefix]:
                        continue

                    sum_key_discrete = f"{prefix}_{key.split('_')[0]}_success"
                    sum_key_cont = f"{prefix}_{key.split('_')[0]}_diff"
                    sum_key_newobj = f"{prefix}_{key.split('_')[0]}_newobj" #edit1

                    cur_sum[sum_key_discrete].append(
                        np.mean(
                            [
                                x["target_true"] > x["target_new"]
                                for x in data[prefix][key]
                            ]
                        )
                    )
                    cur_sum[sum_key_cont].append(
                        np.mean(
                            [
                                np.exp(-x["target_new"]) - np.exp(-x["target_true"])
                                for x in data[prefix][key]
                            ]
                        )
                    )
                    cur_sum[sum_key_newobj].append(
                        np.mean(
                            [
                                np.exp(-x["target_new"])
                                for x in data[prefix][key]
                            ]
                        )
                    ) #edit2


                # Probability metrics for which true should be lower (better) than new
                sum_key_discrete = f"{prefix}_neighborhood_success"
                sum_key_cont = f"{prefix}_neighborhood_diff"
                sum_key_newobj = f"{prefix}_neighborhood_newobj" #edit3
                key = "neighborhood_prompts_probs"
                if prefix in data and key in data[prefix]:
                    cur_sum[sum_key_discrete].append(
                        np.mean(
                            [
                                x["target_true"] < x["target_new"]
                                for x in data[prefix][key]
                            ]
                        )
                    )
                    cur_sum[sum_key_cont].append(
                        np.mean(
                            [
                                np.exp(-x["target_true"]) - np.exp(-x["target_new"])
                                for x in data[prefix][key]
                            ]
                        )
                    )
                    cur_sum[sum_key_newobj].append(
                        np.mean(
                            [
                                np.exp(-x["target_true"])
                                for x in data[prefix][key]
                            ]
                        )
                    ) #edit3

                # zsRE evaluation metrics
                for key in ["rewrite", "paraphrase", "neighborhood"]:
                    sum_key = f"{prefix}_{key}_acc"
                    key = f"{key}_prompts_correct"

                    if prefix not in data or key not in data[prefix]:
                        continue

                    cur_sum[sum_key].append(np.mean(data[prefix][key]))

                # Generation metrics that can be directly averaged
                for key in ["ngram_entropy", "reference_score", "essence_score"]:
                    if prefix in data and key in data[prefix]:
                        cur_sum[f"{prefix}_{key}"].append(data[prefix][key])

        for location in cur_sum_per_location.keys():

            print('location', location)

            cur_sum = cur_sum_per_location[location]
            uncompressed = uncompressed_per_location[location]
            summaries = summaries_per_location[location]

            if len(cur_sum) == 0:
                continue

            num_items = len(cur_sum[next(iter(cur_sum.keys()))])
            metadata = {
                "run_dir": str(run_dir),
                "num_cases": num_items,
            }

            uncompressed.append(dict(cur_sum, **metadata))

            cur_sum = {k: (np.mean(v), np.std(v)) for k, v in cur_sum.items()}
            for prefix in ["pre", "post"]:
                for k_efficacy, k_generalization, k_specificity in [
                    (
                        f"{prefix}_rewrite_success",
                        f"{prefix}_paraphrase_success",
                        f"{prefix}_neighborhood_success",
                    ),
                    (
                        f"{prefix}_rewrite_acc",
                        f"{prefix}_paraphrase_acc",
                        f"{prefix}_neighborhood_acc",
                    ),
                ]:
                    if k_generalization in cur_sum and k_specificity in cur_sum:
                        cur_sum[f"{prefix}_score"] = (
                            hmean(
                                [
                                    cur_sum[k_efficacy][0],
                                    cur_sum[k_generalization][0],
                                    cur_sum[k_specificity][0],
                                ]
                            ),
                            np.nan,
                        )
                        break

            for k, v in cur_sum.items():
                if all(exclude not in k for exclude in ["essence_score", "time"]):
                    # Constant multiplication scales linearly with mean and stddev
                    cur_sum[k] = tuple(np.around(z * 100, 2) for z in v)

            cur_sum.update(metadata)
            pprint(cur_sum)
            summaries.append(cur_sum)

    return uncompressed_per_location if get_uncompressed else summaries_per_location

all_results = {}

for subdir,_, _ in os.walk(alg_name):
    folders = subdir.split('/')
    if len(folders)==3 and folders[1].startswith("parameters"):
        all_results[folders[1]] = pd.DataFrame()
        columns = ['num_cases', 'post_neighborhood_diff', 'post_neighborhood_newobj', 'post_neighborhood_success',
                   'post_paraphrase_diff', 'post_paraphrase_newobj', 'post_paraphrase_success', 'post_rewrite_diff',
                   'post_rewrite_newobj', 'post_rewrite_success', 'post_score',
                   'pre_neighborhood_diff', 'pre_neighborhood_newobj', 'pre_neighborhood_success',
                   'pre_paraphrase_diff', 'pre_paraphrase_newobj', 'pre_paraphrase_success', 'pre_rewrite_diff',
                   'pre_rewrite_newobj', 'pre_rewrite_success', 'pre_score']
        all_results[folders[1]] = pd.DataFrame()
        local_stats = stats(dir_name=subdir, runs=None, first_n_cases=None)
        for results in local_stats:
            run_number = results["run_dir"].split('/')[-1].split('_')[-1]
            with open(subdir +'/' +'run_' + run_number + '/additional_params.json', 'r') as f:
                additional_params = json.load(f)
                all_results[folders[1]].loc[run_number, "layer"] = additional_params["fixed_layer_for_editing"]
            for column in columns:
                if column == 'num_cases':
                    all_results[folders[1]].loc[run_number, column] = results[column]
                else:
                    all_results[folders[1]].loc[run_number, column] = results[column][0]
            all_results[folders[1]]["layer"] = all_results[folders[1]]["layer"].astype(int)
            all_results[folders[1]]['num_cases'] = all_results[folders[1]]['num_cases'].astype(int)
        with open('all_results.pkl', 'wb') as f:
            pickle.dump(all_results, f)
        if trace_subject:
            with open('detailed_stats.pkl', 'wb') as f:
                pickle.dump(dict(detailed_stats(dir_name=subdir, runs=None, first_n_cases=None)), f)


