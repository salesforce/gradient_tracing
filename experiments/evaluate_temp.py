#import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

# new parameters

max_entries = 1024
seed = 42
padding_side = "left"
import torch
overall_precision = torch.float32
model_precision = torch.float16
torch.set_default_dtype(overall_precision)

# end new parameters

import json
import shutil
from time import time
from typing import Tuple, Union
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

#from baselines.efk import EFKHyperParams, EfkRewriteExecutor
#from baselines.ft import FTHyperParams, apply_ft_to_model
#from baselines.kn import KNHyperParams, apply_kn_to_model
#from baselines.mend import MENDHyperParams, MendRewriteExecutor
from dsets import (
    #AttributeSnippets,
    CounterFactDataset,
    CounterFactTrueDataset,
    CounterFactFalseDataset,
    FactDataset
    #MENDQADataset,
    #get_tfidf_vectorizer,
)
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
#from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre
from rome import ROMEHyperParams, apply_rome_to_model
from flexcl import FLEXCLHyperParams, apply_flexcl_to_model, InfError
from util import nethook
from util.globals import *

import numpy as np

ALG_DICT = {
    "ROME": (ROMEHyperParams, apply_rome_to_model),
    #"FT": (FTHyperParams, apply_ft_to_model),
    #"KN": (KNHyperParams, apply_kn_to_model),
    #"MEND": (MENDHyperParams, MendRewriteExecutor().apply_to_model),
    #"KE": (EFKHyperParams, EfkRewriteExecutor().apply_to_model),
    "FLEXCL": (FLEXCLHyperParams, apply_flexcl_to_model)
}

DS_DICT = {
    "cf": (CounterFactDataset, compute_rewrite_quality_counterfact),
    "cft": (CounterFactTrueDataset, compute_rewrite_quality_counterfact),
    "cff": (CounterFactFalseDataset, compute_rewrite_quality_counterfact),
    "fact": (FactDataset, compute_rewrite_quality_counterfact),
    #"zsre": (MENDQADataset, compute_rewrite_quality_zsre),
}


def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    continue_from_run: str,
    #skip_generation_tests: bool,
    conserve_memory: bool,
    dir_name: str,
    fixed_layer_for_editing: int = None,
    minus_end_token_index: int = None
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT[alg_name]

    # Determine run directory
    if continue_from_run is not None:
        run_dir = RESULTS_DIR / dir_name / continue_from_run
        assert (
            run_dir.exists()
        ), f"If continuing from run, {continue_from_run} must exist!"
    else:
        alg_dir = RESULTS_DIR / dir_name
        if alg_dir.exists():
            id_list = [
                int(str(x).split("_")[-1])
                for x in alg_dir.iterdir()
                if str(x).split("_")[-1].isnumeric()
            ]
            run_id = 0 if not id_list else max(id_list) + 1
        else:
            run_id = 0
        run_dir = RESULTS_DIR / dir_name / f"run_{str(run_id).zfill(3)}"
        run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be stored at {run_dir}")

    # Get run hyperparameters
    params_path = (
        run_dir / "params.json"
        if continue_from_run is not None
        else HPARAMS_DIR / alg_name / hparams_fname
    )
    hparams = params_class.from_json(params_path)

    if alg_name == "FLEXCL":
        if fixed_layer_for_editing is not None:
            hparams.fixed_layer_for_editing = fixed_layer_for_editing
        if minus_end_token_index is not None:
            hparams.end_token_index = -minus_end_token_index
    elif alg_name == "ROME":
        if fixed_layer_for_editing is not None:
            hparams.layers = [fixed_layer_for_editing]

    if not (run_dir / "params.json").exists():
        shutil.copyfile(params_path, run_dir / "params.json")
        with open(str(run_dir)+"/extra.txt", 'w') as f:
            f.write("fixed_layer_for_editing: " + str(fixed_layer_for_editing) + ", end_token_index: " +str(-minus_end_token_index))


    print(f"Executing {alg_name} with parameters {hparams}")
         
    # Instantiate vanilla model
    
    print("Instantiating model")
    if model_name == 'vicuna-7b-v1.3':
        model_name='lmsys/'+model_name
    elif model_name == "Mistral-7B-instruct-v0.2":
        model_name= "mistralai/Mistral-7B-instruct-v0.2"
    print('model_name', model_name) 
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype = model_precision)
    #model.cuda(0)
    if alg_name == "FLEXCL":
        location_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype = model_precision)
        location_model.cuda(1)
    
    tok = AutoTokenizer.from_pretrained(model_name, dtype = model_precision)
    if tok.pad_token is None:
        tok.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tok))
    
    # Load data
    #print("Loading dataset, attribute snippets, tf-idf data")
    #snips = AttributeSnippets(DATA_DIR) if not skip_generation_tests else None
    #vec = get_tfidf_vectorizer(DATA_DIR) if not skip_generation_tests else None

    ds_class, ds_eval_method = DS_DICT[ds_name]
    ds = ds_class(DATA_DIR, size=dataset_size_limit, tok=tok)
    ds = np.random.choice(ds, size=max_entries, replace=False, p=None)
    import pickle
    with open(ds_name +'_sub.pkl', 'wb') as f:
        pickle.dump(ds,f)

    # Iterate through dataset
    for record in ds:
        sys.stdout.flush()
        case_id = record["case_id"]
        case_result_path = run_dir / f"case_{case_id}.json"
        if not case_result_path.exists():
            # Compute weight changes + record weights that changed
            start = time()
            args_conserve_memory = (
                dict(return_orig_weights_device=("cpu" if conserve_memory else "cuda"))
                if conserve_memory
                else dict()
            )
            try:
                if alg_name == "FLEXCL":
                    edited_model, weights_copy = apply_algo(
                        model_for_rome=model,
                        location_model=location_model,
                        tok=tok,
                        requests=[record["requested_rewrite"]],
                        case_id=case_id,
                        hparams=hparams,
                        copy=False,
                        return_orig_weights=True,
                        run_dir = str(run_dir),
                        hparams_dir = str(HPARAMS_DIR),
                        notebook_mode = False
                    )
                elif alg_name == "ROME":
                    edited_model, weights_copy = apply_algo(
                        model=model,
                        tok=tok,
                        requests=[record["requested_rewrite"]],
                        hparams=hparams,
                        copy=False,
                        return_orig_weights=True
                    )
            except InfError:
                print("Skipping due to infinite gradient norm")
                continue
            exec_time = time() - start
            print("Execution took", exec_time)

            # Execute evaluation suite
            start = time()
            metrics = {
                "case_id": case_id,
                "requested_rewrite": record["requested_rewrite"],
                "time": exec_time,
                "post": ds_eval_method(edited_model, tok, record, padding_side),#used
            }

            with torch.no_grad():
                for k, v in weights_copy.items():
                    nethook.get_parameter(model, k)[...] = v.to(model.device,copy=True)
            metrics["pre"] = ds_eval_method(model, tok, record, padding_side)#used

            print("Evaluation took", time() - start)

            # Dump metrics in .json
            with open(case_result_path, "w") as f:
                json.dump(metrics, f, indent=1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=["ROME", "FLEXCL"],
        default="ROME",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
        "where a new run_id is generated on each run. "
        "If continuing from previous run, specify the run_id in --continue_from_run.",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        choices=['vicuna-7b-v1.3', "Mistral-7B-instruct-v0.2"],
        default="Mistral-7B-instruct-v0.2",
        help="Model to edit.",
        required=True,
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="Mistral-7B-instruct-v0.2.json",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        required=True,
    )
    parser.add_argument(
        "--ds_name",
        choices=["cf", "cft", "cff", "fact"],
        default="fact",
        help="Dataset to perform evaluations on.",
    )
    parser.add_argument(
        "--continue_from_run",
        type=str,
        default=None,
        help="If continuing from previous run, set to run_id. Otherwise, leave as None.",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=1024,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--fixed_layer_for_editing",
        type=int,
        default=None,
        help="Overrides the fixed_layer_for_editing hyperparameter in the FLEXCL json",
    )
    parser.add_argument(
        "--minus_end_token_index",
        type=int,
        default=None,
        help="Overrides the fixed_layer_for_editing hyperparameter in the FLEXCL json",
    )
    """
    parser.add_argument(
        "--skip_generation_tests",
        dest="skip_generation_tests",
        action="store_true",
        help="Only run fast probability-based tests without slow generation tests. "
        "Useful for quick debugging and hyperparameter sweeps.",
    )
    """
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation at the cost of a minor slowdown. "
        "Backs up model weights on CPU instead of GPU.",
    )
    parser.set_defaults(conserve_memory=False)
    args = parser.parse_args()

    main(
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.ds_name,
        args.dataset_size_limit,
        args.continue_from_run,
        #args.skip_generation_tests,
        args.conserve_memory,
        dir_name=args.alg_name,
        fixed_layer_for_editing=args.fixed_layer_for_editing,
        minus_end_token_index=args.minus_end_token_index
    )
