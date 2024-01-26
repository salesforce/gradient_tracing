import os
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome import repr_tools
from util.globals import *

from .layer_stats import layer_stats
from .rome_hparams import ROMEHyperParams

# Cache variables
inv_mom2_cache = {}


def get_inv_cov(
    model: AutoModelForCausalLM,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
) -> torch.Tensor:
    """
    Retrieves inverse of covariance statistics. Caches result for future use.
    """

    global inv_mom2_cache

    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)

    if key not in inv_mom2_cache:
        print(
            f"Retrieving inverse covariance statistics for {model_name} @ {layer_name}. "
            f"The result will be cached to avoid repetitive computation."
        )
        stats_dir = Path(STATS_DIR)
        ds_name = mom2_dataset
        precision = mom2_dtype
        to_collect = "mom2"
        size_suffix = mom2_n_samples
        file_extension = f"{model_name}/{ds_name}_stats/{layer_name}_{precision}_{'-'.join(sorted([to_collect]))}_{size_suffix}_inverse.pt"
        current_directory = os.getcwd()
        #parent_directory = os.path.dirname(current_directory) #this one is specific when we run things from the notebooks folder, need to fix
        #filename = parent_directory / stats_dir / file_extension
        filename = current_directory / stats_dir / file_extension
        inv_mom2_cache[key] = torch.load(filename, map_location = model.device)

    return inv_mom2_cache[key]


def compute_u(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: ROMEHyperParams,
    layer: int,
    context_templates: List[str]
) -> torch.Tensor:
    """
    Computes the right vector used in constructing the rank-1 update matrix.
    """

    print("Computing left vector (u)...")

    # Compute projection token
    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=hparams.rewrite_module_tmp,
        track="in",
    )
    if "subject_" in hparams.fact_token and hparams.fact_token.index("subject_") == 0:
        word = request["subject"]
        print(f"Selected u projection object {word}")
        cur_repr = repr_tools.get_reprs_at_word_tokens(
            context_templates=[
                templ.format(request["prompt"]) for templ in context_templates
            ],
            words=[word for _ in range(len(context_templates))],
            subtoken=hparams.fact_token[len("subject_") :],
            **word_repr_args,
        ).mean(0)
    elif hparams.fact_token == "last":
        # Heuristic to choose last word. Not a huge deal if there's a minor
        # edge case (e.g. multi-token word) because the function below will
        # take the last token.
        cur_repr = repr_tools.get_reprs_at_idxs(
            contexts=[
                templ.format(request["prompt"].format(request["subject"]))
                for templ in context_templates
            ],
            idxs=[[-1] for _ in range(len(context_templates))],
            **word_repr_args,
        ).mean(0)
        print("Selected u projection token with last token")
    elif str.isdigit(hparams.fact_token):
        cur_repr = repr_tools.get_reprs_at_idxs(
            contexts=[
                templ.format(request["prompt"].format(request["subject"]))
                for templ in context_templates
            ],
            idxs=repr_tools.get_flexible_idxs_in_templates(tok, context_templates, request["prompt"].format(request["subject"]), int(hparams.fact_token)),
            **word_repr_args,
        ).mean(0)
        print("Selected u projection token with last token")
    else:
        raise ValueError(f"fact_token={hparams.fact_token} not recognized")

    # Apply inverse second moment adjustment
    u = cur_repr.to(dtype = torch.get_default_dtype())
    if hparams.mom2_adjustment:
        u = get_inv_cov(
            model,
            hparams.rewrite_module_tmp.format(layer),
            hparams.mom2_dataset,
            hparams.mom2_n_samples,
            hparams.mom2_dtype,
        ).to(dtype = torch.get_default_dtype()) @ u.unsqueeze(1)
        u = u.squeeze()

    return u / u.norm()
