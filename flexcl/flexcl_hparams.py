from dataclasses import dataclass
from typing import List

from util.hparams import HyperParams


@dataclass
class FLEXCLHyperParams(HyperParams):
    # Method
    loss_type: str
    start_token_index: int
    end_token_index: int
    start_layer_for_location: int
    fixed_layer_for_editing: int
    end_layer: int
    trace_subject: bool

    v_num_grad_steps: int
    v_lr: float
    v_loss_layer: int
    v_weight_decay: float
    clamp_norm_factor: float
    kl_factor: float
    mom2_adjustment: bool
    context_template_length_params: List[List[int]]

    # Module templates
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str

    # Statistics
    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str

    kl_format: str
