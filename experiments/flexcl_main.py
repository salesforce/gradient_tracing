import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import rome
from copy import deepcopy
from typing import Dict, List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rome.rome_hparams import ROMEHyperParams
CONTEXT_TEMPLATES_CACHE = None
from flexcl.flexcl_hparams import FLEXCLHyperParams
import json

def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()

class InfError(Exception):
    pass
class GradientTracker:
    '''
    Attaches itself to a pytorch module and records the gradients throughout its submodules.
    Attributes:
        1. state_gradients_dict: state_gradients_dict[name][i] containing the gradient of submodule name in the i-th
        run (since GradientTracker was attached).
        2. reverse_lookup: reverse_lookup[name] is the module with the appropriate name.
    '''
    #reverse_lookup is only needed because the hook has access to the module but not its name: if the module contained
    #its name as an attribute, there would be no need for this and we could actually implement this class faster, as
    #we'd be able to get the name from the module in O(1) time instead of O(n) time.
    def __init__(self, model):
        # Initializes the tracker, sets the requires_grad property of all parameters to be True and zeros it.
        self.model = model
        for param in model.parameters():
            param.requires_grad = True
        self.reverse_lookup = {}
        for name, module in list(self.model.named_modules(remove_duplicate=False))[1:]:
            self.reverse_lookup[name] = module
        self.reset()
        self.handle = torch.nn.modules.module.register_module_full_backward_hook(self.get_state_gradient)

    def get_key_from_value(self, d, val):
        for key, value in d.items():
            if value is val:  # we want the actual object, not just an identical layer, hence 'is' instead of ==
                return key
        return None

    def get_state_gradient(self, module, grad_input, grad_output):
        name = self.get_key_from_value(self.reverse_lookup, module)
        if name is not None:
            self.state_gradients_dict[name].append(grad_output[0].detach()) #.cpu() here caused cpu "memory leak"

    def reset(self):
        self.state_gradients_dict = {}
        for name, module in list(self.model.named_modules(remove_duplicate=False))[1:]:
            self.state_gradients_dict[name] = []
        self.model.zero_grad()

    def compute_grad_norms(self, component: str = 'mlp', run_num: int = 0, num_input_tokens: int = 0, num_layers: int = 0):
        #Computes the norms for all tokens and layers: ignoring some happens in argmax_grad_norm.
        norms = torch.zeros((num_input_tokens, num_layers))
        for layer, layer_gradients in self.state_gradients_dict.items():
            splitted_layer_name = layer.split('.')
            check=False
            if component in ['mlp']:
                if splitted_layer_name[-1]==component:
                    layer_num = int(splitted_layer_name[-2])
                    check=True
            else:
                raise ValueError("Only component='mlp' is currently supported")
            if check:
                all_tokens_gradients = layer_gradients[run_num][0]
                for token_index, gradient in enumerate(all_tokens_gradients):
                    gradient_norm = torch.linalg.vector_norm(gradient).item()
                    norms[token_index, layer_num] = gradient_norm
        output = torch.zeros((num_input_tokens, num_layers))
        for input_token in range(num_input_tokens):
            for layer in range(num_layers):
                output[input_token, layer] = norms[input_token, layer:layer+1].mean()
        return output

    def remove_handle(self):
        self.handle.remove()

    def argmax_grad_norm(self, start_token_index = 5, end_token_index=-4, start_layer = 0, end_layer = 31, component: str = 'mlp', run_num: int = 0, num_input_tokens: int = None, num_layers: int = 32, inf_adjustment = True):
        avg_norm = self.compute_grad_norms(component = component, run_num = run_num, num_input_tokens = num_input_tokens, num_layers = num_layers)[start_token_index:end_token_index+1, start_layer:end_layer+1]
        if inf_adjustment and avg_norm.isinf().any() == True:
                raise InfError("inf grad norm detected")
        max_token, max_layer = np.unravel_index(torch.argmax(avg_norm), avg_norm.shape)
        max_token += start_token_index
        max_layer += start_layer
        return max_token, max_layer

def compute_probability_of_answer(tok, probs, answer, llama:bool = True):
    #probs is assumed to be a shape (1, vocab) tensor.
    if answer[0] != " ":
        # Space required for correct tokenization
        answer = " " + answer
    if not llama:
        token = tok(answer)['input_ids'][0]
    else:
        token = tok(answer, add_special_tokens=False)['input_ids'][1] #changing this to 1 from -1, doesn't matter for wrapped, necessary for unwrapped
    prev_prob = probs[0, token]
    return prev_prob


def apply_flexcl_to_model(
    model_for_rome: AutoModelForCausalLM,
    location_model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    case_id: int,
    hparams: FLEXCLHyperParams,
    copy=False,
    return_orig_weights=False,
    run_dir="",
    hparams_dir="",
    notebook_mode = False
) -> Tuple[AutoModelForCausalLM, List[str]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) an original copy of the weights that changed
    """

    # we assume here that requests contains exactly one entry
    if copy:
        location_model = copy.deepcopy(location_model)

    request = requests[0]
    prompt = request['prompt']
    subject = request['subject']
    prompt_filled = prompt.replace('{}', subject)

    tracker = GradientTracker(location_model)
    location_model.zero_grad()

    inp = tok(prompt_filled, return_tensors='pt').to(location_model.device) #bos included

    out = location_model(**inp)['logits']
    probs = torch.softmax(out[:, -1], dim=1)
    if hparams.loss_type in ['both']:
        prev_correct_answer = request['target_true']['str']
        prev_prob = compute_probability_of_answer(tok, probs, prev_correct_answer)
    if hparams.loss_type in ['new', 'both']:
        new_correct_answer = request['target_new']['str']
        new_prob = compute_probability_of_answer(tok, probs, new_correct_answer)

    if hparams.loss_type == 'new':
        loss = 1-new_prob
    elif hparams.loss_type == 'both':
        loss = 1-new_prob+prev_prob
    loss.backward()

    num_input_tokens = len(inp['input_ids'][0])
    num_layers = len(location_model.model.layers)


    if hparams.trace_subject:
        last_subject_token_index = rome.repr_tools.get_words_idxs_in_templates(tok, [prompt], [subject], "last")[0]
        last_subject_token_index = last_subject_token_index[0]

        first_subject_token_index = rome.repr_tools.get_words_idxs_in_templates(tok, [prompt], [subject], "first")[0]
        first_subject_token_index = first_subject_token_index[0]

        last_subject_token_index += 1 #considers bos
        first_subject_token_index += 1 #considers bos
    if notebook_mode:
        current_directory = os.getcwd()
        parent_directory = os.path.dirname(current_directory)
        hparams_dir = parent_directory + "/hparams"
    ROME_hparams = ROMEHyperParams.from_json(hparams_dir + '/ROME/Mistral-7B-instruct-v0.2.json') #the actual network here doesn't matters

    for flexcl_attribute in hparams.__dict__.items():
        for rome_attribute in ROME_hparams.__dict__.items():
            if flexcl_attribute[0] == rome_attribute[0]:
                ROME_hparams.__dict__[flexcl_attribute[0]] = flexcl_attribute[1]

    ROME_hparams.fact_token, ROME_hparams.layers = tracker.argmax_grad_norm(start_token_index=hparams.start_token_index,
                             end_token_index=hparams.end_token_index,
                             start_layer=hparams.start_layer_for_location,
                             end_layer=hparams.end_layer,
                             component='mlp',
                             run_num=0,
                             num_input_tokens=num_input_tokens,
                             num_layers=num_layers,
                             inf_adjustment=True)
    if isinstance(hparams.fixed_layer_for_editing, int):
        ROME_hparams.layers = hparams.fixed_layer_for_editing

    dictionary = {
        "case_id": case_id,
        "layer": ROME_hparams.layers,
        "fact_token": ROME_hparams.fact_token,
        "total_num_tokens": num_input_tokens, #note: tokens are indexed from 0, so this is 1+last token index. Also: bos included
        "decoded_fact_token": tok.decode(inp['input_ids'][0][ROME_hparams.fact_token]),
        "everything_up_to_fact_token": tok.decode(inp['input_ids'][0][ : ROME_hparams.fact_token+1])
    }
    print("editing at the end of",dictionary["everything_up_to_fact_token"])
    if hparams.trace_subject:
        dictionary["last_subject_token"] = last_subject_token_index
        dictionary["first_subject_token"] = first_subject_token_index
    # Serializing json
    json_object = json.dumps(dictionary, default = np_encoder)
    os.makedirs(run_dir+"/editing_locations",exist_ok=True)
    with open(run_dir+'/editing_locations/editing_location_case_' + str(case_id) +'.json', 'w') as f:
        f.write(json_object)

    ROME_hparams.layers = [ROME_hparams.layers]
    ROME_hparams.fact_token = str(ROME_hparams.fact_token)
    del out
    tracker.remove_handle()
    tracker.reset()
    del tracker
    del loss
    del new_prob
    if copy:
        model_for_rome = deepcopy(model_for_rome)

    return rome.rome_main.apply_rome_to_model(model=model_for_rome, tok=tok, requests=requests, hparams=ROME_hparams, copy=copy, return_orig_weights=return_orig_weights)
