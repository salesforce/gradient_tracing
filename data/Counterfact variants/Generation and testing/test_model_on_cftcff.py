'''
This script takes counterfact_true.json or counterfact_false.json and runs it through vicuna-7b, providing the probabilities
of the output True and False in a new dataset.
'''
dataset_file = 'counterfact_true.json' #can change to counterfact_false.json

model_name='lmsys/vicuna-7b-v1.3'
import torch
import json
import time
import copy
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=False)
model.to('cuda')

with open(dataset_file, 'r') as f:
    dataset = json.load(f)

new_dataset = copy.deepcopy(dataset)

def test(sentence, model = model, false_token = 7700, true_token = 5852):
    #computes the output probabilities given an prompt
    inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
    inputs.pop('token_type_ids')
    inputs = inputs.to(model.device)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits[0, -1, :].float(), dim=0)
    false_prob, true_prob = probs[false_token].item(), probs[true_token].item()
    print('sentence', sentence, 'False prob:', false_prob, 'True prob:', true_prob)
    return false_prob, true_prob

counter = 0
for index, entry in enumerate(dataset):
    counter += 1
    tic = time.perf_counter()
    print('counter', counter, 'begins', tic)
    original = entry['requested_rewrite']['prompt'].format(entry['requested_rewrite']['subject'])
    false_prob, true_prob = test(original)
    new_dataset[counter]['model_false_prob'] = false_prob
    new_dataset[counter]['model_true_prob'] = true_prob
    new_dataset[counter]['paraphrase_probs'] = []
    for paraphrase in entry['paraphrase_prompts']:
        false_prob, true_prob = test(paraphrase)
        new_dataset[counter]['paraphrase_probs'].append({'model_false_prob': false_prob, 'model_true_prob': true_prob})
    new_dataset[counter]['neighborhood_probs'] = []
    for prompt in entry['neighborhood_prompts']:
        false_prob, true_prob = test(prompt)
        new_dataset[counter]['neighborhood_probs'].append({'model_false_prob': false_prob, 'model_true_prob': true_prob})
    toc = time.perf_counter()
    if counter % 1000 == 0:
        with open('new_' + dataset_file + str(counter) + '.json', 'w') as f:
            json.dump(new_dataset, f)
    print('counter', counter, 'ends', toc, 'total time', toc - tic)

with open(dataset_file+'_vicuna_probs'+'.json', 'w') as f:
    json.dump(new_dataset, f)


