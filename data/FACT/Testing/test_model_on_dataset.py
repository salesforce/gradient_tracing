'''
This script takes FACT and runs it through vicuna-7b, providing the probabilities
of the output True and False in a new dataset.
'''
model_name='lmsys/vicuna-7b-v1.3'
import torch
import json
import time
import copy
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
from transformers import AutoModelForCausalLM
model= AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=False) #, device_map="auto"
model.to('cuda')

with open('fact.json', 'r') as f:
    dataset = json.load(f)

new_dataset = copy.deepcopy(dataset)

def test(sentence, model=model, false_token=7700, true_token=5852):
    sentence = 'True or false: '+ sentence +'\nAnswer:'
    inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
    inputs.pop('token_type_ids')
    inputs = inputs.to(model.device)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits[0, -1, :].float(), dim=0)
    false_prob, true_prob = probs[false_token].item(), probs[true_token].item()
    print('sentence', sentence, 'False prob:', false_prob, 'True prob:', true_prob)
    return false_prob, true_prob

counter = 0
for topic, entries in dataset.items():
    counter += 1
    tic = time.perf_counter()
    print('topic', topic, 'begins', tic)
    for index, entry in enumerate(entries):
        for key in entry:
            if key.startswith('original_statement'):
                sentence = entry[key]
                false_prob, true_prob = test(sentence)
                new_dataset[topic][index]['model_false_prob'] = false_prob
                new_dataset[topic][index]['model_true_prob'] = true_prob
                negation = entry['negation']
                false_prob, true_prob = test(negation)
                new_dataset[topic][index]['negation_model_false_prob'] = false_prob
                new_dataset[topic][index]['negation_model_true_prob'] = true_prob
            elif key == 'rephrases':
                for index2, rephrase in enumerate(entry[key]):
                    sentence = rephrase['statement']
                    false_prob, true_prob = test(sentence)
                    new_dataset[topic][index][key][index2]['model_false_prob'] = false_prob
                    new_dataset[topic][index][key][index2]['model_true_prob'] = true_prob
                    negation = rephrase['negation']
                    false_prob, true_prob = test(negation)
                    new_dataset[topic][index][key][index2]['negation_model_false_prob'] = false_prob
                    new_dataset[topic][index][key][index2]['negation_model_true_prob'] = true_prob
            elif key == 'main_terms' or key == "main terms":
                for term_num, term_dict in enumerate(entry[key]):
                    for subkey in term_dict:
                        if subkey.endswith('_statements'):
                            for subindex, subentry in enumerate(term_dict[subkey]):
                                sentence = subentry['statement']
                                false_prob, true_prob = test(sentence)
                                new_dataset[topic][index][key][term_num][subkey][subindex]['model_false_prob'] = false_prob
                                new_dataset[topic][index][key][term_num][subkey][subindex]['model_true_prob'] = true_prob
                                negation = subentry['negation']
                                false_prob, true_prob = test(negation)
                                new_dataset[topic][index][key][term_num][subkey][subindex]['negation_model_false_prob'] = false_prob
                                new_dataset[topic][index][key][term_num][subkey][subindex]['negation_model_true_prob'] = true_prob
    with open('new_dataset'+str(counter)+'.json', 'w') as f:
        json.dump(new_dataset, f)
    toc = time.perf_counter()
    print('topic', topic, 'ends', toc, 'total time', toc-tic)

