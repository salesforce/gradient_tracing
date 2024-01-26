'''
Compute the probability that the correct answer croesses a correctness threshold.
'''

import json
import time
import copy

correct_answer = True
threshold = 0.5
with open('counterfact_true_vicuna_probs.json', 'r') as f:
    dataset = json.load(f)

original_threshold_correct_counter = 0
original_correct_counter = 0
original_counter = 0
gen_threshold_correct_counter = 0
gen_correct_counter = 0
gen_counter = 0
spec_threshold_correct_counter = 0
spec_correct_counter = 0
spec_counter = 0

for entry in dataset:
    original_counter += 1
    true_prob = entry['model_true_prob']
    false_prob = entry['model_false_prob']
    if correct_answer == True:
        correct_prob = true_prob
        incorrect_prob = false_prob
    else:
        correct_prob = false_prob
        incorrect_prob = true_prob
    original_threshold_correct_counter += (1 if correct_prob>threshold else 0)
    original_correct_counter += (1 if correct_prob>incorrect_prob else 0)

    for pair in entry['paraphrase_probs']:
        gen_counter += 1
        true_prob = pair['model_true_prob']
        false_prob = pair['model_false_prob']
        if correct_answer == True:
            correct_prob = true_prob
            incorrect_prob = false_prob
        else:
            correct_prob = false_prob
            incorrect_prob = true_prob
        gen_threshold_correct_counter += (1 if correct_prob > threshold else 0)
        gen_correct_counter += (1 if correct_prob > incorrect_prob else 0)

    for pair in entry['neighborhood_probs']:
        spec_counter += 1
        true_prob = pair['model_true_prob']
        false_prob = pair['model_false_prob']
        if correct_answer == True:
            correct_prob = true_prob
            incorrect_prob = false_prob
        else:
            correct_prob = false_prob
            incorrect_prob = true_prob
        spec_threshold_correct_counter += (1 if correct_prob > threshold else 0)
        spec_correct_counter += (1 if correct_prob > incorrect_prob else 0)

print(original_counter, gen_counter, spec_counter)
print(original_threshold_correct_counter/original_counter, gen_threshold_correct_counter/gen_counter, spec_threshold_correct_counter/spec_counter)
print(original_correct_counter / original_counter, gen_correct_counter / gen_counter, spec_correct_counter / spec_counter)
print((original_correct_counter+gen_correct_counter+spec_correct_counter)/(original_counter+gen_counter+spec_counter))
