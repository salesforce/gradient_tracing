'''
computes the probability that the answer is within the acceptable range (either true or false).
'''

import json
import time
import copy

with open('counterfact_true_vicuna_probs.json', 'r') as f:
    dataset = json.load(f)

total = 0
counter = 0

for entry in dataset:
    total += entry['model_true_prob']
    total += entry['model_false_prob']
    counter += 1
    for category in {'paraphrase_probs','neighborhood_probs'}:
        for pair in entry[category]:
            total += pair['model_true_prob']
            total += pair['model_false_prob']
            counter += 1
print(total, counter, total/counter)

