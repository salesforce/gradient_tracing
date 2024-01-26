'''
Computes the probability of proper format (true+false) response from Vicuna.
'''

import json
import time

with_negation = False
total = 0
counter = 0

with open('vicuna_prob_dataset.json', 'r') as f:
    dataset = json.load(f)

for topic, entries in dataset.items():
    tic = time.perf_counter()
    print('topic', topic, 'begins', tic)
    for index, entry in enumerate(entries):
        for key in entry:
            if key.startswith('original_statement'):
                total += (dataset[topic][index]['model_false_prob'] + dataset[topic][index]['model_true_prob'])
                counter += 1
                if with_negation:
                    total += (dataset[topic][index]['negation_model_false_prob'] + dataset[topic][index]['negation_model_true_prob'])
                    counter += 1
            elif key == 'rephrases':
                for index2, rephrase in enumerate(entry[key]):
                    false_prob = dataset[topic][index][key][index2]['model_false_prob']
                    true_prob = dataset[topic][index][key][index2]['model_true_prob']
                    total += (false_prob+true_prob)
                    counter += 1
                    if with_negation:
                        false_prob = dataset[topic][index][key][index2]['negation_model_false_prob']
                        true_prob = dataset[topic][index][key][index2]['negation_model_true_prob']
                        total += (false_prob + true_prob)
                        counter += 1
            elif key == 'main_terms' or key == 'main terms':
                for term_num, term_dict in enumerate(entry[key]):
                    for subkey in term_dict:
                        if subkey.endswith('_statements'):
                            for subindex, subentry in enumerate(term_dict[subkey]):
                                sentence = subentry['statement']
                                total += (dataset[topic][index][key][term_num][subkey][subindex]['model_false_prob'] + dataset[topic][index][key][term_num][subkey][subindex]['model_true_prob'])
                                counter += 1
                                if with_negation:
                                    false_prob, true_prob = dataset[topic][index][key][term_num][subkey][subindex][
                                        'negation_model_false_prob'], dataset[topic][index][key][term_num][subkey][subindex][
                                        'negation_model_true_prob']
                                    total += (false_prob+true_prob)
                                    counter += 1
print(total, counter, total/counter)

