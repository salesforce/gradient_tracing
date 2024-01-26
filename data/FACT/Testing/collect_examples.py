'''
Collecting examples from FACT prior to sampling and shuffling them for manual accuracy testing.
'''

import json
import time
import pickle
import random

seed=42
random.seed(42)

collection = {'original_statements': [], 'rephrases': [], 'main_terms_same_direction': [], 'main_terms_opposite_direction': []}
negation_collection = {'original_statements': [], 'rephrases': [], 'main_terms_same_direction': [], 'main_terms_opposite_direction': []}

with open('vicuna_prob_dataset.json', 'r') as f:
    dataset = json.load(f)

for topic, entries in dataset.items():
    tic = time.perf_counter()
    print('topic', topic, 'begins', tic)
    for index, entry in enumerate(entries):
        real_truth_value = dataset[topic][index]['truth_value']
        for key in entry:
            if key.startswith('original_statement'):
                truth_value = dataset[topic][index]['truth_value']
                sentence = dataset[topic][index][key]
                collection['original_statements'].append((sentence,truth_value))
                negation = dataset[topic][index]['negation']
                negation_collection['original_statements'].append((negation, not truth_value))
            elif key == 'rephrases':
                for index2, rephrase in enumerate(entry[key]):
                    truth_value = dataset[topic][index]['truth_value']
                    sentence = dataset[topic][index][key][index2]['statement']
                    collection['rephrases'].append((sentence,truth_value))
                    negation = dataset[topic][index][key][index2]['negation']
                    negation_collection['rephrases'].append((negation, not truth_value))
            elif key == 'main_terms' or key == 'main terms':
                for term_num, term_dict in enumerate(entry[key]):
                    for subkey in term_dict:
                        if subkey.endswith('_statements'):
                            for subindex, subentry in enumerate(term_dict[subkey]):
                                truth_value = subentry['truth_value']
                                sentence = subentry['statement']
                                if truth_value == real_truth_value:
                                    collection['main_terms_same_direction'].append((sentence,truth_value))
                                else:
                                    collection['main_terms_opposite_direction'].append((sentence, truth_value))
                                negation = subentry['negation']
                                if truth_value == real_truth_value:
                                    negation_collection['main_terms_same_direction'].append((negation, truth_value))
                                else:
                                    negation_collection['main_terms_opposite_direction'].append((negation, truth_value))
for x in collection.values():
    random.shuffle(x)
for x in negation_collection.values():
    random.shuffle(x)

with open('examples_list.pkl', 'wb') as f:
    pickle.dump([collection,negation_collection],f)

