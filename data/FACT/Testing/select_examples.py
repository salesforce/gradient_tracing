'''
Sampling statements post collection.
'''

import json
import time
import pickle
import numpy as np

seed = 42
num_examples = 100
np.random.seed(seed)

with open('examples_list.pkl', 'rb') as f:
    temp=pickle.load(f)
    collection, negation_collection=temp[0], temp[1]


collection = {'original_statements': [], 'rephrases': [], 'main_terms': []}
negation_collection = {'original_statements': [], 'rephrases': [], 'main_terms': []}

with open('vicuna_prob_dataset.json', 'r') as f:
    dataset = json.load(f)

for topic, entries in dataset.items():
    tic = time.perf_counter()
    print('topic', topic, 'begins', tic)
    for index, entry in enumerate(entries):
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
                                collection['main_terms'].append((sentence,truth_value))
                                negation = subentry['negation']
                                negation_collection['main_terms'].append((negation,not truth_value))

with open('examples_list.pkl', 'wb') as f:
    pickle.dump([collection,negation_collection],f)

