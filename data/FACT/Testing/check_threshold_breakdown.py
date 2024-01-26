'''
Computes correctness statistics broken down by category.
'''

import json
import time
import copy

threshold = 0.5

original_threshold_correct_counter = 0
original_correct_counter = 0
original_counter = 0
gen_threshold_correct_counter = 0
gen_correct_counter = 0
gen_counter = 0
spec_threshold_correct_counter = 0
spec_correct_counter = 0
spec_counter = 0


with open('vicuna_prob_dataset.json', 'r') as f:
    dataset = json.load(f)

for topic, entries in dataset.items():
    tic = time.perf_counter()
    print('topic', topic, 'begins', tic)
    for index, entry in enumerate(entries):
        original_counter += 1
        all_correct = True
        actual_answer = dataset[topic][index]['truth_value']
        for key in entry:
            if key.startswith('original_statement'):
                false_prob, true_prob = dataset[topic][index]['model_false_prob'], dataset[topic][index]['model_true_prob']
                if actual_answer == False:
                    correct_prob = false_prob
                    larger = false_prob > true_prob
                else:
                    correct_prob = true_prob
                    larger = true_prob > false_prob
                correct_above_threshold = correct_prob > threshold
                if correct_above_threshold:
                    original_threshold_correct_counter += 1
                if larger:
                    original_correct_counter += 1

            elif key == 'rephrases':
                for index2, rephrase in enumerate(entry[key]):
                    gen_counter += 1
                    false_prob = dataset[topic][index][key][index2]['model_false_prob']
                    true_prob = dataset[topic][index][key][index2]['model_true_prob']
                    if actual_answer == False:
                        correct_prob = false_prob
                        larger = false_prob > true_prob
                    else:
                        correct_prob = true_prob
                        larger = true_prob > false_prob
                    correct_above_threshold = correct_prob > threshold
                    if correct_above_threshold:
                        gen_threshold_correct_counter += 1
                    if larger:
                        gen_correct_counter += 1

            elif key == 'main_terms' or key == 'main terms':
                for term_num, term_dict in enumerate(entry[key]):
                    for subkey in term_dict:
                        if subkey.endswith('_statements'):
                            for subindex, subentry in enumerate(term_dict[subkey]):
                                sentence = subentry['statement']
                                #print(dataset[topic][index][key][term_num][subkey][subindex])
                                false_prob, true_prob = dataset[topic][index][key][term_num][subkey][subindex]['model_false_prob'], dataset[topic][index][key][term_num][subkey][subindex]['model_true_prob']
                                local_answer = dataset[topic][index][key][term_num][subkey][subindex]['truth_value']
                                if local_answer != actual_answer:
                                    continue
                                else:
                                    spec_counter += 1
                                if actual_answer == False:
                                    correct_prob = false_prob
                                    larger = false_prob > true_prob
                                else:
                                    correct_prob = true_prob
                                    larger = true_prob > false_prob
                                correct_above_threshold = correct_prob > threshold
                                if correct_above_threshold:
                                    spec_threshold_correct_counter += 1
                                if larger:
                                    spec_correct_counter += 1
    toc = time.perf_counter()
    print('topic', topic, 'ends', toc, 'total time', toc-tic)
    print(original_counter, gen_counter, spec_counter)
    print(original_threshold_correct_counter/original_counter, gen_threshold_correct_counter/gen_counter, spec_threshold_correct_counter/spec_counter)
    print(original_correct_counter / original_counter, gen_correct_counter / gen_counter,
          spec_correct_counter / spec_counter)
    print((original_correct_counter + gen_correct_counter + spec_correct_counter) / (
                original_counter + gen_counter + spec_counter))

