'''
Checks whether the probability of Vicuna-7b matching the labeled answer crosses a pre-specified threshold.
'''

import json
import time
import copy

threshold = 0.5
with_negation = True

with open('vicuna_prob_fact.json', 'r') as f:
    dataset = json.load(f)

new_dataset_threshold = copy.deepcopy(dataset)
correct_counter = 0
incorrect_counter = 0
all_correct_counter = 0
entries_counter = 0

for topic, entries in dataset.items():
    tic = time.perf_counter()
    print('topic', topic, 'begins', tic)
    for index, entry in enumerate(entries):
        entries_counter += 1
        all_correct = True
        for key in entry:
            if key.startswith('original_statement'):
                false_prob, true_prob = dataset[topic][index]['model_false_prob'], dataset[topic][index]['model_true_prob']
                actual_answer = dataset[topic][index]['truth_value']
                if actual_answer == False:
                    correct_prob = false_prob
                else:
                    correct_prob = true_prob
                correct_above_threshold = correct_prob > threshold
                if correct_above_threshold:
                    correct_counter += 1
                    #print('correct:', topic, dataset[topic][index][key])
                else:
                    incorrect_counter += 1
                    #print('incorrect:', topic, dataset[topic][index][key])
                    all_correct = False
                new_dataset_threshold[topic][index]['correct_above_threshold'] = correct_above_threshold

                if with_negation:
                    false_prob, true_prob = dataset[topic][index]['negation_model_false_prob'], dataset[topic][index][
                        'negation_model_true_prob']
                    actual_answer = not dataset[topic][index]['truth_value']
                    if actual_answer == False:
                        correct_prob = false_prob
                    else:
                        correct_prob = true_prob
                    correct_above_threshold = correct_prob > threshold
                    if correct_above_threshold:
                        correct_counter += 1
                        # print('correct:', topic, dataset[topic][index][key])
                    else:
                        incorrect_counter += 1
                        # print('incorrect:', topic, dataset[topic][index][key])
                        all_correct = False
                    new_dataset_threshold[topic][index]['negation_correct_above_threshold'] = correct_above_threshold

            elif key == 'rephrases':
                for index2, rephrase in enumerate(entry[key]):
                    false_prob = dataset[topic][index][key][index2]['model_false_prob']
                    true_prob = dataset[topic][index][key][index2]['model_true_prob']
                    actual_answer = dataset[topic][index]['truth_value']
                    if actual_answer == False:
                        correct_prob = false_prob
                    else:
                        correct_prob = true_prob
                    correct_above_threshold = correct_prob > threshold
                    if correct_above_threshold:
                        correct_counter += 1
                        # print('correct:', topic, dataset[topic][index][key])
                    else:
                        incorrect_counter += 1
                        # print('incorrect:', topic, dataset[topic][index][key])
                        all_correct = False
                    new_dataset_threshold[topic][index]['rephrases'][index2]['correct_above_threshold'] = correct_above_threshold

                    if with_negation:

                        false_prob = dataset[topic][index][key][index2]['negation_model_false_prob']
                        true_prob = dataset[topic][index][key][index2]['negation_model_true_prob']
                        actual_answer = not dataset[topic][index]['truth_value']
                        if actual_answer == False:
                            correct_prob = false_prob
                        else:
                            correct_prob = true_prob
                        correct_above_threshold = correct_prob > threshold
                        if correct_above_threshold:
                            correct_counter += 1
                            # print('correct:', topic, dataset[topic][index][key])
                        else:
                            incorrect_counter += 1
                            # print('incorrect:', topic, dataset[topic][index][key])
                            all_correct = False
                        new_dataset_threshold[topic][index]['rephrases'][index2][
                            'negation_correct_above_threshold'] = correct_above_threshold


            elif key == 'main_terms' or key == 'main terms':
                for term_num, term_dict in enumerate(entry[key]):
                    for subkey in term_dict:
                        if subkey.endswith('_statements'):
                            for subindex, subentry in enumerate(term_dict[subkey]):
                                sentence = subentry['statement']
                                print(dataset[topic][index][key][term_num][subkey][subindex])
                                false_prob, true_prob = dataset[topic][index][key][term_num][subkey][subindex]['model_false_prob'], dataset[topic][index][key][term_num][subkey][subindex]['model_true_prob']
                                actual_answer = dataset[topic][index][key][term_num][subkey][subindex]['truth_value']
                                if actual_answer == False:
                                    correct_prob = false_prob
                                else:
                                    correct_prob = true_prob
                                correct_above_threshold = correct_prob > threshold
                                if correct_above_threshold:
                                    correct_counter += 1
                                    #print('correct:', topic, dataset[topic][index][key][term_num][subkey][subindex]['statement'])
                                else:
                                    incorrect_counter += 1
                                    #print('incorrect:', topic, dataset[topic][index][key][term_num][subkey][subindex]['statement'])
                                    all_correct = False
                                new_dataset_threshold[topic][index][key][term_num][subkey][subindex]['correct_above_threshold'] = correct_above_threshold

                                if with_negation:
                                    false_prob, true_prob = dataset[topic][index][key][term_num][subkey][subindex][
                                        'negation_model_false_prob'], dataset[topic][index][key][term_num][subkey][subindex][
                                        'negation_model_true_prob']
                                    actual_answer = not dataset[topic][index][key][term_num][subkey][subindex][
                                        'truth_value']
                                    if actual_answer == False:
                                        correct_prob = false_prob
                                    else:
                                        correct_prob = true_prob
                                    correct_above_threshold = correct_prob > threshold
                                    if correct_above_threshold:
                                        correct_counter += 1
                                        # print('correct:', topic, dataset[topic][index][key][term_num][subkey][subindex]['statement'])
                                    else:
                                        incorrect_counter += 1
                                        # print('incorrect:', topic, dataset[topic][index][key][term_num][subkey][subindex]['statement'])
                                        all_correct = False
                                    new_dataset_threshold[topic][index][key][term_num][subkey][subindex][
                                        'negation_correct_above_threshold'] = correct_above_threshold

        if all_correct:
            all_correct_counter += 1
    print('correct', correct_counter, 'incorrect', incorrect_counter, 'all_correct', all_correct_counter)

with open('fact_threshold_'+str(threshold)+'.json', 'w') as f:
    json.dump(new_dataset_threshold, f, indent=4)
toc = time.perf_counter()
print('topic', topic, 'ends', toc, 'total time', toc-tic)

