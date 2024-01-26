import json
import os
from pathlib import Path

run='run_002'

editing_locations_directory = Path(os.getcwd() + '/editing_locations')
results_directory = Path(os.getcwd() + '/results/FLEXCL/'+run)
editing_locations_dict = {}
for file in editing_locations_directory.glob('*'):
    try:
        file_dict = json.loads(file.read_text())
    except:
        print(file, 'not json!')
        continue
    if file_dict['fact_token'] < file_dict['first_subject_token']:
        location = 'before_subject'
    elif file_dict['first_subject_token'] <= file_dict['fact_token'] < file_dict['last_subject_token']:
        location = 'subject_not_last'
    elif file_dict['fact_token'] == file_dict['last_subject_token']:
        location = 'subject_last'
    elif file_dict['last_subject_token'] < file_dict['fact_token'] < file_dict['total_num_tokens'] - 1:
        location = 'after_subject_before_last'
    else:
        location = 'last'
    overlap = file_dict['last_subject_token'] == file_dict['total_num_tokens'] - 1
    editing_locations_dict[file_dict['case_id']] = {'location': location, 'overlap': overlap,
                                    'token_ratio': (file_dict['fact_token'] + 1) / file_dict['total_num_tokens'],
                                    'layer': file_dict['layer'],
                                    'signed_distance_from_last_subject_token': file_dict['fact_token'] - file_dict[
                                        'last_subject_token'], 'unsigned_distance_from_last_subject_token': abs(
            file_dict['fact_token'] - file_dict['last_subject_token']),
                                    'normalized_signed_distance_from_last_subject_token': (file_dict['fact_token'] -
                                                                                           file_dict[
                                                                                               'last_subject_token']) /
                                                                                          file_dict['total_num_tokens'],
                                    'normalized_unsigned_distance_from_last_subject_token': abs(
                                        file_dict['fact_token'] - file_dict['last_subject_token']) / file_dict[
                                                                                                'total_num_tokens']}
results_dict = {}
for file in results_directory.glob('*'):
    try:
        file_dict = json.loads(file.read_text())
    except:
        print(file, 'not json!')
        continue
    if str(file).endswith('params.json'):
        continue
    good = {'post':[], 'pre': []}
    for key in {'post', 'pre'}:
            for instance in file_dict[key]["neighborhood_prompts_probs"]:
                good[key].append(instance['target_new'] > instance['target_true'])
    results_dict[file_dict['case_id']]=good

output = {'g2g': 0, 'g2b': 0, 'b2b': 0, 'b2g': 0}
for instance in results_dict.values():
    for i in range(len(instance['pre'])):
        if instance['pre'][i]==True and instance['post'][i]==True:
            output['g2g'] += 1
        if instance['pre'][i]==True and instance['post'][i]==False:
            output['g2b'] += 1
        if instance['pre'][i]==False and instance['post'][i]==True:
            output['b2g'] += 1
        if instance['pre'][i]==False and instance['post'][i]==False:
            output['b2b'] += 1

print(output)

total_num = sum(output.values())
for key, x in output.items():
    print(key,x*100/total_num)