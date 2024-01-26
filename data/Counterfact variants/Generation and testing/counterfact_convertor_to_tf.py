import json
import copy
import regex as re
import pickle
'''
takes counterfact.json and creates curatedcounterfact.json, counterfacttrue.json, counterfactfalse.json

original_relations has been collected from counterfact by collecting the prompts used in the edit requests for each relation.
problem_relations are removed
good_relations are kept as is 
corrected_relations are kept after replacing some prompts
for each relation in corrected_relations, translations contains a dictionary translating old prompts to new prompts
'''

def find_last_occurrence(pattern: str,text: str) -> re.Match:
    #finds the last occurrence of pattern in text
    match = None
    for x in re.finditer(pattern, text, overlapped=True):
        match = x
    return match

with open('relations_dicts.pkl', 'rb') as f:
    temp = pickle.load(f)
    original_relations = temp['original_relations']
    good_relations = temp['good_relations']
    problem_relations = temp['problem_relations']
    corrected_relations = temp['corrected_relations']
    translations = temp['translations']

included_relations = good_relations|corrected_relations

with open('counterfact.json', 'r') as f:
    counterfact = json.load(f)

#create curated_counterfact.json

curated_counterfact = []

for request_num, request in enumerate(counterfact):
    print('request_num', request_num)
    relation_id = request['requested_rewrite']['relation_id']
    if relation_id in included_relations.keys():
        new_request = copy.deepcopy(request)
        subject = request['requested_rewrite']['subject']
        if relation_id in translations.keys(): #if correction should be considered
            translation_dict = translations[relation_id]
            edit_prompt = [request['requested_rewrite']['prompt']]
            paraphrase_prompts = request['paraphrase_prompts']
            neighborhood_prompts = request['neighborhood_prompts']
            attribute_prompts = request['attribute_prompts']
            for prompt_collection_name, prompt_collection in {'edit_prompt':edit_prompt, 'paraphrase_prompts': paraphrase_prompts,'neighborhood_prompts':neighborhood_prompts, 'attribute_prompts': attribute_prompts}.items():
                for i, prompt in enumerate(prompt_collection):
                    if prompt_collection_name == 'edit_prompt':
                        for to_be_replaced_unfilled_prompt, replacement_unfilled_prompt in translation_dict.items():
                            if prompt == to_be_replaced_unfilled_prompt:
                                new_prompt = replacement_unfilled_prompt
                                new_request['requested_rewrite']['prompt'] = new_prompt
                                break

                    if prompt_collection_name == 'paraphrase_prompts':
                        for to_be_replaced_unfilled_prompt, replacement_unfilled_prompt in translation_dict.items():
                            to_be_replaced_filled_prompt = to_be_replaced_unfilled_prompt.format(subject)
                            match = find_last_occurrence(re.escape(to_be_replaced_filled_prompt) + '$', prompt)
                            if match is not None:
                                new_prompt = replacement_unfilled_prompt.format(subject)
                                new_request['paraphrase_prompts'][i] = new_prompt
                                break

                    if prompt_collection_name in {'neighborhood_prompts', 'attribute_prompts'}:
                        collection = []
                        for to_be_replaced_unfilled_prompt, replacement_unfilled_prompt in translation_dict.items():
                            open_paren=to_be_replaced_unfilled_prompt.find('{')
                            to_be_replaced_pattern = (re.escape(to_be_replaced_unfilled_prompt[:open_paren])+'{}'+re.escape(to_be_replaced_unfilled_prompt[open_paren+2:])).format('(.*)')+'$'

                            match = re.search(to_be_replaced_pattern, prompt)

                            if match is not None:
                                collection += [(match,replacement_unfilled_prompt)]

                        if collection != []:

                            match, replacement_unfilled_prompt = min(collection, key=lambda x: len(x[0].group(1)))
                            local_subject = match.group(1).strip()

                            new_request[prompt_collection_name][i]=replacement_unfilled_prompt.format(local_subject)
        subject = request['requested_rewrite']['subject']
        for unfilled_prompt in included_relations[relation_id]:
            for num_prompt, paraphrase_prompt in enumerate(new_request['paraphrase_prompts']):
                filled_prompt = unfilled_prompt.format(subject)
                match = find_last_occurrence(re.escape(filled_prompt) + '$', paraphrase_prompt)
                if match is not None:
                    new_prompt = filled_prompt
                    new_request['paraphrase_prompts'][num_prompt] = new_prompt
                    break


        curated_counterfact.append(new_request)

with open('../curated_counterfact.json', 'w') as f:
    json.dump(curated_counterfact,f)



with open('../curated_counterfact.json', 'r') as f:
    curated_counterfact=json.load(f)
t2f_counterfact = []
f2t_counterfact = []

def wrap_prompt(prompt, target):
    return 'True or false: '+prompt+ ' ' +target +'.\nAnswer:'

for request_num, request in enumerate(curated_counterfact):
    print('request_num', request_num)

    relation_id=request['requested_rewrite']['relation_id']
    subject = request['requested_rewrite']['subject']
    target_before = request['requested_rewrite']['target_true']['str']
    target_after = request['requested_rewrite']['target_new']['str']

    neighborhood_prompts = request['neighborhood_prompts']
    attribute_prompts = request['attribute_prompts']

    new_neighborhood_prompts_t = [wrap_prompt(prompt,target_before) for prompt in neighborhood_prompts]
    new_neighborhood_prompts_f = [wrap_prompt(prompt, target_after) for prompt in neighborhood_prompts]
    new_attribute_prompts = [wrap_prompt(prompt,target_after) for prompt in attribute_prompts]
    edit_prompt_t2f = wrap_prompt(request['requested_rewrite']['prompt'],target_before)
    edit_prompt_f2t = wrap_prompt(request['requested_rewrite']['prompt'], target_after)

    paraphrase_prompts_t2f=[]
    paraphrase_prompts_f2t = []
    for prompt in request['paraphrase_prompts']:
        check=False
        candidates=[]
        if relation_id in included_relations.keys():
            candidates += list(included_relations[relation_id])
        if candidates==[]:
            raise ValueError('no candidates')
        for to_be_wrapped in candidates:
            to_be_wrapped = to_be_wrapped.format(subject)
            match = find_last_occurrence(re.escape(to_be_wrapped) + '$', prompt)
            if match is not None:
                new_prompt_t2f = wrap_prompt(to_be_wrapped,target_before)
                new_prompt_f2t = wrap_prompt(to_be_wrapped, target_after)
                paraphrase_prompts_t2f.append(new_prompt_t2f)
                paraphrase_prompts_f2t.append(new_prompt_f2t)
                check=True
        if not check:
            print(prompt)
            raise ValueError('something is wrong')

    request_t2f = copy.deepcopy(request)
    request_t2f['requested_rewrite']['prompt'] = edit_prompt_t2f
    request_t2f['paraphrase_prompts'] = paraphrase_prompts_t2f
    request_t2f['neighborhood_prompts']=new_neighborhood_prompts_t
    request_t2f['attribute_prompts'] = new_attribute_prompts
    request_t2f['requested_rewrite']['target_true']['str'] = 'True'
    request_t2f['requested_rewrite']['target_new']['str'] = 'False'
    request_f2t = copy.deepcopy(request_t2f)
    request_f2t['requested_rewrite']['prompt'] = edit_prompt_f2t
    request_f2t['paraphrase_prompts'] = paraphrase_prompts_f2t
    request_f2t['neighborhood_prompts'] = new_neighborhood_prompts_f
    request_f2t['requested_rewrite']['target_true']['str'] = 'False'
    request_f2t['requested_rewrite']['target_new']['str'] = 'True'

    t2f_counterfact += [request_t2f]
    f2t_counterfact += [request_f2t]

with open('../counterfact_true.json', 'w') as f:
    json.dump(t2f_counterfact,f)

with open('../counterfact_false.json', 'w') as f:
    json.dump(f2t_counterfact,f)


"""
corrected_relations = {
            'P103': {
                'The mother tongue of {} is',
                'The native language of {} is',
                '{} is a native speaker of',
                '{} spoke the language',
                '{} natively speaks', #'{}, a native'
                '{} speaks'},#'{}, speaker of'
            'P1303': {
                'The musical instrument {} plays is the', #'{} performs on the'
                '{} plays',
                '{} plays the',
                'The instrument {} plays is the', #'{} plays the instrument'
                'The musical instrument {} played was the', #'{}, performing on the'
                '{} played the', #'{}, playing the'
                'The instrument {} played was the'}, #'{}, the'
            'P17': {
                "{}'s location is the country of",#'{} is located in'
                '{} is located in the country of',
                '{} is in the country of', #'{}, in'
                '{} is in the nation of', #'{}, located in'
                '{} is located in the nation of'}, #'{}, which is located in'
            'P641': {
                '{} plays professional',  #'What sport does {} play? They play'
                '{} professionally plays',  #'{} is a professional'
                '{} plays',
                '{} professionally plays the sport of', #'{} professionally plays the sport'
                '{} plays the sport of'}, #'{}, the'
            'P176': {
                '{} is a product of',
                '{} is created by',
                '{} is developed by',
                '{} is produced by',
                '{} is made by', #'{}, created by'
                'The developer of {} is',#'{}, developed by'
                'The maker of {} is'},#'{}, produced by'
            'P413': {
                'The position of {} is',#'Which position does {} play? They play as'
                '{} plays as',
                '{} plays in the position of',
                '{}\'s position is',#'{}, the'
                'The position of {} on the field is'}, #'{}, who plays the position'
            'P39': {
                '{} has the position of',
                '{} holds the position of',
                '{} holds the title of',
                '{} has the title of', # '{} is a'
                "{}'s position is",
                "{}'s title is", #'{}, who has the position of'
                'The position of {} is', #'{}, who holds the position of'
                'The title of {} is'}, #'{}, whose position is that of'
            'P159': {
                'The headquarter of {} is in the city of',#'The headquarter of {} is in'
                'The headquarter of {} is located in city of',#'The headquarter of {} is located in'
                'The headquarters of {} is in the city of', #'The headquarters of {} is in'
                '{} is based in the city of', #'{} is based in'
                '{} is headquartered in the city of', #'{} is headquartered in'
                "{}'s headquarters are in the city of", #"{}'s headquarters are in"
                'The city where the headquarter of {} is located is'}, #'{}, whose headquarters are in'
            'P106': {
                'The occupation of {} is',
                'The profession of {} is',
                '{} works as a', #'{} works as'
                "{}'s occupation is",
                "{}'s profession is", #{}'s profession is a
                "{}'s job is", #{}'s profession is an
                "The job of {} is"},  #'{}, who works as'
            'P30': {
                '{} belongs to the continent of',
                '{} is a part of the continent of',
                'The location of {} is the continent of', #'{} is in'
                "{}'s continent is", #'{} is located in'
                '{} is located in the continent of',  # {} is located in the continent
                "{} is in the continent of"},  # '{}, in'
            'P27': {
                '{} has a citizenship from',
                '{} holds a citizenship from',
                '{} is a citizen of',
                "{}'s citizenship is from",  # '{}, a citizen of'
                "{} is currently a citizen of",  # '{}, who has a citizenship from'
                "{} currently has a citizenship from",  # '{}, who holds a citizenship from'
                "{} holds a citizenship from"},  # '{}, who is a citizen of'
            'P138': {
                '{} is called after',
                '{} is named after',
                '{} is named for',
                '{} was called after',
                '{} was named after',
                '{} was named for',
                "{}'s namesake is",#'{}, called after'
                "The namesake of {} is", #'{}, named after'
                "{}'s namesake was",#'{}, named for'
                "The namesake of {} was",#'{}, which is called after'
                "{} is called after its namesake,",#'{}, which is named after'
                "{} is named after its namesake,", #'{}, which is named for'
                "{} was called after its namesake,", #'{}, which was called after'
                "{} was named after its namesake,", #'{}, which was named after'
                "{} is the eponym of"}, #'{}, which was named for'
            'P108': {
                '{} is employed by',
                '{} works for',
                "The employer of {} is", #'{}, of'
                "{}'s employer is", #'{}, who is employed by'
                "The company which {} works for is"}, #'{}, who works for'
            'P36': {
                'The capital city of {} is',
                'The capital of {} is',
                "{}'s capital city is",
                "{}'s current capital city is", #"{}'s capital city,"
                "{}'s capital is",
                "The current capitcal city of {} is", #"{}'s capital,"
                "Currently, the capital of {} is", #'{}, which has the capital'
                "Currently, the capital city of {} is"}, #'{}, which has the capital city'
            'P264': {
                'The music label representing {} is',
                'The music label that is representing {} is',
                '{} is represented by',
                '{} is represented by music label',
                '{} is represented by record label',
                "The record label representing {} is", #'{} label :'
                '{} recorded for',
                "{}'s label is",
                "{}'s music label is",
                "{}'s record company is",
                "{}'s record label is",
                "{} is represented by a record label named", #'{}, released by'
                "{} is represented by a music label named", #'{}, that is represented by'
                "{} is currently represented by"}, #'{}, which is represented by'
}
"""