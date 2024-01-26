import json

with open('fact.json', 'r') as f:
    openai_dataset = json.load(f)

output = []
case_id = 0
for topic, entries in openai_dataset.items():
    for entry in entries:
        new_entry = {}
        new_entry["case_id"] = case_id
        case_id += 1
        new_entry["attribute_prompts"] = []
        new_entry["generation_prompts"] = []
        new_entry["paraphrase_prompts"] = []
        for key in entry:
            if key.startswith('original_statement_'):
                sentence = entry[key]
                truth_value = entry['truth_value']
                prompt = 'True or false: '+ sentence +'\nAnswer:'
                target_true = str(truth_value)
                target_new = str(not truth_value)
                new_entry["requested_rewrite"] = {"prompt": prompt}
                new_entry["requested_rewrite"]["target_true"] ={"str": target_true}
                new_entry["requested_rewrite"]["target_new"] = {"str": target_new}
                new_entry["requested_rewrite"]["subject"] = ""
            elif key == "rephrases":
                for pair in entry[key]:
                    new_entry["paraphrase_prompts"].append('True or false: '+ pair["statement"] +'\nAnswer:')
        output.append(new_entry)

case_id = 0
for topic, entries in openai_dataset.items():
    for entry in entries:
        neighborhood_prompts = []
        key = "main_terms" if ("main_terms" in entry) else "main terms"
        for term in entry[key]:
            for key in term:
                if key.endswith("_statements"):
                    for statement in term[key]:
                        if statement["truth_value"] == entry["truth_value"]:
                            sentence = statement["statement"]
                            neighborhood_prompts.append('True or false: '+ sentence +'\nAnswer:')
        output[case_id]["neighborhood_prompts"] = neighborhood_prompts
        case_id += 1

with open('fact2counterfactformat.json', 'w') as f:
    json.dump(output, f, indent=4)






