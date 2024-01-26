#put in results folder

import json
import pandas as pd
import os
import shutil

alg_name = "ROME"
separate_based_on = ["ds_name", "v_weight_decay", "kl_factor"]


for subdir,_, _ in os.walk(alg_name):
    if subdir.startswith(alg_name + "/run_") and str.isdigit(subdir[-1]):
        print(subdir)
        with open(subdir + '/' + 'params.json') as f:
            params = json.load(f)
        with open(subdir + '/' + 'additional_params.json') as f:
            additional_params = json.load(f)
        collect_separators = {}
        title = "parameters"
        for x in separate_based_on:
            if additional_params[x] is not None:
                collect_separators[x] = additional_params[x]
            else:
                collect_separators[x] = params[x]
            title += '_'
            title += x
            title += '_'
            title += str(collect_separators[x])
        if not os.path.exists(title):
            os.mkdir(title)
        shutil.copytree(subdir, title + "/" + subdir)


