import json
import typing
from pathlib import Path

import torch
from torch.utils.data import Dataset

from util.globals import *

REMOTE_URL = f"{REMOTE_ROOT_URL}/data/dsets/counterfactfalse.json"


class CounterFactFalseDataset(Dataset):
    def __init__(
        self, data_dir: str, size: typing.Optional[int] = None, *args, **kwargs
    ):
        data_dir = Path(data_dir)
        cff_loc = data_dir / "counterfactfalse.json"
        if not cff_loc.exists():
            print(f"{cff_loc} does not exist. Downloading from {REMOTE_URL}")
            data_dir.mkdir(exist_ok=True, parents=True)
            torch.hub.download_url_to_file(REMOTE_URL, cff_loc)

        with open(cff_loc, "r") as f:
            self.data = json.load(f)
        if size is not None:
            self.data = self.data[:size]

        print(f"Loaded dataset with {len(self)} elements")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
