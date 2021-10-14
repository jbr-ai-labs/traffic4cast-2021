import torch
import torch.nn as nn

from collections import OrderedDict


def load_state_dict_from_lightning_checkpoint_(model: nn.Module, path: str,
                                               num_trims: int = 1):
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    lightning_state_dict = checkpoint['state_dict']

    try:
        model.load_state_dict(lightning_state_dict)
    except RuntimeError as e:
        print(e)
        print("    Atttempting loading with keys renaming: ")

        new_state_dict = OrderedDict()

        for key in lightning_state_dict.keys():
            temp_key = key

            for _ in range(num_trims):
                sep_index = temp_key.find(".")
                temp_key = temp_key[sep_index + 1:]

            new_state_dict[temp_key] = lightning_state_dict[key]

        model.load_state_dict(new_state_dict, strict=False)

        print("    Success! ")
