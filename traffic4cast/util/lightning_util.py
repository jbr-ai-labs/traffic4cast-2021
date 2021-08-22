import torch
import torch.nn as nn

from collections import OrderedDict


def load_state_dict_from_lightning_checkpoint_(model: nn.Module, path: str, ):
    print(f"   Loading {path} checkpoint")

    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    lightning_state_dict = checkpoint['state_dict']

    try:
        model.load_state_dict(lightning_state_dict)
    except RuntimeError as e:
        print(e)
        print("    Atttempting loading with keys renaming: ")

        new_state_dict = OrderedDict()

        for key in lightning_state_dict.keys():
            sep_index = key.find(".")
            new_state_dict[key[sep_index + 1:]] = lightning_state_dict[key]

        model.load_state_dict(new_state_dict)

        print("    Success! ")
