import datetime
import torch
import glob

from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

from tqdm.auto import tqdm, trange

from argparse import ArgumentParser, Namespace
from functools import partial

from traffic4cast.util.lightning_util import load_state_dict_from_lightning_checkpoint_
from traffic4cast.competition.competition_constants import CORE_CITES
from traffic4cast.models.baseline_unet import UNet, UNetTransfomer
from traffic4cast.util.h5_util import load_h5_file

from pytorch_lightning import seed_everything


SEED = 111
seed_everything(SEED)


def get_net_and_transform(
    network_name: str,
    checkpoint_path: str,
    *additional_args
) -> Tuple[torch.nn.Module, Callable]:
    if "vanilla_unet" == network_name:
        model = UNet(
            in_channels=12 * 8,
            n_classes=6 * 8,
            depth=5,
            wf=6,
            padding=True,
            up_mode="upconv",
            batch_norm=True
        )
    else:
        raise NotImplementedError

    load_state_dict_from_lightning_checkpoint_(model, checkpoint_path)

    pre_transform: Callable[[torch.Tensor], torch.Tensor] = \
        partial(UNetTransfomer.unet_pre_transform,
                stack_channels_on_time=True, zeropad2d=(6, 6, 1, 0),
                batch_dim=True, from_numpy=False)

    return model, pre_transform


def run_pseudolabeling(
    net: torch.nn.Module,
    pre_transform: Callable,
    test_data: torch.Tensor,
    num_rounds: int,
    max_epochs: int,
    batch_size: int,
    num_workers: int,
    learning_rate: float,
):
    for round_idx in range(num_rounds):
        # label test data
        print(" Pseudo labeling... ")

        for epoch_idx in trange(max_epochs):
            print(
                f" Training on re-labeled data, epoch {epoch_idx + 1} of {max_epochs}: ")
            # train on the test data


def main(params: Namespace):
    now = datetime.datetime.now().strftime("%m%d_%H:%M")
    experiment_name = f"{params.city}_{now}_{params.net}_PL_{params.num_rounds}X{params.max_epochs}"

    competition_file = glob.glob(
        f"{params.dataset_path}/{params.city}/*test_{params.competition}.h5",
        recursive=True)[0]

    print(f' Running {params.net} model on {competition_file}')

    test_data = load_h5_file(competition_file, to_torch=True)

    print(f' Loading {params.net} for {params.city}')
    model, pre_transform = get_net_and_transform(
        params.net, params.checkpoint_path)
    model.to(params.device)

    try:
        run_pseudolabeling(
            model, pre_transform, test_data,
            params.num_rounds, params.max_epochs,
            params.batch_size, params.num_workers, params.learning_rate)

    except KeyboardInterrupt:
        print(" Keyboard Interrupt triggered, saving model's weights... ")
        torch.save(model.state_dict(), experiment_name + ".pth")
        exit(0)

    torch.save(model.state_dict(), experiment_name + ".pth")


if __name__ == "__main__":
    # TODO: move configuration to *.yaml with Hydra
    parser = ArgumentParser(add_help=False)

    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--competition", type=str, default='temporal')
    parser.add_argument("--city", choices=CORE_CITES, type=str, required=True)

    parser.add_argument("--net", default="vanilla_unet", type=str, choices=[
        "vanilla_unet", "unet2020", "fitvid", "unet+rnn", "transformer", "naive_repeat_last"
    ], )
    parser.add_argument("--checkpoint_path", type=str, required=True)

    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--device", type=str, choices=['cuda', 'cpu'], default='cpu')

    parser.add_argument("--optimizer", default="adam", type=str)

    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_workers", default=8, type=int)

    args = parser.parse_args()
    main(args)
