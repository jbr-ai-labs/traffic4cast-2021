import datetime
import torch
import torch.nn.functional as F
import time
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
    device: str,
    num_rounds: int,
    max_epochs: int,
    batch_size: int,
    num_workers: int,
    learning_rate: float,
):
    net.to(device)

    for round_idx in range(num_rounds):
        # label test data
        print(f" Pseudo labeling, round {round_idx + 1} of {num_rounds}... ")

        net.eval()

        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(test_data),
            batch_size=batch_size, shuffle=False, num_workers=num_workers)

        pseudo_labels = list()

        with torch.no_grad():
            for test_batch in tqdm(test_loader):
                test_data_batch = test_batch[0]
                test_data_batch = pre_transform(test_data_batch)
                test_data_batch = test_data_batch.to(device)

                batch_prediction = net(test_data_batch)

                pseudo_labels.append(
                    batch_prediction.cpu().detach().clamp(0, 255))

        labeled_test_data = torch.cat(pseudo_labels, dim=0)

        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(test_data, labeled_test_data),
            batch_size=batch_size, num_workers=num_workers, shuffle=True,
        )

        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

        start_time = time.time()
        print(
            f" Training on re-labeled data, {max_epochs} epochs : ")
        net.train()

        for epoch_idx in range(max_epochs):
            loss_values = list()

            for batch_idx, (data, target) in enumerate(loader):
                data, target = pre_transform(data).to(device), target.to(device)
                optimizer.zero_grad()
                output = net(data)

                loss = F.mse_loss(output, target)
                loss.backward()

                loss_values.append(loss.detach().cpu().item())
                optimizer.step()

                print("[Epoch %d  Batch %d]  batch_loss %.10f  average_loss %.10f  elapsed %.2fs" % (
                    epoch_idx, batch_idx, loss_values[-1], sum(loss_values) / len(loss_values), time.time() - start_time
                ))


def main(params: Namespace):
    now = datetime.datetime.now().strftime("%m%d_%H:%M")
    experiment_name = f"{params.city}_{now}_{params.net}_PL_{params.num_rounds}X{params.max_epochs}"

    competition_file = glob.glob(
        f"{params.dataset_path}/{params.city}/*test_{params.competition}.h5",
        recursive=True)[0]

    print(f' Running {params.net} model on {competition_file}')

    test_data = load_h5_file(competition_file, to_torch=True)

    print(f' Loading {params.checkpoint_path} for {params.city}')
    model, pre_transform = get_net_and_transform(
        params.net, params.checkpoint_path)

    try:
        run_pseudolabeling(
            model, pre_transform, test_data,
            params.device, params.num_rounds, params.max_epochs,
            params.batch_size, params.num_workers, params.learning_rate)

    except KeyboardInterrupt:
        print(" Keyboard Interrupt triggered, saving model's weights... ")
        torch.save(model.state_dict(), 'weights/' + experiment_name + ".pth")
        exit(0)

    torch.save(model.state_dict(), 'weights/' + experiment_name + ".pth")


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
