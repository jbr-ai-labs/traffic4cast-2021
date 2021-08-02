# coding: utf-8
__author__ = "sevakon: https://github.com/sevakon"

import torch
import numpy as np
from torch.utils.data import DataLoader

from tqdm.auto import tqdm
from argparse import ArgumentParser, Namespace

from traffic4cast.data.dataset import T4CDataset
from traffic4cast.util.h5_util import write_data_to_h5
from traffic4cast.metrics.masking import get_static_mask
from traffic4cast.competition.competition_constants import (
    TRAIN_CITIES,
    CORE_CITES,
)


def log_mask_stats(mask: np.ndarray, mask_name: str, use_channels: bool = True):
    print(f"{mask_name} static mask stats: ")

    if use_channels:
        for index in range(8):
            print(f"Channel {index}:")
            print(np.unique(mask[:, :, index], return_counts=True))

    else:
        print(np.unique(mask, return_counts=True))


def main(args: Namespace):
    dataset = T4CDataset(
        root_dir=args.dataset_path,
        file_filter=f"{args.city}/training/2019*8ch.h5",
    )

    loader = DataLoader(dataset, batch_size=args.batch_size,
                        num_workers=args.num_workers, shuffle=False)

    max_num_iterations = len(loader) \
        if args.limit_batches is None else args.limit_batches

    _, map_h, map_w, map_ch = dataset[0][0].shape

    static_map = torch.zeros((map_h, map_w, map_ch))

    write_data_to_h5(static_map.numpy(), args.output_path, verbose=True)

    iterations_completed = 0

    for (inp_frames, out_frames) in tqdm(loader, total=max_num_iterations):
        concatenated = torch.cat((inp_frames, out_frames), dim=1).view(
            -1, map_h, map_w, map_ch)

        summed_over_batch = torch.mean(concatenated, dim=0)
        batch_static_map = torch.where(summed_over_batch > 0, 1, 0)

        static_map = torch.where(batch_static_map + static_map > 0, 1, 0)

        iterations_completed += 1

        if iterations_completed > max_num_iterations:
            break

    static_map_uint8 = torch.tensor(static_map.clone().detach(), dtype=torch.uint8).numpy()

    log_mask_stats(static_map_uint8, "Generated", use_channels=True)

    write_data_to_h5(static_map_uint8, args.output_path, verbose=True)

    provided_static_map = get_static_mask(args.city, args.dataset_path)

    log_mask_stats(provided_static_map, "Provided", use_channels=True)

    combined_static_map = np.where(
        provided_static_map + static_map_uint8 > 0, 1, 0)

    log_mask_stats(combined_static_map, "Provided + Generated", use_channels=True)

    write_data_to_h5(combined_static_map, args.output_path_combined, verbose=True)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--city",
                        choices=TRAIN_CITIES + CORE_CITES,
                        type=str, required=True)

    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--limit_batches", default=None, type=int)

    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--output_path_combined", type=str, required=True)

    args = parser.parse_args()
    main(args)
