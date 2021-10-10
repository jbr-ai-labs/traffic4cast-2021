import typing as t
import numpy as np
import torch 
import os

from functools import partial
from tqdm.auto import tqdm

from traffic4cast.data.dataset import T4CDataset
from traffic4cast.models.baseline_unet import UNet, UNetTransfomer
from traffic4cast.competition.competition_constants import (
    TRAIN_CITIES,
    CORE_CITES,
    EXTENDED_CITIES
)

from argparse import ArgumentParser, Namespace


def calculate_mean_maps(dataset, limit=None) -> t.Tuple[np.ndarray, np.ndarray]:
    # TODO: could be parallelized, however hit the read bottleneck
    input_mean_traffic_map = np.zeros((495, 436, 8))
    output_mean_traffic_map = np.zeros((495, 436, 8))
    
    input_counter, output_counter = 0, 0
    
    for d_idx in tqdm(range(len(dataset))):
        before, after = dataset[d_idx]
        input_counter += before.shape[0]
        output_counter += after.shape[0]
        
        input_mean_traffic_map += before.sum(0).numpy()
        output_mean_traffic_map += after.sum(0).numpy()     
        
        if limit is not None:
            if d_idx + 1 >= limit:
                break
        
    input_mean_traffic_map /= input_counter
    output_mean_traffic_map /= output_counter
        
    print(input_counter, len(dataset) * 6)
    print(output_counter, len(dataset) * 12)
    
    return input_mean_traffic_map, output_mean_traffic_map


def generate_mean_traffic_maps(
    dataset_path: str,
    city: str, year: str, 
    output_folder: str,
    limit_num_samples: t.Optional[int] = None,
):
    dataset = T4CDataset(
        root_dir=dataset_path,
        file_filter=f"{city}/training/{year}*8ch.h5",
        transform=partial(UNetTransfomer.unet_pre_transform,
                          stack_channels_on_time=False, batch_dim=False),
    )
    
    mean_traffic_maps = calculate_mean_maps(dataset, limit_num_samples)
    
    input_maps_path = os.path.join(output_folder, f"{city}_{year}_mean_input_traffic_map.npy")
    output_maps_path = os.path.join(output_folder, f"{city}_{year}_mean_output_traffic_map.npy")
    
    print(f' Writing mean input traffic map of shape {mean_traffic_maps[0].shape} to {input_maps_path}')  
    unique_values = np.unique(mean_traffic_maps[0])
    print(f"  {len(unique_values)} unique values in prediction in the range [{np.min(mean_traffic_maps[0])}, {np.max(mean_traffic_maps[0])}]")
    
    with open(input_maps_path, 'wb') as f:
        np.save(f, mean_traffic_maps[0])
        
    print(f' Writing mean output traffic map of shape {mean_traffic_maps[1].shape} to {output_maps_path}')        
    unique_values = np.unique(mean_traffic_maps[1])
    print(f"  {len(unique_values)} unique values in prediction in the range [{np.min(mean_traffic_maps[1])}, {np.max(mean_traffic_maps[1])}]")
    
    with open(output_maps_path, 'wb') as f:
        np.save(f, mean_traffic_maps[1])
    

if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--city",
                        choices=TRAIN_CITIES + CORE_CITES,
                        type=str, required=True)
    parser.add_argument("--year", type=int, 
                        choices=[2019, 2020], default=2019)
    parser.add_argument("--limit_num_samples", type=int, default=None)
    parser.add_argument("--output_folder", type=str, required=True)
    
    args = parser.parse_args()
    
    generate_mean_traffic_maps(
        args.dataset_path, args.city, 
        str(args.year), args.output_folder, 
        args.limit_num_samples)