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


def calculate_std_maps(
    dataset, input_mean_traffic_map, 
    output_mean_traffic_map, mean_traffic_map, limit=None
) -> t.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    input_std_traffic_map = np.zeros((495, 436, 8))
    output_std_traffic_map = np.zeros((495, 436, 8))
    std_traffic_map = np.zeros((495, 436, 8))
    
    input_counter, output_counter = 0, 0
    
    for d_idx in tqdm(range(len(dataset))):
        before, after = dataset[d_idx]
        
        input_counter += before.shape[0]
        for i in range(before.shape[0]):
            before_map = before[i, :, :, :].numpy()
            before_residual = (before_map - input_mean_traffic_map)
            input_std_traffic_map += before_residual ** 2
            
            residual = (before_map - mean_traffic_map)
            std_traffic_map += residual ** 2
        
        output_counter += after.shape[0]
        for i in range(after.shape[0]):
            after_map = after[i, :, :, :].numpy()
            after_residual = (after_map - output_mean_traffic_map)
            output_std_traffic_map += after_residual ** 2  
            
            residual = (after_map - mean_traffic_map)
            std_traffic_map += residual ** 2
               
        if limit is not None:
            if d_idx + 1 >= limit:
                break
        
    input_std_traffic_map = np.sqrt(input_std_traffic_map / (input_counter - 1))
    output_std_traffic_map = np.sqrt(output_std_traffic_map / (output_counter - 1))
    std_traffic_map = np.sqrt(std_traffic_map / (input_counter + output_counter - 1))
        
    print(input_counter, len(dataset) * 6)
    print(output_counter, len(dataset) * 12)
    
    return input_std_traffic_map, output_std_traffic_map, std_traffic_map


def write_map_to_file(
    traffic_map: np.ndarray, filename: str, mapname: str
) -> None:
    print(f' Writing {mapname} of shape {traffic_map.shape} to {filename}')  
    unique_values = np.unique(traffic_map)
    print(f"  {len(unique_values)} unique values in prediction in the range [{np.min(traffic_map)}, {np.max(traffic_map)}]")
    
    with open(filename, 'wb') as f:
        np.save(f, traffic_map)    
        
        
def load_map_from_file(filename: str, mapname: str) -> np.ndarray:
    with open(filename, 'rb') as f:
        a = np.load(f)
    print(f' Loaded {mapname} of shape {a.shape} from {filename}')  
    return a


def generate_std_traffic_maps(
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
    
    input_maps_path = os.path.join(output_folder, f"{city}_{year}_mean_input_traffic_map.npy")
    output_maps_path = os.path.join(output_folder, f"{city}_{year}_mean_output_traffic_map.npy")
    maps_path = os.path.join(output_folder, f"{city}_{year}_mean_traffic_map.npy")
    
    input_mean_traffic_map = load_map_from_file(input_maps_path, 'mean input traffic map')
    output_mean_traffic_map = load_map_from_file(output_maps_path, 'mean output traffic map')
    mean_traffic_map = load_map_from_file(maps_path, 'mean traffic map')
    
    std_traffic_maps = calculate_std_maps(
        dataset, input_mean_traffic_map, 
        output_mean_traffic_map, mean_traffic_map, 
        limit_num_samples)
    
    input_maps_path = os.path.join(output_folder, f"{city}_{year}_std_input_traffic_map.npy")
    output_maps_path = os.path.join(output_folder, f"{city}_{year}_std_output_traffic_map.npy")
    maps_path = os.path.join(output_folder, f"{city}_{year}_std_traffic_map.npy")
    
    write_map_to_file(std_traffic_maps[0], input_maps_path, 'std input traffic map')
    write_map_to_file(std_traffic_maps[1], output_maps_path, 'std output traffic map')
    write_map_to_file(std_traffic_maps[2], maps_path, 'std traffic map')
    

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
    
    generate_std_traffic_maps(
        args.dataset_path, args.city, 
        str(args.year), args.output_folder, 
        args.limit_num_samples)