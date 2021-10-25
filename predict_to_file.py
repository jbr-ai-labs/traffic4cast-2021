import os
import glob
import torch
import numpy as np
from argparse import ArgumentParser, Namespace
from tqdm.auto import tqdm, trange
from functools import partial

from traffic4cast.models.pretrained_unet import PretrainedEncoderUNet
from traffic4cast.models.baseline_unet import UNet, UNetTransfomer
from traffic4cast.competition.competition_constants import (
    CORE_CITES,
)
from traffic4cast.util.lightning_util import load_state_dict_from_lightning_checkpoint_
from traffic4cast.util.h5_util import load_h5_file


def load_map_from_file(filename: str, mapname: str) -> np.ndarray:
    with open(filename, 'rb') as f:
        a = np.load(f)
    print(f' Loaded {mapname} of shape {a.shape} from {filename}')
    return a


def load_maps(city, year):
    # i_path = f'maps/{city}_{year}_mean_input_traffic_map.npy'
    # o_path = f'maps/{city}_{year}_mean_output_traffic_map.npy'
    path = f'maps/{city}_{year}_mean_traffic_map.npy'

    # i = load_map_from_file(i_path, 'input')
    # o = load_map_from_file(o_path, 'output')
    m = load_map_from_file(path, 'mean')

    # return (i, o, m)
    return (None, None, m)


def get_core_cities_maps(dataset_path: str, city: str):
    competition_file = f'{dataset_path}/{city}/{city}_test_temporal.h5'
    batch_start, batch_end = 0, 100
    test_data: np.ndarray = load_h5_file(competition_file,
                                         sl=slice(batch_start, batch_end),
                                         to_torch=False)

    mean_traffic_2020_map = test_data.reshape(-1, 495, 436, 8).mean(0)
    mlb_2019_maps = load_maps(city, 2019)

    mean_traffic_2019_map = mlb_2019_maps[2]

    return mean_traffic_2019_map, mean_traffic_2020_map


def get_domain_multipliers(dataset_path: str, city: str):
    mean_map_2019, mean_map_2020 = get_core_cities_maps(dataset_path, city)

    mean_value_per_channel_2019 = mean_map_2019.mean(axis=(0, 1))
    mean_value_per_channel_2020 = mean_map_2020.mean(axis=(0, 1))

    mean_value_per_channel_2019_over_2020 = (
            mean_value_per_channel_2019 / mean_value_per_channel_2020)

    return mean_value_per_channel_2019_over_2020


def get_domain_multipliers_per_pixel(dataset_path: str, city: str):
    mean_map_2019, mean_map_2020 = get_core_cities_maps(dataset_path, city)

    mean_value_per_pixel_2019 = mean_map_2019
    mean_value_per_pixel_2020 = mean_map_2020

    mean_value_per_pixel_2019_over_2020 = np.where(
        mean_value_per_pixel_2020 > 0,
        mean_value_per_pixel_2019 /
        mean_value_per_pixel_2020, 1)

    mean_value_per_pixel_2019_over_2020 = np.where(
        mean_value_per_pixel_2019_over_2020 > 1, mean_value_per_pixel_2019_over_2020, 1)

    print(mean_value_per_pixel_2019_over_2020.shape)

    return mean_value_per_pixel_2019_over_2020


def load_model_and_padding(model, model_path):
    print(f' Load {model} from {model_path}...')
    if model == 'unet':
        model = UNet(
            in_channels=12 * 8,
            n_classes=6 * 8,
            depth=5,
            wf=6,
            padding=True,
            up_mode="upconv",
            batch_norm=True
        )
        padding = (6, 6, 1, 0)
    elif model == 'densenet_unet':
        model = PretrainedEncoderUNet(
            encoder='densenet201',
            in_channels=12 * 8,
            n_classes=6 * 8, depth=5
        )
        padding = (6, 6, 9, 8) # input image 512 x 448
    elif model == 'effnetb5_unet':
        model = PretrainedEncoderUNet(
            encoder="efficientnet-b5",
            in_channels=12 * 8,
            n_classes=6 * 8, depth=6
        )
        padding = (6, 6, 9, 8) # input image 512 x 448
    else:
        raise ValueError
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    except:
        # Load from PyTorch-Lightning checkpoint then
        load_state_dict_from_lightning_checkpoint_(model, model_path)

    return model, padding


def predict_to_file(
    dataset_path: str,
    city: str, model: str, model_path: str,
    output_folder: str, domain_adaptation: str,
    batch_size : int = 1, num_tests_per_file : int = 100, device: str ='cpu',
):
    competition_file = glob.glob(f"{dataset_path}/{city}/{city}_test_temporal.h5",
                                  recursive=True)[0]
    print(f' Running on {competition_file}')

    if domain_adaptation == 'mean_by_channel_and_pixel':
        mean_value_per_pixel_2019_over_2020 = \
            get_domain_multipliers_per_pixel(dataset_path, city)
        print(' Domain Adaptation values per pixel, 2020 -> 2019: ', mean_value_per_pixel_2019_over_2020)

    elif domain_adaptation != 'none':
        mean_value_per_channel_2019_over_2020 = get_domain_multipliers(
            dataset_path, city)
        print(' Domain Adaptation values per channel, 2020 -> 2019: ', *mean_value_per_channel_2019_over_2020)

    prediction_file_name = f'{city}_{model}_{domain_adaptation}.npy'
    raw_prediction_file_path = os.path.join(output_folder, prediction_file_name)

    print(f' Writing raw logits to {raw_prediction_file_path}')

    model, padding = load_model_and_padding(model, model_path)
    model.eval()
    model = model.to(device)

    pre_transform = partial(
        UNetTransfomer.unet_pre_transform,
        stack_channels_on_time=True, zeropad2d=padding,
        batch_dim=True, from_numpy=True)

    print(padding)

    zeropad2d = torch.nn.ZeroPad2d(padding)

    post_transform = partial(
        UNetTransfomer.unet_post_transform,
        stack_channels_on_time=True, crop=padding, batch_dim=True)

    assert num_tests_per_file % batch_size == 0, f"num_tests_per_file={num_tests_per_file} must be a multiple of batch_size={batch_size}"

    num_batches = num_tests_per_file // batch_size
    prediction = np.zeros(shape=(num_tests_per_file, 6, 495, 436, 8), dtype=np.uint8)

    input_multiplier = 1
    output_multiplier = 1

    if domain_adaptation == 'mean_by_channel_and_pixel':

        input_multiplier = np.zeros(
            (mean_value_per_pixel_2019_over_2020.shape[0],
             mean_value_per_pixel_2019_over_2020.shape[1],
             mean_value_per_pixel_2019_over_2020.shape[2] * 12,
             ))

        output_multiplier = np.zeros(
            (mean_value_per_pixel_2019_over_2020.shape[0],
             mean_value_per_pixel_2019_over_2020.shape[1],
             mean_value_per_pixel_2019_over_2020.shape[2] * 6,
             ))

        for i in range(12):
            input_multiplier[:, :, 8 * i:8 * (i + 1)] = mean_value_per_pixel_2019_over_2020

        for i in range(6):
            output_multiplier[:, :, 8 * i:8 * (i + 1)] = np.where(
                mean_value_per_pixel_2019_over_2020 == 0, 1, 1 / mean_value_per_pixel_2019_over_2020)

        input_multiplier = torch.from_numpy(input_multiplier).permute(2, 0, 1).to(device).unsqueeze(0)
        output_multiplier = torch.from_numpy(output_multiplier).permute(2, 0, 1).to(device).unsqueeze(0)

    elif domain_adaptation.startswith('mean_by_channel'):
        input_multiplier = np.tile(mean_value_per_channel_2019_over_2020, 12)
        output_multiplier = np.tile(1 / mean_value_per_channel_2019_over_2020, 6)

        input_multiplier = torch.from_numpy(input_multiplier).to(device)
        output_multiplier = torch.from_numpy(output_multiplier).to(device)

    elif domain_adaptation == 'mean_overall':
        input_multiplier = mean_value_per_channel_2019_over_2020.mean()
        output_multiplier = 1 / mean_value_per_channel_2019_over_2020.mean()

    if domain_adaptation == 'mean_by_channel_output_one':
        output_multiplier = torch.ones_like(output_multiplier)
    elif domain_adaptation == 'mean_by_channel_output_one_speed':
        for i in range(len(mean_value_per_channel_2019_over_2020)):
            if i % 2 == 1:
                # speed channels equal to 1
                mean_value_per_channel_2019_over_2020[i] = 1

        output_multiplier = np.tile(1 / mean_value_per_channel_2019_over_2020, 6)
        output_multiplier = torch.from_numpy(output_multiplier).to(device)

    if domain_adaptation.startswith('mean_by_channel') \
            and domain_adaptation != 'mean_by_channel_and_pixel':
        output_multiplier = output_multiplier.view(1, -1, 1, 1)
        input_multiplier = input_multiplier.view(1, -1, 1, 1)


    if domain_adaptation == 'mean_by_channel_and_pixel':
        input_multiplier = zeropad2d(input_multiplier)
        output_multiplier= zeropad2d(output_multiplier)
        print(output_multiplier.shape)
        print(input_multiplier.shape)
    else:
        print(output_multiplier)
        print(input_multiplier)

    with torch.no_grad():
        for i in trange(num_batches):
            batch_start = i * batch_size
            batch_end = batch_start + batch_size
            test_data: np.ndarray = load_h5_file(competition_file,
                                                 sl=slice(batch_start,
                                                          batch_end),
                                                 to_torch=False)
            if pre_transform is not None:
                test_data = pre_transform(test_data, city=city)
            else:
                test_data = torch.from_numpy(test_data)
                test_data = test_data.to(dtype=torch.float)

            test_data = test_data.to(device)
            test_data *= input_multiplier

            batch_prediction = model(test_data)

            batch_prediction *= output_multiplier

            if post_transform is not None:
                batch_prediction = post_transform(batch_prediction, city=city).detach().cpu().numpy()
            else:
                batch_prediction = batch_prediction.cpu().detach().numpy()

            batch_prediction = np.clip(batch_prediction, 0, 255)
            prediction[batch_start:batch_end] = batch_prediction

    print(
        f" values in prediction in the range [{np.min(prediction)}, {np.max(prediction)}]")

    with open(raw_prediction_file_path, 'wb') as f:
        np.save(f, prediction)


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)

    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--city", choices=CORE_CITES, type=str, required=True)
    parser.add_argument("--model", choices=['unet', 'effnetb5_unet', 'densenet_unet'])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_folder", type=str, default='.')
    parser.add_argument("--domain_adaptation", type=str, default='none',
                        choices=['none', 'mean_by_channel',
                                 'mean_by_channel_and_pixel', 'mean_overall',
                                 'mean_by_channel_output_one', 'mean_by_channel_output_one_speed'])
    parser.add_argument("--device", type=str, choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument("--batch_size", type=int, default=4)

    params = parser.parse_args()

    predict_to_file(params.dataset_path, params.city, params.model,
                    params.model_path, params.output_folder,
                    params.domain_adaptation, params.batch_size,
                    100, params.device)
