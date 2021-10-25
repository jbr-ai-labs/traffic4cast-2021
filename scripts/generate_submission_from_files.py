import typing as t
import numpy as np
import argparse
import datetime
import tempfile
import zipfile
import torch
import glob
import re
import os

from pathlib import Path

from traffic4cast.competition.competition_constants import CORE_CITES
from traffic4cast.util.h5_util import load_h5_file, write_data_to_h5


def package_ensembling_submission_from_prediction_file(
    data_raw_path: str,
    model_str: str,
    aggregate_fn: str,
    prediction_file_dict: t.Dict,
    submission_output_dir: Path,
    h5_compression_params: t.Dict = None,
    city_masks_path: t.Optional[t.Dict] = None,
):
    competition = 'temporal'
    tstamp = datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d%H%M")

    competition_files = glob.glob(
        f"{data_raw_path}/**/*test_{competition}.h5", recursive=True)

    if h5_compression_params is None:
        h5_compression_params = {}

    if submission_output_dir is None:
        submission_output_dir = Path(".")
    submission_output_dir.mkdir(exist_ok=True, parents=True)
    submission = submission_output_dir / f"submission_{model_str}_{competition}_{tstamp}.zip"
    print(submission)

    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(submission, "w") as z:
            for competition_file in competition_files:
                print(f"  loading predictions on {competition_file}")
                city = re.search(r".*/([A-Z]+)_test_", competition_file).group(1)

                city_mask = None
                if city_masks_path is not None:
                    mask_path = city_masks_path[city]
                    city_mask = load_h5_file(mask_path)

                    mask_torch = torch.from_numpy(city_mask)

                    if mask_torch.shape[0] > mask_torch.shape[2]:
                        mask_torch = torch.moveaxis(mask_torch, 2, 0)

                    mask_torch_unsqueezed = torch.unsqueeze(mask_torch, 0)
                    summed_mask = torch.sum(mask_torch_unsqueezed[0], dim=0)
                    mask_2d = torch.where(summed_mask > 0, 1, 0)

                    print(mask_2d.shape)

                ensemble_prediction = []

                for file in prediction_file_dict[city]:
                    print(file)

                    with open(file, 'rb') as f:
                        prediction = np.load(f)
                        ensemble_prediction.append(prediction)

                    print(prediction.dtype, prediction.shape)

                ensemble_prediction = np.stack(ensemble_prediction)

                if aggregate_fn == 'mean':
                    prediction = ensemble_prediction.mean(axis=0).astype(np.uint8)
                elif aggregate_fn == 'median':
                    prediction = np.median(ensemble_prediction, axis=0).astype(np.uint8)
                else:
                    raise ValueError()

                print(prediction.dtype, prediction.shape)

                if city_mask is not None:
                    prediction = torch.from_numpy(prediction)

                    prediction *= mask_2d.view(1, 1, 495, 436, 1)

                    prediction = prediction.numpy().astype(np.uint8)

                print(prediction.dtype, prediction.shape)

                unique_values = np.unique(prediction)
                print(f"  {len(unique_values)} unique values in prediction in the range [{np.min(prediction)}, {np.max(prediction)}]")

                temp_h5 = os.path.join(temp_dir, os.path.basename(competition_file))
                arcname = os.path.join(*competition_file.split(os.sep)[-2:])
                print(f"  writing h5 file {temp_h5}")

                write_data_to_h5(prediction, temp_h5, **h5_compression_params)
                print(f"  adding {temp_h5} as {arcname}")

                z.write(temp_h5, arcname=arcname)

            print(z.namelist())

    submission_mb_size = os.path.getsize(submission) / (1024 * 1024)
    print(f"Submission {submission} with {submission_mb_size:.2f}MB")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--dataset_path", type=str, required=True)
    args = parser.parse_args()

    # TODO:
    generated_combined_with_test_mask = {
        city: f'maps/{city}_combined_with_test_static.h5'
        for city in CORE_CITES
    }

    city_to_predictions = {
        "BERLIN": [
            "predictions/BERLIN_densenet_unet_none.npy",
            "predictions/BERLIN_effnetb5_unet_none.npy",
            "predictions/BERLIN_unet_none.npy",
            "predictions/BERLIN_densenet_unet_mean_by_channel_and_pixel.npy",
            "predictions/BERLIN_effnetb5_unet_mean_by_channel_and_pixel.npy",
            "predictions/BERLIN_unet_mean_by_channel_and_pixel.npy",
        ],
        "CHICAGO": [
            "predictions/CHICAGO_densenet_unet_none.npy",
            "predictions/CHICAGO_effnetb5_unet_none.npy",
            "predictions/CHICAGO_unet_none.npy",
            "predictions/CHICAGO_densenet_unet_mean_by_channel_and_pixel.npy",
            "predictions/CHICAGO_effnetb5_unet_mean_by_channel_and_pixel.npy",
            "predictions/CHICAGO_unet_mean_by_channel_and_pixel.npy",
        ],
        "MELBOURNE": [
            "predictions/MELBOURNE_densenet_unet_none.npy",
            "predictions/MELBOURNE_effnetb5_unet_none.npy",
            "predictions/MELBOURNE_unet_none.npy",
            "predictions/MELBOURNE_densenet_unet_mean_by_channel_and_pixel.npy",
            "predictions/MELBOURNE_effnetb5_unet_mean_by_channel_and_pixel.npy",
            "predictions/MELBOURNE_unet_mean_by_channel_and_pixel.npy",
        ],
        "ISTANBUL": [
            "predictions/ISTANBUL_unet_none.npy",
            "predictions/ISTANBUL_effnetb5_unet_none.npy",
            "predictions/ISTANBUL_unet_none.npy",
            "predictions/ISTANBUL_unet_mean_by_channel_and_pixel.npy",
            "predictions/ISTANBUL_effnetb5_unet_mean_by_channel_and_pixel.npy",
            "predictions/ISTANBUL_effnetb5_unet_mean_by_channel_and_pixel.npy",
        ],

    }

    package_ensembling_submission_from_prediction_file(
        args.dataset_path,
        "all_unets_da_none_mpcpm1_mean", 'mean',
        city_to_predictions,
        submission_output_dir=Path('submission/'),
        h5_compression_params={"compression_level": 6},
        city_masks_path=generated_combined_with_test_mask,
    )
