#  Copyright 2021 Institute of Advanced Research in Artificial Intelligence (IARAI) GmbH.
#  IARAI licenses this file to You under the Apache License, Version 2.0
#  (the "License"); you may not use this file except in compliance with
#  the License. You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import argparse
import datetime
import glob
import logging
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Union

import numpy as np
import psutil
import torch
import torch_geometric

from tqdm.auto import trange
from functools import partial

from traffic4cast.util.h5_util import load_h5_file
from traffic4cast.util.h5_util import write_data_to_h5

from traffic4cast.models.baseline_unet import UNet, UNetTransfomer


def package_submission(
    data_raw_path: str,
    competition: str,
    model_str: str,
    model: Union[torch.nn.Module, Dict[str, torch.nn.Module]],
    device: str,
    submission_output_dir: Path,
    batch_size=10,
    num_tests_per_file=100,
    h5_compression_params: dict = None,
    city_masks_path: Optional[dict] = None,
    **additional_transform_args,
) -> Path:
    tstamp = datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d%H%M")

    if h5_compression_params is None:
        h5_compression_params = {}

    if submission_output_dir is None:
        submission_output_dir = Path(".")
    submission_output_dir.mkdir(exist_ok=True, parents=True)
    submission = submission_output_dir / f"submission_{model_str}_{competition}_{tstamp}.zip"
    print(submission)

    competition_files = glob.glob(f"{data_raw_path}/**/*test_{competition}.h5",
                                  recursive=True)

    assert len(competition_files) > 0

    model_dict = None
    if isinstance(model, dict):
        model_dict = model

    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(submission, "w") as z:
            for competition_file in competition_files:
                print(f"  running model on {competition_file}")
                city = re.search(r".*/([A-Z]+)_test_", competition_file).group(
                    1)

                if model_dict is not None:
                    print(f'   loading model for {city}')
                    model = model_dict[city]
                model = model.to(device)
                model.eval()

                pre_transform: Callable[[np.ndarray], Union[
                    torch.Tensor, torch_geometric.data.Data]] = \
                    partial(UNetTransfomer.unet_pre_transform,
                            stack_channels_on_time=True, zeropad2d=(6, 6, 1, 0),
                            batch_dim=True, from_numpy=True)

                post_transform: Callable[[Union[
                                              torch.Tensor, torch_geometric.data.Data]], np.ndarray] = \
                    partial(UNetTransfomer.unet_post_transform,
                            stack_channels_on_time=True, crop=(6, 6, 1, 0),
                            batch_dim=True)

                if city_masks_path is not None:
                    mask_path = city_masks_path[city]
                    city_mask = load_h5_file(mask_path)

                    mask_torch = torch.from_numpy(city_mask)

                    if mask_torch.shape[0] > mask_torch.shape[2]:
                        mask_torch = torch.moveaxis(mask_torch, 2, 0)

                    mask_torch_unsqueezed = torch.unsqueeze(mask_torch, 0)

                    zeropad2d = (6, 6, 1, 0)  # TODO: move to config
                    if zeropad2d is not None:
                        padding = torch.nn.ZeroPad2d(zeropad2d)
                        mask_torch_unsqueezed = padding(mask_torch_unsqueezed)

                    summed_mask = torch.sum(mask_torch_unsqueezed[0], dim=0)
                    mask_2d = torch.where(summed_mask > 0, 1, 0)

                    print(f"    loaded city mask with shape: {mask_2d.shape}")

                assert num_tests_per_file % batch_size == 0, f"num_tests_per_file={num_tests_per_file} must be a multiple of batch_size={batch_size}"

                num_batches = num_tests_per_file // batch_size
                prediction = np.zeros(
                    shape=(num_tests_per_file, 6, 495, 436, 8), dtype=np.uint8)

                with torch.no_grad():
                    for i in trange(num_batches):
                        batch_start = i * batch_size
                        batch_end = batch_start + batch_size
                        test_data: np.ndarray = load_h5_file(competition_file,
                                                             sl=slice(
                                                                 batch_start,
                                                                 batch_end),
                                                             to_torch=False)
                        additional_data = load_h5_file(
                            competition_file.replace("test", "test_additional"),
                            sl=slice(batch_start, batch_end), to_torch=False)

                        if pre_transform is not None:
                            test_data: Union[
                                torch.Tensor, torch_geometric.data.Data] = pre_transform(
                                test_data, city=city,
                                **additional_transform_args)
                        else:
                            test_data = torch.from_numpy(test_data)
                            test_data = test_data.to(dtype=torch.float)
                        test_data = test_data.to(device)

                        additional_data = torch.from_numpy(additional_data)
                        additional_data = additional_data.to(device)

                        batch_prediction = model(test_data, city=city,
                                                 additional_data=additional_data)

                        if mask_2d is not None:
                            # print(torch.count_nonzero(batch_prediction))
                            batch_prediction = batch_prediction * mask_2d
                            # print(torch.count_nonzero(batch_prediction))

                        if post_transform is not None:
                            batch_prediction = post_transform(
                                batch_prediction, city=city, **additional_transform_args)
                        else:
                            batch_prediction = batch_prediction.cpu().detach().numpy()
                        batch_prediction = np.clip(batch_prediction, 0, 255)
                        # clipping is important as assigning float array to uint8
                        # array has not the intended effect.... (see `test_submission.test_assign_reload_floats)

                        prediction[batch_start:batch_end] = batch_prediction

                unique_values = np.unique(prediction)
                print(
                    f"  {len(unique_values)} unique values in prediction in the range [{np.min(prediction)}, {np.max(prediction)}]")

                # plt.hist(prediction.flatten())
                # plt.show()

                temp_h5 = os.path.join(temp_dir,
                                       os.path.basename(competition_file))
                arcname = os.path.join(*competition_file.split(os.sep)[-2:])
                print(f"  writing h5 file {temp_h5}")

                write_data_to_h5(prediction, temp_h5, **h5_compression_params)
                print(f"  adding {temp_h5} as {arcname}")

                z.write(temp_h5, arcname=arcname)

            print(z.namelist())

    submission_mb_size = os.path.getsize(submission) / (1024 * 1024)
    print(f"Submission {submission} with {submission_mb_size:.2f}MB")

    return submission


def create_parser() -> argparse.ArgumentParser:
    """Creates the argument parser for this program."""
    parser = argparse.ArgumentParser(description=("This programs creates a submission."))
    parser.add_argument("--checkpoint", type=str, help="Torch checkpoint file", required=True, default=None)
    parser.add_argument("--model_str", type=str, help="The `model_str` in the config", required=False, default="unet")
    parser.add_argument("--data_raw_path", type=str, help="Path of raw data", required=False, default="./data/raw")
    parser.add_argument("--batch_size", type=int, help="Batch size for evaluation", required=False, default=10)
    parser.add_argument("--device", type=str, help="Device", required=False, default="cpu")
    parser.add_argument(
        "--submission_output_dir", type=str, default=None, required=False, help="If given, submission is stored to this directory instead of current.",
    )
    return parser


def main(model_str: str, checkpoint: str, batch_size: int, device: str, data_raw_path: str, submission_output_dir: Optional[str] = None):
    t4c_apply_basic_logging_config()
    model_class = configs[model_str]["model_class"]
    model_config = configs[model_str].get("model_config", {})
    model = model_class(**model_config)
    load_torch_model_from_checkpoint(checkpoint=checkpoint, model=model)
    competitions = ["temporal", "spatiotemporal"]
    for competition in competitions:
        package_submission(
            data_raw_path=data_raw_path,
            competition=competition,
            model=model,
            model_str=model_str,
            batch_size=batch_size,
            device=device,
            h5_compression_params={"compression_level": 6},
            submission_output_dir=Path(submission_output_dir if submission_output_dir is not None else "."),
        )


if __name__ == "__main__":
    parser = create_parser()
    try:
        params = parser.parse_args()
        main(**vars(params))
    except Exception as e:
        print(f"There was an error during execution, please review: {e}")
        parser.print_help()
        exit(1)
