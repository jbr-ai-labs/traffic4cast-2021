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
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

import random

from traffic4cast.competition.competition_constants import MAX_TEST_SLOT_INDEX
from traffic4cast.competition.prepare_test_data.prepare_test_data import prepare_test
from traffic4cast.util.h5_util import load_h5_file


class T4CDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        file_filter: str = None,
        limit: Optional[int] = None,
        folds_to_use: Optional[Tuple[int]] = None,
        n_splits: Optional[int] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        use_npy: bool = False,
    ):
        """torch dataset from training data.

        Parameters
        ----------
        root_dir
            data root folder, by convention should be `data/raw`, see `data/README.md`. All `**/training/*8ch.h5` will be added to the dataset.
        file_filter: str
            filter files under `root_dir`, defaults to `"**/training/*ch8.h5`
        limit
            truncate dataset size
        transform
            transform applied to both the input and label
        """
        self.root_dir = root_dir
        self.limit = limit
        self.files = []
        self.file_filter = file_filter
        self.folds_to_use = folds_to_use
        self.n_splits = n_splits
        self.use_npy = use_npy
        if self.file_filter is None:
            self.file_filter = "**/training/*8ch.h5"
            if self.use_npy:
                self.file_filter = "**/training_npy/*.npy"
        self.transform = transform
        self._load_dataset()

    def _load_dataset(self):
        files = sorted(list(Path(self.root_dir).rglob(self.file_filter)))
        # random shuffling based on before-hand seed
        random.Random(123).shuffle(files)

        if self.folds_to_use is not None and self.n_splits is not None:
            # K-Fold Selection
            fold_sizes = np.full(self.n_splits, len(files) // self.n_splits, dtype=int)
            fold_sizes[:len(files) % self.n_splits] += 1

            current = 0
            for fold_idx, fold_size in enumerate(fold_sizes):
                start, stop = current, current + fold_size
                current = stop

                if fold_idx in self.folds_to_use:
                    self.files.extend(files[start:stop])

            return

        self.files = files

    def _load_h5_file(self, fn, sl: Optional[slice]):
        if self.use_npy:
            return np.load(fn)
        else:
            return load_h5_file(fn, sl=sl)

    def __len__(self):
        size_240_slots_a_day = len(self.files) * MAX_TEST_SLOT_INDEX
        if self.limit is not None:
            return min(size_240_slots_a_day, self.limit)
        return size_240_slots_a_day

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        if idx > self.__len__():
            raise IndexError("Index out of bounds")

        file_idx = idx // MAX_TEST_SLOT_INDEX
        start_hour = idx % MAX_TEST_SLOT_INDEX

        two_hours = self._load_h5_file(self.files[file_idx], sl=slice(start_hour, start_hour + 12 * 2 + 1))

        input_data, output_data = prepare_test(two_hours)

        input_data = self._to_torch(input_data)
        output_data = self._to_torch(output_data)

        if self.transform is not None:
            input_data = self.transform(input_data)
            output_data = self.transform(output_data)

        return input_data, output_data

    def _to_torch(self, data):
        data = torch.from_numpy(data)
        data = data.to(dtype=torch.float)
        return data


class ExpandedDataset(Dataset):

    def __init__(
        self,
        initial_dataset: Dataset,
        desired_length: int,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    ):
        self.dataset = initial_dataset
        self.length = desired_length
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        if idx >= len(self.dataset):
            idx = idx % len(self.dataset)

        item = self.dataset[idx][0]

        if self.transform is not None:
            item = self.transform(item)

        return item


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)
