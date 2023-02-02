# Copyright 2023 AI Singapore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
from omegaconf import DictConfig

import albumentations as A
import cv2
import pandas as pd

from .. import config

if config.TORCH_AVAILABLE:
    import torch
    import torchvision.transforms as T
    from torch.utils.data import Dataset
else:
    raise ImportError("Called a torch-specific function but torch is not installed.")

if config.TF_AVAILABLE:
    import tensorflow as tf
else:
    raise ImportError(
        "Called a tensorflow-specific function but tensorflow is not installed."
    )

TransformTypes = Optional[Union[A.Compose, T.Compose]]


class ImageClassificationDataset(Dataset):
    """Template for Image Classification Dataset."""

    def __init__(
        self,
        cfg: DictConfig,
        df: Optional[pd.DataFrame] = None,
        stage: str = "train",
        transforms: TransformTypes = None,
        **kwargs,
    ) -> None:
        """"""

        super().__init__(**kwargs)
        self.cfg: DictConfig = cfg
        self.df = df
        self.stage = stage
        self.transforms = transforms

        self.image_path = df[cfg.dataset.image_path_col_name].values
        self.targets = (
            df[cfg.dataset.target_col_name].values if stage != "test" else None
        )

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.df)

    def __getitem__(
        self, index: int
    ) -> Union[torch.FloatTensor, Union[torch.FloatTensor, torch.LongTensor]]:
        """ """
        image_path = self.image_path[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.apply_image_transforms(image)

        # Get target for all modes except for test dataset.
        # If test, replace target with dummy ones as placeholder.
        target = self.targets[index] if self.stage != "test" else torch.ones(1)
        # target = self.apply_target_transforms(target)

        # TODO: consider stage to be private since it is only used internally.
        if self.stage in ["train", "valid", "debug"]:
            return image, target
        elif self.stage == "test":
            return image
        else:
            raise ValueError(f"Invalid stage {self.stage}.")

    def apply_image_transforms(
        self, image: torch.Tensor, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Apply transforms to the image."""
        if self.transforms and isinstance(self.transforms, A.Compose):
            image = self.transforms(image=image)["image"]
        elif self.transforms and isinstance(self.transforms, T.Compose):
            image = self.transforms(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1)  # convert HWC to CHW
        return torch.tensor(image, dtype=dtype)

    # pylint: disable=no-self-use # not yet!
    def apply_target_transforms(
        self, target: torch.Tensor, dtype: torch.dtype = torch.long
    ) -> torch.Tensor:
        """Apply transforms to the target.
        Note:
            This is useful for tasks such as segmentation object detection where
            targets are in the form of bounding boxes, segmentation masks etc.
        """
        return torch.tensor(target, dtype=dtype)


class Generator(object):
    def __init__(self, batch_size, name_x, name_y):

        data_f = None  # h5py.File(open_directory, "r")

        self.x = data_f[name_x]
        self.y = data_f[name_y]

        if len(self.x.shape) == 4:
            self.shape_x = (None, self.x.shape[1], self.x.shape[2], self.x.shape[3])

        if len(self.x.shape) == 3:
            self.shape_x = (None, self.x.shape[1], self.x.shape[2])

        if len(self.y.shape) == 4:
            self.shape_y = (None, self.y.shape[1], self.y.shape[2], self.y.shape[3])

        if len(self.y.shape) == 3:
            self.shape_y = (None, self.y.shape[1], self.y.shape[2])

        self.num_samples = self.x.shape[0]
        self.batch_size = batch_size
        self.epoch_size = self.num_samples // self.batch_size + 1 * (
            self.num_samples % self.batch_size != 0
        )

        self.pointer = 0
        self.sample_nums = np.arange(0, self.num_samples)
        np.random.shuffle(self.sample_nums)

    def data_generator(self):

        for batch_num in range(self.epoch_size):

            x = []
            y = []

            for elem_num in range(self.batch_size):

                sample_num = self.sample_nums[self.pointer]

                x += [self.x[sample_num]]
                y += [self.y[sample_num]]

                self.pointer += 1

                if self.pointer == self.num_samples:
                    self.pointer = 0
                    np.random.shuffle(self.sample_nums)
                    break

            x = np.array(x, dtype=np.float32)
            y = np.array(y, dtype=np.float32)

            yield x, y

    def get_dataset(self):
        dataset = tf.data.Dataset.from_generator(
            self.data_generator,
            output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.RaggedTensorSpec(shape=(2, None), dtype=tf.int32),
            ),
        )
        dataset = dataset.prefetch(1)

        return dataset
