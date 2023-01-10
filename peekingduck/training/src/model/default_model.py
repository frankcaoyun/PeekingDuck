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

import functools
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import torch
import torchinfo
from torch import nn
from torchinfo.model_statistics import ModelStatistics
from omegaconf import DictConfig

# from configs.base_params import PipelineConfig


class Model(ABC, nn.Module):
    """Model Base Class."""

    def __init__(self, pipeline_config: DictConfig) -> None:
        super().__init__()
        self.backbone: Optional[nn.Module]
        self.head: Optional[nn.Module]
        self.model: nn.Module
        self.pipeline_config = pipeline_config

    @abstractmethod
    def create_model(self) -> nn.Module:
        """Create the model.
        Note that users can implement anything they want, as long
        as the shape matches.
        """
        raise NotImplementedError("Please implement your own model.")

    @abstractmethod
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""

    def load_backbone(self) -> nn.Module:
        """Load the backbone of the model."""

    def modify_head(self) -> nn.Module:
        """Modify the head of the model."""

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize the weights of the model.

        Note need to be private since it is only used internally.
        """

    def extract_embeddings(self, inputs: torch.Tensor) -> torch.Tensor:
        """Extract the embeddings from the model.

        NOTE:
            Users can use feature embeddings to do metric learning or clustering,
            as well as other tasks.

        Sample Implementation:
            ```python
            def extract_embeddings(self, inputs: torch.Tensor) -> torch.Tensor:
                return self.backbone(inputs)
            ```
        """

    def model_summary(
        self, input_size: Optional[Tuple[int, int, int, int]] = None, **kwargs: Any
    ) -> ModelStatistics:
        """Wrapper for torchinfo package to get the model summary."""
        if input_size is None:
            input_size = (
                1,
                3,
                self.pipeline_config.augmentation.image_size,
                self.pipeline_config.augmentation.image_size,
            )
        return torchinfo.summary(self.model, input_size=input_size, **kwargs)

    def get_last_layer(self) -> Tuple[list, int, nn.Module]:
        """Get the last layer information of a PyTorch Model.

        NOTE:
            This is only correct if the last layer is a linear layer and is the head.
            The easy way for timm is actually to use the `reset_classifier` method to
            remove the head and then add a new head.
        """
        # propagate through the model to get the last layer name
        for name, _param in self.backbone.named_modules():
            last_layer_name = name

        last_layer_attributes = last_layer_name.split(".")  # + ['in_features']
        # reduce applies to a list recursively and reduce it to a single value
        linear_layer = functools.reduce(getattr, last_layer_attributes, self.backbone)
        in_features = linear_layer.in_features
        last_layer_name = ".".join(last_layer_attributes)
        return last_layer_name, linear_layer, in_features