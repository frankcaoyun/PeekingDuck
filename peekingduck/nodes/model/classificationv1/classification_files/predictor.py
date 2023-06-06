# Copyright 2022 AI Singapore
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

import sys
from typing import Any, Dict

import numpy as np
from omegaconf import DictConfig
import torch
import torch.nn as nn
from torchvision import transforms

# sys.path.append("/home/frank/git-repo/aiap/PeekingDuck/peekingduck/training/")
from peekingduck.training.src.model.pytorch_model import PTClassificationModel
from peekingduck.nodes.model.classificationv1.classification_files.constants import (
    IMG_MEAN,
    IMG_STD,
)


class PTPredictor:
    def __init__(self, cfg: DictConfig, weights_dir: str):
        self._create_model(cfg)
        self._load_model_weights(weights_dir)
        self._change_to_inference_mode()
        self._define_img_transformer()

    def _create_model(self, cfg: DictConfig) -> None:
        self.model = PTClassificationModel(cfg)

    def _load_model_weights(self, weights_dir: str) -> None:
        # load the state dict based on the way PeekingDuck training pipeline saves the model
        self.model.load_state_dict(torch.load(weights_dir)["model_state_dict"])
        self.model.to(
            self.model.model_config.device
        )  # model_config is attached when initiating the model

    def _change_to_inference_mode(self) -> None:
        self.model.eval()

    def _define_img_transformer(self) -> None:
        self.img_transformer = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMG_MEAN,  # Mean values for each channel
                    std=IMG_STD,
                ),  # Standard deviation values for each channel
            ]
        )

    def predict_label_and_score_from_image(self, frame: np.ndarray):
        print("frame shape before processing: ", frame.shape)
        frame = self.img_transformer(frame).to(self.model.model_config.device)
        print("frame shape after processing: ", frame.shape)
        print(self.model.model)
        with torch.no_grad():
            logits = self.model.model(frame)
        # print(logits.shape)
        softmax = nn.Softmax(dim=1)
        scores = softmax(logits)
        predicted_probs = scores.squeeze(0)
        pred_index = torch.argmax(predicted_probs).item()
        pred_score = predicted_probs[pred_index].item()
        return pred_index, pred_score


class TFPredictor:
    pass
    # raise NotImplementedError("TFPredictor not implemented!")
