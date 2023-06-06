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

from typing import Any, Dict

import numpy as np
from omegaconf import OmegaConf, DictConfig

from peekingduck.nodes.model.classificationv1.classification_files.predictor import (
    PTPredictor,
    TFPredictor,
)


class ClassificationModel:
    """
    Generic classification model based on config.
    This class should take in the config and instantiate a PyTorch or TensorFlow model instance based on the config.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.framework = cfg["framework"]
        self.dictconfig = self._convert_dict_to_dictconfig(cfg[self.framework])
        self.weights_dir = cfg["weights_dir"]

        if cfg["framework"] == "pytorch":
            self.predictor = PTPredictor(self.dictconfig, self.weights_dir)
        elif cfg["framework"] == "tensorflow":
            self.predictor = TFPredictor(self.dictconfig, self.weights_dir)

    def _convert_dict_to_dictconfig(self, cfg: Dict[str, Any]) -> DictConfig:
        return OmegaConf.create(cfg)

    def predict(self, frame: np.ndarray):
        return self.predictor.predict_label_and_score_from_image(frame)
