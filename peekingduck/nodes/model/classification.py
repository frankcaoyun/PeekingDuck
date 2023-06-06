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

from typing import Any, Dict, List, Optional, Union

import cv2

from peekingduck.nodes.abstract_node import AbstractNode
from peekingduck.nodes.model.classificationv1 import classification_model


class Node(AbstractNode):
    """Initializes an user-defined classification model node to perform classification.

    Inputs:
        |img_data|

    Outputs:
        |pred_label|

        |pred_score|

    Configs:
        To be completed.

    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.model = classification_model.ClassificationModel(self.config)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Takes an image as input and returns the class and score of the prediction."""
        image = cv2.cvtColor(inputs["img"], cv2.COLOR_BGR2RGB)
        pred_label, pred_score = self.model.predict(image)

        outputs = {"pred_label": pred_label, "pred_score": pred_score}
        return outputs

    def _get_config_types(self) -> Dict[str, Any]:
        """Returns dictionary mapping the node's config keys to respective types."""
        return {
            "weights_parent_dir": Optional[str],
            "class_label_map": Dict[int, Any],
        }
        # raise NotImplementedError("Function not implemented.")
