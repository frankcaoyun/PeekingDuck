"""
YOLOv6 model node
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np

from peekingduck.nodes.abstract_node import AbstractNode
from peekingduck.nodes.model.yolov6v1 import yolov6_model
# the renaming of the model folder is important!
# otherwise python will import from the python file instead of the folder

class Node(AbstractNode):
    """Initializes and uses YOLOV6 to infer from an image frame.

    Args:
        config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.logger.info(f"model loaded with configs: config")
        self.model =  yolov6_model.YOLOV6Model(self.config)# take in the model.

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """This node does ___.

        Args:
            inputs (dict): Dictionary with keys "__", "__".

        Returns:
            outputs (dict): Dictionary with keys "__".
        """
        # print('config: \n\n\n', self.config)
        # # the input for this node is an image numpy array

        bboxes, labels, scores = self.model.predict(inputs["img"])
        bboxes = np.clip(bboxes, 0, 1)

        outputs = {"bboxes": bboxes, "bbox_labels": labels, "bbox_scores": scores}

        return outputs