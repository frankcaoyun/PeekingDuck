"""
YOLOv6 custom node
"""

from typing import Any, Dict

import numpy as np

from peekingduck.pipeline.nodes.abstract_node import AbstractNode
from custom_nodes.model.yolov6_custom import yolov6_model


class Node(AbstractNode):
    """This is a template class of how to write a node for PeekingDuck.

    Args:
        config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

        # initialize/load any configs and models here
        # configs can be called by self.<config_name> e.g. self.filepath
        self.logger.info(f"model loaded with configs: config")
        
        self.model =  yolov6_model.YOLOV6Model(self.config)# take in the model.
        # the config files need to be further adjusted. probably hard coding first



    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """This node does ___.

        Args:
            inputs (dict): Dictionary with keys "__", "__".

        Returns:
            outputs (dict): Dictionary with keys "__".
        """
        # print('config: \n\n\n', self.config)
        # the input for this node is an image numpy array

        bboxes, labels, scores = self.model.predict(inputs["img"])
        bboxes = np.clip(bboxes, 0, 1)
        

        outputs = {"bboxes": bboxes, "bbox_labels": labels, "bbox_scores": scores}

        return outputs