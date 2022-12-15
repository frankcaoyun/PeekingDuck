"""YOLOv6 model"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

import os, sys
# print('test!!!', os.path.abspath(__file__))
# print('test2!', os.path.dirname(os.path.abspath(__file__)))


# ROOT = os.getcwd()
ROOT = os.path.dirname(os.path.abspath(__file__)) # the parrent folder of the current file
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    # print(sys.path, 'here we go...\n\n\n')

from tools import infer

class YOLOV6Model:

    def __init__(self, config) -> None:
        """initiate the model with all the model parameters"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        print(config)
        # initialize the inferer here
        self.inferer = infer.init_inferer(weights=os.path.join(ROOT, 'weights/yolov6n.pt'),
            # source = image,
            yaml=os.path.join(ROOT, 'data/coco.yaml'))

    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # print(image)
        """Predicts bboxes from image.

        Args:
            image (np.ndarray): Input image frame.

        Returns:
            (Tuple[np.ndarray, np.ndarray, np.ndarray]): Returned tuple
            contains:
            - An array of detection bboxes
            - An array of human-friendly detection class names
            - An array of detection scores

        Raises:
            TypeError: The provided `image` is not a numpy array.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a np.ndarray")

        # bboxes, classes, scores = infer.run(
        #     weights=os.path.join(ROOT, 'weights/yolov6n.pt'),
        #     # source=r'/home/frank/git-repo/aiap/PeekingDuck/data/images/image1.jpg',
        #     source = image,
        #     yaml=os.path.join(ROOT, 'data/coco.yaml')
        #     )

        bboxes, classes, scores = infer.run(self.inferer, source=image)

        return bboxes, classes, scores

        # return NotImplementedError

    