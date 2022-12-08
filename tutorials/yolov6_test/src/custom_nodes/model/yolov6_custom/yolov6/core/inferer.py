#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import os.path as osp
import math
from tqdm import tqdm
import numpy as np
import cv2
import torch
from PIL import ImageFont

from yolov6.utils.events import LOGGER, load_yaml
from yolov6.layers.common import DetectBackend
from yolov6.data.data_augment import letterbox
from yolov6.utils.nms import non_max_suppression
from yolov6.utils.torch_utils import get_model_info

import sys
# root_path = '/home/frank/git-repo/aiap/PeekingDuck'
# sys.path.insert(0, str(root_path))
# print(sys.path)
from peekingduck.pipeline.utils.bbox.transforms import xyxy2xyxyn # this is not working yet

class Inferer:
    def __init__(self, source, weights, device, yaml, img_size, half):
        import glob
        from yolov6.data.datasets import IMG_FORMATS

        self.__dict__.update(locals())

        # Init model
        self.device = device
        # self.img_size = source.shape # assign the original shape of the image
        # print('source shape:', source.shape)
        self.img_size = list(source.shape[:2]) # take the height and width. drop the channel
        cuda = self.device != 'cpu' and torch.cuda.is_available()
        self.device = torch.device('cuda:0' if cuda else 'cpu')
        self.model = DetectBackend(weights, device=self.device)
        self.stride = self.model.stride
        self.class_names = load_yaml(yaml)['names']
        self.img_size = self.check_img_size(self.img_size, s=self.stride)  # check image size
        # print('image size after correction:', self.img_size)
        # add in the source image array
        self.source = source
        self.half = half

        # Half precision
        if half & (self.device.type != 'cpu'):
            self.model.model.half()
        else:
            self.model.model.float()
            half = False

        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, *self.img_size).to(self.device).type_as(next(self.model.model.parameters())))  # warmup

        # since the source is now an image array, no ned to do this
        # # Load data
        # if os.path.isdir(source):
        #     img_paths = sorted(glob.glob(os.path.join(source, '*.*')))  # dir
        # elif os.path.isfile(source):
        #     img_paths = [source]  # files
        # else:
        #     raise Exception(f'Invalid path: {source}')
        # self.img_paths = [img_path for img_path in img_paths if img_path.split('.')[-1].lower() in IMG_FORMATS]

        # Switch model to deploy status
        self.model_switch(self.model, self.img_size)

    def model_switch(self, model, img_size):
        ''' Model switch to deploy status '''
        from yolov6.layers.common import RepVGGBlock
        for layer in model.modules():
            if isinstance(layer, RepVGGBlock):
                layer.switch_to_deploy()

        LOGGER.info("Switch model to deploy modality.")

    def infer(self, conf_thres, iou_thres, classes, agnostic_nms, max_det, save_dir, save_txt, save_img, hide_labels, hide_conf):
        ''' Model Inference and results visualization '''

        # print(self.source) # this is the image array
        img, img_src = self.precess_image(self.source, self.img_size, self.stride, self.half) # return the image tensor and original image
        img = img.to(self.device)
        if len(img.shape) == 3:
            img = img[None] # expand for batch dim
         
        # predict
        pred_results = self.model(img)
        # detect
        det = non_max_suppression(pred_results, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0] # this is a tensor containing the bboxes, classes and scores

        # return the bboxes, classes and scores, pass to infer
        det_np = det.cpu().detach().numpy()
        bboxes = det_np[:, :4] # These are absolute positions.
        # Convert to range between 0 and 1
        bboxes = xyxy2xyxyn(bboxes, *self.img_size) # normalize
        
        # classes = det_np[:, 5]
        classes = np.array([self.class_names[int(i)] for i in det_np[:, 5]]) # convert the numbers to label names
        scores = det_np[:, 4]
        # print(bboxes, classes, scores)
        return bboxes, classes, scores


        # for img_path in tqdm(self.img_paths):
        #     img, img_src = self.precess_image(img_path, self.img_size, self.stride, self.half)
        #     img = img.to(self.device)
        #     if len(img.shape) == 3:
        #         img = img[None]
        #         # expand for batch dim
        #     pred_results = self.model(img)
        #     det = non_max_suppression(pred_results, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0] # this is a tensor

        #     save_path = osp.join(save_dir, osp.basename(img_path))  # im.jpg
        #     txt_path = osp.join(save_dir, 'labels', osp.splitext(osp.basename(img_path))[0])

        #     gn = torch.tensor(img_src.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        #     img_ori = img_src

        #     # check image and font
        #     assert img_ori.data.contiguous, 'Image needs to be contiguous. Please apply to input images with np.ascontiguousarray(im).'
        #     self.font_check()

        #     if len(det):
        #         det[:, :4] = self.rescale(img.shape[2:], det[:, :4], img_src.shape).round()
        #         print(det[:, :4], det[:, 4], det[:, 5]) # bbox, score and class
        #         '''
        #         for *xyxy, conf, cls in reversed(det):
        #             if save_txt:  # Write to file
        #                 xywh = (self.box_convert(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        #                 line = (cls, *xywh, conf)
        #                 with open(txt_path + '.txt', 'a') as f:
        #                     f.write(('%g ' * len(line)).rstrip() % line + '\n')

        #             if save_img:
        #                 class_num = int(cls)  # integer class
        #                 label = None if hide_labels else (self.class_names[class_num] if hide_conf else f'{self.class_names[class_num]} {conf:.2f}')

        #                 self.plot_box_and_label(img_ori, max(round(sum(img_ori.shape) / 2 * 0.003), 2), xyxy, label, color=self.generate_colors(class_num, True))

        #         img_src = np.asarray(img_ori) # this is the image with annotated box and label

        #         # Save results (image with detections)
        #         if save_img:
        #             cv2.imwrite(save_path, img_src)
        #         '''
        #         # return the bboxes, classes and scores, pass to infer
        #         det_np = det.cpu().detach().numpy()
        #         bboxes = det_np[:, :4] # These are absolute positions.
        #         # Convert to decimal
                
        #         print('image_size:', self.img_size)
        #         bboxes = xyxy2xyxyn(bboxes, *self.img_size)
                
        #         # classes = det_np[:, 5]
        #         classes = np.array([self.class_names[int(i)] for i in det_np[:, 5]]) # convert the numbers to label names
        #         scores = det_np[:, 4]
        #         # print(bboxes, classes, scores)
        #         return bboxes, classes, scores

    @staticmethod
    def precess_image(img_src, img_size, stride, half):
        '''Process image before image inference.'''
        # try:
        #     # img_src = cv2.imread(path)
        #     cap = cv2.VideoCapture(path)
        #     _, img_src = cap.read()
        #     assert img_src is not None, f'Invalid image: {path}'
        # except Exception as e:
        #     LOGGER.warning(e)

        # takes in the image array and output the processed image
        image = letterbox(img_src, img_size, stride=stride)[0]

        # Convert
        image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        image = torch.from_numpy(np.ascontiguousarray(image))
        image = image.half() if half else image.float()  # uint8 to fp16/32
        image /= 255  # 0 - 255 to 0.0 - 1.0

        return image, img_src

    @staticmethod
    def rescale(ori_shape, boxes, target_shape):
        '''Rescale the output to the original image shape'''
        ratio = min(ori_shape[0] / target_shape[0], ori_shape[1] / target_shape[1])
        padding = (ori_shape[1] - target_shape[1] * ratio) / 2, (ori_shape[0] - target_shape[0] * ratio) / 2

        boxes[:, [0, 2]] -= padding[0]
        boxes[:, [1, 3]] -= padding[1]
        boxes[:, :4] /= ratio

        boxes[:, 0].clamp_(0, target_shape[1])  # x1
        boxes[:, 1].clamp_(0, target_shape[0])  # y1
        boxes[:, 2].clamp_(0, target_shape[1])  # x2
        boxes[:, 3].clamp_(0, target_shape[0])  # y2

        return boxes

    def check_img_size(self, img_size, s=32, floor=0):
        """Make sure image size is a multiple of stride s in each dimension, and return a new shape list of image."""
        if isinstance(img_size, int):  # integer i.e. img_size=640
            new_size = max(self.make_divisible(img_size, int(s)), floor)
        elif isinstance(img_size, list):  # list i.e. img_size=[640, 480]
            new_size = [max(self.make_divisible(x, int(s)), floor) for x in img_size]
        else:
            raise Exception(f"Unsupported type of img_size: {type(img_size)}")

        if new_size != img_size:
            print(f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}')
        return new_size if isinstance(img_size,list) else [new_size]*2

    def make_divisible(self, x, divisor):
        # Upward revision the value x to make it evenly divisible by the divisor.
        return math.ceil(x / divisor) * divisor

    @staticmethod
    def plot_box_and_label(image, lw, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        # Add one xyxy box to image with label
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
        if label:
            tf = max(lw - 1, 1)  # font thickness
            w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
            outside = p1[1] - h - 3 >= 0  # label fits outside box
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, lw / 3, txt_color,
                        thickness=tf, lineType=cv2.LINE_AA)

    @staticmethod
    # def font_check(font='./yolov6/utils/Arial.ttf', size=10): # change font location
    def font_check(font='./yolov6/utils/Arial.ttf', size=4):
        # font = os.path.join(os.path)
        font = 'src/custom_nodes/model/yolov6_custom/yolov6/utils/Arial.ttf'
        # Return a PIL TrueType Font, downloading to CONFIG_DIR if necessary
        assert osp.exists(font), f'font path not exists: {font}'
        try:
            return ImageFont.truetype(str(font) if font.exists() else font.name, size)
        except Exception as e:  # download if missing
            return ImageFont.truetype(str(font), size)

    @staticmethod
    def box_convert(x):
        # Convert boxes with shape [n, 4] from [x1, y1, x2, y2] to [x, y, w, h] where x1y1=top-left, x2y2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
        y[:, 2] = x[:, 2] - x[:, 0]  # width
        y[:, 3] = x[:, 3] - x[:, 1]  # height
        return y

    @staticmethod
    def generate_colors(i, bgr=False):
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        palette = []
        for iter in hex:
            h = '#' + iter
            palette.append(tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4)))
        num = len(palette)
        color = palette[int(i) % num]
        return (color[2], color[1], color[0]) if bgr else color
