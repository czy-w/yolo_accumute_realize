import os
import numpy as np
import copy
import colorsys
from timeit import default_timer as timer
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image,ImageFont, ImageDraw
from yolo3 import yolo_body,yolo_eval
from utils.utils import letterbox_image


class YOLO(object):
    _defaults = {
        "model_path": "model_data/yolo_weights.h5",
        "anchors_path": "model_data/yolo_anchors.txt",
        "class_path": "model_data/coco_classes.txt",
        "score": 0.5,
        "iou": 0.3,
        "model_image_size": (416, 415)
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._default:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name'" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_sessions()
        self.boxes, self.scores, self.classes=self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names =f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def generate(self):
        model_path = os.path.expanduser(self.model)