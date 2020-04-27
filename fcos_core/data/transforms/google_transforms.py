'''
author: jdnie
date: 2019/07/04
source code: https://github.com/tensorflow/tpu/tree/master/models/official/detection/utils/autoaugment_utils.py
'''
'''
All contributions by Jiadong Nie:
Copyright (c) 2020 Jiadong Nie
All rights reserved.
'''

import random
import torch
from PIL import Image
from PIL import ImageOps
from PIL import ImageEnhance
import math
import cv2
import numpy as np

_MAX_LEVEL = 10
_REPLACE = (128, 128, 128)


class Cutout(object):
    def __init__(self, prob, level):
        self.prob = prob
        self.level = level
        self.cutout_const = 100
        self.pad_size = int(self.level / _MAX_LEVEL * self.cutout_const)

    def __call__(self, image, target):
        if random.random() > self.prob:
            return image, target

        image_np = np.asarray(image).copy()
        ct_x = int(random.random() * image.size[0])
        ct_y = int(random.random() * image.size[1])
        l = ct_x - self.pad_size
        if l < 0:
            l = 0
        r = ct_x + self.pad_size
        if r > image.size[0]:
            r = image.size[0]
        t = ct_y - self.pad_size
        if t < 0:
            t = 0
        b = ct_y + self.pad_size
        if b > image.size[1]:
            b = image.size[1]
        image_np[t:b, l:r, :] = _REPLACE
        image = Image.fromarray(image_np)
        return image, target


class TranslateX_BBox(object):
    def __init__(self, prob, level):
        self.prob = prob
        self.level = level
        self.translate_const = 250
        self.pixel = int(self.level / _MAX_LEVEL * self.translate_const)

    def __call__(self, image, target):
        if random.random() > self.prob:
            return image, target

        if random.random() < 0.5:
            pixel = self.pixel
        else:
            pixel = -self.pixel

        image = Image.Image.transform(image, image.size, Image.AFFINE, (1, 0, pixel, 0, 1, 0),
                                      fillcolor=_REPLACE)
        boxes = np.array(target.bbox.numpy(), dtype=np.int32)
        boxes[:, 0] -= pixel
        boxes[:, 2] -= pixel
        target = target.reset_bbox(boxes)
        # google _check_bbox_area default is 0.05, if bbox is empty, keep at least 0.05*w/h size.
        # delete empty bbox directly
        target = target.clip_to_image(remove_empty=True)
        return image, target


class TranslateY_BBox(object):
    def __init__(self, prob, level):
        self.prob = prob
        self.level = level
        self.translate_const = 250
        self.pixel = int(self.level / _MAX_LEVEL * self.translate_const)

    def __call__(self, image, target):
        if random.random() > self.prob:
            return image, target

        if random.random() < 0.5:
            pixel = self.pixel
        else:
            pixel = -self.pixel

        image = Image.Image.transform(image, image.size, Image.AFFINE, (1, 0, 0, 0, 1, pixel),
                                      fillcolor=_REPLACE)
        boxes = np.array(target.bbox.numpy(), dtype=np.int32)
        boxes[:, 1] -= pixel
        boxes[:, 3] -= pixel
        target = target.reset_bbox(boxes)
        # google _check_bbox_area default is 0.05, if bbox is empty, keep at least 0.05*w/h size.
        # delete empty bbox directly
        target = target.clip_to_image(remove_empty=True)
        return image, target


class TranslateY_Only_BBoxes(object):
    def __init__(self, prob, level):
        self.prob = prob / 3
        self.level = level

        # google use voc/coco datasets, default 120
        # VOC:
        # w: [avg: 0.34, min: 0.00, max: 1.00],
        # h: [avg: 0.45, min: 0.01, max: 1.00]
        # COCO:
        # w: [avg: 0.180, min: 0.000, max: 1.000],
        # h: [avg: 0.227, min: 0.000, max: 1.000]
        # self.translate_bbox_const = 120

        # w: [avg: 0.056, min: 0.025, max: 0.180],
        # h: [avg: 0.067, min: 0.031, max: 0.144]
        self.translate_bbox_const = 40

        self.pixel = int(self.level / _MAX_LEVEL * self.translate_bbox_const)

    def __call__(self, image, target):
        bboxes = np.array(target.bbox.numpy(), dtype=np.int32)
        np.random.shuffle(bboxes)

        for bbox in bboxes:
            if random.random() > self.prob:
                continue

            if random.random() < 0.5:
                pixel = self.pixel
            else:
                pixel = -self.pixel

            mask = np.zeros((image.size[1], image.size[0]), np.uint8)
            mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 255
            mask = Image.fromarray(mask)
            bbox_img = Image.new("RGB", image.size, color=_REPLACE)
            bbox_img.paste(image, mask=mask)
            bbox_img = Image.Image.transform(
                bbox_img, bbox_img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixel), fillcolor=_REPLACE)
            image.paste(bbox_img, mask=mask)
        return image, target


class Equalize(object):
    def __init__(self, prob, level):
        self.prob = prob
        self.level = level

    def __call__(self, image, target):
        if random.random() > self.prob:
            return image, target

        image = ImageOps.equalize(image)
        return image, target


class ShearX_BBox(object):
    def __init__(self, prob, level):
        self.prob = prob
        self.level = level / _MAX_LEVEL * 0.3

    def __call__(self, image, target):
        if random.random() > self.prob:
            return image, target

        if random.random() < 0.5:
            level = self.level
        else:
            level = -self.level

        image = Image.Image.transform(image, image.size, Image.AFFINE, (1, level, 0, 0, 1, 0),
                                      fillcolor=_REPLACE)
        bboxes = np.array(target.bbox.numpy(), dtype=np.int32)
        coordinates = np.vstack([bboxes[:, 0], bboxes[:, 1],
                                 bboxes[:, 0], bboxes[:, 3],
                                 bboxes[:, 2], bboxes[:, 1],
                                 bboxes[:, 2], bboxes[:, 3]]).transpose()
        coordinates = coordinates.reshape(-1, 2)
        new_coordinates = np.matmul(coordinates, np.array([[1, 0], [-level, 1]]))
        new_coordinates = new_coordinates.reshape(-1, 4, 2)
        max_x = np.max(new_coordinates[:, :, 0], axis=1)
        max_y = np.max(new_coordinates[:, :, 1], axis=1)
        min_x = np.min(new_coordinates[:, :, 0], axis=1)
        min_y = np.min(new_coordinates[:, :, 1], axis=1)
        new_bboxes = np.vstack([min_x, min_y, max_x, max_y]).transpose()
        target = target.reset_bbox(new_bboxes.astype(np.int32))
        target = target.clip_to_image(remove_empty=True)
        return image, target


class ShearY_BBox(object):
    def __init__(self, prob, level):
        self.prob = prob
        self.level = level / _MAX_LEVEL * 0.3

    def __call__(self, image, target):
        if random.random() > self.prob:
            return image, target

        if random.random() < 0.5:
            level = self.level
        else:
            level = -self.level

        image = Image.Image.transform(image, image.size, Image.AFFINE, (1, 0, 0, level, 1, 0),
                                      fillcolor=_REPLACE)
        bboxes = np.array(target.bbox.numpy(), dtype=np.int32)
        coordinates = np.vstack([bboxes[:, 0], bboxes[:, 1],
                                 bboxes[:, 0], bboxes[:, 3],
                                 bboxes[:, 2], bboxes[:, 1],
                                 bboxes[:, 2], bboxes[:, 3]]).transpose()
        coordinates = coordinates.reshape(-1, 2)
        new_coordinates = np.matmul(coordinates, np.array([[1, -level], [0, 1]]))
        new_coordinates = new_coordinates.reshape(-1, 4, 2)
        max_x = np.max(new_coordinates[:, :, 0], axis=1)
        max_y = np.max(new_coordinates[:, :, 1], axis=1)
        min_x = np.min(new_coordinates[:, :, 0], axis=1)
        min_y = np.min(new_coordinates[:, :, 1], axis=1)
        new_bboxes = np.vstack([min_x, min_y, max_x, max_y]).transpose()
        target = target.reset_bbox(new_bboxes.astype(np.int32))
        target = target.clip_to_image(remove_empty=True)
        return image, target


class Sharpness(object):
    def __init__(self, prob, level):
        self.prob = prob
        self.level = (level/_MAX_LEVEL) * 1.8 + 0.1

    def __call__(self, image, target):
        if random.random() > self.prob:
            return image, target

        image = ImageEnhance.Sharpness(image).enhance(self.level)
        return image, target


class Color(object):
    def __init__(self, prob, level):
        self.prob = prob
        self.level = (level/_MAX_LEVEL) * 1.8 + 0.1

    def __call__(self, image, target):
        if random.random() > self.prob:
            return image, target

        degenerate = image.convert("L").convert("RGB")
        image = Image.blend(degenerate, image, self.level)

        return image, target


class Rotate_BBox(object):
    def __init__(self, prob, level):
        self.prob = prob
        self.level = (level/_MAX_LEVEL) * 30

    def __call__(self, image, target):
        if random.random() > self.prob:
            return image, target

        if random.random() < 0.5:
            level = self.level
        else:
            level = -self.level

        image = image.rotate(level, fillcolor=_REPLACE)
        w, h = image.size
        bboxes = np.array(target.bbox.numpy(), dtype=np.int32)
        coordinates = np.vstack([h / 2 - bboxes[:, 1], bboxes[:, 0] - w / 2,
                                 h / 2 - bboxes[:, 3], bboxes[:, 0] - w / 2,
                                 h / 2 - bboxes[:, 1], bboxes[:, 2] - w / 2,
                                 h / 2 - bboxes[:, 3], bboxes[:, 2] - w / 2]).transpose()
        coordinates = coordinates.reshape(-1, 2)
        degrees_to_radians = math.pi / 180.0
        radians = level * degrees_to_radians
        new_coordinates = np.matmul(
            np.array([[math.cos(radians), math.sin(radians)],
                      [-math.sin(radians), math.cos(radians)]]),
            coordinates.transpose())
        new_coordinates = new_coordinates.reshape(2, -1, 4)
        max_x = np.max(new_coordinates[1, :, :], axis=1) + w / 2
        max_y = h / 2 - np.min(new_coordinates[0, :, :], axis=1)
        min_x = np.min(new_coordinates[1, :, :], axis=1) + w / 2
        min_y = h / 2 - np.max(new_coordinates[0, :, :], axis=1)
        new_bboxes = np.vstack([min_x, min_y, max_x, max_y]).transpose()
        target = target.reset_bbox(new_bboxes.astype(np.int32))
        target = target.clip_to_image(remove_empty=True)
        return image, target


class Policy(object):
    def __init__(self, policies):
        self.policies = policies
        self.policie_num = len(self.policies)

    def __call__(self, image, target):
        idx = int(random.random() * self.policie_num)
        for t in self.policies[idx]:
            image, target = t(image, target)

        # ################# debug ####################
        # rgb_image = np.array(image)
        # bboxes = np.array(target.bbox.numpy(), dtype=np.int).tolist()
        # for bbox in bboxes:
        #     rgb_image = cv2.rectangle(rgb_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
        #                               (0, 255, 0), 2)
        # cv2.imwrite("/home/niejiadong/test/{}.jpg".format(random.randint(0, 10000)),
        #             cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
        return image, target


Policy_v0 = Policy(
    [
            [TranslateX_BBox(0.6, 4), Equalize(0.8, 10)],
            [TranslateY_Only_BBoxes(0.2, 2), Cutout(0.8, 8)],
            [Sharpness(0.0, 8), ShearX_BBox(0.4, 0)],
            [ShearY_BBox(1.0, 2), TranslateY_Only_BBoxes(0.6, 6)],
            [Rotate_BBox(0.6, 10), Color(1.0, 6)]
    ]
)


if __name__ == '__main__':
    from fcos_core.data.transforms.transforms import Compose
    from fcos_core.structures.bounding_box import BoxList

    image = Image.open("/home/nie/lena.jpg").convert("RGB")
    target = BoxList(torch.tensor([[80, 92, 152, 194]], dtype=torch.float32),
                     image.size, mode="xyxy")
    target.add_field("labels", torch.tensor([1]))
    # transform = Compose(
    #     [
    #         Rotate_BBox(1.0, 5),
    #     ]
    # )
    for i in range(100):
        _image, _target = Policy_v0(image=image, target=target)
        rgb_image = np.array(_image)
        bboxes = np.array(_target.bbox.numpy(), dtype=np.int).tolist()
        for bbox in bboxes:
            rgb_image = cv2.rectangle(rgb_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                                      (0, 255, 0), 2)
        cv2.imwrite("/home/nie/test/{}.jpg".format(i),
                    cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))