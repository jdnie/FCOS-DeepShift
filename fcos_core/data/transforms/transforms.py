# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random

import torch
import torchvision
from torchvision.transforms import functional as F

from PIL import Image
import numpy as np

import imgaug as ia
from imgaug import augmenters as iaa

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target=None):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        if isinstance(target, list):
            target = [t.resize(image.size) for t in target]
        elif target is None:
            return image
        else:
            target = target.resize(image.size)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            target = target.transpose(0)
        if random.random() < self.prob:
            image = F.vflip(image)
            target = target.transpose(1)
        return image, target


class ImgJitter(object):
    def __init__(self, prob=0.3):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < 0.2:
            return image, target
        if random.random() < self.prob:
            image = F.adjust_brightness(image, 1 + 0.4*(random.random()-0.4))
        if random.random() < self.prob:
            image = F.adjust_hue(image, 0.1*(random.random()-0.5))
        if random.random() < self.prob:
            image = F.adjust_contrast(image, 1 + 0.3*(random.random()-0.5))
        if random.random() < self.prob:
            image = F.adjust_saturation(image, 1 + (random.random()-0.2))
        if random.random() < self.prob:
            image = F.adjust_gamma(image, 1 + 0.4*(random.random()-0.4))
        return image, target


class ImgAug_Private(object):
    def __init__(self):
        ia.seed(random.randint(0, 65535))
        sometimes = lambda aug: iaa.Sometimes(0.35, aug)
        sometimes10 = lambda aug: iaa.Sometimes(0.05, aug)
        self.seq = iaa.Sequential(
            [
                sometimes(iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    # scale images to 80-120% of their size, individually per axis
                    translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                    # translate by -20 to +20 percent (per axis)
                    rotate=(-45, 45),  # rotate by -45 to +45 degrees
                    shear=(-5, 5),  # shear by -16 to +16 degrees
                    order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                    cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                    mode="ALL"  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
                sometimes(iaa.CropAndPad(
                    percent=((-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1)),
                    pad_mode="ALL",
                    pad_cval=(0, 255)
                )),
                sometimes(iaa.GaussianBlur((0, 1))),
                sometimes(iaa.AdditiveGaussianNoise(scale=(0, 5), per_channel=True)),
                sometimes10(iaa.ContrastNormalization((0.8, 1.2))),
                sometimes10(iaa.ContrastNormalization((0.9, 1.1), per_channel=True)),
                # sometimes10(iaa.DirectedEdgeDetect(alpha=(0, 0.1), direction=(0.0, 1))),
                # sometimes10(iaa.Emboss(alpha=(0, 0.1))),
                sometimes(iaa.Sharpen((0, 1)))

                # iaa.PiecewiseAffine(scale=(0.01, 0.03))

            ])

    def __call__(self, image, target):
        if random.random() < 0.5:
            return image, target
        try:
            seq_det = self.seq.to_deterministic()

            image_r = np.array(image)
            image_r = seq_det.augment_images([image_r])[0]

            boxes = np.array(target.bbox.numpy(), dtype=np.int32).tolist()
            bbs = ia.BoundingBoxesOnImage([ia.BoundingBox(box[0], box[1], box[2], box[3]) for box in boxes],
                                          shape=image_r.shape)
            bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
            scaled_box = []
            for box in bbs_aug.remove_out_of_image().clip_out_of_image().bounding_boxes:
                scaled_box.append([box.x1, box.y1, box.x2, box.y2])

            target = target.reset_bbox(scaled_box)
            image_r = Image.fromarray(np.uint8(image_r))
            image = image_r

        except Exception as e:
            print(e)

        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target=None):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image
        return image, target


if __name__ == '__main__':
    import cv2
    from fcos_core.structures.bounding_box import BoxList

    image = Image.open("/home/nie/lena.jpg").convert("RGB")
    target = BoxList(torch.tensor([[80, 92, 152, 194]], dtype=torch.float32),
                     image.size, mode="xyxy")
    target.add_field("labels", torch.tensor([1]))
    transform = Compose(
        [
            # Resize(256, 256),
            # RandomHorizontalFlip(0.5),
            ImgJitter(),
            ImgAug_Private()

        ]
    )
    for i in range(10):
        _image, _target = transform(image=image, target=target)
        rgb_image = np.array(_image)
        bboxes = np.array(_target.bbox.numpy(), dtype=np.int).tolist()
        for bbox in bboxes:
            rgb_image = cv2.rectangle(rgb_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                                      (0, 255, 0), 2)
        # cv2.imwrite("/home/niejiadong/test/{}.jpg".format(i),
        #             cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
        cv2.imshow('lena', cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)