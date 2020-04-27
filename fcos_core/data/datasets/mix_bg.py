'''
author: niejiadong
date: 2019/07/09
'''
'''
All contributions by Jiadong Nie:
Copyright (c) 2020 Jiadong Nie
All rights reserved.
'''

import torch
import torch.utils.data
import random
from PIL import Image
import numpy as np
import cv2
from torchvision.transforms import functional as F


class MixBG(torch.utils.data.Dataset):
    def __init__(self, dataset, transforms=None):
        self.dataset = dataset
        self.prob = 0.5
        self.replace = (128, 128, 128)
        self.transforms = transforms
        self.dataset_len = len(dataset)
        print("Use mix bg datasets !!!")

    def __getitem__(self, index):
        img1, target1, index1 = self.dataset[index]
        if random.random() > self.prob:
            if self.transforms is not None:
                img1, target1 = self.transforms(img1, target1)
            return img1, target1, index1
        # print("mix_bg!!!")

        index2 = np.random.choice(np.delete(np.arange(len(self)), index))
        img2, target2, _ = self.dataset[index2]
        size = img1.size  # width, height
        img2 = F.resize(img2, (size[1], size[0]))
        target2 = target2.resize(size)

        rectrange_num = random.randint(5, 15)
        x = np.random.randint(0, size[0] // 4, (rectrange_num, 2))
        y = np.random.randint(0, size[1] // 4, (rectrange_num, 2))
        color = np.random.randint(0, 255, rectrange_num)
        x1 = np.min(x, axis=1)
        x2 = np.max(x, axis=1)
        y1 = np.min(y, axis=1)
        y2 = np.max(y, axis=1)
        mask = np.ones((size[1] // 4, size[0] // 4), np.uint8) * 128
        masks = []
        for i in range(rectrange_num):
            _mask = mask.copy()
            _mask[y1[i]:y2[i], x1[i]:x2[i]] = color[i]
            masks.append(_mask)
        masks = np.vstack(masks).reshape((-1, size[1] // 4, size[0] // 4))
        masks = np.average(masks, axis=0).astype(np.uint8)
        # masks = cv2.resize(masks, None, fx=4, fy=4)
        masks = cv2.resize(masks, size)
        masks = np.where(masks > 128, 255, 0)

        bboxes1 = np.array(target1.bbox.numpy(), dtype=np.int32)
        bboxes2 = np.array(target2.bbox.numpy(), dtype=np.int32)
        bbox_mask1 = masks.astype(np.uint8)
        bbox_mask2 = (255 - masks).astype(np.uint8)
        for bbox in bboxes1:
            bbox_mask1[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 255
        for bbox in bboxes2:
            bbox_mask2[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 0

        final_img = Image.new("RGB", size, color=self.replace)
        mask2 = Image.fromarray(bbox_mask2)
        mask1 = Image.fromarray(bbox_mask1)
        final_img.paste(img2, mask=mask2)
        final_img.paste(img1, mask=mask1)

        if self.transforms is not None:
            final_img, target1 = self.transforms(final_img, target1)

        # ################# debug ####################
        # rgb_image = np.array(final_img)
        # cv2.imwrite("/home/niejiadong/test/{}.jpg".format(random.randint(0, 10000)),
        #             cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
        # bboxes = np.array(target1.bbox.numpy(), dtype=np.int).tolist()
        # for bbox in bboxes:
        #     rgb_image = cv2.rectangle(rgb_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
        #                               (0, 255, 0), 2)
        # cv2.imwrite("/home/niejiadong/test/{}.jpg".format(random.randint(0, 10000)),
        #             cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))

        # masks = np.array(masks)
        # cv2.imwrite("/home/niejiadong/test/mask_{}.jpg".format(index), masks)
        # img1 = np.array(img1)
        # cv2.imwrite("/home/niejiadong/test/img1_{}.jpg".format(index), cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))
        # img2 = np.array(img2)
        # cv2.imwrite("/home/niejiadong/test/img2_{}.jpg".format(index), cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))
        # rgb_image = np.array(final_img)
        # bboxes = np.array(target1.bbox.numpy(), dtype=np.int).tolist()
        # for bbox in bboxes:
        #     rgb_image = cv2.rectangle(rgb_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
        #                               (0, 255, 0), 2)
        # cv2.imwrite("/home/niejiadong/test/final_{}.jpg".format(index), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))

        return final_img, target1, index1

    def __len__(self):
        return self.dataset_len

    def get_img_info(self, index):
        return self.dataset.get_img_info(index)


if __name__ == "__main__":
    from fcos_core.data.datasets.voc import PascalVOCDataset
    # import fcos_core.data.transforms.transforms as T
    # transform = T.Compose(
    #     [
    #         T.Resize(640, 640),
    #         T.RandomHorizontalFlip(0.5),
    #         T.ImgJitter(),
    #         T.ImgAug_Private(),
    #         # gT.Policy_v0,
    #         T.ToTensor(),
    #         # T.Normalize(mean=[102.9801, 115.9465, 122.7717], std=[1., 1., 1.], to_bgr255=True),
    #     ]
    # )
    dataset = PascalVOCDataset(data_dir="/home/nie/f/dataset/voc/train/VOC2007",
                               split="trainval",
                               transforms=None)
    mix_bg = MixBG(dataset)
    for i in range(100):
        img, target, index = mix_bg[i]