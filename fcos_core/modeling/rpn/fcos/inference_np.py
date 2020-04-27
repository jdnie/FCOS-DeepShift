import numpy as np
import cv2


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class PreProcessor(object):
    def __init__(self, cfg):
        super(PreProcessor, self).__init__()
        self.mean = cfg.INPUT.PIXEL_MEAN
        self.std = cfg.INPUT.PIXEL_STD
        self.resize_shape = (cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST)

    def transform_single_image(self, image):
        image = image - np.array(self.mean)
        image /= np.array(self.std)
        image = np.array(image).transpose((2, 0, 1))
        image = image[np.newaxis, :].astype('float32')
        return image

    def transform_mul_image(self, image):
        image = image - np.array(self.mean)
        image /= np.array(self.std)
        image = np.array(image).transpose((0, 3, 1, 2))
        return image

    def procesing(self, img):
        if isinstance(img, list):
            for idx, i in enumerate(img):
                if i.shape[:2] != self.resize_shape:
                    img[idx] = cv2.resize(i, self.resize_shape)
            img = np.array(img)
            return self.transform_mul_image(img)
        if img.shape.__len__() == 3:
            if img.shape[:2] != self.resize_shape:
                img = cv2.resize(img, self.resize_shape)
            return self.transform_single_image(img)
        if img.shape.__len__() == 4:
            return self.transform_mul_image(img)


class PostProcessor(object):
    """
    Performs processing including pre and post
    """
    def __init__(self, cfg, min_size=0):
        super(PostProcessor, self).__init__()
        self.resize_shape = [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST]
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.pre_nms_thresh = cfg.MODEL.FCOS.INFERENCE_TH
        self.pre_nms_top_n = cfg.MODEL.FCOS.PRE_NMS_TOP_N
        self.nms_thresh = cfg.MODEL.FCOS.NMS_TH
        self.fpn_post_nms_top_n = cfg.TEST.DETECTIONS_PER_IMG
        self.min_size = min_size
        self.num_classes = cfg.MODEL.FCOS.NUM_CLASSES

        self.locations = []
        self.compute_locations(self.resize_shape, self.fpn_strides)

    def compute_locations_per_level(self, h, w, stride):
        shifts_x = np.arange(
            0, w * stride, step=stride,
            dtype=np.float32
        )
        shifts_y = np.arange(
            0, h * stride, step=stride,
            dtype=np.float32
        )
        shift_y, shift_x = np.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = np.stack((shift_y, shift_x), axis=1) + stride // 2
        return locations

    def compute_locations(self, img_shape, fpn_strides):
        for level, fpn_stride in enumerate(fpn_strides):
            h, w = [i // fpn_stride for i in img_shape]
            locations_per_level = self.compute_locations_per_level(
                h, w, fpn_stride
            )
            self.locations.append(locations_per_level)

    def nms(self, classes, boxes, scores, nms_thresh):
        order = np.argsort(scores)[::-1]
        classes = classes[order]
        boxes = boxes[order]
        scores = scores[order]
        areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)

        i = 0
        while True:
            if i >= len(scores) - 1:
                break
            x1 = boxes[:, 0]
            y1 = boxes[:, 1]
            x2 = boxes[:, 2]
            y2 = boxes[:, 3]

            xx1 = np.where(x1[i] > x1, x1[i], x1)
            xx2 = np.where(x2[i] < x2, x2[i], x2)
            yy1 = np.where(y1[i] > y1, y1[i], y1)
            yy2 = np.where(y2[i] < y2, y2[i], y2)
            w = xx2 - xx1 + 1
            h = yy2 - yy1 + 1
            w = np.where(w > 0, w, 0)
            h = np.where(h > 0, h, 0)
            inter = w * h
            iou = inter / (areas[i] + areas - inter)
            keep = np.where(iou > nms_thresh, False, True)
            keep[i] = True

            classes = classes[keep]
            boxes = boxes[keep]
            scores = scores[keep]
            areas = areas[keep]

            i += 1
        return classes, boxes, scores

    def forward_for_single_feature_map(
            self, locations, box_cls,
            box_regression, centerness,
            image_sizes):
        N, C, H, W = box_cls.shape

        # put in the same format as locations
        box_cls = box_cls.reshape(N, C, H, W).transpose(0, 2, 3, 1)
        box_cls = box_cls.reshape(N, -1, C)
        box_cls = sigmoid(box_cls)

        box_regression = box_regression.reshape(N, 4, H, W).transpose(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, 4)
        centerness = centerness.reshape(N, 1, H, W).transpose(0, 2, 3, 1)
        centerness = centerness.reshape(N, -1)
        centerness = sigmoid(centerness)

        candidate_inds = box_cls > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.reshape(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clip(max=self.pre_nms_top_n)

        # multiply the classification scores with centerness scores
        box_cls = box_cls * centerness[:, :, None]

        results = []
        for i in range(N):
            per_box_cls = box_cls[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]

            per_candidate_nonzeros = np.nonzero(per_candidate_inds)
            per_box_loc = per_candidate_nonzeros[0]
            per_class = per_candidate_nonzeros[1] + 1

            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc]
            per_locations = locations[per_box_loc]

            per_pre_nms_top_n = pre_nms_top_n[i]
            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                top_k_indices = np.flip(np.argsort(per_box_cls))[:per_pre_nms_top_n]
                per_box_cls = per_box_cls[top_k_indices]
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]

            detections = np.stack([
                per_locations[:, 0] - per_box_regression[:, 0],
                per_locations[:, 1] - per_box_regression[:, 1],
                per_locations[:, 0] + per_box_regression[:, 2],
                per_locations[:, 1] + per_box_regression[:, 3],
            ], axis=1)

            x1 = detections[:, 0]
            y1 = detections[:, 1]
            x2 = detections[:, 2]
            y2 = detections[:, 3]
            keep = np.where((x1 > 0) & (x1 < image_sizes[0] - 1) &
                            (y1 > 0) & (y1 < image_sizes[1] - 1) &
                            (x2 > 0) & (x2 < image_sizes[0] - 1) &
                            (y2 > 0) & (y2 < image_sizes[1] - 1) &
                            (x1 < x2) & (y1 < y2), True, False)
            if self.min_size > 0:
                w = x2 - x1
                h = y2 - y1
                keep = keep & np.where(w > self.min_size & h > self.min_size)
            per_box_cls = per_box_cls[keep]
            per_class = per_class[keep]
            detections = detections[keep]

            results.append([per_box_cls, per_class, detections])
        return results

    def procesing(self, output):
        box_cls = output[:5]
        box_regression = output[5:10]
        centerness = output[10:]

        sampled_boxes = []
        for _, (l, o, b, c) in enumerate(zip(self.locations, box_cls, box_regression, centerness)):
            sampled_boxes.append(
                self.forward_for_single_feature_map(
                    l, o, b, c, self.resize_shape
                )
            )

        classes_list = []
        boxes_list = []
        scores_list = []
        for N in range(sampled_boxes[0].__len__()):
            classes = []
            boxes = []
            scores = []
            for i in range(sampled_boxes.__len__()):
                single_layer_res = sampled_boxes[i][N]
                try:
                    per_box_cls, per_class, detections = single_layer_res
                    classes.append(per_class)
                    boxes.append(detections)
                    scores.append(per_box_cls)
                except:
                    pass
            classes = np.concatenate(classes)
            boxes = np.concatenate(boxes)
            scores = np.concatenate(scores)

            i_classes = []
            i_boxes = []
            i_scores = []
            for i in range(1, self.num_classes):
                keep = np.where(classes == i, True, False)
                _classes = classes[keep]
                _boxes = boxes[keep]
                _scores = scores[keep]
                _classes, _boxes, _scores = \
                    self.nms(_classes, _boxes, _scores, self.nms_thresh)
                i_classes.append(_classes)
                i_boxes.append(_boxes)
                i_scores.append(_scores)

            classes_list.append(np.concatenate(i_classes))
            boxes_list.append(np.concatenate(i_boxes))
            scores_list.append(np.concatenate(i_scores))

        return classes_list, boxes_list, scores_list