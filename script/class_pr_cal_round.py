from __future__ import division
import os
import glob
import cv2
import sys
sys.path.append("../")
from fcos_core.config import cfg
import torch
from fcos_core.config import cfg
from fcos_core.modeling.detector import build_detection_model
from fcos_core.utils.checkpoint import DetectronCheckpointer
from torchvision import transforms as T
from fcos_core.structures.image_list import to_image_list
import os
import time
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import copy
import time
import shutil
import argparse
from fcos_core.deepshift.convert import convert_to_shift, round_shift_weights, count_layer_type


def progress(percent, width=80):
    '''进度打印功能'''
    if percent >= 100:
        percent = 100

    show_str = ('[%%-%ds]' % width) % (int(width * percent / 100) * "#")  # 字符串拼接的嵌套使用
    print('\r%s %d%%' % (show_str, percent), end='')


def query_free_gpu(free_memory_th=10000):
    ''' 查询空闲显存大于阈值的一个GPU '''
    gpus = os.popen("nvidia-smi -q -d Memory |grep -A4 GPU|grep Free|awk '{print $3}'").read().split("\n")
    gpus = [int(i) for i in gpus if i]
    max_free_memory = max(gpus)
    if max_free_memory < free_memory_th:
        print("There is no free gpu: ", gpus)
        return ""
    for i, gpu in enumerate(gpus):
        if gpu == max_free_memory:
            return str(i)


def get_cls(cls_name, labels_dict):
    for cls_id in range(len(labels_dict)):
        if labels_dict[str(cls_id)]['name'] == str(cls_name):
            break
    return cls_id


def get_iou(box1, box2):
    # x1, y1, x2, y2
    w1_1, h1_1, w2_1, h2_1 = box1
    box1_s = (h2_1 - h1_1) * (w2_1 - w1_1)
    w1_2, h1_2, w2_2, h2_2 = box2
    box2_s = (h2_2 - h1_2) * (w2_2 - w1_2)

    min_flag = 0
    area_min = box1_s
    if box1_s > box2_s:
        area_min = box2_s
        min_flag = 1

    if h1_1 > h2_2 or h2_1 < h1_2:
        h1 = 0
        h2 = 0
    else:
        h1 = max(h1_1, h1_2)
        h2 = min(h2_1, h2_2)

    if w1_1 > w2_2 or w2_1 < w1_2:
        w1 = 0
        w2 = 0
    else:
        w1 = max(w1_1, w1_2)
        w2 = min(w2_1, w2_2)

    box_s = (h2 - h1) * (w2 - w1)
    try:
        iou = (box_s / (box1_s + box2_s - box_s))
    except ZeroDivisionError: # What fuck boxes ??!!
        print(box1, box2)
        raise ZeroDivisionError

    # print(box1_s,box2_s,box_s)
    return area_min, box_s, min_flag, iou


def soft_nms(detection_classes, detection_boxes, detection_scores,
             confidence_threshold=0.3):
    boxes = []
    classes = []
    scores = []

    if not len(detection_boxes):
        return detection_classes, detection_boxes, detection_scores

    while True:
        idx = np.argmax(detection_scores)
        t_box = detection_boxes[idx]
        tc = int(detection_classes[idx])
        ts = float(detection_scores[idx])

        del detection_classes[idx]
        del detection_boxes[idx]
        del detection_scores[idx]

        if ts < confidence_threshold:
            break

        boxes.append(t_box)
        classes.append(tc)
        scores.append(ts)

        if not len(detection_boxes):
            break

        for box_idx in range(len(detection_boxes)):
            cur_box = detection_boxes[box_idx]
            _, iou_area, _, iou_score = get_iou(t_box, cur_box)

            if detection_classes[box_idx] == tc:
                detection_scores[box_idx] =\
                    np.exp(-(iou_score * iou_score * 2))*detection_scores[box_idx]
            else:
                if iou_score > 0.8:
                    # np.exp(-(iou_score * iou_score * 2)) * detection_scores[box_idx]
                    detection_scores[box_idx] =\
                        (1 + 2 * np.exp(-(iou_score * iou_score * 2))) / 3 * detection_scores[box_idx]

    return classes, boxes, scores


def run_accCal(model_path,
               test_base_path,
               save_base_path,
               labels_dict,
               config_file,
               input_size=640,
               confidence_thresholds=(0.3, )):
    save_res_path = os.path.join(save_base_path, 'all')
    if os.path.exists(save_res_path):
        shutil.rmtree(save_res_path)
    os.mkdir(save_res_path)

    save_recall_path = os.path.join(save_base_path, 'recall')
    if os.path.exists(save_recall_path):
        shutil.rmtree(save_recall_path)
    os.mkdir(save_recall_path)

    save_ero_path = os.path.join(save_base_path, 'ero')
    if os.path.exists(save_ero_path):
        shutil.rmtree(save_ero_path)
    os.mkdir(save_ero_path)

    save_ori_path = os.path.join(save_base_path, 'ori')
    if os.path.exists(save_ori_path):
        shutil.rmtree(save_ori_path)
    os.mkdir(save_ori_path)

    test_img_path = os.path.join(test_base_path, 'VOC2007/JPEGImages')
    test_ano_path = os.path.join(test_base_path, 'VOC2007/Annotations')
    img_list = glob.glob(test_img_path + '/*.jpg')

    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHT = model_path
    cfg.TEST.IMS_PER_BATCH = 1  # only test single image
    cfg.freeze()
    dbg_cfg = cfg

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)
    model, conversion_count = convert_to_shift(
            model,
            cfg.DEEPSHIFT_DEPTH,
            cfg.DEEPSHIFT_TYPE,
            convert_weights=True,
            use_kernel=cfg.DEEPSHIFT_USEKERNEL,
            rounding=cfg.DEEPSHIFT_ROUNDING,
            shift_range=cfg.DEEPSHIFT_RANGE)
    model = round_shift_weights(model)
    print("############# conversion_count: {} ".format(conversion_count))
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.load(cfg.MODEL.WEIGHT)
    model.eval()

    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
    )
    transform = T.Compose(
        [
            T.ToPILImage(),
            T.Resize(input_size),
            T.ToTensor(),
            T.Lambda(lambda x: x * 255),
            normalize_transform,
        ]
    )

    sad_accuracy = [0] * len(confidence_thresholds)
    sad_precision = [0] * len(confidence_thresholds)
    sad_recall = [0] * len(confidence_thresholds)
    spend_time = []
    for idx, img_name in enumerate(img_list):
        progress(int(idx/len(img_list) * 100))
        base_img_name = os.path.split(img_name)[-1]
        frame = cv2.imread(img_name)
        ori_frame = copy.deepcopy(frame)

        h, w = frame.shape[:2]
        image = transform(frame)
        image_list = to_image_list(image, cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(cfg.MODEL.DEVICE)

        start_time = time.time()
        with torch.no_grad():
            predictions = model(image_list)
        prediction = predictions[0].to("cpu")
        end_time = time.time()
        spend_time.append(end_time - start_time)

        prediction = prediction.resize((w, h)).convert("xyxy")
        # scores = prediction.get_field("scores")
        # keep = torch.nonzero(scores > confidence_threshold).squeeze(1)
        # prediction = prediction[keep]
        scores = prediction.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        prediction = prediction[idx]
        scores = prediction.get_field("scores").numpy()
        labels = prediction.get_field("labels").numpy()
        bboxes = prediction.bbox.numpy().astype(np.int32)
        bboxes_area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])

        for ii, confidence_threshold in enumerate(confidence_thresholds):
            _keep = np.where((scores > confidence_threshold) & (bboxes_area > 0), True, False)
            _scores = scores[_keep].tolist()
            _labels = labels[_keep].tolist()
            _bboxes = bboxes[_keep].tolist()
            _labels, _bboxes, _scores = soft_nms(_labels, _bboxes, _scores, confidence_threshold)

            if ii == 0:
                for i, b in enumerate(_bboxes):
                    # save all
                    frame = cv2.rectangle(frame,
                                          (b[0], b[1]), (b[2], b[3]),
                                          (100, 220, 200), 2)
                    frame = cv2.putText(frame,
                                        str(_labels[i]) + '-' + str(int(_scores[i] * 100)),
                                        (b[0], b[1]), 1, 1,
                                        (0, 0, 255), 1)
                # cv2.imwrite(os.path.join(save_res_path, base_img_name), frame)

            boxes_list_tmp = copy.deepcopy(_bboxes)
            classes_list_tmp = copy.deepcopy(_labels)
            score_list_tmp = copy.deepcopy(_scores)

            fg_cnt = 0
            recall_flag = False
            xml_name = base_img_name[:-4] + '.xml'
            anno_path = os.path.join(test_ano_path, xml_name)
            tree = ET.parse(anno_path)
            root = tree.getroot()
            rc_box = []
            for siz in root.findall('size'):
                width_ = siz.find('width').text
                height_ = siz.find('height').text
            if not int(width_) or not int(height_):
                width_ = w
                height_ = h
            for obj in root.findall('object'):
                name = obj.find('name').text
                # class_tmp = get_cls(name, labels_dict)
                for bndbox in obj.findall('bndbox'):
                    xmin = bndbox.find('xmin').text
                    ymin = bndbox.find('ymin').text
                    xmax = bndbox.find('xmax').text
                    ymax = bndbox.find('ymax').text
                    tmp_bbox = [int(int(xmin) * w / int(width_)),
                                int(int(ymin) * h / int(height_)),
                                int(int(xmax) * w / int(width_)),
                                int(int(ymax) * h / int(height_))]
                map_flag = False
                for bbox_idx in range(len(boxes_list_tmp)):
                    min_area, box_s, min_flag, iou_score = \
                        get_iou(tmp_bbox, boxes_list_tmp[bbox_idx])
                    if iou_score > 0.3:
                        map_flag = True
                        del classes_list_tmp[bbox_idx]
                        del boxes_list_tmp[bbox_idx]
                        del score_list_tmp[bbox_idx]
                        break
                # 如果没找到匹配，属于漏检，算到召回率/检出率中
                if not map_flag:
                    recall_flag = True
                    rc_box.append(tmp_bbox)
                fg_cnt = fg_cnt + 1

            if recall_flag:
                sad_recall[ii] += 1
                if ii == 0:
                    for box_idx in range(len(rc_box)):
                        x1, y1, x2, y2 = rc_box[box_idx]
                        rca_frame = cv2.rectangle(frame,
                                                  (int(x1), int(y1)), (int(x2), int(y2)),
                                                  (255, 0, 0), 4)
                    cv2.imwrite(os.path.join(save_recall_path, base_img_name), rca_frame)
                    shutil.copy(img_name, os.path.join(save_ori_path, base_img_name))
                    shutil.copy(anno_path, os.path.join(save_ori_path, xml_name))
                # print("sad_recall: " + str(sad_recall))

            # 如果有多出来的，属于误检，ground_truth中没有这个框，算到准确率中
            if len(classes_list_tmp) > 0:
                sad_precision[ii] += 1
                if ii == 0:
                    for box_idx in range(len(boxes_list_tmp)):
                        x1, y1, x2, y2 = boxes_list_tmp[box_idx]
                        ero_frame = cv2.rectangle(frame,
                                                  (int(x1), int(y1)), (int(x2), int(y2)),
                                                  (0, 0, 255), 4)
                        err_rect_name = base_img_name[:-4] + '_' + str(box_idx) + '.jpg'
                        cv2.imwrite(os.path.join(save_ero_path, err_rect_name),
                                    ori_frame[y1: y2, x1: x2, :])
                    cv2.imwrite(os.path.join(save_ero_path, base_img_name), ero_frame)
                    shutil.copy(img_name, os.path.join(save_ori_path, base_img_name))
                    shutil.copy(anno_path, os.path.join(save_ori_path, xml_name))

            if not recall_flag and len(classes_list_tmp) == 0:
                sad_accuracy[ii] += 1

            # print("cur sad: " + str(sad))
            # print("fg_cnt: " + str(fg_cnt))
            # print("pred_cnt: " + str(len(classes_list_tmp)))

    # 单图所有框都检测正确才正确率，少一个框算漏检，多一个框算误检，不看mAP
    print('\nfps is : ', 1 / np.average(spend_time))
    for ii, confidence_threshold in enumerate(confidence_thresholds):
        print("confidence th is : {}".format(confidence_threshold))
        accuracy = float(sad_accuracy[ii] / len(img_list))
        print("accuracy is : {}".format(accuracy))
        precision = 1 - float(sad_precision[ii] / len(img_list))
        print("precision is : {}".format(precision))
        recall = 1 - float(sad_recall[ii] / len(img_list))
        print("recall is : {}\n".format(recall))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='model path',
                        default="2/save_model/model_0005000.pth")
    parser.add_argument('--test_base_path', type=str, help='test base path',
                        default="/home/nie/f/dataset")
    parser.add_argument('--save_base_path', type=str, help='save base name',
                        default='/home/nie/f/test/')
    parser.add_argument('--config_file', type=str, help='config file',
                        default="2/fcos.yaml")
    parser.add_argument('--confidence_thresholds', type=float, help='confidence threshold', nargs='+',
                        default=[0.3, 0.4, 0.5, 0.6, 0.8, 0.2, 0.1])
    args = parser.parse_args()
    print(args)
    labels_dict = {
        "1": {"name": "FG"}
    }
    free_gpu = query_free_gpu()
    os.environ['CUDA_VISIBLE_DEVICES'] = free_gpu
    # pwd_path = os.popen("pwd").read().split('\n')[0]
    run_accCal(model_path=args.model_path,
               test_base_path=args.test_base_path,
               save_base_path=args.save_base_path,
               labels_dict=labels_dict,
               config_file=args.config_file,
               confidence_thresholds=args.confidence_thresholds)


if __name__ == '__main__':
    main()
