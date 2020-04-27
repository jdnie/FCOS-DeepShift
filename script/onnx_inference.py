import caffe2.python.onnx.backend as backend
import onnx
import cv2
from maskrcnn_benchmark.modeling.rpn.fcos import inference_np
from maskrcnn_benchmark.config import cfg
import os
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np
import xml.etree.ElementTree as ET
import copy
import time
import glob
import shutil


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

    # print(box1_s,box2_s,box_s)
    return area_min, box_s, min_flag, (box_s / (box1_s + box2_s - box_s))


def soft_nms(detection_classes, detection_boxes, detection_scores,
             confidence_threshold=0.3):
    boxes = []
    classes = []
    scores = []

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
            detection_scores[box_idx] = np.exp(-(iou_score * iou_score)/0.5)*detection_scores[box_idx]
        else:
            if iou_score > 0.8:
                detection_scores[box_idx] = np.exp(-(iou_score * iou_score) / 0.5) * detection_scores[box_idx]

    return classes, boxes, scores


def run_accCal(model_path,
               test_base_path,
               save_base_path,
               labels_dict,
               config_file,
               confidence_threshold=0.3):
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

    model = onnx.load(model_path)
    rep = backend.prepare(model)

    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHT = model_path
    cfg.TEST.IMS_PER_BATCH = 1  # only test single image
    cfg.freeze()
    dbg_cfg = cfg

    pre = inference_np.PreProcessor(cfg)
    post = inference_np.PostProcessor(cfg)

    sad_accuracy = 0
    sad_precision = 0
    sad_recall = 0
    spend_time = []
    for idx, img_name in enumerate(img_list):
        progress(int(idx / len(img_list) * 100))
        base_img_name = os.path.split(img_name)[-1]
        frame = cv2.imread(img_name)
        ori_frame = copy.deepcopy(frame)

        h, w = frame.shape[:2]
        image = pre.procesing(frame)

        start_time = time.time()
        predictions = rep.run(image.astype(np.float32))
        labels, bboxes, scores = post.procesing(predictions)
        end_time = time.time()
        spend_time.append(end_time - start_time)

        keep = np.where(scores[0] > confidence_threshold, True, False)
        if np.sum(keep) == 0:
            continue
        labels = labels[0][keep]
        bboxes = bboxes[0][keep]
        scores = scores[0][keep]
        idx = np.argsort(scores)
        labels = labels[idx].tolist()
        bboxes = bboxes[idx].astype(np.int32).tolist()
        scores = scores[idx].tolist()

        labels, bboxes, scores = soft_nms(labels, bboxes, scores, confidence_threshold)

        for i, b in enumerate(bboxes):
            # save all
            frame = cv2.rectangle(frame,
                                  (b[0], b[1]), (b[2], b[3]),
                                  (100, 220, 200), 2)
            frame = cv2.putText(frame,
                                str(labels[i]) + '-' + str(int(scores[i] * 100)),
                                (b[0], b[1]), 1, 1,
                                (0, 0, 255), 1)
        # cv2.imwrite(os.path.join(save_res_path, base_img_name), frame)

        boxes_list_tmp = copy.deepcopy(bboxes)
        classes_list_tmp = copy.deepcopy(labels)
        score_list_tmp = copy.deepcopy(scores)

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
                if iou_score > 0.15:
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
            sad_recall = sad_recall + 1
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
            sad_precision = sad_precision + 1
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
            sad_accuracy += 1

        # print("cur sad: " + str(sad))
        # print("fg_cnt: " + str(fg_cnt))
        # print("pred_cnt: " + str(len(classes_list_tmp)))

    print('\nfps is : ', 1 / np.average(spend_time))
    accuracy = float(sad_accuracy / len(img_list))
    print("accuracy is : " + str(accuracy))
    precision = 1 - float(sad_precision / len(img_list))
    print("precision is : " + str(precision))
    recall = 1 - float(sad_recall / len(img_list))
    print("recall is : " + str(recall))


if __name__ == '__main__':
    labels_dict = {
        "1": {"name": "FG"}
    }
    free_gpu = query_free_gpu()
    os.environ['CUDA_VISIBLE_DEVICES'] = free_gpu
    pwd_path = os.popen("pwd").read().split('\n')[0]
    test_base_path = '/home/niejiadong/CommodityDepository/2.detect/jiangxiaobai/398/2019000002_Test/'
    # test_base_path = "/home/niejiadong/Challenge2019/VOCdevkitTest"
    # test_base_path = "/home/niejiadong/CommodityDepository/4.detect_trains/jiangxiaobai"
    run_accCal(model_path=os.path.join(pwd_path, "save_model/frozen.onnx"),
               test_base_path=test_base_path,
               save_base_path='/home/niejiadong/test/',
               labels_dict=labels_dict,
               config_file=os.path.join(pwd_path, "../configs/fcos/fcos_efficientnet_b3_voc.yaml"))