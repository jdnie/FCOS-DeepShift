import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
import os
from torch.autograd import Variable


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


def pytorch_2_onnx(model_path, onnx_path, config_file, input_size=640):
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHT = model_path
    cfg.TEST.IMS_PER_BATCH = 1  # only test single image
    cfg.MODEL.DEVICE = 'cpu'
    cfg.FROZEN_ONNX = True

    def_cfg = cfg
    cfg.freeze()

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.load(cfg.MODEL.WEIGHT)
    model.eval()

    example = Variable(torch.rand(1, 3, input_size, input_size)).to(cfg.MODEL.DEVICE)

    torch.onnx._export(model, example, onnx_path, export_params=True)


if __name__ == "__main__":
    # free_gpu = query_free_gpu()
    # os.environ['CUDA_VISIBLE_DEVICES'] = free_gpu
    pwd_path = os.popen("pwd").read().split('\n')[0]
    pytorch_2_onnx(
        model_path=os.path.join(pwd_path, 'save_model/model_final.pth'),
        onnx_path=os.path.join(pwd_path, 'save_model/frozen.onnx'),
        config_file=os.path.join(pwd_path, "../configs/fcos/fcos_efficientnet_b3_voc.yaml")
        # config_file=os.path.join(pwd_path, "../configs/fcos/fcos_dla34_2x_voc.yaml")
    )
