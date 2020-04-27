# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from fcos_core.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from fcos_core.config import cfg
from fcos_core.data import make_data_loader
from fcos_core.solver import make_lr_scheduler
from fcos_core.solver import make_optimizer
from fcos_core.engine.inference import inference
from fcos_core.engine.trainer import do_train
from fcos_core.modeling.detector import build_detection_model
from fcos_core.utils.checkpoint import DetectronCheckpointer
from fcos_core.utils.collect_env import collect_env_info
from fcos_core.utils.comm import synchronize, \
    get_rank, is_pytorch_1_1_0_or_later
from fcos_core.utils.imports import import_file
from fcos_core.utils.logger import setup_logger
from fcos_core.utils.miscellaneous import mkdir

from fcos_core.deepshift.convert import convert_to_shift, round_shift_weights, count_layer_type

# from fcos_core.layers.misc import Conv2d as fConv2d
# def convert_to_shift_dbg(model, shift_depth, shift_type, convert_all_linear=True, convert_weights=False, freeze_sign = False, use_kernel=False, use_cuda=True, rounding='deterministic', shift_range=(-15, 0)):
#     conversion_count = 0
#     for name, module in reversed(model._modules.items()):
#         l = len(list(module.children()))
#         t = type(module)
#         if len(list(module.children())) > 0:
#             # recurse
#             model._modules[name], num_converted = convert_to_shift_dbg(model=module, shift_depth=shift_depth-conversion_count, shift_type=shift_type, convert_all_linear=convert_all_linear, convert_weights=convert_weights, freeze_sign = freeze_sign, use_kernel=use_kernel, use_cuda = use_cuda, rounding = rounding, shift_range = shift_range)
#             conversion_count += num_converted
#         if type(module) == torch.nn.Linear and (convert_all_linear == True or conversion_count < shift_depth):            
#             if convert_all_linear == False:
#                 conversion_count += 1

#         if (type(module) == torch.nn.Conv2d or type(module) == fConv2d) and conversion_count < shift_depth:            
#             conversion_count += 1

#     return model, conversion_count

def train(cfg, local_rank, distributed, iter_clear, ignore_head):
    model = build_detection_model(cfg)
    # model, conversion_count = convert_to_shift_dbg(
    #         model,
    #         cfg.DEEPSHIFT_DEPTH,
    #         cfg.DEEPSHIFT_TYPE,
    #         convert_weights=True,
    #         use_kernel=cfg.DEEPSHIFT_USEKERNEL,
    #         rounding=cfg.DEEPSHIFT_ROUNDING,
    #         shift_range=cfg.DEEPSHIFT_RANGE)

    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    if cfg.MODEL.USE_SYNCBN:
        assert is_pytorch_1_1_0_or_later(), \
            "SyncBatchNorm is only available in pytorch >= 1.1.0"
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    output_dir = cfg.OUTPUT_DIR
    save_to_disk = get_rank() == 0
    if iter_clear:
        load_opt = False
        load_sch = False
    else:
        load_opt = True
        load_sch = True
    if ignore_head:
        load_body = True
        load_fpn = True
        load_head = False
    else:
        load_body = True
        load_fpn = True
        load_head = True
    # 预加载模型或者是通常的模型，或者是deepshift模型
    if cfg.MODEL.WEIGHT:
        checkpointer = DetectronCheckpointer(
            cfg, model, None, None, output_dir, save_to_disk
        )

        extra_checkpoint_data = checkpointer.load(
            cfg.MODEL.WEIGHT, load_opt=False, load_sch=False,
            load_body=load_body, load_fpn=load_fpn, load_head=load_head)
        
        model, conversion_count = convert_to_shift(
            model,
            cfg.DEEPSHIFT_DEPTH,
            cfg.DEEPSHIFT_TYPE,
            convert_weights=True,
            use_kernel=cfg.DEEPSHIFT_USEKERNEL,
            rounding=cfg.DEEPSHIFT_ROUNDING,
            shift_range=cfg.DEEPSHIFT_RANGE)
        
        optimizer = make_optimizer(cfg, model)
        scheduler = make_lr_scheduler(cfg, optimizer)

        checkpointer = DetectronCheckpointer(
            cfg, model, optimizer, scheduler, output_dir, save_to_disk
        )
    else:
        model, conversion_count = convert_to_shift(
            model,
            cfg.DEEPSHIFT_DEPTH,
            cfg.DEEPSHIFT_TYPE,
            convert_weights=True,
            use_kernel=cfg.DEEPSHIFT_USEKERNEL,
            rounding=cfg.DEEPSHIFT_ROUNDING,
            shift_range=cfg.DEEPSHIFT_RANGE)
        
        optimizer = make_optimizer(cfg, model)
        scheduler = make_lr_scheduler(cfg, optimizer)

        checkpointer = DetectronCheckpointer(
            cfg, model, optimizer, scheduler, output_dir, save_to_disk
        )

        extra_checkpoint_data = checkpointer.load(
            cfg.MODEL.WEIGHT, load_opt=False, load_sch=False,
            load_body=load_body, load_fpn=load_fpn, load_head=load_head)
    
    conv2d_layers_count = count_layer_type(model, torch.nn.Conv2d)
    linear_layers_count = count_layer_type(model, torch.nn.Linear)
    print("###### conversion_count: {}, not convert conv2d layer: {}, linear layer: {}".format(
        conversion_count, conv2d_layers_count, linear_layers_count))

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )

    arguments = {}
    arguments["iteration"] = 0

    arguments.update(extra_checkpoint_data)

    if iter_clear:
        arguments["iteration"] = 0

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
    )

    model = round_shift_weights(model)
    torch.save({"model": model.state_dict()}, os.path.join(output_dir, "model_final_round.pth"))

    return model


def run_test(cfg, model, distributed):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.FCOS_ON or cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="script/2/fcos.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "--iter-clear",
        dest="iter_clear",
        help="clear iteration to 0, re-finetune the model",
        action="store_true",
    )
    parser.add_argument(
        "--ignore-head",
        dest="ignore_head",
        help="ignore head when load checkpoint",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # ======================dbg=================================
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    # args.skip_test = True
    # args.iter_clear = True
    # ======================dbg=================================

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("fcos_core", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    model = train(cfg, args.local_rank, args.distributed, args.iter_clear, args.ignore_head)

    if not args.skip_test:
        run_test(cfg, model, args.distributed)


if __name__ == "__main__":
    main()
