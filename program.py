# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from argparse import ArgumentParser, RawDescriptionHelpFormatter
import os
import random
import time
import shutil
import traceback
import yaml
import logging
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)

import numpy as np
from copy import deepcopy
from torch.utils import data
import torch
import torch.nn as nn
from tqdm import tqdm
from flags import Flags
from utils import initial_logger, save_checkpoint, load_checkpoint, create_module, weight_init
from trainer import TrainerRec
from torch import optim


class ArgsParser(ArgumentParser):
    def __init__(self):
        super(ArgsParser, self).__init__(
            formatter_class=RawDescriptionHelpFormatter)
        self.add_argument("-c", "--config", help="configuration file to use")

    def parse_args(self, argv=None):
        args = super(ArgsParser, self).parse_args(argv)
        assert args.config is not None, \
            "Please specify --config=configure_file_path."
        return args


def build_config():
    args = ArgsParser().parse_args()
    flags = Flags(args.config).get()
    log_file_path = os.path.join(flags.Global.save_model_dir, time.strftime('%Y%m%d_%H%M%S') + '.log')
    os.makedirs(flags.Global.save_model_dir, exist_ok=True)
    logger = initial_logger(log_file_path)
    return flags


def build_model(flags):
    # build network
    model_infor = flags.Architecture.function
    print('model_infor', model_infor)
    model = create_module(model_infor)(flags)
    return model


def build_data_loader(flags=None, mode=None):
    assert mode in ["train", "validation", "test"], "Nonsupport mode:{}".format(mode)
    if mode == "train":
        dataloader_infor = deepcopy(flags.TrainReader.dataloader)
    elif mode == "validation":
        dataloader_infor = deepcopy(flags.EvalReader.dataloader) 
    dataloader = create_module(dataloader_infor)(flags)
    return dataloader


def build_optimizer(flags, model):
    if flags.Optimizer.function == 'sgd':
        optimizer = optim.SGD(model.parameters(), 
                                lr=flags.Optimizer.base_lr, 
                                momentum=flags.Optimizer.momentum, 
                                weight_decay=flags.Optimizer.weight_decay)
    if flags.Optimizer.function == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=flags.Optimizer.base_lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=flags.Optimizer.base_lr)
    return optimizer


def build_pretrained_weights(flags, model, optimizer):
    # 是否加载之前训练的模型
    pretrain_weights = flags.Global.pretrain_weights
    to_use_device = flags.Global.device
    if pretrain_weights and os.path.exists(pretrain_weights):
        model, _resumed_optimizer, global_state = load_checkpoint(model, pretrain_weights, to_use_device, optimizer)
        if flags.Global.resumed_optimizer and _resumed_optimizer is not None:
            optimizer = _resumed_optimizer
    else:
        global_state = {}
        model.apply(weight_init)
    return model, optimizer, global_state


def build_device(flags):
    if flags.Global.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = flags.Global.gpu_num
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_count = torch.cuda.device_count()
        else:
            device = torch.device("cpu")
            gpu_count = 0
    else:
        device = torch.device("cpu")
        gpu_count = 0
    return device, gpu_count


def build_loss(flags):
    loss_params = flags.Loss.function
    loss = create_module(loss_params)(params=flags.Loss)
    return loss


def build_trainer(model, optimizer, loss, train_loader, val_loader, \
                  device, flags, global_state):
    if flags.Global.algorithm in ['CRNN', 'FAN', 'GRCNN', 'DAN', 'SAR']:
        trainer = TrainerRec(
            device=device,
            model=model,
            optimizer=optimizer,
            loss=loss,
            val_loader=val_loader,
            train_loader=train_loader,
            flags=flags,
            global_state=global_state
        )
    return trainer