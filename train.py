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
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import logging
from program import build_model, build_config, build_data_loader, build_device, build_optimizer, build_loss, build_trainer, build_pretrained_weights

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def main():
    
    flags = build_config()

    model = build_model(flags)

    optimizer = build_optimizer(flags, model)

    model, optimizer, global_state = build_pretrained_weights(flags, model, optimizer)

    train_loader = build_data_loader(flags, mode='train')
    val_loader = build_data_loader(flags, mode='validation')

    device, gpu_count = build_device(flags)
    if gpu_count > 1:
        model = nn.DataParallel(model)

    loss = build_loss(flags)

    trainer = build_trainer(
        device=device,
        model=model,
        optimizer=optimizer,
        loss=loss,
        val_loader=val_loader,
        train_loader=train_loader,
        flags=flags,
        global_state=global_state
    )
    trainer.train()


if __name__ == '__main__':
    main()
