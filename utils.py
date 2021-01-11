# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import importlib
import sys
import torch
import torch.nn as nn
import torch.nn.init as init
import Levenshtein


def load_checkpoint(_model, pretrained_weights, to_use_device, _optimizer=None):
    global_state = {}
    state = torch.load(pretrained_weights, map_location=to_use_device)
    _model.load_state_dict(state['state_dict'])
    if _optimizer is not None:
        _optimizer.load_state_dict(state['optimizer'])
    if 'global_state' in state:
        global_state = state['global_state']
    
    return _model, _optimizer, global_state


def save_checkpoint(checkpoint_path, model, optimizer, **kwargs):
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    state.update(kwargs)
    torch.save(state, checkpoint_path)


def initial_logger(log_file_path):
    """
    ARGS
    log_file_path: string, path to the logging file
    """
    # logging settings
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    # file handler
    log_file_handler = logging.FileHandler(log_file_path)
    log_file_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_file_handler)
    # stream handler (stdout)
    log_stream_handler = logging.StreamHandler(sys.stdout)
    log_stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_stream_handler)
    logging.info('Logging file is %s' % log_file_path)


def create_module(module_str):
    tmpss = module_str.split(",")
    assert len(tmpss) == 2, "Error formate\
        of the module path: {}".format(module_str)
    module_name, function_name = tmpss[0], tmpss[1]
    somemodule = importlib.import_module(module_name, __package__)
    function = getattr(somemodule, function_name)
    return function


class RecMetric:
    def __init__(self, converter):
        """
        文本识别相关指标计算类

        :param converter: 用于label转换的转换器
        """
        self.converter = converter

    def __call__(self, predictions, labels):
        n_correct = 0
        norm_edit_dis = 0.0
        predictions = predictions.softmax(dim=2).detach().cpu().numpy()
        preds_str = self.converter.decode(predictions)
        show_str = []
        for (pred, pred_conf), target in zip(preds_str, labels):
            if max(len(pred), len(target)) == 0:
                continue
            else:
                norm_edit_dis += Levenshtein.distance(pred, target) / max(len(pred), len(target))
                show_str.append(f'{pred} -> {target}')
                if pred == target:
                    n_correct += 1
        return {'n_correct': n_correct, 'norm_edit_dis': norm_edit_dis, 'show_str': show_str}


def weight_init(m):
    """
    Usage:
        model = Model()
        model.apply(weight_init)
    """
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


def get_characters(dict_path):
    character_str = ''
    with open(dict_path, 'rb') as f:
        lines = f.readlines()
        for i in lines:
            tmp_char = i.decode('utf-8').strip('\n').strip('\r\n')
            character_str += tmp_char
    
    return character_str

