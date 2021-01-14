# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

import torch
import torch.nn as nn
import torch.nn.functional as F
from character import CTCLabelConverter


class GRCNN(nn.Module):
    def __init__(self, flags):
        super(GRCNN, self).__init__()
        self.inplanes = 1 if flags.Global.image_shape[0] == 1 else 3
        self.input_size = flags.SeqRNN.input_size
        self.hidden_size = flags.SeqRNN.hidden_size
        self.converter = CTCLabelConverter(flags)
        self.num_classes = self.converter.char_num

        self.feature_extractor = GRCNN_FeatureExtractor(self.inplanes, self.input_size)
        self.reshape_layer = ReshapeLayer()
        self.sequence_layer = SequenceLayer(self.input_size, self.hidden_size)
        self.linear_layer = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, inputs):
        x = self.feature_extractor(inputs)
        x = self.reshape_layer(x)
        x = self.sequence_layer(x)
        outputs = self.linear_layer(x)

        return outputs


class GRCNN_FeatureExtractor(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(GRCNN_FeatureExtractor, self).__init__()
        output_channels = [int(output_channel / 8), int(output_channel / 4), 
                                   int(output_channel / 2), output_channel]
        self.conv1 = nn.Conv2d(input_channel, output_channels[0], 3, 1, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2, 2, 0)
        self.grcl1 = GRCL(output_channels[0], output_channels[0], num_iterations=5, kernel_size=3, pad=1)
        self.maxpool2 = nn.MaxPool2d(2, 2, 0)
        self.grcl2 = GRCL(output_channels[0], output_channels[1], num_iterations=5, kernel_size=3, pad=1)
        self.maxpool3 = nn.MaxPool2d(2, stride=(2, 1), padding=(0, 1))
        self.grcl3 = GRCL(output_channels[1], output_channels[2], num_iterations=5, kernel_size=3, pad=1)
        self.maxpool4 = nn.MaxPool2d(2, stride=(2, 1), padding=(0, 1))
        self.conv2 = nn.Conv2d(output_channels[2], output_channels[3], kernel_size=(2, 2), stride=(1, 1), bias=False)
        self.bn = nn.BatchNorm2d(512)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.grcl1(x)
        x = self.maxpool2(x)
        x = self.grcl2(x)
        x = self.maxpool3(x)
        x = self.grcl3(x)
        x = self.maxpool4(x)
        x = self.conv2(x)
        x = self.bn(x)
        outputs = self.relu(x)

        return outputs


class GRCL(nn.Module):
    def __init__(self, input_channel, output_channel, num_iterations, 
                 kernel_size, pad):
        super(GRCL, self).__init__()
        self.wgf_u = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=False)
        self.wgr_x = nn.Conv2d(output_channel, output_channel, 1, 1, 0, bias=False)
        self.wf_u = nn.Conv2d(input_channel, output_channel, kernel_size, 1, pad, bias=False)
        self.wg_x = nn.Conv2d(output_channel, output_channel, kernel_size, 1, pad, bias=False)

        self.bn_x = nn.BatchNorm2d(output_channel)

        self.num_iterations = num_iterations
        self.GRCL = [GRCL_unit(output_channel) for _ in range(num_iterations)]
        self.GRCL = nn.Sequential(*self.GRCL)

    def forward(self, inputs):
        self.wgf = self.wgf_u(inputs)
        self.wf = self.wf_u(inputs)
        x = F.relu(self.bn_x(self.wf))

        for i in range(self.num_iterations):
            x = self.GRCL[i](self.wgf, self.wgr_x(x), self.wf, self.wg_x(x))

        return x


class GRCL_unit(nn.Module):
    def __init__(self, output_channel):
        super(GRCL_unit, self).__init__()
        self.wgf_bn = nn.BatchNorm2d(output_channel)
        self.wgr_bn = nn.BatchNorm2d(output_channel)
        self.wf_bn = nn.BatchNorm2d(output_channel)
        self.wx_bn = nn.BatchNorm2d(output_channel)
        self.gx_bn = nn.BatchNorm2d(output_channel)

    def forward(self, wgf, wgx, wf, wx):
        G = F.sigmoid(self.wgf_bn(wgf) + self.wgr_bn(wgx))
        wf = self.wf_bn(wf)
        wx = self.wx_bn(wx)

        x = F.relu(wf + self.gx_bn(wx * G))

        return x


class ReshapeLayer(nn.Module):
    def __init__(self):
        super(ReshapeLayer, self).__init__()

    def forward(self, inputs):
        B, C, H, W = inputs.size()
        inputs = inputs.reshape(B, C, H * W)
        inputs = inputs.permute(0, 2, 1)
        return inputs


class SequenceLayer(nn.Module):
    def __init__(self, num_inputs, num_hiddens):
        super(SequenceLayer, self).__init__()
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.rnn_1 = nn.LSTM(self.num_inputs, self.num_hiddens, bidirectional=True, batch_first=True)
        self.rnn_2 = nn.LSTM(self.num_hiddens, self.num_hiddens, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(self.num_hiddens * 2, self.num_hiddens)

    def forward(self, inputs):
        self.rnn_1.flatten_parameters()
        recurrent, _ = self.rnn_1(inputs)
        inputs = self.linear(recurrent)
        self.rnn_2.flatten_parameters()
        recurrent, _ = self.rnn_2(inputs)
        outputs = self.linear(recurrent)
        return outputs
