# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))
from character import CTCLabelConverter
import torch
import torch.nn as nn
import torch.nn.functional as F


class CRNN(nn.Module):
    def __init__(self, flags):
        super(CRNN, self).__init__()
        self.inplanes = 1 if flags.Global.image_shape[0] == 1 else 3
        self.num_inputs = flags.SeqRNN.input_size
        self.num_hiddens = flags.SeqRNN.hidden_size
        self.converter = CTCLabelConverter(flags)
        self.num_classes = self.converter.char_num

        self.feature_extractor = BackBone(self.inplanes)
        self.reshape_layer = ReshapeLayer()
        self.sequence_layer = SequenceLayer(self.num_inputs, self.num_hiddens)
        self.linear_layer = nn.Linear(self.num_hiddens, self.num_classes)

    def forward(self, inputs):
        x = self.feature_extractor(inputs)
        x = self.reshape_layer(x)
        x = self.sequence_layer(x)
        outputs = self.linear_layer(x)

        return outputs


class BackBone(nn.Module):
    def __init__(self, inplanes):
        super(BackBone, self).__init__()
        self.inplanes = inplanes
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(self.inplanes, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(512, 512, kernel_size=(2, 2), stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        return self.feature_extractor(inputs)


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





