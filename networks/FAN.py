# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))
from character import AttnLabelConverter
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FAN(nn.Module):
    def __init__(self, flags):
        super(FAN, self).__init__()
        self.inplanes = 1 if flags.Global.image_shape[0] == 1 else 3
        self.num_inputs = flags.SeqRNN.input_size
        self.num_hiddens = flags.SeqRNN.hidden_size
        self.converter = AttnLabelConverter(flags)
        self.num_classes = self.converter.char_num

        self.block = BasicBlock
        self.layers = flags.Architecture.layers
        self.feature_extractor = ResNet(self.inplanes, self.num_inputs, self.block, self.layers)
        self.reshape_layer = ReshapeLayer()
        self.sequence_layer = Attention(self.num_inputs, self.num_hiddens, self.num_classes)

    def forward(self, inputs, text):
        outputs = self.feature_extractor(inputs)
        outputs = self.reshape_layer(outputs)
        outputs = self.sequence_layer(outputs, text)
        return outputs


class Attention(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Attention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.generator = nn.Linear(hidden_size, num_classes)
        self.attention_cell = AttentionCell(input_size, hidden_size, num_classes)

    def _char_one_hot(self, input_char, onehot_dim):
        input_char = input_char.unsqueeze(1)
        batch_size = input_char.size(0)
        one_hot = torch.FloatTensor(batch_size, onehot_dim).zero_().to(device)
        one_hot = one_hot.scatter_(1, input_char, 1)
        return one_hot

    def forward(self, batch_H, text, is_train=True):
        batch_size = batch_H.size(0)
        num_steps = text.size(1)

        output_hiddens = torch.FloatTensor(batch_size, num_steps, self.hidden_size).zero_().to(device)
        hidden = (torch.FloatTensor(batch_size, self.hidden_size).zero_().to(device),
                  torch.FloatTensor(batch_size, self.hidden_size).zero_().to(device))
        
        if is_train:
            for i in range(num_steps):
                one_hot = self._char_one_hot(text[:, i], self.num_classes)
                hidden, alpha = self.attention_cell(hidden, batch_H, one_hot)
                output_hiddens[:, i, :] = hidden[0]
                probs = self.generator(output_hiddens)
        else:
            targets = torch.FloatTensor(batch_size, self.num_classes).zero_().to(device)
            probs = torch.FloatTensor(batch_size, num_steps, self.num_classes).zero_().to(device)
            
            for i in range(num_steps):
                one_hot = self._char_one_hot(targets, self.num_classes)
                hidden, alpha = self.attention_cell(hidden, batch_H, one_hot)
                prob = self.generator(hidden[0])
                probs[:, i, :] = prob
                _, next_input = prob.max(axis=1)
                targets = next_input
        
        return probs


class ReshapeLayer(nn.Module):
    def __init__(self):
        super(ReshapeLayer, self).__init__()

    def forward(self, inputs):
        B, C, H, W = inputs.size()
        inputs = inputs.reshape(B, C, H * W)
        inputs = inputs.permute(0, 2, 1)
        return inputs


class AttentionCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_embeddings):
        super(AttentionCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size
        self.rnn = nn.LSTMCell(input_size + num_embeddings, hidden_size)

    def forward(self, prev_hidden, batch_H, one_hot):
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = self.h2h(prev_hidden[0]).unsqueeze(1)
        e = self.score(torch.tanh(batch_H_proj + prev_hidden_proj))

        alpha = F.softmax(e, dim=1)
        context = torch.bmm(alpha.permute(0, 2, 1), batch_H).squeeze(1)
        concat_context = torch.cat([context, one_hot], dim=1)
        cur_hidden = self.rnn(concat_context, prev_hidden)

        return cur_hidden, alpha


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = self._conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = self._conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def _conv3x3(self, inplanes, planes):
        return nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                         padding=1)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual
        x = self.relu(x)

        return x


class ResNet(nn.Module):
    def __init__(self, input_channel, output_channel, block, layers):
        super(ResNet, self).__init__()
        self.output_channel_blocks = [int(output_channel / 4), int(output_channel / 2), output_channel, output_channel]
        self.inplanes = int(output_channel / 8)

        self.conv0_1 = nn.Conv2d(input_channel, int(output_channel / 16), 
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(int(output_channel / 16))
        self.relu = nn.ReLU(inplace=True)
        self.conv0_2 = nn.Conv2d(int(output_channel / 16), int(output_channel / 8),
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0_2 = nn.BatchNorm2d(int(output_channel / 8))
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.layer1 = self._make_layer(block, self.output_channel_blocks[0], layers[0])
        self.conv1 = nn.Conv2d(self.output_channel_blocks[0], self.output_channel_blocks[0],
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.output_channel_blocks[0])
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.layer2 = self._make_layer(block, self.output_channel_blocks[1], layers[1])
        self.conv2 = nn.Conv2d(self.output_channel_blocks[1], self.output_channel_blocks[1],
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.output_channel_blocks[1])
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1))
        self.layer3 = self._make_layer(block, self.output_channel_blocks[2], layers[2])
        self.conv3 = nn.Conv2d(self.output_channel_blocks[2], self.output_channel_blocks[2],
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.output_channel_blocks[2])
        self.layer4 = self._make_layer(block, self.output_channel_blocks[3], layers[3])
        self.conv4 = nn.Conv2d(self.output_channel_blocks[3], self.output_channel_blocks[3],
                               kernel_size=(2, 2), stride=(2, 1), padding=(0, 1), bias=False)
        self.bn4 = nn.BatchNorm2d(self.output_channel_blocks[3])
        self.conv5 = nn.Conv2d(self.output_channel_blocks[3], self.output_channel_blocks[3],
                               kernel_size=(2, 2), stride=(1, 1), padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(self.output_channel_blocks[3])

    def _make_layer(self, block, planes, blocks):
        downsample = None
        if self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(planes)
            )

        layers = []
        layers.append(block(self.inplanes, planes, downsample))
        self.inplanes = planes
        for i in range(blocks):
            layers.append(block(self.inplanes, planes))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv0_1(x)
        x = self.bn0_1(x)
        x = self.relu(x)
        x = self.conv0_2(x)
        x = self.bn0_2(x)
        x = self.relu(x)

        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.maxpool2(x)
        x = self.layer2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.maxpool3(x)
        x = self.layer3(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.layer4(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        return x






