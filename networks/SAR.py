# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))
from character import AttnLabelConverter
import torch
import torch.nn as nn
import torch.nn.functional as F

from program import build_config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SAR(nn.Module):
    def __init__(self, flags):
        super(SAR, self).__init__()
        self.inplanes = 1 if flags.Global.image_shape[0] == 1 else 3
        self.input_size = flags.SeqRNN.input_size
        self.en_hidden_size = flags.SeqRNN.en_hidden_size
        self.de_hidden_size = flags.SeqRNN.de_hidden_size
        self.converter = AttnLabelConverter(flags)
        self.num_classes = self.converter.char_num

        self.block = BasicBlock
        self.layers = flags.Architecture.layers
        self.feature_extractor = ResNet(self.inplanes, self.input_size, self.block, self.layers)
        self.lstm_encoder = LSTMEncoder(self.input_size, self.en_hidden_size)
        self.lstm_decoder = LSTMDecoder(self.input_size, self.en_hidden_size, self.de_hidden_size, self.num_classes)

    def forward(self, inputs, text):
        vis_features = self.feature_extractor(inputs)
        holistic_features = self.lstm_encoder(vis_features)
        outputs = self.lstm_decoder(vis_features, holistic_features, text)
        return outputs


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

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
    def __init__(self, in_channel, out_channel, block, layers):
        super(ResNet, self).__init__()
        self.out_channel_lst = [int(out_channel / 8), int(out_channel / 4), int(out_channel / 2), out_channel]
        self.input_planes = self.out_channel_lst[1]
        self.sequential_layer1 = nn.Sequential(
            nn.Conv2d(in_channel, self.out_channel_lst[0], 3, 1, 1),
            nn.BatchNorm2d(self.out_channel_lst[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channel_lst[0], self.out_channel_lst[1], 3, 1, 1),
            nn.BatchNorm2d(self.out_channel_lst[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.layer1 = self._make_layer(block, self.out_channel_lst[2], layers[0])
        self.sequential_layer2 = nn.Sequential(
            nn.Conv2d(self.out_channel_lst[2], self.out_channel_lst[2], 3, 1, 1),
            nn.BatchNorm2d(self.out_channel_lst[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.layer2 = self._make_layer(block, self.out_channel_lst[2], layers[1])
        self.sequential_layer3 = nn.Sequential(
            nn.Conv2d(self.out_channel_lst[2], self.out_channel_lst[2], 3, 1, 1),
            nn.BatchNorm2d(self.out_channel_lst[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )
        self.layer3 = self._make_layer(block, self.out_channel_lst[3], layers[2])
        self.sequential_layer4 = nn.Sequential(
            nn.Conv2d(self.out_channel_lst[3], self.out_channel_lst[3], 3, 1, 1),
            nn.BatchNorm2d(self.out_channel_lst[3]),
            nn.ReLU(inplace=True)
        )
        self.layer4 = self._make_layer(block, self.out_channel_lst[3], layers[3])
        self.sequential_layer5 = nn.Sequential(
            nn.Conv2d(self.out_channel_lst[3], self.out_channel_lst[3], 3, 1, 1),
            nn.BatchNorm2d(self.out_channel_lst[3]),
            nn.ReLU(inplace=True)
        )

    def _make_layer(self, block, planes, blocks):
        downsample = None
        if self.input_planes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.input_planes, planes, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(planes)
            )
        layers = []
        layers.append(block(self.input_planes, planes, downsample))
        self.input_planes = planes
        for i in range(1, blocks):
            layers.append(block(self.input_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, inputs):
        x = self.sequential_layer1(inputs)
        x = self.layer1(x)
        x = self.sequential_layer2(x)
        x = self.layer2(x)
        x = self.sequential_layer3(x)
        x = self.layer3(x)
        x = self.sequential_layer4(x)
        x = self.layer4(x)
        outputs = self.sequential_layer5(x)

        return outputs


class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_cell = nn.LSTMCell(input_size, hidden_size)
        self.avgpool = nn.AdaptiveAvgPool2d((None, 1))

    def forward(self, features):
        batch_size = features.size(0)
        num_steps = features.size(3)
        features = self.avgpool(features.permute(0, 1, 3, 2)).squeeze(3)
        hidden = (torch.FloatTensor(batch_size, self.hidden_size).zero_().to(device),
                  torch.FloatTensor(batch_size, self.hidden_size).zero_().to(device))
        for i in range(num_steps):
            hidden = self.sequence_cell(features[:, :, i], hidden)
        holistic_feature = hidden

        return holistic_feature


class LSTMDecoder(nn.Module):
    def __init__(self, input_size, en_hidden_size, de_hidden_size, num_classes):
        super(LSTMDecoder, self).__init__()
        self.input_size = input_size
        self.en_hidden_size = en_hidden_size
        self.de_hidden_size = de_hidden_size
        self.num_classes = num_classes
        self.generator = nn.Linear(input_size+de_hidden_size, num_classes)
        self.rnn = nn.LSTMCell(num_classes, de_hidden_size)
        self.attn_cell = AttentionCell(input_size, en_hidden_size, de_hidden_size)

    def _char_one_hot(self, input_char, onehot_dim):
        input_char = input_char.unsqueeze(1)
        batch_size = input_char.size(0)
        one_hot = torch.FloatTensor(batch_size, onehot_dim).zero_().to(device)
        one_hot = one_hot.scatter_(1, input_char, 1)
        return one_hot

    def forward(self, vis_features, holistic_features, text, is_train=True):
        batch_size = text.size(0)
        num_steps = text.size(1)

        output_hiddens = torch.FloatTensor(batch_size, num_steps, \
                            self.input_size+self.de_hidden_size).zero_().to(device)
        hidden = holistic_features
        if is_train:
            for i in range(num_steps):
                target = self._char_one_hot(text[:, i], self.num_classes)
                hidden = self.rnn(target, hidden)
                g = self.attn_cell(hidden[0], vis_features)
                output_hiddens[:, i, :] = torch.cat([hidden[0], g], dim=1)
            probs = self.generator(output_hiddens)
        else:
            probs = torch.FloatTensor(batch_size, num_steps, \
                        self.num_classes).zero_().to(device)
            target = torch.FloatTensor(batch_size, self.num_classes).zero_().to(device)
            for i in range(num_steps):
                hidden = self.rnn(target, hidden)
                g = self.attn_cell(hidden[0], vis_features)
                concat_feature = torch.cat([hidden[0], g], dim=1)
                prob = self.generator(concat_feature)
                probs[:, i, :] = prob
                _, next_input = prob.max(axis=1)
                target = self._char_one_hot(next_input)
        return probs
                

class AttentionCell(nn.Module):
    def __init__(self, input_size, en_hidden_size, de_hidden_size):
        super(AttentionCell, self).__init__()
        self.conv1 = nn.Conv2d(en_hidden_size, de_hidden_size, 1, 1, 0)
        self.conv2 = nn.Conv2d(input_size, de_hidden_size, 3, 1, 1)
        self.conv3 = nn.Conv2d(de_hidden_size, 1, 1, 1, 0)

    def forward(self, hidden, features):
        batch_size = features.size(0)
        num_channel = features.size(1)
        feature_h = features.size(2)
        feature_w = features.size(3)
        hidden = hidden.unsqueeze(2).unsqueeze(3)
        hidden = self.conv1(hidden)
        hidden = hidden.expand(-1, -1, feature_h, feature_w)
        de_features = self.conv2(features)
        e = self.conv3(torch.tanh(hidden + de_features))
        e = e.squeeze(1).view(-1, feature_h * feature_w)
        alpha = F.softmax(e, dim=1).view(batch_size, 1, feature_h, feature_w)
        out_feature = (features * alpha).view(batch_size, num_channel, -1).sum(2)
        return out_feature



if __name__ == "__main__":
    flags = build_config()
    test_tensor = torch.randn(10, 1, 32, 128).to(device)
    text = torch.randint(1000, (10, 40)).to(device)
    sar = SAR(flags).to(device)
    outputs = sar.forward(test_tensor, text)
    print(outputs.size())





            



        



    
        
