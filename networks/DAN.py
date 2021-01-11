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
from character import AttnLabelConverter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DAN(nn.Module):
    def __init__(self, flags):
        super(DAN, self).__init__()
        self.input_shape = flags.Global.image_shape
        self.inplanes = self.input_shape[0]
        self.strides = [(2,2), (1,1), (2,2), (1,1), (1,1)]
        self.compress_layer = flags.Architecture.compress_layer
        self.block = BasicBlock
        self.layers = flags.Architecture.layers
        self.maxT = flags.Global.batch_max_length
        self.depth = flags.CAM.depth
        self.num_channel = flags.CAM.num_channel
        self.converter = AttnLabelConverter(flags)
        self.num_class = self.converter.char_num

        self.feature_extractor = ResNet(self.inplanes, self.block, self.strides,
                                        self.layers, self.compress_layer)
        self.scales = Feature_Extractor(self.input_shape, self.block, self.strides, 
                                        self.layers, self.compress_layer).Iwantshapes()
        self.cam_module = CAM(self.scales, self.maxT, self.depth, self.num_channel)
        self.decoder = DTD(self.num_class, self.num_channel)

    def forward(self, inputs, text):
        output_features = self.feature_extractor(inputs)
        x = self.cam_module(output_features)
        outputs = self.decoder(output_features[-1], x, text)
        return outputs


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, inputs):
        residual = inputs
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        if self.downsample is not None:
            residual = self.downsample(residual)
        outputs = residual + x
        outputs = self.relu(outputs)

        return outputs


class ResNet(nn.Module):
    def __init__(self, inplanes, block, strides, layers, compress_layer=True):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, 32, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.inplanes = 32
        self.layer1 = self._make_layer(32, block, strides[0], layers[0])
        self.layer2 = self._make_layer(64, block, strides[1], layers[1])
        self.layer3 = self._make_layer(128, block, strides[2], layers[2])
        self.layer4 = self._make_layer(256, block, strides[3], layers[3])
        self.layer5 = self._make_layer(512, block, strides[4], layers[4])

        self.compress_layer = compress_layer        
        if compress_layer:
            # for handwritten
            self.layer6 = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=(3, 1), padding=(0, 0), stride=(1, 1)),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace = True))

    def _make_layer(self, planes, block, stride, num_layer):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample))
        self.inplanes = planes
        for i in range(1, num_layer):
            layers.append(block(self.inplanes, planes))
        layers = nn.Sequential(*layers)
        return layers

    def forward(self, inputs):
        output_features = []
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        tmp_size = x.size()[2:]
        x = self.layer1(x)
        if x.size()[2:] != tmp_size:
            tmp_size = x.size()[2:]
            output_features.append(x)
        x = self.layer2(x)
        if x.size()[2:] != tmp_size:
            tmp_size = x.size()[2:]
            output_features.append(x)
        x = self.layer3(x)
        if x.size()[2:] != tmp_size:
            tmp_size = x.size()[2:]
            output_features.append(x)
        x = self.layer4(x)
        if x.size()[2:] != tmp_size:
            tmp_size = x.size()[2:]
            output_features.append(x)
        x = self.layer5(x)
        if x.size()[2:] != tmp_size:
            tmp_size = x.size()[2:]
            output_features.append(x)
        if not self.compress_layer:
            output_features.append(x)
        else:
            if x.size()[2:] != tmp_size:
                tmp_size = x.size()[2:]
                output_features.append(x)
            x = self.layer6(x)
            output_features.append(x)
        return output_features


class Feature_Extractor(nn.Module):
    def __init__(self, input_shape, block, strides, layers, compress_layer=True):
        super(Feature_Extractor, self).__init__()
        self.model = ResNet(input_shape[0], block, strides, layers, compress_layer)
        self.input_shape = input_shape

    def forward(self, input):
        features = self.model(input)
        return features

    def Iwantshapes(self):
        pseudo_input = torch.rand(1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        features = self.model(pseudo_input)
        return [feat.size()[1:] for feat in features]



class CAM(nn.Module):
    def __init__(self, scales, maxT, depth, num_channel):
        super(CAM, self).__init__()
        fpn = []
        for i in range(1, len(scales)):
            assert (scales[i-1][1] / scales[i][1]) % 1 == 0, 'layer scale error from {} to {}'.format(i-1, i)
            assert (scales[i-1][2] / scales[i][2]) % 1 == 0, 'layer scale error from {} to {}'.format(i-1, i)
            ksize = [3, 3, 5]
            r_h = int(scales[i-1][1] / scales[i][1])
            r_w = int(scales[i-1][2] / scales[i][2])
            k_h = 1 if scales[i-1][1] == 1 else ksize[r_h-1]
            k_w = 1 if scales[i-1][2] == 1 else ksize[r_w-1]
            fpn.append(nn.Sequential(
                nn.Conv2d(scales[i-1][0], scales[i][0],
                    kernel_size=(k_h, k_w),
                    stride=(r_h, r_w),
                    padding=(int((k_h-1)/2), int((k_w-1)/2))),
                nn.BatchNorm2d(scales[i][0]),
                nn.ReLU(True)
            ))
        self.fpn = nn.Sequential(*fpn)

        assert depth % 2 == 0
        in_shape = scales[-1]
        conv_ksizes = []
        strides = []
        deconv_ksizes = []
        h, w = in_shape[1], in_shape[2]
        for i in range(0, int(depth / 2)):
            stride = [2] if 2 ** (depth / 2 - i) <= h else [1]
            stride = stride + [2] if 2 ** (depth / 2 - i) <= w else stride + [1]
            strides.append(stride)
            conv_ksizes.append([3, 3])
            deconv_ksizes.append([_ ** 2 for _ in stride])
        convs = [nn.Sequential(
            nn.Conv2d(in_shape[0], num_channel,
                      kernel_size=tuple(conv_ksizes[0]),
                      stride=tuple(strides[0]),
                      padding=(int((conv_ksizes[0][0]-1)/2), int((conv_ksizes[0][1]-1)/2))),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(True))]
        for i in range(1, int(depth / 2)):
            convs.append(nn.Sequential(
                nn.Conv2d(num_channel, num_channel,
                        kernel_size=tuple(conv_ksizes[i]),
                        stride=tuple(strides[i]),
                        padding=(int((conv_ksizes[i][0]-1)/2), int((conv_ksizes[i][1]-1)/2))),
                nn.BatchNorm2d(num_channel),
                nn.ReLU(True))
            )
        self.convs = nn.Sequential(*convs)
        deconvs = []
        for i in range(1, int(depth / 2)):
            deconvs.append(nn.Sequential(
                nn.ConvTranspose2d(num_channel, num_channel,
                                   kernel_size=tuple(deconv_ksizes[int(depth/2)-i]),
                                   stride=tuple(strides[int(depth/2)-i]),
                                   padding=(int(deconv_ksizes[int(depth/2)-i][0]/4.), int(deconv_ksizes[int(depth/2)-i][1]/4.))),
                nn.BatchNorm2d(num_channel),
                nn.ReLU(True)
            ))
        deconvs.append(nn.Sequential(
                nn.ConvTranspose2d(num_channel, maxT,
                                   kernel_size=tuple(deconv_ksizes[0]),
                                   stride=tuple(strides[0]),
                                   padding=(int(deconv_ksizes[0][0]/4.), int(deconv_ksizes[0][1]/4.))),
                nn.Sigmoid()
            ))
        self.deconvs = nn.Sequential(*deconvs)

    def forward(self, inputs):
        x = inputs[0]
        for i in range(len(self.fpn)):
            x = self.fpn[i](x) + inputs[i+1]
        conv_layers = []
        for i in range(len(self.convs)):
            x = self.convs[i](x)
            conv_layers.append(x)
        for i in range(0, len(self.deconvs)-1):
            x = self.deconvs[i](x) + conv_layers[len(conv_layers) - 2 - i]
        x = self.deconvs[-1](x)

        return x


class DTD(nn.Module):
    # LSTM DTD
    def __init__(self, nclass, nchannel, dropout = 0.3):
        super(DTD,self).__init__()
        self.num_classes = nclass
        self.nchannel = nchannel
        self.pre_lstm = nn.LSTM(nchannel, int(nchannel / 2), bidirectional=True, batch_first=True)
        self.rnn = nn.GRUCell(nchannel + self.num_classes, nchannel)
        self.generator = nn.Sequential(
                            nn.Dropout(p = dropout),
                            nn.Linear(nchannel, nclass)
                        )

    def _char_one_hot(self, input_char, onehot_dim):
        input_char = input_char.unsqueeze(1)
        batch_size = input_char.size(0)
        one_hot = torch.FloatTensor(batch_size, onehot_dim).zero_().to(device)
        one_hot = one_hot.scatter_(1, input_char, 1)
        return one_hot

    def forward(self, feature, A, text, test = False):
        nB, nC, nH, nW = feature.size()
        nT = A.size()[1]
        # Normalize
        A = A / A.view(nB, nT, -1).sum(2).view(nB,nT,1,1)
        # weighted sum
        C = feature.view(nB, 1, nC, nH, nW) * A.view(nB, nT, 1, nH, nW)
        C = C.view(nB,nT,nC,-1).sum(3)
        self.pre_lstm.flatten_parameters()
        C, _ = self.pre_lstm(C)
        C = F.dropout(C, p = 0.3, training=self.training)
        if not test:
            num_steps = text.size(1)

            gru_res = torch.zeros(nB, num_steps, self.nchannel).type_as(C.data)        
            hidden = torch.zeros(nB, self.nchannel).type_as(C.data)
            
            for i in range(0, num_steps):
                prev_emb = self._char_one_hot(text[:, i], self.num_classes)
                hidden = self.rnn(torch.cat((C[:, i, :], prev_emb), dim = 1),
                                 hidden)
                gru_res[:, i, :] = hidden
            probs = self.generator(gru_res)
        else:
            targets = torch.FloatTensor(nB, self.num_classes).zero_().type_as(C.data)
            probs = torch.FloatTensor(nB, num_steps, self.num_classes).zero_().type_as(C.data)

            for i in range(num_steps):
                one_hot = self._char_one_hot(targets, self.num_classes)
                hidden = self.rnn(torch.cat((C[:, i, :], prev_emb), dim = 1),
                                 hidden)
                prob = self.generator(hidden)
                probs[:, i, :] = prob
                _, next_input = prob.max(axis=1)
                targets = next_input
        
        return probs

            



