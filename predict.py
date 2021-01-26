# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import time
import shutil
import traceback
import yaml
import logging
import math
import sys

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from program import build_config, build_model, build_optimizer, \
                          build_pretrained_weights, build_device
from character import CTCLabelConverter, AttnLabelConverter


class Recoginizer(object):
    def __init__(self):
        config = build_config()
        model = build_model(config)
        device, gpu_count = build_device(config)
        optimizer = build_optimizer(config, model)
        if gpu_count > 1:
            model = nn.DataParallel(model)
        model, optimizer, global_state = build_pretrained_weights(config, model, optimizer)
        self.device = device
        if config.Global.loss_type == 'ctc':
            self.converter = CTCLabelConverter(config)
        else:
            self.converter = AttnLabelConverter(config)
        self.model = model.to(self.device)

        self.keep_ratio_with_pad = config.TrainReader.padding
        self.channel = config.Global.image_shape[0]
        self.imgH = config.Global.image_shape[1]
        self.imgW = config.Global.image_shape[2]

    def preprocess(self, image):
        self.transform = transforms.ToTensor()

        if self.keep_ratio_with_pad:
            w, h = image.size
            ratio = w / float(h)
            if math.ceil(ratio * self.imgH) > self.imgW:
                resized_image = image.resize((self.imgW, self.imgH), Image.BICUBIC)
                resized_image = self.transform(resized_image)
                imgP = resized_image.sub(0.5).div(0.5)
            else:
                resized_W = math.ceil(ratio * self.imgH)
                resized_image = image.resize((resized_W, self.imgH), Image.BICUBIC)
                resized_image = self.transform(resized_image)
                resized_image = resized_image.sub(0.5).div(0.5)

                c, h, w = resized_image.size()
                imgP = torch.FloatTensor(*(self.channel, self.imgH, self.imgW)).fill_(0)
                imgP[:, :, :w] = resized_image
                imgP[:, :, w:] = resized_image[:, :, w - 1].unsqueeze(2).expand(c, h, self.imgW - w)
        else:
            resized_image = image.resize((self.imgW, self.imgH), Image.BICUBIC)
            resized_image = self.transform(resized_image)
            imgP = resized_image.sub(0.5).div(0.5)

        imgP = imgP.unsqueeze(0)
        return imgP

    def predict(self, image_tensor):
        self.model.eval()
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            outputs = self.model(image_tensor)
            outputs = outputs.softmax(dim=2).detach().cpu().numpy()
            preds_str = self.converter.decode(outputs)

        return preds_str

    def __call__(self, image):
        image_tensor = self.preprocess(image)
        preds_str = self.predict(image_tensor)

        return preds_str



if __name__ == "__main__":
    config = build_config()
    text_recognizer = Recoginizer()
    img_path = config.Global.infer_img

    start_time = time.time()
    if os.path.isdir(img_path):
        for file in os.listdir(img_path):
            if file.endswith('jpg') or file.endswith('jpeg') \
                or file.endswith('png'):
                print(f'当前处理的图片是: {file}')
                img_file_path = os.path.join(img_path, file)
                image = Image.open(img_file_path)
                if config.Global.image_shape[0] == 1:
                    image = image.convert('L')
                else:
                    image = image.convert('RGB')
                preds_str = text_recognizer(image)
                print(f'识别结果是: {preds_str}')
        end_time = time.time()
        print(f'一共耗时为: {end_time - start_time}')
    elif os.path.isfile(img_path):
        file = os.path.basename(img_path)
        if file.endswith('jpg') or file.endswith('jpeg') \
                or file.endswith('png'):
            print(f'当前处理的图片是: {file}')
            image = Image.open(img_path)
            if config.Global.image_shape[0] == 1:
                image = image.convert('L')
            else:
                image = image.convert('RGB')
            preds_str = text_recognizer(image)
            print(f'识别结果是: {preds_str}')

