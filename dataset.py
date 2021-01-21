# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import re
import six
import math
import lmdb
import cv2
import random
import torch
import logging

from natsort import natsorted
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, ConcatDataset, Subset, DataLoader
from torch._utils import _accumulate
import torchvision.transforms as transforms
from utils import get_characters


class BatchBalancedDataset(object):

    def __init__(self, params):
        """
        Modulate the data ratio in the batch.
        For example, when select_data is "MJ-ST" and batch_ratio is "0.5-0.5",
        the 50% of the batch is filled with MJ and the other 50% of the batch is filled with ST.
        """
        self.train_data = params.TrainReader.lmdb_sets_dir
        self.select_data = params.TrainReader.select_data.split('-')
        self.batch_ratio = params.TrainReader.batch_ratio.split('-')
        self.total_data_usage_ratio = params.TrainReader.total_data_usage_ratio
        self.keep_ratio_with_pad = params.TrainReader.padding
        self.batch_size = params.TrainReader.batch_size
        self.data_augment = params.TrainReader.augment
        self.imgH = params.Global.image_shape[1]
        self.imgW = params.Global.image_shape[2]
        self.workers = params.TrainReader.num_workers
        self.characters = get_characters(params.Global.character_dict_path)
        dashed_line = '-' * 80
        logging.info(dashed_line + '\n')
        logging.info(f'dataset_root: {self.train_data}\nopt.select_data: {self.select_data}\nopt.batch_ratio: {self.batch_ratio}\n')
        assert len(self.select_data) == len(self.batch_ratio)

        _AlignCollate = AlignCollate(imgH=self.imgH, imgW=self.imgW, keep_ratio_with_pad=self.keep_ratio_with_pad, data_augment=self.data_augment)
        self.data_loader_list = []
        self.dataloader_iter_list = []
        batch_size_list = []
        Total_batch_size = 0
        for selected_d, batch_ratio_d in zip(self.select_data, self.batch_ratio):
            _batch_size = max(round(self.batch_size * float(batch_ratio_d)), 1)
            logging.info(dashed_line + '\n')
            _dataset, _dataset_log = hierarchical_dataset(root=self.train_data, characters=self.characters, params=params, select_data=[selected_d])
            total_number_dataset = len(_dataset)
            logging.info(_dataset_log)
            """
            The total number of data can be modified with opt.total_data_usage_ratio.
            ex) opt.total_data_usage_ratio = 1 indicates 100% usage, and 0.2 indicates 20% usage.
            See 4.2 section in our paper.
            """
            number_dataset = int(total_number_dataset * float(self.total_data_usage_ratio))
            dataset_split = [number_dataset, total_number_dataset - number_dataset]
            indices = range(total_number_dataset)
            _dataset, _ = [Subset(_dataset, indices[offset - length:offset])
                           for offset, length in zip(_accumulate(dataset_split), dataset_split)]
            selected_d_log = f'num total samples of {selected_d}: {total_number_dataset} x {self.total_data_usage_ratio} (total_data_usage_ratio) = {len(_dataset)}\n'
            selected_d_log += f'num samples of {selected_d} per batch: {self.batch_size} x {float(batch_ratio_d)} (batch_ratio) = {_batch_size}'
            logging.info(selected_d_log + '\n')
            batch_size_list.append(str(_batch_size))
            Total_batch_size += _batch_size

            _data_loader = torch.utils.data.DataLoader(
                _dataset, batch_size=_batch_size,
                shuffle=True,
                num_workers=int(self.workers),
                collate_fn=_AlignCollate, pin_memory=True)
            self.data_loader_list.append(_data_loader)
            self.dataloader_iter_list.append(iter(_data_loader))

        Total_batch_size_log = f'{dashed_line}\n'
        batch_size_sum = '+'.join(batch_size_list)
        Total_batch_size_log += f'Total_batch_size: {batch_size_sum} = {Total_batch_size}\n'
        Total_batch_size_log += f'{dashed_line}'
        self.batch_size = Total_batch_size

        logging.info(Total_batch_size_log + '\n')

    def get_batch(self):
        batch = {'img': [], 'label': []}

        for i, data_loader_iter in enumerate(self.dataloader_iter_list):
            try:
                image, text = data_loader_iter.next()
                batch['img'].append(image)
                batch['label'] += text
            except Exception:
                self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
                image, text = self.dataloader_iter_list[i].next()
                batch['img'].append(image)
                batch['label'] += text
            # except ValueError:
            #     pass

        batch['img'] = torch.cat(batch['img'], 0)

        return batch


def evaldataloader(params):
    root = params.EvalReader.lmdb_sets_dir
    select_data = params.EvalReader.select_data
    keep_ratio_with_pad = params.EvalReader.padding
    batch_size = params.EvalReader.batch_size
    imgH = params.Global.image_shape[1]
    imgW = params.Global.image_shape[2]
    num_workers = params.EvalReader.num_workers
    characters = get_characters(params.Global.character_dict_path)

    AlignCollate_valid = AlignCollate(imgH=imgH, imgW=imgW, keep_ratio_with_pad=keep_ratio_with_pad)
    valid_dataset, valid_dataset_log = hierarchical_dataset(root=root, characters=characters, params=params, select_data=[select_data])
    logging.info(valid_dataset_log)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size,
        shuffle=True,  # 'True' to check training progress with validation function.
        num_workers=int(num_workers),
        collate_fn=AlignCollate_valid, pin_memory=True)
    return valid_loader


def hierarchical_dataset(root, characters, params, select_data='/'):
    """ select_data='/' contains all sub-directory of root directory """
    dataset_list = []
    dataset_log = f'dataset_root:    {root}\t dataset: {select_data[0]}'
    dataset_log += '\n'
    for dirpath, dirnames, filenames in os.walk(root+'/'):
        if not dirnames:
            select_flag = False
            for selected_d in select_data:
                if selected_d in dirpath:
                    select_flag = True
                    break

            if select_flag:
                dataset = LmdbDataset(dirpath, characters, params)
                sub_dataset_log = f'sub-directory:\t/{os.path.relpath(dirpath, root)}\t num samples: {len(dataset)}'
                dataset_log += f'{sub_dataset_log}\n'
                dataset_list.append(dataset)

    concatenated_dataset = ConcatDataset(dataset_list)
    selected_d_log = f'num total samples of total dataset is {len(concatenated_dataset)}\n'
    dataset_log += f'{selected_d_log}\n'
    
    return concatenated_dataset, dataset_log


class LmdbDataset(Dataset):

    def __init__(self, root, characters, params):

        self.root = root
        self.params = params
        self.imgH = params.Global.image_shape[1]
        self.imgW = params.Global.image_shape[2]
        self.data_filtering_off = params.Global.data_filtering_off
        self.batch_max_length = params.Global.batch_max_length
        self.character = characters
        self.rgb = params.Global.image_shape[0] == 3
        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print('cannot create lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples

            if self.data_filtering_off:
                # for fast check or benchmark evaluation with no filtering
                self.filtered_index_list = [index + 1 for index in range(self.nSamples)]
            else:
                """ Filtering part
                If you want to evaluate IC15-2077 & CUTE datasets which have special character labels,
                use --data_filtering_off and only evaluate on alphabets and digits.
                see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L190-L192
                """
                self.filtered_index_list = []
                for index in range(self.nSamples):
                    index += 1  # lmdb starts with 1
                    label_key = 'label-%09d'.encode() % index
                    label = txn.get(label_key).decode('utf-8')

                    if len(label) >= self.batch_max_length:
                        # print(f'The length of the label is longer than max_length: length
                        # {len(label)}, {label} in dataset {self.root}')
                        continue

                    if '###' in label:
                        continue

                    # By default, images containing characters which are not in opt.character are filtered.
                    # You can add [UNK] token to `opt.character` in utils.py instead of this filtering.
                    out_of_char = f'[^{self.character}]'
                    if re.search(out_of_char, label.lower()):
                        continue

                    self.filtered_index_list.append(index)

                self.nSamples = len(self.filtered_index_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index = self.filtered_index_list[index]

        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                if self.rgb:
                    img = Image.open(buf).convert('RGB')  # for color image
                    width, height = img.size
                    # if height > width:
                    #     rotated_arr = np.rot90(np.array(img))
                    #     img = Image.fromarray(np.uint8(rotated_arr))
                else:
                    img = Image.open(buf).convert('L')
                    width, height = img.size
                    if height > width:
                        rotated_arr = np.rot90(np.array(img))
                        img = Image.fromarray(np.uint8(rotated_arr))
            except IOError:
                print(f'Corrupted image for {index}')
                # make dummy image and dummy label for corrupted image.
                if self.rgb:
                    img = Image.new('RGB', (self.imgW, self.imgH))
                else:
                    img = Image.new('L', (self.imgW, self.imgH))
                label = '[dummy_label]'

            # if not self.opt.sensitive:
            #     label = label.lower()

            # We only train and evaluate on alphanumerics (or pre-defined character set in train.py)
            out_of_char = f'[^{self.character}]'
            label = re.sub(out_of_char, '', label)

        return (img, label)


class RawDataset(Dataset):

    def __init__(self, root, params):
        self.params = params
        self.imgH = params.Global.image_shape[1]
        self.imgW = params.Global.image_shape[2]
        self.rgb = params.Global.image_shape == 3
        self.image_path_list = []
        for dirpath, dirnames, filenames in os.walk(root):
            for name in filenames:
                _, ext = os.path.splitext(name)
                ext = ext.lower()
                if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                    self.image_path_list.append(os.path.join(dirpath, name))

        self.image_path_list = natsorted(self.image_path_list)
        self.nSamples = len(self.image_path_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):

        try:
            if self.rgb:
                img = Image.open(self.image_path_list[index]).convert('RGB')  # for color image
            else:
                img = Image.open(self.image_path_list[index]).convert('L')

        except IOError:
            print(f'Corrupted image for {index}')
            # make dummy image and dummy label for corrupted image.
            if self.rgb:
                img = Image.new('RGB', (self.imgW, self.imgH))
            else:
                img = Image.new('L', (self.imgW, self.imgH))

        return (img, self.image_path_list[index])


class ResizeNormalize(object):

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class NormalizePAD(object):

    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

        return Pad_img


class AlignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False, data_augment=False):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad
        self.data_augment = data_augment

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)

        if self.data_augment:
            batch_images = []
            augment = DataAugment()
            for i, image in enumerate(images):
                image_arr = augment.apply(np.array(image))
                output_image = Image.fromarray(np.uint8(image_arr))
                batch_images.append(output_image)
            if len(batch_images) > 0:
                images = batch_images
            else:
                print('images length less than 0')

        if self.keep_ratio_with_pad:  # same concept with 'Rosetta' paper
            resized_max_w = self.imgW
            input_channel = 3 if images[0].mode == 'RGB' else 1
            transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

            resized_images = []
            for image in images:
                w, h = image.size
                ratio = w / float(h)
                if math.ceil(self.imgH * ratio) > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = math.ceil(self.imgH * ratio)

                resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
                resized_images.append(transform(resized_image))
                # resized_image.save('./image_test/%d_test.jpg' % w)

            image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

        else:
            transform = ResizeNormalize((self.imgW, self.imgH))
            image_tensors = [transform(image) for image in images]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors, labels


## 数据增强模块
class DataAugment(object):
    def __init__(self):
        pass

    def apply(self, img):
        funcs = [self.add_erode, self.add_dilate, self.apply_sp_noise, self.apply_gauss_blur,
                 self.apply_emboss, self.apply_sharp, self.apply_curve, self.affine_transform]

        if np.random.random() < 0.5:
            return img

        augment_func = np.random.choice(funcs)
        return augment_func(img)

    def add_erode(self, img):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        img = cv2.erode(img, kernel)
        return img

    def add_dilate(self, img):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        img = cv2.dilate(img, kernel)
        return img

    def apply_sp_noise(self, img):
        """
                Salt and pepper noise. Replaces random pixels with 0 or 255.
                """
        s_vs_p = 0.5
        amount = np.random.uniform(0.004, 0.01)
        out = np.copy(img)
        # Salt mode
        num_salt = np.ceil(amount * img.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in img.shape]
        out[coords] = 255.

        # Pepper mode
        num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in img.shape]
        out[coords] = 0
        return out

    def apply_gauss_blur(self, img, ks=None):
        if ks is None:
            ks = [3, 5]
        ksize = random.choice(ks)

        sigmas = [0, 1, 2, 3, 4, 5, 6, 7]
        sigma = 0
        if ksize <= 3:
            sigma = random.choice(sigmas)
        img = cv2.GaussianBlur(img, (ksize, ksize), sigma)
        return img

    def apply_emboss(self, img):
        emboss_kernal = np.array([
            [-2, -1, 0],
            [-1, 1, 1],
            [0, 1, 2]
        ])
        return cv2.filter2D(img, -1, emboss_kernal)

    def apply_sharp(self, img):
        sharp_kernel = np.array([
            [-1, -1, -1],
            [-1, 9, -1],
            [-1, -1, -1]
        ])
        return cv2.filter2D(img, -1, sharp_kernel)

    def apply_curve(self, img):
        bg_height = img.shape[0]
        bg_width = img.shape[1]
        word_height = bg_height - 6
        word_width = bg_width - 6

        text_x = int((bg_width - word_width) / 2)
        text_y = int((bg_height - word_height) / 2)

        text_box_pnts = [
            [text_x, text_y],
            [text_x + word_width, text_y],
            [text_x + word_width, text_y + word_height],
            [text_x, text_y + word_height]
        ]

        max_val = np.random.uniform(1, 5)

        h = img.shape[0]
        w = img.shape[1]

        img_x = np.zeros((h, w), np.float32)
        img_y = np.zeros((h, w), np.float32)

        xmin = text_box_pnts[0][0]
        xmax = text_box_pnts[1][0]
        ymin = text_box_pnts[0][1]
        ymax = text_box_pnts[2][1]

        remap_y_min = ymin
        remap_y_max = ymax

        def _remap_y(x, max_val):
            return int(max_val * np.math.sin(2 * 3.14 * x / 360))

        for y in range(h):
            for x in range(w):
                remaped_y = y + _remap_y(x, max_val)

                if y == ymin:
                    if remaped_y < remap_y_min:
                        remap_y_min = remaped_y

                if y == ymax:
                    if remaped_y > remap_y_max:
                        remap_y_max = remaped_y

                # 某一个位置的 y 值应该为哪个位置的 y 值
                img_y[y, x] = remaped_y
                # 某一个位置的 x 值应该为哪个位置的 x 值
                img_x[y, x] = x

        remaped_text_box_pnts = [
            [xmin, remap_y_min],
            [xmax, remap_y_min],
            [xmax, remap_y_max],
            [xmin, remap_y_max]
        ]

        # TODO: use cuda::remap
        dst = cv2.remap(img, img_x, img_y, cv2.INTER_CUBIC)
        return dst

    def affine_transform(self, image):
        """
            Conduct same affine transform for both image and polygon for data augmentation.
            """
        height, width = image.shape
        center_x, center_y = width / 2, height / 2

        angle = np.random.uniform(-0.5, 0.5)
        shear_x, shear_y = (np.random.uniform(-0.05, 0.05), np.random.uniform(-0.05, 0.05))

        rad = math.radians(angle)
        sin, cos = math.sin(rad), math.cos(rad)  # x, y
        abs_sin, abs_cos = abs(sin), abs(cos)

        new_width = ((height * abs_sin) + (width * abs_cos))
        new_height = ((height * abs_cos) + (width * abs_sin))

        new_width += np.abs(shear_y * new_height)
        new_height += np.abs(shear_x * new_width)

        new_width = int(new_width)
        new_height = int(new_height)

        M = np.array(
            [[cos, sin + shear_y, new_width / 2 - center_x + (1 - cos) * center_x - (sin + shear_y) * center_y],
             [-sin + shear_x, cos, new_height / 2 - center_y + (sin - shear_x) * center_x + (1 - cos) * center_y]])

        rotatedImage = cv2.warpAffine(image, M, (new_width, new_height), flags=cv2.INTER_CUBIC,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        return rotatedImage


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)
