# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import torch.nn as nn
from torch.optim import lr_scheduler
import random
import time
import shutil
import traceback
import logging
from tqdm import tqdm
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)

import torch
from utils import save_checkpoint, load_checkpoint, RecMetric, create_module
from character import CTCLabelConverter, AttnLabelConverter


class TrainerRec(object):
    def __init__(self, device, model, optimizer, loss, val_loader, \
                  train_loader, flags, global_state):
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss
        self.train_loader = train_loader
        self.eval_loader = val_loader
        self.to_use_device = device
        self.flags = flags.Global
        self.global_state = global_state
        if flags.Global.loss_type == 'ctc':
            self.converter = CTCLabelConverter(flags)
        else:
            self.converter = AttnLabelConverter(flags)

    def train(self):
        self.metric = RecMetric(self.converter)
        self.model = self.model.to(self.to_use_device)
        logging.info(self.to_use_device)
        logging.info('Training...')
        all_step = self.flags.num_iters
        if len(self.global_state) > 0:
            best_model = self.global_state['best_model']
            global_step = self.global_state['global_step']
        else:
            best_model = {'best_acc': 0, 'eval_loss': 0, 'eval_acc': 0, 'norm_edit_dis': 0}
            global_step = 0
        try:
            while True:
                self.model.train()
                start_time = time.time()
                batch_data = self.train_loader.get_batch()
                cur_batch_size = batch_data['img'].shape[0]
                targets, targets_lengths = self.converter.encode(batch_data['label'])
                batch_data['targets'] = targets
                batch_data['targets_lengths'] = targets_lengths
                batch_data['img'] = batch_data['img'].to(self.to_use_device)
                batch_data['targets'] = batch_data['targets'].to(self.to_use_device)
                
                self.optimizer.zero_grad()
                if self.flags.loss_type == 'ctc':
                    predicts = self.model.forward(batch_data['img'])
                else:
                    predicts = self.model.forward(batch_data['img'], batch_data['targets'][:, :-1])
                loss = self.loss_func(predicts, batch_data)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optimizer.step()

                acc_dict = self.metric(predicts, batch_data['label'])
                acc = acc_dict['n_correct'] / cur_batch_size
                norm_edit_dis = 1 - acc_dict['norm_edit_dis'] / cur_batch_size
                if (global_step + 1) % self.flags.print_batch_step == 0:
                    interval_batch_time = time.time() - start_time
                    logging.info(f"[{global_step + 1} / {all_step}] - "
                                f"loss:{loss:.4f} - "
                                f"acc:{acc:.4f} - "
                                f"norm_edit_dis:{norm_edit_dis:.4f} - "
                                f"interval_batch_time:{interval_batch_time:.4f} - ")
                if (global_step + 1) >= self.flags.eval_batch_step and (global_step + 1) % self.flags.eval_batch_step == 0:
                    self.global_state['global_step'] = global_step
                    eval_dict = self.evaluate()
                    if eval_dict['eval_acc'] > best_model['best_acc']:
                        best_model.update(eval_dict)
                        
                        self.global_state['best_model'] = best_model
                        model_save_path = f"{self.flags.save_model_dir}/best_acc.pth"
                        save_checkpoint(model_save_path, self.model, self.optimizer, global_state=self.global_state)
                    if not self.flags.highest_acc_save_type:
                        model_save_path = f"{self.flags.save_model_dir}/iter_{global_step + 1}.pth"
                        save_checkpoint(model_save_path, self.model, self.optimizer, global_state=self.global_state)
                    
                if global_step == self.flags.num_iters:
                    print('end the training')
                    raise StopIteration
                global_step += 1
        except KeyboardInterrupt:
            save_checkpoint(os.path.join(self.flags.save_model_dir, 'final.pth'), self.model, self.optimizer, global_state=self.global_state)
        except:
            error_msg = traceback.format_exc()
            logging.error(error_msg)
        finally:
            for k, v in best_model.items():
                logging.info(f'{k}: {v}')

    def evaluate(self):
        logging.info('start evaluate')
        self.model.eval()
        nums = 0
        result_dict = {'eval_loss': 0., 'eval_acc': 0., 'norm_edit_dis': 0.}
        show_str = []
        with torch.no_grad():
            for (img, label) in tqdm(self.eval_loader):
                batch_data = {}
                batch_data['img'], batch_data['label'] = img, label
                targets, targets_lengths = self.converter.encode(batch_data['label'])
                batch_data['targets'] = targets
                batch_data['targets_lengths'] = targets_lengths
                batch_data['img'] = batch_data['img'].to(self.to_use_device)
                batch_data['targets'] = batch_data['targets'].to(self.to_use_device)
                if self.flags.loss_type == 'ctc':
                    output = self.model.forward(batch_data['img'])
                else:
                    output = self.model.forward(batch_data['img'], batch_data['targets'][:, :-1])
                loss = self.loss_func(output, batch_data)

                nums += batch_data['img'].shape[0]
                acc_dict = self.metric(output, batch_data['label'])
                result_dict['eval_loss'] += loss.item()
                result_dict['eval_acc'] += acc_dict['n_correct']
                result_dict['norm_edit_dis'] += acc_dict['norm_edit_dis']
                show_str.extend(acc_dict['show_str'])

        result_dict['eval_loss'] /= len(self.eval_loader)
        result_dict['eval_acc'] /= nums
        result_dict['norm_edit_dis'] = 1 - result_dict['norm_edit_dis'] / nums
        logging.info(f"eval_loss:{result_dict['eval_loss']}")
        logging.info(f"eval_acc:{result_dict['eval_acc']}")
        logging.info(f"norm_edit_dis:{result_dict['norm_edit_dis']}")

        for s in show_str[:10]:
            logging.info(s)
        self.model.train()
        return result_dict


