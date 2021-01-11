# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn
import torch


class CTCLoss(nn.Module):
    def __init__(self, params, reduction='mean'):
        super().__init__()
        blank_idx = params.blank_idx
        self.loss_func = torch.nn.CTCLoss(blank=blank_idx, reduction=reduction, zero_infinity=True)

    def forward(self, pred, args):
        batch_size = pred.size(0)
        label, label_length = args['targets'], args['targets_lengths']
        pred = pred.log_softmax(2)
        pred = pred.permute(1, 0, 2)
        preds_lengths = torch.tensor([pred.size(0)] * batch_size, dtype=torch.long)
        loss = self.loss_func(pred, label.cuda(), preds_lengths.cuda(), label_length.cuda())
        return loss


class AttnLoss(nn.Module):
    def __init__(self, params):
        super(AttnLoss, self).__init__()
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, pred, args):
        label, label_length = args['targets'], args['targets_lengths']
        label = label[:, 1:]
        loss = self.loss_func(pred.view(-1, pred.size(-1)), label.contiguous().view(-1))
        return loss
        