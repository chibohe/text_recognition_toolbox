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


class FAN(nn.Module):
    def __init__(self, flags):
        super(FAN, self).__init__()


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
        one_hot = torch.FloatTensor(batch_size, onehot_dim).zero_()
        one_hot = one_hot.scatter_(1, input_char, 1)
        return one_hot

    def forward(self, batch_H, text, is_train=True):
        batch_size = batch_H.size(0)
        num_steps = text.size(1) - 1

        output_hiddens = torch.FloatTensor(batch_size, num_steps, self.num_classes).zero_()
        hidden = (torch.FloatTensor(batch_size, self.hidden_size).zero_(),
                  torch.FloatTensor(batch_size, self.hidden_size).zero_())
        
        if is_train:
            for i in range(num_steps):
                one_hot = self._char_one_hot(text[:, i], self.num_classes)
                hidden, alpha = self.attention_cell(hidden, batch_H, one_hot)
                output_hiddens[:, i, :] = hidden[0]
                probs = self.generator(output_hiddens)
        else:
            targets = torch.FloatTensor(batch_size, self.num_classes).zero_()
            probs = torch.FloatTensor(batch_size, num_steps, self.num_classes).zero_()
            
            for i in range(num_steps):
                one_hot = self._char_one_hot(targets, self.num_classes)
                hidden, alpha = self.attention_cell(hidden, batch_H, one_hot)
                prob = self.generator(hidden[0])
                probs[:, i, :] = prob
                _, next_input = prob.max(axis=1)
                targets = next_input
        
        return probs


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






