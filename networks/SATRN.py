# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
__dir__ = os.path.abspath(os.path.dirname(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from character import AttnLabelConverter
from program import build_config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SATRN(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, flags):
        super(SATRN, self).__init__()
        self.inplanes = 1 if flags.Global.image_shape[0] == 1 else 3
        self.converter = AttnLabelConverter(flags)
        self.num_classes = self.converter.char_num
        self.d_model = flags.Transformer.model_dims
        self.d_ff = flags.Transformer.feedforward_dims
        self.num_encoder = flags.Transformer.num_encoder
        self.num_decoder = flags.Transformer.num_decoder
        self.h = flags.Transformer.num_head
        self.dropout = flags.Transformer.dropout_rate

        c = copy.deepcopy
        self.attn = MultiHeadedAttention(self.h, self.d_model)
        self.laff = LocalityAwareFeedForward(self.d_model, self.d_ff, self.d_model)
        self.ff = PointwiseFeedForward(self.d_model, self.d_ff)
        self.position1d = PositionalEncoding(self.d_model, self.dropout)
        self.position2d = A2DPE(self.d_model, self.dropout)

        self.encoder = Encoder(EncoderLayer(self.d_model, c(self.attn), c(self.laff), self.dropout), 
                            self.num_encoder)
        self.decoder = Decoder(DecoderLayer(self.d_model, c(self.attn), c(self.attn), 
                             c(self.ff), self.dropout), self.num_decoder)
        self.src_embed = nn.Sequential(
                            ShallowCNN(self.inplanes, self.d_model),
                            self.position2d)
        self.tgt_embed = nn.Sequential(
                            Embeddings(self.num_classes, self.d_model),
                            self.position1d)
        self.generator = Generator(self.d_model, self.num_classes)
        
    def forward(self, src, tgt):
        "Take in and process masked src and target sequences."
        features = self.decode(self.encode(src), tgt)
        outputs = self.generator(features)
        return outputs
    
    def encode(self, src):
        return self.encoder(self.src_embed(src))
    
    def decode(self, memory, tgt):
        return self.decoder(self.tgt_embed(tgt), memory)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return self.proj(x)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory):
        for layer in self.layers:
            x = layer(x, memory)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m))
        return self.sublayer[2](x, self.feed_forward)


class ShallowCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ShallowCNN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, inputs):
        return self.feature_extractor(inputs)


# 1D positional encoding
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).to(device)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# adaptive 2D positional encoding
class A2DPE(nn.Module):
    def __init__(self, d_model, dropout, max_len=1000):
        super(A2DPE, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

        pe_h = torch.zeros(max_len, d_model).to(device)
        pe_w = torch.zeros(max_len, d_model).to(device)
        div_term = torch.exp(torch.arange(0, 2, d_model) * -math.log(10000) / d_model)
        position_h = torch.arange(0, max_len).unsqueeze(1)
        position_w = torch.arange(0, max_len).unsqueeze(1)
        pe_h[:, 0::2] = torch.sin(position_h * div_term)
        pe_h[:, 1::2] = torch.cos(position_h * div_term)
        pe_w[:, 0::2] = torch.sin(position_w * div_term)
        pe_w[:, 1::2] = torch.cos(position_w * div_term)
        self.register_buffer('pe_h', pe_h)
        self.register_buffer('pe_w', pe_w)
        
        self.pool_h = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, 1))
        self.transform_h = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        self.transform_w = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

    def forward(self, x):
        # tensor shape
        nbatches = x.size(0)
        pe_h = self.pe_h[:x.size(2), :].unsqueeze(0).unsqueeze(2)
        pe_w = self.pe_w[:x.size(3), :].unsqueeze(0).unsqueeze(1)
        x_h = self.pool_h(x).permute(0, 2, 3, 1)
        x_w = self.pool_w(x).permute(0, 2, 3, 1)
        alpha = self.transform_h(x_h)
        beta = self.transform_w(x_w)
        pe = alpha * pe_h + beta * pe_w
        outputs = (x.permute(0, 2, 3, 1) + pe).view(nbatches, -1, self.d_model)
        return outputs


def attention(query, key, value, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
                / np.sqrt(d_k)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_model = d_model
        self.d_k = d_model // h
        self.h = h
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.linears = clones(nn.Linear(d_model, d_model), 4)

    def forward(self, query, key, value):
        nbatches = query.size(0)
        # reshape query, key, value and apply multiheadedattention 
        # along h_dim and w_dim
        query = query.view(nbatches, -1, self.d_model)
        key = key.view(nbatches, -1, self.d_model)
        value = value.view(nbatches, -1, self.d_model)

        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) \
                                for l, x in zip(self.linears, (query, key, value))]
        
        x, self.attn = attention(query, key, value, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        x = self.linears[-1](x)

        return x


class LocalityAwareFeedForward(nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels):
        super(LocalityAwareFeedForward, self).__init__()
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inter_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        nbatches = x.size(0)
        x = x.permute(0, 2, 1).unsqueeze(-1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        outputs = self.relu(x)
        outputs = outputs.squeeze(-1).transpose(1, 2)

        return outputs


class PointwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PointwiseFeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))


class Embeddings(nn.Module):
    def __init__(self, vocab, d_model):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) / np.sqrt(self.d_model)




        
