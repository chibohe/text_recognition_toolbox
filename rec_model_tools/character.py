# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CTCLabelConverter(object):
    """ Convert between text-label and text-index for baidu warpctc """

    def __init__(self, flags):
        # character (str): set of the possible characters.
        flags = flags.Global
        self.character_type = flags.character_type
        self.loss_type = flags.loss_type
        if self.character_type == 'en':
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        elif self.character_type == 'ch':
            character_dict_path = flags.character_dict_path
            add_space = False
            if hasattr(flags, 'use_space_char'):
                add_space = flags.use_space_char
            self.character_str = ""
            with open(character_dict_path, 'rb') as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip("\n").strip("\r\n")
                    self.character_str += line
            if add_space:
                self.character_str += " "
            dict_character = list(self.character_str)
        elif self.character_type == "en_sensitive":
            # same with ASTER setting (use 94 char).
            self.character_str = string.printable[:-6]
            dict_character = list(self.character_str)
        else:
            self.character_str = None
        assert self.character_str is not None, \
            "Nonsupport type of the character: {}".format(self.character_str)
        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = ['[blank]'] + dict_character  # dummy '[CTCblank]' token for CTCLoss (index 0)
        self.char_num = len(self.character)

    def encode(self, text):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]
        # text = ''.join(text)
        # text = [self.dict[char] for char in text]
        d = []
        batch_max_length = max(length)
        for s in text:
            t = [self.dict[char] for char in s]
            t.extend([0] * (batch_max_length - len(s)))
            d.append(t)
        return (torch.tensor(d, dtype=torch.long), torch.tensor(length, dtype=torch.long))

    def decode(self, preds, raw=False):
        """ convert text-index into text-label. """
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        result_list = []
        for word, prob in zip(preds_idx, preds_prob):
            if raw:
                result_list.append((''.join([self.character[int(i)] for i in word]), prob))
            else:
                result = []
                conf = []
                for i, index in enumerate(word):
                    if word[i] != 0 and (not (i > 0 and word[i - 1] == word[i])):
                        result.append(self.character[int(index)])
                        conf.append(prob[i])
                result_list.append((''.join(result), conf))
        return result_list


class AttnLabelConverter(object):
    def __init__(self, flags):
        # character (str): set of the possible characters.
        flags = flags.Global
        self.character_type = flags.character_type
        self.loss_type = flags.loss_type
        if self.character_type == 'en':
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        elif self.character_type == 'ch':
            character_dict_path = flags.character_dict_path
            add_space = False
            if hasattr(flags, 'use_space_char'):
                add_space = flags.use_space_char
            self.character_str = ""
            with open(character_dict_path, 'rb') as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip("\n").strip("\r\n")
                    self.character_str += line
            if add_space:
                self.character_str += " "
            dict_character = list(self.character_str)
        elif self.character_type == "en_sensitive":
            # same with ASTER setting (use 94 char).
            self.character_str = string.printable[:-6]
            dict_character = list(self.character_str)
        else:
            self.character_str = None
        assert self.character_str is not None, \
            "Nonsupport type of the character: {}".format(self.character_str)

        self.character = ['[Go]', '[s]'] + dict_character  # '[Go]' for the start token, '[s]' for the end token
        self.dict = {}
        for i, char in enumerate(self.character):
            # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
            self.dict[char] = i 
        self.char_num = len(self.character)

    def encode(self, text):
        length = [len(s)+1 for s in text]
        batch_max_length = max(length) + 1
        batch_size = len(length)
        outputs = torch.LongTensor(batch_size, batch_max_length).fill_(0)
        for i in range(batch_size):
            curr_text = list(text[i])
            curr_text.append('[s]')
            curr_text = [self.dict[char] for char in curr_text]
            outputs[i, 1: len(curr_text)+1] = torch.LongTensor(curr_text)
        return (outputs, torch.IntTensor(length))

    def decode(self, preds):
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text_conf = []
        for idx, prob in zip(preds_idx, preds_prob):
            curr_text = [self.character[index] for index in idx]
            text_conf.append((curr_text, prob))
        result_list = []
        for text, prob in text_conf:
            end_index = ''.join(text).find('[s]')
            text = text[: end_index]
            prob = prob[: end_index]
            result_list.append((''.join(text), prob))
        return result_list
            
