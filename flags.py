# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import yaml
import collections


def dict_to_namedtuple(d):
    FLAGSTuple = collections.namedtuple('FLAGS', sorted(d.keys()))

    for k, v in d.items():
        if type(v) is dict:
            d[k] = dict_to_namedtuple(v)
        elif type(v) is str:
            d[k] = v

    nt = FLAGSTuple(**d)

    return nt


class Flags():

    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            d = yaml.safe_load(f)

        self.flags = dict_to_namedtuple(d)

    def get(self):
        return self.flags