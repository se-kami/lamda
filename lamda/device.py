#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device
