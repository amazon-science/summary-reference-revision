# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch.nn as nn


class LossDropper:
    """
    Adapted from https://github.com/ddkang/loss_dropper/blob/master/loss_dropper/dropper.py
    Paper: Improved Natural Language Generation via Loss Truncation
    """
    def __init__(
            self,
            dropc=0.4,
            min_count=10000,
            recompute=10000,
            verbose=True
    ):
        super().__init__()
        self.keepc = 1. - dropc
        self.count = 0
        self.min_count = min_count

        self.recompute = recompute
        self.last_computed = 0
        self.percentile_val = 100000000.
        self.cur_idx = 0
        self.verbose = verbose

        self.vals = np.zeros(self.recompute, dtype=np.float32)

    def should_keep(self, loss):
        assert loss.numel() == 1
        loss_float = loss.detach().cpu().numpy()
        self.last_computed += 1
        self.count += loss.numel()
        if self.count < self.recompute:
            self.vals[self.count - 1] = loss_float
            self.cur_idx += 1
            return True
        else:
            self.vals[self.cur_idx] = loss_float
            self.cur_idx += 1
            if self.cur_idx >= len(self.vals):
                self.cur_idx = 0
        if self.count < self.min_count:
            return True

        if self.last_computed > self.recompute:
            self.percentile_val = np.percentile(self.vals, self.keepc * 100)
            if self.verbose:
                print('Using cutoff', self.percentile_val)
            self.last_computed = 0
        return loss_float < self.percentile_val
