import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import gumbel_softmax, load_flops_lut
from utils import weights_init, load_flops_lut
from torch.nn import DataParallel
import time
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from utils import Tensorboard, weights_init, load_flops_lut, AvgrageMeter, load_flops_lut, CosineDecayLR


class MixedOp(nn.Module):
    """Mixed operation.
    Weighted sum of blocks.
    """
    def __init__(self, blocks):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for op in blocks:
            self._ops.append(op)

    def forward(self, x, weights):
        tmp = []
        for i, op in enumerate(self._ops):
            r = op(x)
            w = weights[..., i].reshape((-1, 1, 1, 1))
            res = w * r
            tmp.append(res)
        return sum(tmp)


class FBNet(nn.Module):
    def __init__(self, num_classes, blocks,
                 init_theta=1.0,
                 speed_f='./speed.txt',
                 energy_f='./energy.txt',
                 flops_f='./flops.txt',
                 alpha=0,
                 beta=0,
                 gamma=0,
                 delta=0,
                 eta=0,
                 criterion=nn.CrossEntropyLoss(),
                 dim_feature=1984):
        super(FBNet, self).__init__()

        init_func = lambda x: nn.init.constant_(x, init_theta)
        self._eta = eta
        self._flops = None
        self._blocks = blocks
        self._criterion = criterion
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._delta = delta
        self._criterion = nn.CrossEntropyLoss().cuda()

        self.theta = []
        self._ops = nn.ModuleList()
        self._blocks = blocks

        tmp = []
        input_conv_count = 0
        for b in blocks:
            if isinstance(b, nn.Module):
                tmp.append(b)
                input_conv_count += 1
            else:
                break
        self._input_conv = nn.Sequential(*tmp)
        self._input_conv_count = input_conv_count

        for b in blocks:
            if isinstance(b, list):
                num_block = len(b)
                theta = nn.Parameter(torch.ones((num_block,)).cuda(), requires_grad=True)
                init_func(theta)
                self.theta.append(theta)
                self._ops.append(MixedOp(b))
                input_conv_count += 1

        tmp = []
        for b in blocks[input_conv_count:]:
            if isinstance(b, nn.Module):
                tmp.append(b)
                input_conv_count += 1
            else:
                break
        self._output_conv = nn.Sequential(*tmp)

        # خواندن سرعت
        with open(speed_f, 'r') as f:
            _speed = f.readlines()
        self._speed = [[float(t) for t in s.strip().split(' ')] for s in _speed]

        # خواندن انرژی
        with open(energy_f, 'r') as f:
            _energy = f.readlines()
        self._energy = [[float(t) for t in s.strip().split(' ')] for s in _energy]

        # نرمال‌سازی سرعت
        max_len = max([len(s) for s in self._speed])
        iden_s = sum(s[max_len - 1] for s in self._speed if len(s) == max_len) / sum(1 for s in self._speed if len(s) == max_len)
        for i in range(len(self._speed)):
            if len(self._speed[i]) == max_len - 1:
                self._speed[i].append(iden_s)

        # نرمال‌سازی انرژی
        max_len = max([len(s) for s in self._energy])
        iden_s = sum(s[max_len - 1] for s in self._energy if len(s) == max_len) / sum(1 for s in self._energy if len(s) == max_len)
        for i in range(len(self._energy)):
            if len(self._energy[i]) == max_len - 1:
                self._energy[i].append(iden_s)

        self._speed = torch.tensor(self._speed, requires_grad=False)
        self._energy = torch.tensor(self._energy, requires_grad=False)

        # FLOPs LUT
        self._flops = load_flops_lut(flops_f) if os.path.exists(flops_f) else None
        if self._flops is not None:
            self._flops = torch.tensor(self._flops, requires_grad=False)

        self.classifier = nn.Linear(dim_feature, num_classes)

    def forward(self, input, target, temperature=5.0, theta_list=None):
        self.rounds_per_layer = []

        batch_size = input.size()[0]
        self.batch_size = batch_size
        data = self._input_conv(input)
        theta_idx = 0
        lat = []
        ener = []
        flops_acc = []  # 🔵 برای محاسبه FLOPs

        for l_idx in range(self._input_conv_count, len(self._blocks)):
            block = self._blocks[l_idx]
            if isinstance(block, list):
                blk_len = len(block)

                if theta_list is None:
                    theta = self.theta[theta_idx]
                else:
                    theta = theta_list[theta_idx]

                t = theta.repeat(batch_size, 1)
                weight = nn.functional.gumbel_softmax(t, temperature)

                # --- FLOPs ---
                if self._flops is not None:
                    flops = self._flops[theta_idx][:blk_len].to(weight.device)
                    flops_ = weight * flops.repeat(batch_size, 1)
                    flops_acc.append(torch.sum(flops_))

                # --- Latency & Energy ---
                speed = self._speed[theta_idx][:blk_len].to(weight.device)
                energy = self._energy[theta_idx][:blk_len].to(weight.device)
                lat_ = weight * speed.repeat(batch_size, 1)
                ener_ = weight * energy.repeat(batch_size, 1)
                lat.append(torch.sum(lat_))
                ener.append(torch.sum(ener_))

                # --- Hardware Rounds ---
                # flops_ بر حسب GigaOps است → باید به Ops تبدیل کنیم
                ops_this_layer = torch.sum(flops_).item() * 1e9

                # ظرفیت هر PE
                pe_capacity = 50000
                num_pe = 20
                total_capacity = num_pe * pe_capacity

                # چند دور طول می‌کشد تا این لایه روی سخت‌افزار اجرا شود
                rounds = int((ops_this_layer + total_capacity - 1) // total_capacity)
                self.rounds_per_layer.append(rounds)

                data = self._ops[theta_idx](data, weight)
                theta_idx += 1
            else:
                break

        # --- خروجی نهایی ---
        data = self._output_conv(data)
        lat = sum(lat)
        ener = sum(ener)
        data = nn.functional.avg_pool2d(data, data.size()[2:])
        data = data.reshape((batch_size, -1))
        logits = self.classifier(data)

        self.ce = self._criterion(logits, target).sum()
        self.lat_loss = lat / batch_size
        self.ener_loss = ener / batch_size
        self.loss = self.ce + self._alpha * self.lat_loss.pow(self._beta) + \
                    self._gamma * self.ener_loss.pow(self._delta)

        self.flops_loss = sum(flops_acc) / batch_size if len(flops_acc) > 0 \
            else torch.tensor(0.0, device=input.device)  # 🔵 FLOPs

        pred = torch.argmax(logits, dim=1)
        self.acc = torch.sum(pred == target).float() / batch_size

        # --- در انتهای forward ---
        self.max_rounds = max(self.rounds_per_layer) if len(self.rounds_per_layer) > 0 else 0

        # اضافه کردن penalty برای max_rounds
        rounds_loss = torch.tensor(self.max_rounds, dtype=torch.float32, device=input.device)

        self.loss = (self.ce
                     + self._alpha * self.lat_loss.pow(self._beta)
                     + self._gamma * self.ener_loss.pow(self._delta)
                     + self._eta * rounds_loss  # 🔵 پارامتر جدید برای کنترل اهمیت max_rounds
                     )
        return self.loss, self.ce, self.lat_loss, self.acc, self.ener_loss
