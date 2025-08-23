import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import gumbel_softmax, load_flops_lut


class FBNet(nn.Module):
    def __init__(self, num_classes, blocks,
                 init_theta=1.0,
                 speed_f='./speed.txt',
                 energy_f='./energy.txt',
                 flops_f='./flops_lut.txt',   # مسیر LUT FLOPs
                 alpha=0,
                 beta=0,
                 gamma=0,
                 delta=0,
                 criterion=nn.CrossEntropyLoss()):
        super(FBNet, self).__init__()

        self._blocks = blocks
        self._criterion = criterion
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._delta = delta

        # سرعت و انرژی
        #self._speed = torch.load(speed_f) if os.path.exists(speed_f) else None
        #self._energy = torch.load(energy_f) if os.path.exists(energy_f) else None
        if os.path.exists(speed_f):
            with open(speed_f, 'r') as f:
                _speed = f.readlines()
            self._speed = [[float(t) for t in s.strip().split()] for s in _speed]
            self._speed = torch.tensor(self._speed, requires_grad=False)
        else:
            self._speed = None

# خواندن انرژی
        if os.path.exists(energy_f):
            with open(energy_f, 'r') as f:
                _energy = f.readlines()
            self._energy = [[float(t) for t in s.strip().split()] for s in _energy]
            self._energy = torch.tensor(self._energy, requires_grad=False)
        else:
            self._energy = None
        # FLOPs LUT
            self._flops = load_flops_lut(flops_f) if os.path.exists(flops_f) else None

        # theta
        self.theta = nn.ParameterList()
        for blk in self._blocks:
            if isinstance(blk, list):
                self.theta.append(nn.Parameter(torch.ones(len(blk)) * init_theta))

        # input conv
        self._input_conv = self._blocks[0]
        self._input_conv_count = 1

        # output conv
        self._output_conv = self._blocks[-1]

        # classifier
        self.classifier = nn.Linear(self._output_conv.out_channels, num_classes)

    def forward(self, input, target, temperature=5.0, theta_list=None):
        batch_size = input.size()[0]
        self.batch_size = batch_size
        data = self._input_conv(input)
        theta_idx = 0
        lat = []
        ener = []
        flops_acc = []  # 🔵 اضافه شده: برای محاسبه FLOPs

        for l_idx in range(self._input_conv_count, len(self._blocks)):
            block = self._blocks[l_idx]
            if isinstance(block, list):
                blk_len = len(block)
                if theta_list is None:
                    theta = self.theta[theta_idx]
                else:
                    theta = theta_list[theta_idx]
                t = theta.repeat(batch_size, 1)
                weight = F.gumbel_softmax(t, temperature)

                speed = self._speed[theta_idx][:blk_len].to(weight.device) if self._speed is not None else None
                energy = self._energy[theta_idx][:blk_len].to(weight.device) if self._energy is not None else None

                if speed is not None:
                    lat_ = weight * speed.repeat(batch_size, 1)
                    lat.append(torch.sum(lat_))

                if energy is not None:
                    ener_ = weight * energy.repeat(batch_size, 1)
                    ener.append(torch.sum(ener_))

                # 🔵 اضافه شده: محاسبه FLOPs از LUT
                if self._flops is not None:
                    flops_row = self._flops[theta_idx][:blk_len].to(weight.device)
                    flops_blk = weight * flops_row.repeat(batch_size, 1)
                    flops_acc.append(torch.sum(flops_blk))

                data = self._ops[theta_idx](data, weight)
                theta_idx += 1
            else:
                break

        data = self._output_conv(data)
        lat = sum(lat) if len(lat) > 0 else torch.tensor(0.0, device=input.device)
        ener = sum(ener) if len(ener) > 0 else torch.tensor(0.0, device=input.device)
        self.flops_loss = sum(flops_acc) / batch_size if len(flops_acc) > 0 else torch.tensor(0.0, device=input.device)  # 🔵 اضافه شده

        data = F.avg_pool2d(data, data.size()[2:])
        data = data.reshape((batch_size, -1))
        logits = self.classifier(data)

        self.ce = self._criterion(logits, target).sum()
        self.lat_loss = lat / batch_size
        self.ener_loss = ener / batch_size
        self.loss = self.ce + self._alpha * self.lat_loss.pow(self._beta) + self._gamma * self.ener_loss.pow(self._delta)

        pred = torch.argmax(logits, dim=1)
        self.acc = torch.sum(pred == target).float() / batch_size

        return self.loss, self.ce, self.lat_loss, self.acc, self.ener_loss, self.flops_loss


class Trainer:
    def __init__(self, network,w_lr=0.01,
             w_mom=0.9,
             w_wd=1e-4,
             t_lr=0.001,
             t_wd=3e-3,
             t_beta=(0.5, 0.999),
             init_temperature=5.0,
             temperature_decay=0.965,
             logger=logging,
             lr_scheduler={'T_max' : 200},
             gpus=[0],
             save_theta_prefix='',
             save_tb_log=''):
                 
                self.network = network
                theta_params = network.theta
                self.theta_optimizer = torch.optim.Adam(theta_params, lr=1e-3)
                self.net_optimizer = torch.optim.SGD(network.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
