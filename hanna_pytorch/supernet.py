import logging
import os
import time
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# رسم هیت‌مپ را فقط در صورت نیاز فعال می‌کنیم
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from torch.nn import DataParallel
from utils import AvgrageMeter, weights_init, CosineDecayLR, Tensorboard
# نکته: اگر util دیگری برای LUT داری، کافی‌ست جایگزین کنی
# از فایل متنی ساده (space-separated) استفاده می‌کنیم

# -------------------------
# Utils کوچک برای خواندن LUT
# -------------------------
def _load_txt_lut(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    lut = [[float(t) for t in s.strip().split()] for s in lines]
    # padding یک ستون کم‌بود را همسان می‌کنیم (مثل کد قبلی)
    if len(lut) > 0:
        max_len = max(len(r) for r in lut)
        if max_len > 0:
            # مقدار ستون آخر در ردیف‌های کامل را میانگین می‌گیریم
            tails = [r[max_len - 1] for r in lut if len(r) == max_len]
            tail_mean = np.mean(tails) if len(tails) > 0 else 0.0
            for r in lut:
                if len(r) == max_len - 1:
                    r.append(float(tail_mean))
    return lut


class MixedOp(nn.Module):
    """Mixed operation: weighted sum of blocks."""
    def __init__(self, blocks):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for op in blocks:
            self._ops.append(op)

    def forward(self, x, weights):
        # weights: [B, O]
        outs = []
        for i, op in enumerate(self._ops):
            r = op(x)                                 # [B, C, H, W]
            w = weights[..., i].reshape((-1, 1, 1, 1))# [B, 1, 1, 1]
            outs.append(w * r)
        return sum(outs)


class FBNet(nn.Module):
    def __init__(self,
                 num_classes,
                 blocks,
                 init_theta=1.0,
                 # ---------- Files ----------
                 flops_f='./flops.txt',         # [FLOPS-ONLY] تنها ورودی لازم
                 # speed_f='./speed.txt',       # [DISABLED-LATENCY] نگه‌داشتن برای بازگشت
                 # energy_f='./energy.txt',     # [DISABLED-ENERGY]  نگه‌داشتن برای بازگشت
                 # ---------- Loss Coeffs ----------
                 alpha=0.0,                     # [FLOPS-ONLY] scale ترم FLOPs
                 beta=1.0,                      # [FLOPS-ONLY] power ترم FLOPs
                 gamma=0.0,                     # [DISABLED-ENERGY]
                 delta=0.0,                     # [DISABLED-ENERGY]
                 eta=0.0,                       # [DISABLED-ROUNDS]
                 criterion=nn.CrossEntropyLoss(),
                 dim_feature=1984):
        super(FBNet, self).__init__()

        # پارامترهای لا‌س
        self._alpha = float(alpha)
        self._beta  = float(beta)
        self._gamma = float(gamma)  # [DISABLED-ENERGY]
        self._delta = float(delta)  # [DISABLED-ENERGY]
        self._eta   = float(eta)    # [DISABLED-ROUNDS]

        self._criterion = criterion.cuda()
        self._blocks = blocks

        # θ های لایه‌های Mixed
        self.theta = []
        self._ops = nn.ModuleList()

        # ورودیِ ثابت
        init_func = lambda x: nn.init.constant_(x, init_theta)
        tmp = []
        input_conv_count = 0
        for b in blocks:
            if isinstance(b, nn.Module):
                tmp.append(b); input_conv_count += 1
            else:
                break
        self._input_conv = nn.Sequential(*tmp)
        self._input_conv_count = input_conv_count

        # Mixed layers
        for b in blocks:
            if isinstance(b, list):
                num_block = len(b)
                t = nn.Parameter(torch.ones((num_block,), device='cuda', dtype=torch.float32),
                                 requires_grad=True)
                init_func(t)
                self.theta.append(t)
                self._ops.append(MixedOp(b))
                input_conv_count += 1

        # خروجی ثابت
        tmp = []
        for b in blocks[input_conv_count:]:
            if isinstance(b, nn.Module):
                tmp.append(b); input_conv_count += 1
            else:
                break
        self._output_conv = nn.Sequential(*tmp)

        # -------------- [DISABLED-LATENCY/ENERGY] --------------
        # self._speed  = torch.tensor(_load_txt_lut(speed_f),  requires_grad=False).cuda()
        # self._energy = torch.tensor(_load_txt_lut(energy_f), requires_grad=False).cuda()
        # -------------------------------------------------------

        # ---------------- FLOPs LUT (فعال) ---------------------
        assert flops_f is not None and os.path.exists(flops_f), \
            f"FLOPs LUT file not found: {flops_f}"
        self._flops = torch.tensor(_load_txt_lut(flops_f),
                                   requires_grad=False,
                                   dtype=torch.float32).cuda()
        # -------------------------------------------------------

        self.classifier = nn.Linear(dim_feature, num_classes).cuda()

    def forward(self, input, target, temperature=5.0, theta_list=None):
        batch_size = input.size(0)
        self.batch_size = batch_size

        data = self._input_conv(input)
        theta_idx = 0

        # [DISABLED-LATENCY] lat_terms = []
        # [DISABLED-ENERGY]  ener_terms = []
        flops_terms = []  # [FLOPS-ONLY]

        for l_idx in range(self._input_conv_count, len(self._blocks)):
            block = self._blocks[l_idx]
            if not isinstance(block, list):
                break

            blk_len = len(block)
            theta = self.theta[theta_idx] if theta_list is None else theta_list[theta_idx]

            # Gumbel-Softmax بر batch تکرار می‌شود
            w = F.gumbel_softmax(theta.repeat(batch_size, 1), temperature)  # [B, O]

            # [DISABLED-LATENCY]
            # speed_row = self._speed[theta_idx][:blk_len]       # [O]
            # lat_b = w * speed_row.unsqueeze(0).expand(batch_size, -1)
            # lat_terms.append(torch.sum(lat_b))

            # [DISABLED-ENERGY]
            # energy_row = self._energy[theta_idx][:blk_len]     # [O]
            # ener_b = w * energy_row.unsqueeze(0).expand(batch_size, -1)
            # ener_terms.append(torch.sum(ener_b))

            # [FLOPS-ONLY]
            flops_row = self._flops[theta_idx][:blk_len]          # [O]
            flops_b = w * flops_row.unsqueeze(0).expand(batch_size, -1)  # [B, O]
            flops_terms.append(torch.sum(flops_b))

            # عبور داده از MixedOp
            data = self._ops[theta_idx](data, w)
            theta_idx += 1

        data = self._output_conv(data)
        data = F.avg_pool2d(data, data.size()[2:])
        data = data.reshape((batch_size, -1))
        logits = self.classifier(data)

        # --- losses ---
        self.ce = self._criterion(logits, target).sum()

        # [FLOPS-ONLY] به‌جای lat_loss همان FLOPs را گزارش می‌دهیم
        flops_total = sum(flops_terms) if len(flops_terms) > 0 else torch.tensor(0.0, device=input.device)
        self.lat_loss  = flops_total / batch_size
        self.ener_loss = torch.tensor(0.0, device=input.device)  # انرژی نداریم

        # [DISABLED-ROUNDS] می‌توانستیم penalty بر rounds اضافه کنیم
        # rounds_loss = torch.tensor(0.0, device=input.device)

        # نهایی
        self.loss = self.ce + self._alpha * (self.lat_loss ** self._beta)
        pred = torch.argmax(logits, dim=1)
        self.acc = torch.sum(pred == target).float() / batch_size

        return self.loss, self.ce, self.lat_loss, self.acc, self.ener_loss


class Trainer(object):
    """Training network parameters and theta separately. (FLOPs-only logs)"""
    def __init__(self, network,
                 w_lr=0.01, w_mom=0.9, w_wd=1e-4,
                 t_lr=0.001, t_wd=3e-3, t_beta=(0.5, 0.999),
                 init_temperature=5.0, temperature_decay=0.965,
                 logger=logging, lr_scheduler={'T_max': 200},
                 gpus=[0], save_theta_prefix='', save_tb_log=''):
        assert isinstance(network, FBNet)
        network.apply(weights_init)
        network = network.train().cuda()

        # DataParallel
        if isinstance(gpus, str):
            gpus = [int(i) for i in gpus.strip().split(',')]
        network = DataParallel(network, device_ids=gpus)

        self.gpus = gpus
        self._mod = network
        self.theta = network.module.theta
        self.w = network.parameters()
        self._tem_decay = temperature_decay
        self.temp = init_temperature
        self.logger = logger
        self.tensorboard = Tensorboard('logs/' + (save_tb_log if save_tb_log else 'default_log'))
        self.save_theta_prefix = save_theta_prefix

        self._acc_avg = AvgrageMeter('acc')
        self._ce_avg  = AvgrageMeter('ce')
        self._lat_avg = AvgrageMeter('flops')  # [FLOPS-ONLY] برچسب
        self._loss_avg= AvgrageMeter('loss')
        self._ener_avg= AvgrageMeter('ener')

        self.w_opt = torch.optim.SGD(self.w, w_lr, momentum=w_mom, weight_decay=w_wd)
        self.w_sche = CosineDecayLR(self.w_opt, **lr_scheduler)
        self.t_opt = torch.optim.Adam(self.theta, lr=t_lr, betas=t_beta, weight_decay=t_wd)

    def train_w(self, input, target, decay_temperature=False):
        self.w_opt.zero_grad()
        loss, ce, lat, acc, ener = self._mod(input, target, self.temp)  # lat == FLOPs
        loss.backward()
        self.w_opt.step()
        if decay_temperature:
            tmp = self.temp
            self.temp *= self._tem_decay
            self.logger.info("Change temperature from %.5f to %.5f" % (tmp, self.temp))
        return loss.item(), ce.item(), lat.item(), acc.item(), ener.item()

    def train_t(self, input, target, decay_temperature=False):
        self.t_opt.zero_grad()
        loss, ce, lat, acc, ener = self._mod(input, target, self.temp)  # lat == FLOPs
        loss.backward()
        self.t_opt.step()
        if decay_temperature:
            tmp = self.temp
            self.temp *= self._tem_decay
            self.logger.info("Change temperature from %.5f to %.5f" % (tmp, self.temp))
        return loss.item(), ce.item(), lat.item(), acc.item(), ener.item()

    def decay_temperature(self, decay_ratio=None):
        tmp = self.temp
        self.temp *= (self._tem_decay if decay_ratio is None else decay_ratio)
        self.logger.info("Change temperature from %.5f to %.5f" % (tmp, self.temp))

    def _step(self, input, target, epoch, step, log_frequence, func):
        input = input.cuda(); target = target.cuda()
        loss, ce, lat, acc, ener = func(input, target)  # lat == FLOPs

        # batch size امن با DataParallel
        try:
            batch_size = self._mod.module.batch_size
        except AttributeError:
            batch_size = self._mod.batch_size

        self._acc_avg.update(acc)
        self._ce_avg.update(ce)
        self._lat_avg.update(lat)    # FLOPs
        self._loss_avg.update(loss)
        self._ener_avg.update(ener)

        if step > 1 and (step % log_frequence == 0):
            self.toc = time.time()
            speed = (batch_size * log_frequence) / (self.toc - self.tic)
            self.tensorboard.log_scalar('Total Loss', self._loss_avg.getValue(), step)
            self.tensorboard.log_scalar('Accuracy',   self._acc_avg.getValue(), step)
            self.tensorboard.log_scalar('FLOPs',      self._lat_avg.getValue(), step)  # [FLOPS-ONLY]
            self.logger.info("Epoch[%d] Batch[%d] Speed: %.2f samples/sec %s %s %s %s %s"
                             % (epoch, step, speed, self._loss_avg,
                                self._acc_avg, self._ce_avg, self._lat_avg, self._ener_avg))
            # reset
            for avg in [self._loss_avg, self._acc_avg, self._ce_avg, self._lat_avg, self._ener_avg]:
                avg.reset()
            self.tic = time.time()

    def search(self, train_w_ds, train_t_ds, total_epoch=10, start_w_epoch=2, log_frequence=100):
        assert start_w_epoch >= 1, "Start to train w"
        self.tic = time.time()
        # warmup: w
        for epoch in range(start_w_epoch):
            self.logger.info("Start to train w for epoch %d" % epoch)
            for step, (input, target) in enumerate(train_w_ds):
                self._step(input, target, epoch, step, log_frequence, lambda x, y: self.train_w(x, y, False))
                self.w_sche.step()
                self.tensorboard.log_scalar('Learning rate curve', self.w_sche.last_epoch, self.w_opt.param_groups[0]['lr'])

        # main search: (t) then (w)
        self.tic = time.time()
        for epoch in range(total_epoch):
            self.logger.info("Start to train theta for epoch %d" % (epoch + start_w_epoch))
            for step, (input, target) in enumerate(train_t_ds):
                self._step(input, target, epoch + start_w_epoch, step, log_frequence, lambda x, y: self.train_t(x, y, False))
                self.save_theta(f'./theta-result/{self.save_theta_prefix}_theta_epoch_{epoch+start_w_epoch}.txt', epoch)

            self.decay_temperature()
            self.logger.info("Start to train w for epoch %d" % (epoch + start_w_epoch))
            for step, (input, target) in enumerate(train_w_ds):
                self._step(input, target, epoch + start_w_epoch, step, log_frequence, lambda x, y: self.train_w(x, y, False))
                self.w_sche.step()
        self.tensorboard.close()

    def save_theta(self, save_path='theta.txt', epoch=0, plot=False, annot=False):
        """Save theta values. Ensures the parent directory exists."""
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        res = []
        with p.open('w') as f:
            for i, t in enumerate(self.theta):
                t_list = list(t.detach().cpu().numpy())
                # برای هم‌تراز کردن هیت‌مپ اگر کمتر از 9 بود
                if len(t_list) < 9:
                    t_list += [0.0] * (9 - len(t_list))
                max_index = int(np.argmax(t_list))
                self.tensorboard.log_scalar(f'Layer {i}', max_index + 1, epoch)
                res.append(t_list)
                f.write(' '.join(str(v) for v in t_list) + '\n')

        if plot:
            try:
                val = np.array(res, dtype=np.float32)
                ax = sns.heatmap(val, cbar=True, annot=annot)
                ax.figure.savefig(p.with_suffix('.png'))
                plt.close(ax.figure)
            except Exception as e:
                self.logger.warning(f"save_theta heatmap failed: {e}")
        return res

    # خروجی معماری نهایی (argmax θ)
    def export_final_architecture(self, out_json="final_arch.json", print_table=True):
        # دسترسی به مدل زیری
        net = self._mod.module if hasattr(self._mod, "module") else self._mod

        # لایه‌های Mixed از روی blocks
        mixed_candidates = [b for b in net._blocks if isinstance(b, list)]

        final_ops = []
        final_specs = []
        rows = []
        for layer_idx, t in enumerate(self.theta):
            t_cpu = t.detach().cpu()
            num_real_ops = len(mixed_candidates[layer_idx])
            t_use = t_cpu[:num_real_ops] if t_cpu.shape[0] > num_real_ops else t_cpu
            best_op = int(torch.argmax(t_use).item())
            final_ops.append(best_op)

            op_mod = mixed_candidates[layer_idx][best_op]
            op_name = type(op_mod).__name__
            spec = OrderedDict(name=op_name)
            for attr in ["kernel_size", "stride", "groups", "expand", "expansion", "in_channels", "out_channels"]:
                if hasattr(op_mod, attr):
                    v = getattr(op_mod, attr)
                    try:
                        spec[attr] = int(v) if isinstance(v, (int, np.integer)) else (tuple(v) if isinstance(v, (list, tuple)) else v)
                    except:
                        spec[attr] = str(v)
            final_specs.append(spec)

            if print_table:
                rows.append({
                    "layer": layer_idx,
                    "chosen_idx": best_op,
                    "scores": [float(x) for x in t_use.tolist()],
                    "op_name": op_name
                })

        payload = {
            "num_mixed_layers": len(mixed_candidates),
            "selected_ops": final_ops,
            "selected_specs": final_specs,
        }
        with open(out_json, "w") as f:
            import json
            json.dump(payload, f, indent=2)

        if print_table:
            print(f"\n=== FBNet Final Architecture ({len(final_ops)} mixed layers) ===")
            for r in rows:
                print(f"Layer {r['layer']:02d}: op={r['chosen_idx']}  name={r['op_name']}  scores={r['scores']}")
            print(f"Saved final architecture → {out_json}")
        return payload
