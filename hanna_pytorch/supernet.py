# supernet.py  — FBNet supernet with optional Latency/Energy/FLOPs LUTs + Trainer
import os
import time
import json
import logging
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# NOTE: این‌ها باید در utils موجود باشند
from utils import (
    gumbel_softmax, weights_init, load_flops_lut,
    Tensorboard, AvgrageMeter, CosineDecayLR
)

# =========================
# Helpers for LUT handling
# =========================
def _load_matrix_lut(path: str):
    """
    Load a matrix-like LUT where each line contains numeric values
    separated by space/tab/comma. Returns list[list[float]] or None if
    file doesn't exist or is empty.
    """
    if not path or not isinstance(path, str) or not os.path.exists(path):
        return None
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            # auto-separate
            s = s.replace(",", " ")
            parts = [p for p in s.split() if p]
            try:
                rows.append([float(x) for x in parts])
            except Exception:
                # skip malformed line
                continue
    return rows if rows else None


def _pad_last_column_to_same_length(mat):
    """
    If some rows are len=max_len-1, append the average of the last column of full-length rows.
    """
    if not mat:
        return mat
    max_len = max(len(r) for r in mat)
    full_last_vals = [r[max_len - 1] for r in mat if len(r) == max_len]
    if full_last_vals:
        iden = float(sum(full_last_vals) / len(full_last_vals))
        for r in mat:
            if len(r) == max_len - 1:
                r.append(iden)
    return mat


# =========================
# Mixed operation container
# =========================
class MixedOp(nn.Module):
    """Weighted sum of candidate blocks (MixedOp)."""
    def __init__(self, blocks):
        super().__init__()
        self._ops = nn.ModuleList()
        for op in blocks:
            self._ops.append(op)

    def forward(self, x, weights):
        tmp = []
        for i, op in enumerate(self._ops):
            r = op(x)
            w = weights[..., i].reshape((-1, 1, 1, 1))
            tmp.append(w * r)
        return sum(tmp)


# =========================
# FBNet supernet
# =========================
class FBNet(nn.Module):
    def __init__(self, num_classes, blocks,
                 init_theta=1.0,
                 # LUT paths (optional)
                 speed_f='./speed.txt',
                 energy_f='./energy.txt',
                 flops_f='./flops.txt',
                 # loss scalars
                 alpha=0.0, beta=0.0, gamma=0.0, delta=0.0, eta=0.0,
                 criterion=nn.CrossEntropyLoss(),
                 dim_feature=1984):
        super().__init__()

        # store basics
        self._blocks = blocks
        self._alpha = float(alpha)
        self._beta  = float(beta)
        self._gamma = float(gamma)
        self._delta = float(delta)
        self._eta   = float(eta)
        self._criterion = criterion  # moved device-sync to forward
        self.dim_feature = dim_feature

        # theta & ops assembly
        self.theta = []
        self._ops = nn.ModuleList()

        # init helper
        def init_func(x): nn.init.constant_(x, init_theta)

        # split blocks into: input_conv (prefix nn.Module), mixed lists, output_conv (suffix nn.Module)
        tmp = []
        input_conv_count = 0
        for b in blocks:
            if isinstance(b, nn.Module):
                tmp.append(b); input_conv_count += 1
            else:
                break
        self._input_conv = nn.Sequential(*tmp)
        self._input_conv_count = input_conv_count

        for b in blocks:
            if isinstance(b, list):
                num_block = len(b)
                t = nn.Parameter(torch.ones((num_block,)), requires_grad=True)
                init_func(t)
                self.theta.append(t)
                self._ops.append(MixedOp(b))
                input_conv_count += 1

        tmp = []
        for b in blocks[input_conv_count:]:
            if isinstance(b, nn.Module):
                tmp.append(b); input_conv_count += 1
            else:
                break
        self._output_conv = nn.Sequential(*tmp)

        # ---------- LUTs (optional & robust) ----------
        speed_mat  = _load_matrix_lut(speed_f)
        energy_mat = _load_matrix_lut(energy_f)

        speed_mat  = _pad_last_column_to_same_length(speed_mat)  if speed_mat  is not None else None
        energy_mat = _pad_last_column_to_same_length(energy_mat) if energy_mat is not None else None

        if energy_mat is None and speed_mat is not None:
            energy_mat = [[0.0] * len(r) for r in speed_mat]
        if speed_mat is None and energy_mat is not None:
            speed_mat = [[0.0] * len(r) for r in energy_mat]

        self._speed  = torch.tensor(speed_mat,  requires_grad=False) if speed_mat  is not None else None
        self._energy = torch.tensor(energy_mat, requires_grad=False) if energy_mat is not None else None

        fl = load_flops_lut(flops_f) if os.path.exists(flops_f) else None
        self._flops = torch.tensor(fl, requires_grad=False) if fl is not None else None

        # classifier head
        self.classifier = nn.Linear(dim_feature, num_classes)

        # small init log
        mode = "minimal"
        if self._flops is not None and self._speed is not None:
            mode = "flops+latency"
        elif self._flops is not None:
            mode = "flops-only"
        elif self._speed is not None:
            mode = "latency-only"
        print(f"[FBNet] init → mode={mode}, alpha={self._alpha}, beta={self._beta}, gamma={self._gamma}, delta={self._delta}, eta={self._eta}")
        if self._flops is not None:  print(f"[FBNet] FLOPs LUT entries   : {self._flops.shape[0]}")
        if self._speed  is not None: print(f"[FBNet] Latency LUT entries : {self._speed.shape[0]}")
        if self._energy is not None: print(f"[FBNet] Energy LUT entries  : {self._energy.shape[0]}")

    def forward(self, input, target, temperature=5.0, theta_list=None):
        device = input.device
        # keep criterion on same device
        self._criterion = self._criterion.to(device)

        self.rounds_per_layer = []
        batch_size = input.size(0)
        self.batch_size = batch_size

        data = self._input_conv(input)
        theta_idx = 0
        lat_terms, ener_terms, flops_acc = [], [], []

        for l_idx in range(self._input_conv_count, len(self._blocks)):
            block = self._blocks[l_idx]
            if isinstance(block, list):
                blk_len = len(block)

                theta = self.theta[theta_idx] if theta_list is None else theta_list[theta_idx]
                # NOTE: gumbel_softmax از utils هم هست، ولی اینجا از torch استفاده می‌کنیم
                t = theta.to(device).repeat(batch_size, 1)
                weight = F.gumbel_softmax(t, tau=temperature, hard=False)

                # ----- FLOPs -----
                flops_ = None
                if self._flops is not None:
                    flops = self._flops[theta_idx][:blk_len].to(device)
                    flops_ = weight * flops.repeat(batch_size, 1)
                    flops_acc.append(torch.sum(flops_))

                # ----- Latency & Energy -----
                if self._speed is not None:
                    speed = self._speed[theta_idx][:blk_len].to(device)
                    lat_ = weight * speed.repeat(batch_size, 1)
                    lat_terms.append(torch.sum(lat_))
                if self._energy is not None:
                    energy = self._energy[theta_idx][:blk_len].to(device)
                    ener_ = weight * energy.repeat(batch_size, 1)
                    ener_terms.append(torch.sum(ener_))

                # ----- Hardware rounds (optional; only if FLOPs present) -----
                if flops_ is not None:
                    ops_this_layer = torch.sum(flops_).item() * 1e9  # assuming flops in GOp
                    pe_capacity = 50000
                    num_pe = 20
                    total_capacity = num_pe * pe_capacity
                    rounds = int((ops_this_layer + total_capacity - 1) // total_capacity)
                    self.rounds_per_layer.append(rounds)

                # mixed op forward
                data = self._ops[theta_idx](data, weight)
                theta_idx += 1
            else:
                break

        # tail
        data = self._output_conv(data)
        data = F.avg_pool2d(data, data.size()[2:])
        data = data.reshape((batch_size, -1))
        logits = self.classifier(data)

        # terms
        ce   = self._criterion(logits, target).sum()
        lat  = (sum(lat_terms)  if len(lat_terms)  > 0 else torch.tensor(0.0, device=device)) / batch_size
        ener = (sum(ener_terms) if len(ener_terms) > 0 else torch.tensor(0.0, device=device)) / batch_size
        flop =  (sum(flops_acc) if len(flops_acc) > 0 else torch.tensor(0.0, device=device)) / batch_size

        # accuracy
        pred = torch.argmax(logits, dim=1)
        acc = torch.sum(pred == target).float() / batch_size

        # optional rounds penalty
        max_rounds = max(self.rounds_per_layer) if len(self.rounds_per_layer) > 0 else 0
        rounds_loss = torch.tensor(float(max_rounds), dtype=torch.float32, device=device)

        loss = (ce
                + self._alpha * (lat ** self._beta if self._beta != 0 else lat)
                + self._gamma * (ener ** self._delta if self._delta != 0 else ener)
                + self._eta   * rounds_loss)

        # expose terms
        self.ce = ce
        self.lat_loss = lat
        self.ener_loss = ener
        self.flops_loss = flop
        self.acc = acc
        self.loss = loss
        self.max_rounds = max_rounds

        return loss, ce, lat, acc, ener


# =========================
# Trainer
# =========================
from torch.nn import DataParallel

class Trainer(object):
    """Training network parameters and theta separately."""
    def __init__(self, network,
                 w_lr=0.01, w_mom=0.9, w_wd=1e-4,
                 t_lr=0.001, t_wd=3e-3, t_beta=(0.5, 0.999),
                 init_temperature=5.0, temperature_decay=0.965,
                 logger=logging, lr_scheduler={'T_max': 200},
                 gpus=[0], save_theta_prefix='', save_tb_log=''):
        assert isinstance(network, FBNet)
        network.apply(weights_init)
        network = network.train().cuda()

        if isinstance(gpus, str):
            gpus = [int(i) for i in gpus.strip().split(',') if i != '']
        self.gpus = gpus
        network = DataParallel(network, device_ids=gpus) if len(gpus) > 0 else network
        self._mod = network

        # parameters
        self.theta = network.module.theta if hasattr(network, "module") else network.theta
        self.w = network.parameters()

        self._tem_decay = temperature_decay
        self.temp = init_temperature
        self.logger = logger

        self.tensorboard = Tensorboard('logs/' + (save_tb_log if save_tb_log else 'default_log'))
        self.save_theta_prefix = save_theta_prefix

        self._acc_avg = AvgrageMeter('acc')
        self._ce_avg  = AvgrageMeter('ce')
        self._lat_avg = AvgrageMeter('lat')
        self._loss_avg= AvgrageMeter('loss')
        self._ener_avg= AvgrageMeter('ener')

        self.w_opt = torch.optim.SGD(self.w, lr=w_lr, momentum=w_mom, weight_decay=w_wd)
        self.w_sche = CosineDecayLR(self.w_opt, **lr_scheduler)

        self.t_opt = torch.optim.Adam(self.theta, lr=t_lr, betas=t_beta, weight_decay=t_wd)

    def train_w(self, input, target, decay_temperature=False):
        """Update model (w) parameters."""
        self.w_opt.zero_grad(set_to_none=True)
        loss, ce, lat, acc, ener = self._mod(input, target, self.temp)
        loss.backward()
        self.w_opt.step()
        if decay_temperature:
            tmp = self.temp
            self.temp *= self._tem_decay
            self.logger.info("Change temperature from %.5f to %.5f" % (tmp, self.temp))
        return loss.item(), ce.item(), lat.item(), acc.item(), ener.item()

    def train_t(self, input, target, decay_temperature=False):
        """Update theta parameters."""
        self.t_opt.zero_grad(set_to_none=True)
        loss, ce, lat, acc, ener = self._mod(input, target, self.temp)
        loss.backward()
        self.t_opt.step()
        if decay_temperature:
            tmp = self.temp
            self.temp *= self._tem_decay
            self.logger.info("Change temperature from %.5f to %.5f" % (tmp, self.temp))
        return loss.item(), ce.item(), lat.item(), acc.item(), ener.item()

    def decay_temperature(self, decay_ratio=None):
        tmp = self.temp
        self.temp *= (decay_ratio if decay_ratio is not None else self._tem_decay)
        self.logger.info("Change temperature from %.5f to %.5f" % (tmp, self.temp))

    def _step(self, input, target, epoch, step, log_frequence, func):
        """Perform one step of training with logging."""
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        loss, ce, lat, acc, ener = func(input, target)

        # batch size from model
        try:
            batch_size = self._mod.module.batch_size
        except AttributeError:
            batch_size = getattr(self._mod, "batch_size", input.size(0))

        # update meters
        self._acc_avg.update(acc)
        self._ce_avg.update(ce)
        self._lat_avg.update(lat)
        self._loss_avg.update(loss)
        self._ener_avg.update(ener)

        if step == 0:
            self.tic = time.time()

        if step > 0 and (step % log_frequence == 0):
            self.toc = time.time()
            speed = 1.0 * (batch_size * log_frequence) / (self.toc - self.tic)
            self.tensorboard.log_scalar('Total Loss', self._loss_avg.getValue(), step)
            self.tensorboard.log_scalar('Accuracy',   self._acc_avg.getValue(),  step)
            self.tensorboard.log_scalar('Latency',    self._lat_avg.getValue(),  step)
            self.tensorboard.log_scalar('Energy',     self._ener_avg.getValue(), step)
            self.logger.info("Epoch[%d] Batch[%d] Speed: %.6f samples/sec %s %s %s %s %s"
                             % (epoch, step, speed, self._loss_avg,
                                self._acc_avg, self._ce_avg, self._lat_avg, self._ener_avg))
            # reset meters
            for m in [self._loss_avg, self._acc_avg, self._ce_avg, self._lat_avg, self._ener_avg]:
                m.reset()
            self.tic = time.time()

    def search(self, train_w_ds, train_t_ds, total_epoch=10, start_w_epoch=5, log_frequence=100):
        """Search (alternating w/theta)."""
        assert start_w_epoch >= 1, "Start to train w must be >= 1"

        # warm-up w
        self.logger.info("Warmup: train w for %d epoch(s)" % start_w_epoch)
        self.tic = time.time()
        for epoch in range(start_w_epoch):
            self.logger.info("Start to train w for epoch %d" % epoch)
            for step, (inp, tgt) in enumerate(train_w_ds):
                self._step(inp, tgt, epoch, step, log_frequence, lambda x, y: self.train_w(x, y, False))
                self.w_sche.step()
                # log lr (optional)
                self.tensorboard.log_scalar('Learning rate curve', self.w_sche.last_epoch, self.w_opt.param_groups[0]['lr'])

        # alternation
        self.tic = time.time()
        for epoch in range(total_epoch):
            # theta
            self.logger.info("Start to train theta for epoch %d" % (epoch + start_w_epoch))
            for step, (inp, tgt) in enumerate(train_t_ds):
                self._step(inp, tgt, epoch + start_w_epoch, step, log_frequence, lambda x, y: self.train_t(x, y, False))
                self.save_theta('./theta-result/%s_theta_epoch_%d.txt' % (self.save_theta_prefix, epoch + start_w_epoch), epoch)
            self.decay_temperature()

            # w
            self.logger.info("Start to train w for epoch %d" % (epoch + start_w_epoch))
            for step, (inp, tgt) in enumerate(train_w_ds):
                self._step(inp, tgt, epoch + start_w_epoch, step, log_frequence, lambda x, y: self.train_w(x, y, False))
                self.w_sche.step()

        self.tensorboard.close()

    def save_theta(self, save_path='theta.txt', epoch=0, plot=False, annot=False):
        """Save theta values to text (fast). Optionally draw a heatmap (slow)."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        res = []
        with open(save_path, 'w') as f:
            for i, t in enumerate(self.theta):
                t_list = list(t.detach().cpu().numpy())
                # align length if you need a fixed width (optionally append a zero)
                if len(t_list) < 9:
                    t_list.append(0.0)

                max_index = int(np.argmax(t_list))
                # log winner index to TB
                self.tensorboard.log_scalar(f'Layer {i}', max_index + 1, epoch)

                res.append(t_list)
                f.write(' '.join(str(v) for v in t_list) + '\n')

        if plot:
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                import seaborn as sns
                val = np.array(res, dtype=np.float32)
                ax = sns.heatmap(val, cbar=True, annot=annot, square=False)
                ax.set_xlabel('op'); ax.set_ylabel('layer')
                ax.set_xticks(range(min(val.shape[1], 9)))
                ax.set_yticks(range(min(val.shape[0], 30)))
                fig = ax.get_figure()
                fig.tight_layout()
                fig.savefig(save_path[:-3] + 'png', dpi=150)
                plt.close(fig)
            except Exception as e:
                self.logger.warning(f"save_theta heatmap failed: {e}")

        return res

    def export_final_architecture(self, out_json="final_arch.json", print_table=True):
        """
        After supernet training, call:
            trainer.export_final_architecture("final_arch.json")
        Exports:
          - selected op index per MixedOp
          - block specs (if introspectable)
          - number of mixed layers
        """
        # unwrap module
        net = self._mod.module if hasattr(self._mod, "module") else self._mod

        # gather mixed candidates from self._blocks
        mixed_candidates = []
        for b in net._blocks:
            if isinstance(b, list):
                mixed_candidates.append(b)

        final_ops, final_names, layer_rows = [], [], []
        for layer_idx, t in enumerate(self.theta):
            t_cpu = t.detach().cpu()
            if t_cpu.ndim != 1:
                raise RuntimeError(f"theta at layer {layer_idx} has unexpected shape: {tuple(t_cpu.shape)}")

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
            final_names.append(spec)

            if print_table:
                probs_row = [float(x) for x in t_use.tolist()]
                layer_rows.append({
                    "layer": layer_idx,
                    "chosen_idx": best_op,
                    "scores": probs_row,
                    "op_name": op_name
                })

        payload = {
            "num_mixed_layers": len(mixed_candidates),
            "selected_ops": final_ops,
            "selected_specs": final_names,
        }

        with open(out_json, "w") as f:
            json.dump(payload, f, indent=2)

        if print_table:
            print(f"\n=== FBNet Final Architecture ({len(final_ops)} mixed layers) ===")
            for row in layer_rows:
                print(f"Layer {row['layer']:02d}: op={row['chosen_idx']}  name={row['op_name']}  scores={row['scores']}")
            print(f"Saved final architecture → {out_json}")

        return payload
