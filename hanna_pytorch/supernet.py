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
import json
from collections import OrderedDict
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
class Trainer(object):
  """Training network parameters and theta separately.
  """
  def __init__(self, network,
               w_lr=0.01,
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
    assert isinstance(network, FBNet)
    network.apply(weights_init)
    network = network.train().cuda()
    if isinstance(gpus, str):
      gpus = [int(i) for i in gpus.strip().split(',')]
    network = DataParallel(network, gpus)
    self.gpus = gpus
    self._mod = network
    theta_params = network.module.theta
    mod_params = network.parameters()
    self.theta = theta_params
    self.w = mod_params
    self._tem_decay = temperature_decay
    self.temp = init_temperature
    self.logger = logger
    #self.tensorboard = Tensorboard('logs/'+save_tb_log)
    self.tensorboard = Tensorboard('logs/' + (save_tb_log if save_tb_log is not None else 'default_log'))

    self.save_theta_prefix = save_theta_prefix

    self._acc_avg = AvgrageMeter('acc')
    self._ce_avg = AvgrageMeter('ce')
    self._lat_avg = AvgrageMeter('lat')
    self._loss_avg = AvgrageMeter('loss')
    self._ener_avg = AvgrageMeter('ener')

    self.w_opt = torch.optim.SGD(
                    mod_params,
                    w_lr,
                    momentum=w_mom,
                    weight_decay=w_wd)
    
    self.w_sche = CosineDecayLR(self.w_opt, **lr_scheduler)

    self.t_opt = torch.optim.Adam(
                    theta_params,
                    lr=t_lr, betas=t_beta,
                    weight_decay=t_wd)

  def train_w(self, input, target, decay_temperature=False):
    """Update model parameters.
    """
    self.w_opt.zero_grad()
    loss, ce, lat, acc,ener = self._mod(input, target, self.temp)
    loss.backward()
    self.w_opt.step()
    if decay_temperature:
      tmp = self.temp
      self.temp *= self._tem_decay
      self.logger.info("Change temperature from %.5f to %.5f" % (tmp, self.temp))
    return loss.item(), ce.item(), lat.item(), acc.item(),ener.item()
  
  def train_t(self, input, target, decay_temperature=False):
    """Update theta.
    """
    self.t_opt.zero_grad()
    loss, ce, lat, acc,ener = self._mod(input, target, self.temp)
    loss.backward()
    self.t_opt.step()
    if decay_temperature:
      tmp = self.temp
      self.temp *= self._tem_decay
      self.logger.info("Change temperature from %.5f to %.5f" % (tmp, self.temp))
    return loss.item(), ce.item(), lat.item(), acc.item(),ener.item()
  
  def decay_temperature(self, decay_ratio=None):
    tmp = self.temp
    if decay_ratio is None:
      self.temp *= self._tem_decay
    else:
      self.temp *= decay_ratio
    self.logger.info("Change temperature from %.5f to %.5f" % (tmp, self.temp))
  
  def _step(self, input, target, 
            epoch, step,
            log_frequence,
            func):
    """Perform one step of training.
    """
    input = input.cuda()
    target = target.cuda()
    loss, ce, lat, acc ,ener= func(input, target)

    # Get status
    #batch_size = self.module._mod.batch_size
    try:
     batch_size = self._mod.module.batch_size
    except AttributeError:
     batch_size = self._mod.batch_size
    self._acc_avg.update(acc)
    self._ce_avg.update(ce)
    self._lat_avg.update(lat)
    self._loss_avg.update(loss)
    self._ener_avg.update(ener)

    if step > 1 and (step % log_frequence == 0):
      self.toc = time.time()
      speed = 1.0 * (batch_size * log_frequence) / (self.toc - self.tic)
      self.tensorboard.log_scalar('Total Loss', self._loss_avg.getValue(), step)
      self.tensorboard.log_scalar('Accuracy',self._acc_avg.getValue(),step)
      self.tensorboard.log_scalar('Latency',self._lat_avg.getValue(),step)
      self.tensorboard.log_scalar('Energy',self._ener_avg.getValue(),step)
      self.logger.info("Epoch[%d] Batch[%d] Speed: %.6f samples/sec %s %s %s %s %s" 
              % (epoch, step, speed, self._loss_avg, 
                 self._acc_avg, self._ce_avg, self._lat_avg,self._ener_avg))
      map(lambda avg: avg.reset(), [self._loss_avg, self._acc_avg, 
                                    self._ce_avg, self._lat_avg,self._ener_avg])
      self.tic = time.time()
  import torch

  def print_architecture(model, lut_ops=None):
    """
    نمایش معماری انتخاب‌شده از FBNet سوپرنت
    model: شیء FBNet
    lut_ops: لیستی از نام بلوک‌ها (اختیاری، برای نمایش بهتر)
    """
    print("=== Selected Architecture ===")
    for i, theta in enumerate(model.thetas):
        op_id = torch.argmax(theta).item()   # انتخاب قوی‌ترین آپراتور
        if lut_ops is not None and op_id < len(lut_ops):
            op_name = lut_ops[op_id]
        else:
            op_name = f"op_{op_id}"
        print(f"Layer {i}: {op_name}")
    print("=============================")
  def search(self, train_w_ds,
            train_t_ds,
            total_epoch=10,
            start_w_epoch=5,
            log_frequence=100):
    """Search model.
    """
    assert start_w_epoch >= 1, "Start to train w"
    self.tic = time.time()
    for epoch in range(start_w_epoch):
      self.logger.info("Start to train w for epoch %d" % epoch)
      for step, (input, target) in enumerate(train_w_ds):
        self._step(input, target, epoch, 
                   step, log_frequence,
                   lambda x, y: self.train_w(x, y, False))
        self.w_sche.step()
        self.tensorboard.log_scalar('Learning rate curve',self.w_sche.last_epoch,self.w_opt.param_groups[0]['lr'])
        #print(self.w_sche.last_epoch, self.w_opt.param_groups[0]['lr'])

    self.tic = time.time()
    for epoch in range(total_epoch):
      self.logger.info("Start to train theta for epoch %d" % (epoch+start_w_epoch))
      for step, (input, target) in enumerate(train_t_ds):
        self._step(input, target, epoch + start_w_epoch, 
                   step, log_frequence,
                   lambda x, y: self.train_t(x, y, False))
        self.save_theta('./theta-result/%s_theta_epoch_%d.txt' % 
                    (self.save_theta_prefix, epoch+start_w_epoch), epoch)
      self.decay_temperature()
      self.logger.info("Start to train w for epoch %d" % (epoch+start_w_epoch))
      for step, (input, target) in enumerate(train_w_ds):
        self._step(input, target, epoch + start_w_epoch, 
                   step, log_frequence,
                   lambda x, y: self.train_w(x, y, False))
        self.w_sche.step()
      self.tensorboard.close()

  def save_theta(self, save_path='theta.txt', epoch=0):
    # ایجاد پوشه در صورت عدم وجود
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    res = []
    with open(save_path, 'w') as f:
        for i, t in enumerate(self.theta):
            t_list = list(t.detach().cpu().numpy())
            if len(t_list) < 9:
                t_list.append(0.00)
            max_index = t_list.index(max(t_list))
            self.tensorboard.log_scalar('Layer %s' % str(i), max_index + 1, epoch)
            res.append(t_list)
            s = ' '.join([str(tmp) for tmp in t_list])
            f.write(s + '\n')

        val = np.array(res)
        ax = sns.heatmap(val, cbar=True, annot=True)
        ax.figure.savefig(save_path[:-3] + 'png')
        # self.tensorboard.log_image('Theta Values', val, epoch)
        plt.close()

    return res




  def export_final_architecture(self, out_json="final_arch.json", print_table=True):
        """
        بعد از اتمام trainِ سوپرنت صدا بزن:
            trainer.export_final_architecture("final_arch.json")
        خروجی:
          - فهرست ایندکس انتخاب‌شده برای هر MixedOp
          - نام بلاک انتخاب‌شده (از روی خود blocks پروژه)
          - تعداد لایه‌های Mixed و کل بلاک‌های کاندید
        """
        # دسترسی به مدل زیری (زیر DataParallel)
        if hasattr(self._mod, "module"):
            net = self._mod.module
        else:
            net = self._mod

        # جمع کردن کاندیدهای هر Mixed لایه از روی self._blocks
        # هر entry که list باشد یعنی MixedOp با چند بلاک کاندید
        mixed_candidates = []
        for b in net._blocks:
            if isinstance(b, list):
                mixed_candidates.append(b)

        # استخراج θها از self.theta که قبلاً در __init__ تنظیم شده
        final_ops = []
        final_names = []
        layer_rows = []
        for layer_idx, t in enumerate(self.theta):
            # t: nn.Parameter با سایز [num_ops]
            t_cpu = t.detach().cpu()
            if t_cpu.ndim != 1:
                raise RuntimeError(f"theta at layer {layer_idx} has unexpected shape: {tuple(t_cpu.shape)}")

            num_ops = t_cpu.shape[0]
            # گاهی در save_theta یه صفر اِضافه برای Align طول‌ها می‌گذاری؛
            # اینجا مطمئن می‌شیم فقط به اندازهٔ واقعی کاندیدها argmax بگیریم
            if layer_idx >= len(mixed_candidates):
                raise RuntimeError("More thetas than MixedOp layers detected.")
            num_real_ops = len(mixed_candidates[layer_idx])
            if num_ops > num_real_ops:
                t_use = t_cpu[:num_real_ops]
            else:
                t_use = t_cpu

            best_op = int(torch.argmax(t_use).item())
            final_ops.append(best_op)

            # استخراج اسم خوانا از ماژول کاندید
            op_mod = mixed_candidates[layer_idx][best_op]
            op_name = type(op_mod).__name__  # اگر کلاس‌ها نام‌گذاری معنادار دارند
            # تلاش برای کشف جزییات متداول (kernel_size, expand, groups, stride) اگر وجود داشته باشند
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
                # احتمال/لوگیت‌ها رو برای شفافیت نگه می‌داریم
                probs_row = [float(x) for x in t_use.tolist()]
                layer_rows.append({
                    "layer": layer_idx,
                    "chosen_idx": best_op,
                    "scores": probs_row,
                    "op_name": op_name
                })

        payload = {
            "num_mixed_layers": len(mixed_candidates),
            "selected_ops": final_ops,          # ایندکس انتخاب‌شده برای هر لایه
            "selected_specs": final_names,      # نام و مشخصات هر بلاک انتخاب‌شده (اگر قابل استخراج بود)
        }

        with open(out_json, "w") as f:
            json.dump(payload, f, indent=2)

        if print_table:
            print(f"\n=== FBNet Final Architecture ({len(final_ops)} mixed layers) ===")
            for row in layer_rows:
                print(f"Layer {row['layer']:02d}: op={row['chosen_idx']}  name={row['op_name']}  scores={row['scores']}")
            print(f"Saved final architecture → {out_json}")

        return payload
