import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
import logging

np.random.seed(0)
sns.set()
from utils import AvgrageMeter, weights_init, \
                  CosineDecayLR, Tensorboard

# ====== DataParallel fallback: اگر data_parallel نبود، از torch.nn استفاده کن ======
try:
  from data_parallel import DataParallel as _DP
except Exception:
  from torch.nn import DataParallel as _DP
# ================================================================================

# ====== کمکی: خواندن LUT متنی و یکسان‌سازی طول سطرها ======
def _load_txt_lut(path):
  with open(path, 'r') as f:
    lines = f.readlines()
  lut = [[float(x) for x in line.strip().split()] for line in lines if line.strip()]
  if not lut:
    return []
  max_len = max(len(r) for r in lut)
  if max_len <= 0:
    return lut
  # میانگینِ ستون آخرِ سطرهای کامل برای پد کردن سطرهای کوتاه
  full_tails = [r[max_len - 1] for r in lut if len(r) == max_len]
  tail_mean = float(np.mean(full_tails)) if full_tails else 0.0
  for r in lut:
    if len(r) == max_len - 1:
      r.append(tail_mean)
  return lut
# ================================================================================

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
               # --------- (OLD) latency/energy inputs (غیرفعال) ----------
               speed_f='./speed.txt',     # OLD: latency LUT path
               energy_f='./energy.txt',   # OLD: energy  LUT path
               alpha=0,                   # OLD: scale latency term
               beta=0,                    # OLD: power latency term
               gamma=0,                   # OLD: scale energy term
               delta=0,                   # OLD: power energy term
               # --------- (NEW) FLOPs & Rounds --------------------------
               flops_f='./flops.txt',     # NEW: FLOPs LUT path (الزامی)
               flops_unit='gflops',       # NEW: 'gflops' | 'mflops' | 'flops'
               eta=1e-2,                   # NEW: وزن پنالتی Rounds
               rounds_agg='max',          # NEW: 'max' | 'sum' | 'mean'
               pe_capacity=50000,         # NEW: ظرفیت هر PE (ops/round)
               num_pe=20,                 # NEW: تعداد PE موازی
               discrete_rounds=False,     # NEW: True→ ceil(no-grad), False→ نرم
               dim_feature=1984):
    super(FBNet, self).__init__()
    init_func = lambda x: nn.init.constant_(x, init_theta)
    
    # --------- (OLD) نگه می‌داریم اما در لاس استفاده نمی‌کنیم ----------
    self._alpha = alpha
    self._beta  = beta
    self._gamma = gamma
    self._delta = delta
    # ---------------------------------------------------------------

    # ====== (NEW) تنظیمات FLOPs / Rounds ======
    self._eta         = float(eta)
    self._rounds_agg  = str(rounds_agg).lower()
    self._pe_capacity = int(pe_capacity)
    self._num_pe      = int(num_pe)
    u = str(flops_unit).lower()
    if   u in ('gflops','gops','giga'):
      self._flops_scale = 1e9
    elif u in ('mflops','mops','mega'):
      self._flops_scale = 1e6
    else:
      self._flops_scale = 1.0
    self._discrete_rounds = bool(discrete_rounds)
    # ==========================================

    self._criterion = nn.CrossEntropyLoss().cuda()

    self.theta = []
    self._ops = nn.ModuleList()
    self._blocks = blocks

    # --------- stem ----------
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

    # --------- mixed ----------
    for b in blocks:
      if isinstance(b, list):
        num_block = len(b)
        theta = nn.Parameter(torch.ones((num_block, ), dtype=torch.float32, device='cuda'),
                             requires_grad=True)
        init_func(theta)
        self.theta.append(theta)
        self._ops.append(MixedOp(b))
        input_conv_count += 1

    # --------- head ----------
    tmp = []
    for b in blocks[input_conv_count:]:
      if isinstance(b, nn.Module):
        tmp.append(b)
        input_conv_count += 1
      else:
        break
    self._output_conv = nn.Sequential(*tmp)

    # ================= (OLD) latency/energy خواندن و نرمال‌سازی (غیرفعال) =================
    # with open(speed_f, 'r') as f:
    #   _speed = f.readlines()
    # self._speed = [[float(t) for t in s.strip().split(' ')] for s in _speed]
    # max_len = max([len(s) for s in self._speed])
    # iden_s = sum(s[max_len - 1] for s in self._speed if len(s) == max_len) / \
    #          sum(1 for s in self._speed if len(s) == max_len)
    # for i in range(len(self._speed)):
    #   if len(self._speed[i]) == max_len - 1:
    #     self._speed[i].append(iden_s)
    # self._speed = torch.tensor(self._speed, requires_grad=False).cuda()

    # with open(energy_f, 'r') as f:
    #   _energy = f.readlines()
    # self._energy = [[float(t) for t in s.strip().split(' ')] for s in _energy]
    # max_len = max([len(s) for s in self._energy])
    # iden_s = sum(s[max_len - 1] for s in self._energy if len(s) == max_len) / \
    #          sum(1 for s in self._energy if len(s) == max_len)
    # for i in range(len(self._energy)):
    #   if len(self._energy[i]) == max_len - 1:
    #     self._energy[i].append(iden_s)
    # self._energy = torch.tensor(self._energy, requires_grad=False).cuda()
    # ======================================================================================

    # ================= (NEW) FLOPs LUT =================
    assert flops_f and os.path.exists(flops_f), f"FLOPs LUT not found: {flops_f}"
    self._flops = torch.tensor(_load_txt_lut(flops_f), requires_grad=False,
                               dtype=torch.float32, device='cuda')
    # ===================================================

    self.classifier = nn.Linear(dim_feature, num_classes).cuda()

  def forward(self, input, target, temperature=5.0, theta_list=None):
    batch_size = input.size()[0]
    self.batch_size = batch_size
    data = self._input_conv(input)
    theta_idx = 0

    # (NEW) برای Rounds: میانگین FLOPs لایه‌ها (برحسب واحد LUT)
    per_layer_flops_mean = []  # eg. GFLOPs per-sample per-layer

    for l_idx in range(self._input_conv_count, len(self._blocks)):
      block = self._blocks[l_idx]
      if isinstance(block, list):
        blk_len = len(block)
        theta = self.theta[theta_idx] if theta_list is None else theta_list[theta_idx]
        t = theta.repeat(batch_size, 1)
        weight = F.gumbel_softmax(t, temperature)

        # --------- (NEW) FLOPs expectation برای این لایه ----------
        flops_row = self._flops[theta_idx][:blk_len]                      # [O]
        flops_b   = weight * flops_row.unsqueeze(0).expand(batch_size, -1)# [B,O]
        # مجموع روی batch → تقسیم بر batch برای per-sample mean
        per_layer_flops_mean.append(torch.sum(flops_b) / batch_size)      # واحد LUT

        # --------- MixedOp ----------
        data = self._ops[theta_idx](data, weight)
        theta_idx += 1
      else:
        break

    data = self._output_conv(data)
    data = F.avg_pool2d(data, data.size()[2:])
    data = data.reshape((batch_size, -1))
    logits = self.classifier(data)

    # --- Cross-Entropy ---
    self.ce = self._criterion(logits, target).sum()

    # --- (NEW) Rounds penalty: از روی FLOPs و ظرفیت سخت‌افزار ---
    total_capacity = float(self._pe_capacity * self._num_pe)  # ops در هر round
    if len(per_layer_flops_mean) > 0 and total_capacity > 0:
      rounds_per_layer = []
      for l_flops in per_layer_flops_mean:
        ops_layer = l_flops * self._flops_scale  # واحد: ops
        ratio = ops_layer / total_capacity       # rounds پیوسته (قابل گرادیان)
        if self._discrete_rounds:
          rounds_layer = torch.ceil(ratio.detach())  # بدون گرادیان (پله‌ای)
        else:
          rounds_layer = ratio
        rounds_per_layer.append(rounds_layer)
      stack_r = torch.stack(rounds_per_layer)
      if   self._rounds_agg == 'sum':
        self.rounds_loss = stack_r.sum()
      elif self._rounds_agg == 'mean':
        self.rounds_loss = stack_r.mean()
      else:  # 'max'
        self.rounds_loss = stack_r.max()
    else:
      self.rounds_loss = torch.tensor(0.0, device=input.device)

    # --- (OLD) latency / energy (غیرفعال) ---
    # self.lat_loss = lat / batch_size
    # self.ener_loss = ener / batch_size

    # --- (NEW) برای سازگاری با Trainer: lat_loss = Rounds, ener_loss = 0 ---
    self.lat_loss  = self.rounds_loss
    self.ener_loss = torch.tensor(0.0, device=input.device)

    # --- (NEW) Loss نهایی: CE + η · Rounds ---
    self.loss = self.ce + self._eta * self.rounds_loss

    # --- accuracy ---
    pred = torch.argmax(logits, dim=1)
    self.acc = torch.sum(pred == target).float() / batch_size
    return self.loss, self.ce, self.lat_loss, self.acc, self.ener_loss

class Trainer(object):
  """Training network parameters and theta separately.  (Rounds-aware)"""
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
    network = _DP(network, device_ids=gpus)   # ← fallback-safe DP
    self.gpus = gpus
    self._mod = network
    # >>> اگر DataParallel است، theta را از module بگیر:
    self.theta = network.module.theta if hasattr(network, "module") else network.theta
    self.w = network.parameters()
    self._tem_decay = temperature_decay
    self.temp = init_temperature
    self.logger = logger
    self.tensorboard = Tensorboard('logs/'+(save_tb_log if save_tb_log else 'default_log'))
    self.save_theta_prefix = save_theta_prefix

    self._acc_avg = AvgrageMeter('acc')
    self._ce_avg  = AvgrageMeter('ce')
    self._lat_avg = AvgrageMeter('rounds')  # ← الان rounds را لاگ می‌کنیم
    self._loss_avg= AvgrageMeter('loss')
    self._ener_avg= AvgrageMeter('ener')

    self.w_opt = torch.optim.SGD(self.w, w_lr, momentum=w_mom, weight_decay=w_wd)
    self.w_sche = CosineDecayLR(self.w_opt, **lr_scheduler)
    self.t_opt = torch.optim.Adam(self.theta, lr=t_lr, betas=t_beta, weight_decay=t_wd)

  def train_w(self, input, target, decay_temperature=False):
    self.w_opt.zero_grad()
    loss, ce, lat, acc, ener = self._mod(input, target, self.temp)  # lat = rounds
    loss.backward()
    self.w_opt.step()
    if decay_temperature:
      tmp = self.temp
      self.temp *= self._tem_decay
      self.logger.info("Change temperature from %.5f to %.5f" % (tmp, self.temp))
    return loss.item(), ce.item(), lat.item(), acc.item(), ener.item()
  
  def train_t(self, input, target, decay_temperature=False):
    self.t_opt.zero_grad()
    loss, ce, lat, acc, ener = self._mod(input, target, self.temp)  # lat = rounds
    loss.backward()
    self.t_opt.step()
    if decay_temperature:
      tmp = self.temp
      self.temp *= self._tem_decay
      self.logger.info("Change temperature from %.5f to %.5f" % (tmp, self.temp))
    return loss.item(), ce.item(), lat.item(), acc.item(), ener.item()
  
  def decay_temperature(self, decay_ratio=None):
    tmp = self.temp
    if decay_ratio is None:
      self.temp *= self._tem_decay
    else:
      self.temp *= decay_ratio
    self.logger.info("Change temperature from %.5f to %.5f" % (tmp, self.temp))
  
  def _step(self, input, target, epoch, step, log_frequence, func):
    input = input.cuda()
    target = target.cuda()
    loss, ce, lat, acc ,ener= func(input, target)   # lat = rounds

    # batch_size امن با DP
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
      self.tensorboard.log_scalar('Accuracy', self._acc_avg.getValue(), step)
      self.tensorboard.log_scalar('Rounds',   self._lat_avg.getValue(), step)
      self.logger.info("Epoch[%d] Batch[%d] Speed: %.6f samples/sec %s %s %s %s %s" 
              % (epoch, step, speed, self._loss_avg, 
                 self._acc_avg, self._ce_avg, self._lat_avg, self._ener_avg))
      # reset
      for avg in [self._loss_avg, self._acc_avg, self._ce_avg, self._lat_avg, self._ener_avg]:
        avg.reset()
      self.tic = time.time()
  
  def search(self, train_w_ds, train_t_ds, total_epoch=90, start_w_epoch=10, log_frequence=100):
    assert start_w_epoch >= 1, "Start to train w"
    self.tic = time.time()
    for epoch in range(start_w_epoch):
      self.logger.info("Start to train w for epoch %d" % epoch)
      for step, (input, target) in enumerate(train_w_ds):
        self._step(input, target, epoch, step, log_frequence,
                   lambda x, y: self.train_w(x, y, False))
        self.w_sche.step()
        self.tensorboard.log_scalar('Learning rate curve',
                                    self.w_sche.last_epoch, self.w_opt.param_groups[0]['lr'])

    self.tic = time.time()
    for epoch in range(total_epoch):
      self.logger.info("Start to train theta for epoch %d" % (epoch+start_w_epoch))
      for step, (input, target) in enumerate(train_t_ds):
        self._step(input, target, epoch + start_w_epoch, step, log_frequence,
                   lambda x, y: self.train_t(x, y, False))
        self.save_theta('./theta-result/%s_theta_epoch_%d.txt' % 
                    (self.save_theta_prefix, epoch+start_w_epoch), epoch)
      self.decay_temperature()
      self.logger.info("Start to train w for epoch %d" % (epoch+start_w_epoch))
      for step, (input, target) in enumerate(train_w_ds):
        self._step(input, target, epoch + start_w_epoch, step, log_frequence,
                   lambda x, y: self.train_w(x, y, False))
        self.w_sche.step()
    self.tensorboard.close()

  def save_theta(self, save_path='theta.txt',epoch=0):
    """Save theta + heatmap."""
    res = []
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
      for i,t in enumerate(self.theta):
        t_list = list(t.detach().cpu().numpy())
        if(len(t_list) < 9): t_list.append(0.00)  # هم‌ترازسازی اختیاری
        max_index = t_list.index(max(t_list))
        self.tensorboard.log_scalar('Layer %s'% str(i),max_index+1, epoch)
        res.append(t_list)
        s = ' '.join([str(tmp) for tmp in t_list])
        f.write(s + '\n')

      val = np.array(res)
      ax = sns.heatmap(val,cbar=True,annot=True)
      ax.figure.savefig(save_path[:-3]+'png')
      plt.close()
    return res
