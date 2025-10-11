# train_fixed.py — نسخهٔ اصلاح‌شده (جدا‌سازی theta/w و محدودیت batch در هر epoch)

import os, sys, logging, argparse, importlib, inspect, time
import numpy as np
import torch
from torch import nn
import torchvision.datasets as dset
import torchvision.transforms as transforms

# مسیر سورس
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)

# کلاس‌ها/توابع از پروژه
from supernet import Trainer, FBNet
from candblks import get_blocks
from utils import _logger, _set_file

# -------------------------
# Config پایه (مثل فایل شما)
# -------------------------
class Config(object):
    num_cls_used = 0
    init_theta = 1.0
    # آپتیمایزر
    w_lr = 0.1; w_mom = 0.9; w_wd = 1e-4
    t_lr = 0.01; t_wd = 5e-4; t_beta = (0.9, 0.999)
    # دما
    init_temperature = 5.0; temperature_decay = 0.956
    # حلقه‌ها
    model_save_path = './term_output'
    total_epoch = 90
    start_w_epoch = 1
    train_portion = 0.8

config = Config()
logging.basicConfig(level=logging.INFO)

# -------------------------
# آرگومان‌ها
# -------------------------
parser = argparse.ArgumentParser(description="Train FBNet with separated theta/w and optional batch-limit per epoch.")
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--log-frequence', type=int, default=100)
parser.add_argument('--gpus', type=str, default='0')
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--tb-log', type=str, default='run_fbnet_fix')
parser.add_argument('--warmup', type=int, default=config.start_w_epoch)
parser.add_argument('--total-epochs', type=int, default=config.total_epoch)

# cost / LUTs
parser.add_argument('--alpha', type=float, default=0.0)
parser.add_argument('--beta', type=float, default=1.0)
parser.add_argument('--gamma', type=float, default=0.0)
parser.add_argument('--delta', type=float, default=0.0)
parser.add_argument('--energy-file', type=str, default='new_rpi_energy.txt')
parser.add_argument('--latency-file', type=str, default='rpi_speed.txt')
parser.add_argument('--flops-file', type=str, default='flops.txt')

# محدودیت تعداد batch در هر epoch (0 => نامحدود / تمام epoch)
parser.add_argument('--max-batches', type=int, default=0,
                    help='If >0, limit number of batches per epoch to this value (faster debugging).')

args = parser.parse_args()

# paths & logs
args.model_save_path = '%s/%s/' % (config.model_save_path, args.tb_log)
os.makedirs(args.model_save_path, exist_ok=True)
_set_file(os.path.join(args.model_save_path, 'log.log'))

# -------------------------
# دیتاست و transforms
# -------------------------
CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR_STD  = [0.24703233, 0.24348505, 0.26158768]

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])

train_data = dset.CIFAR10(root='./data', train=True, download=True, transform=train_transform)

train_queue = torch.utils.data.DataLoader(
  train_data, batch_size=args.batch_size,
  shuffle=True, pin_memory=True,
  num_workers=max(0, min(args.num_workers, 16)),
)

val_queue = torch.utils.data.DataLoader(
  train_data, batch_size=args.batch_size,
  shuffle=False, pin_memory=True,
  num_workers=max(0, min(args.num_workers // 2, 8)),
)

# -------------------------
# LimitedDataLoader wrapper — هر epoch تا max_batches می‌دهد (قابل iterate مجدد)
# -------------------------
from itertools import islice

class LimitedDataLoader:
    def __init__(self, dataloader, max_batches=0):
        self._dl = dataloader
        self.max_batches = int(max_batches) if max_batches is not None else 0

    def __iter__(self):
        if self.max_batches and self.max_batches > 0:
            return islice(iter(self._dl), self.max_batches)
        else:
            return iter(self._dl)

    def __len__(self):
        if self.max_batches and self.max_batches > 0:
            return min(len(self._dl), self.max_batches)
        else:
            try:
                return len(self._dl)
            except Exception:
                return 0

# -------------------------
# Loaderها را ممکن است محدود کنیم
# -------------------------
if args.max_batches and args.max_batches > 0:
    train_loader_used = LimitedDataLoader(train_queue, args.max_batches)
    val_loader_used = LimitedDataLoader(val_queue, args.max_batches)
    _logger.info(f"Using LimitedDataLoader: max_batches={args.max_batches}")
else:
    train_loader_used = train_queue
    val_loader_used = val_queue

# -------------------------
# FBNet سازنده
# -------------------------
import supernet
importlib.reload(supernet)
print("Using supernet from:", supernet.__file__)

blocks = get_blocks(cifar10=True)
model = FBNet(num_classes=config.num_cls_used if config.num_cls_used > 0 else 10,
              blocks=blocks,
              init_theta=config.init_theta,
              alpha=args.alpha,
              beta=args.beta,
              gamma=args.gamma,
              delta=args.delta,
              speed_f=args.latency_file,
              energy_f=args.energy_file,
              flops_f=args.flops_file)

# -------------------------
# FixedTrainer: زیرکلاسِ Trainer که پارامترهای theta و weight را جدا می‌کند
# -------------------------
class FixedTrainer(Trainer):
    def __init__(self, *a, **kw):
        # ابتدا __init__ اصلی را اجرا کن (تا بقیهٔ stateها درست ساخته شود)
        super().__init__(*a, **kw)

        # از unwrap کردن module استفاده می‌کنیم تا به net اصلی دسترسی داشته باشیم
        net_for_params = self._mod.module if hasattr(self._mod, "module") else self._mod

        # فهرست theta از شبکه (باید توسط FBNet ساخته شده باشد)
        if not hasattr(net_for_params, "theta"):
            raise RuntimeError("Network does not have attribute 'theta' — cannot separate parameters.")

        self.theta = net_for_params.theta
        theta_ids = {id(t) for t in self.theta}

        # جمع‌آوری وزن‌ها (تمام پارامترها به جز theta)
        w_params = [p for p in net_for_params.parameters() if id(p) not in theta_ids]

        # debug prints
        num_theta = sum(p.numel() for p in self.theta)
        num_w = sum(p.numel() for p in w_params)
        print(f"[FixedTrainer] theta params: {num_theta}  weight params: {num_w}")

        # بازسازی اپتیمایزرها فقط برای پارامترهای صحیح
        # Note: قبلا parent __init__ مولدهای دیگری ساخته بود؛ ما آن‌ها را بازنویسی می‌کنیم.
        import torch.optim as optim
        self.w = w_params
        self.w_opt = optim.SGD(self.w, lr=kw.get('w_lr', 0.01) if 'w_lr' in kw else 0.01,
                               momentum=kw.get('w_mom', 0.9) if 'w_mom' in kw else 0.9,
                               weight_decay=kw.get('w_wd', 1e-4) if 'w_wd' in kw else 1e-4)
        # حفظ scheduler قبلی interface
        try:
            self.w_sche = CosineDecayLR(self.w_opt, **kw.get('lr_scheduler', {'T_max':200}))
        except Exception:
            # اگر چیزی نامناسب بود، ignore کن
            self.w_sche = None

        # theta optimizer (Adam) مجددا تنظیم می‌شود
        self.t_opt = torch.optim.Adam(self.theta,
                                      lr=kw.get('t_lr', 0.001) if 't_lr' in kw else 0.001,
                                      betas=kw.get('t_beta', (0.5, 0.999)) if 't_beta' in kw else (0.5,0.999),
                                      weight_decay=kw.get('t_wd', 3e-3) if 't_wd' in kw else 3e-3)

    # override train_w to be explicit (همان منطق parent، با اطمینان از اینکه w_opt فقط روی weight هاست)
    def train_w(self, input, target, decay_temperature=False):
        self.w_opt.zero_grad(set_to_none=True)
        loss, ce, lat, acc, ener = self._mod(input, target, self.temp)
        loss.backward()

        # debug: مقدار نُرم گرادیان weightها
        total_grad_norm = 0.0
        any_grad = False
        for p in self.w:
            if p.grad is not None:
                any_grad = True
                try:
                    total_grad_norm += float(p.grad.data.norm(2).item())
                except Exception:
                    pass
        if not any_grad:
            self.logger.warning("[FixedTrainer] No gradients for weight params!")
        else:
            self.logger.info(f"[FixedTrainer] w grad norm sum: {total_grad_norm:.6f}")

        self.w_opt.step()
        if decay_temperature:
            tmp = self.temp
            self.temp *= self._tem_decay
            self.logger.info("Change temperature from %.5f to %.5f" % (tmp, self.temp))
        return loss.item(), ce.item(), lat.item(), acc.item(), ener.item()

    def train_t(self, input, target, decay_temperature=False):
        self.t_opt.zero_grad(set_to_none=True)
        loss, ce, lat, acc, ener = self._mod(input, target, self.temp)
        loss.backward()

        # debug theta grads
        t_grad_norm = 0.0
        any_t_grad = False
        for p in self.theta:
            if p.grad is not None:
                any_t_grad = True
                try:
                    t_grad_norm += float(p.grad.data.norm(2).item())
                except Exception:
                    pass
        if not any_t_grad:
            self.logger.warning("[FixedTrainer] No gradients for theta params!")
        else:
            self.logger.info(f"[FixedTrainer] theta grad norm sum: {t_grad_norm:.6f}")

        self.t_opt.step()
        if decay_temperature:
            tmp = self.temp
            self.temp *= self._tem_decay
            self.logger.info("Change temperature from %.5f to %.5f" % (tmp, self.temp))
        return loss.item(), ce.item(), lat.item(), acc.item(), ener.item()

# -------------------------
# trainer ساخته و اجرا
# -------------------------
trainer = FixedTrainer(
    network=model,
    w_lr=config.w_lr, w_mom=config.w_mom, w_wd=config.w_wd,
    t_lr=config.t_lr, t_wd=config.t_wd, t_beta=config.t_beta,
    init_temperature=config.init_temperature, temperature_decay=config.temperature_decay,
    logger=_logger, lr_scheduler={'T_max':400, 'logger':_logger, 'alpha':1e-4,
                                  'warmup_step':100, 't_mul':1.5, 'lr_mul':0.98},
    gpus=args.gpus, save_tb_log=args.tb_log, save_theta_prefix=args.tb_log
)

# اگر از LimitedDataLoader استفاده می‌کنیم trainer.search انتظار iterable دارد — درست است.
trainer.search(
    train_loader_used,   # train_w_ds
    val_loader_used,     # train_t_ds (برای سادگی از validation یا همان train استفاده شده)
    total_epoch=args.total_epochs,
    start_w_epoch=args.warmup,
    log_frequence=args.log_frequence
)

# -------------------------
# خروجی معماری نهایی
# -------------------------
out_json = os.path.join(args.model_save_path, "final_arch.json")
try:
    trainer.export_final_architecture(out_json=out_json, print_table=True)
except Exception:
    net = trainer._mod.module if hasattr(trainer._mod, "module") else trainer._mod
    selected_ops = [int(torch.argmax(t.detach().cpu()).item()) for t in net.theta]
    import json
    with open(out_json, "w") as f:
        json.dump({"selected_ops": selected_ops}, f, indent=2)
    print("\n=== FBNet Final Architecture (fallback) ===")
    for i, op in enumerate(selected_ops):
        print(f"Layer {i:02d}: op={op}")
    print(f"Saved final architecture → {out_json}")
