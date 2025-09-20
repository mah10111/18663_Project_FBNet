# train_cifar10_fp.py  — CIFAR-10 + AMP (fp32/fp16/bf16) + انتخاب صریح FLOPs یا Latency

import os, sys, time, logging, argparse, json, importlib, inspect
import numpy as np
import torch
from torch import nn
import torchvision.datasets as dset
import torchvision.transforms as transforms

# --- مسیر سورس را قبل از ایمپورت‌ها اضافه کن
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)

from supernet import Trainer, FBNet
from candblks import get_blocks
from utils import _logger, _set_file

# -------------------------
# پیکربندی پایه
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
    total_epoch = 10
    start_w_epoch = 2

config = Config()
logging.basicConfig(level=logging.INFO)

# -------------------------
# آرگومان‌ها
# -------------------------
parser = argparse.ArgumentParser(description="FBNet supernet on CIFAR-10 with AMP (fp32/fp16/bf16).")

# اجرای کلی
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--log-frequence', type=int, default=100)
parser.add_argument('--gpus', type=str, default='0')
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--tb-log', type=str, default='run_fbnet_amp')
parser.add_argument('--warmup', type=int, default=config.start_w_epoch)

# انتخاب معیار هزینه
parser.add_argument('--cost-mode', choices=['flops','latency'], default='flops',
                    help='انتخاب معیار هزینه: فقط flops یا فقط latency(+energy)')

# پنالتی تعداد روندها (rounds)
parser.add_argument('--eta', type=float, default=0.0, help='Scaling factor for rounds penalty')

# دقت شناور
parser.add_argument('--dtype', type=str, default='fp16', choices=['fp32','fp16','bf16'],
                    help='Numerics for AMP: fp32 (off), fp16, or bf16')

# FLOPs
parser.add_argument('--flops-file', type=str, default='flops.txt',
                    help='FLOPs LUT (per layer, per op)')
parser.add_argument('--lambda-flops', type=float, default=1e-2,
                    help='scale for FLOPs loss term (alpha)')
parser.add_argument('--beta', type=float, default=1.0,
                    help='power for cost term (beta) — هم برای FLOPs و هم Latency استفاده می‌شود')

# Latency/Energy
parser.add_argument('--latency-file', type=str, default='rpi_speed.txt',
                    help='Latency LUT (per layer, per op)')
parser.add_argument('--energy-file', type=str, default=None,
                    help='Energy LUT (اختیاری). اگر خالی باشد، انرژی صفر فرض می‌شود.')
parser.add_argument('--alpha', type=float, default=1e-2,
                    help='scale for latency loss term (alpha)')

args = parser.parse_args()

# paths & logs
args.model_save_path = os.path.join(config.model_save_path, args.tb_log)
os.makedirs(args.model_save_path, exist_ok=True)
_set_file(os.path.join(args.model_save_path, 'log.log'))

# -------------------------
# دقت شناور و بهینه‌سازی matmul
# -------------------------
torch.backends.cuda.matmul.allow_tf32 = (args.dtype in ['fp32','bf16'])
torch.backends.cudnn.allow_tf32 = (args.dtype in ['fp32','bf16'])

# -------------------------
# دیتاست و دیتالودر
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
    train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True,
    num_workers=max(0, min(args.num_workers, 4)),
    persistent_workers=True if args.num_workers > 0 else False,
)

val_queue = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size, shuffle=False, pin_memory=True,
    num_workers=max(0, min(args.num_workers // 2, 2)),
    persistent_workers=True if (args.num_workers // 2) > 0 else False,
)

# -------------------------
# سازندهٔ داینامیک FBNet (با انتخاب صریح هزینه)
# -------------------------
import supernet
importlib.reload(supernet)
print("Using supernet from:", supernet.__file__)

FBNet = supernet.FBNet
sig = inspect.signature(FBNet.__init__)
params = set(sig.parameters.keys())

blocks = get_blocks(cifar10=True)
num_classes = config.num_cls_used if config.num_cls_used > 0 else 10

def resolve_here(p):
    if not p: return p
    return p if os.path.isabs(p) else os.path.join(here, p)

flops_path   = resolve_here(args.flops_file)
latency_path = resolve_here(args.latency_file)
energy_path  = resolve_here(args.energy_file) if args.energy_file else None

print(f"[PATH] cwd={os.getcwd()}")
print(f"[PATH] flops_file   = {flops_path}   exists={os.path.exists(flops_path)}")
print(f"[PATH] latency_file = {latency_path} exists={os.path.exists(latency_path)}")
print(f"[PATH] energy_file  = {energy_path}  exists={os.path.exists(energy_path) if energy_path else 'n/a'}")

common_kwargs = dict(
    num_classes=num_classes,
    blocks=blocks,
    init_theta=config.init_theta,
    dim_feature=1984,
    eta=args.eta,
)

if args.cost_mode == 'flops':
    assert 'flops_f' in params, "این نسخهٔ FBNet از flops_f پشتیبانی نمی‌کند."
    assert os.path.exists(flops_path), f"Missing FLOPs LUT: {flops_path}"
    model = FBNet(**common_kwargs,
                  flops_f=flops_path,
                  alpha=args.lambda_flops,  # وزن FLOPs
                  beta=args.beta)
    print(">> Built FBNet (FLOPs-only).")

elif args.cost_mode == 'latency':
    assert 'speed_f' in params, "این نسخهٔ FBNet از speed_f پشتیبانی نمی‌کند."
    assert os.path.exists(latency_path), f"Missing latency LUT: {latency_path}"
    # Energy اختیاری است؛ اگر مسیرش نباشد FBNet آن را صفر می‌گیرد.
    model = FBNet(**common_kwargs,
                  speed_f=latency_path,
                  energy_f=energy_path if (energy_path and os.path.exists(energy_path)) else None,
                  alpha=args.alpha,  # وزن Latency
                  beta=args.beta,
                  gamma=0.0, delta=0.0)
    print(">> Built FBNet (latency-only).")

# -------------------------
# AmpTrainer: زیرکلاس Trainer با autocast/GradScaler
# -------------------------
from torch.cuda.amp import autocast, GradScaler

class AmpTrainer(Trainer):
    def __init__(self, *a, dtype='fp16', **kw):
        super().__init__(*a, **kw)
        self.dtype = dtype
        self.autocast_dtype = None
        if dtype == 'fp16':
            self.autocast_dtype = torch.float16
        elif dtype == 'bf16':
            self.autocast_dtype = torch.bfloat16
        self.scaler = GradScaler(enabled=(dtype == 'fp16'))

    def _maybe_autocast(self):
        return autocast(dtype=self.autocast_dtype) if self.autocast_dtype else torch.cuda.amp.autocast(enabled=False)

    def train_w(self, input, target, decay_temperature=False):
        self.w_opt.zero_grad(set_to_none=True)
        with self._maybe_autocast():
            loss, ce, lat, acc, ener = self._mod(input, target, self.temp)
        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
            self.scaler.step(self.w_opt)
            self.scaler.update()
        else:
            loss.backward()
            self.w_opt.step()
        if decay_temperature:
            tmp = self.temp
            self.temp *= self._tem_decay
            self.logger.info("Change temperature from %.5f to %.5f" % (tmp, self.temp))
        return loss.item(), ce.item(), lat.item(), acc.item(), ener.item()

    def train_t(self, input, target, decay_temperature=False):
        self.t_opt.zero_grad(set_to_none=True)
        with self._maybe_autocast():
            loss, ce, lat, acc, ener = self._mod(input, target, self.temp)
        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
            self.scaler.step(self.t_opt)
            self.scaler.update()
        else:
            loss.backward()
            self.t_opt.step()
        if decay_temperature:
            tmp = self.temp
            self.temp *= self._tem_decay
            self.logger.info("Change temperature from %.5f to %.5f" % (tmp, self.temp))
        return loss.item(), ce.item(), lat.item(), acc.item(), ener.item()

# -------------------------
# ترینر و آموزش
# -------------------------
trainer = AmpTrainer(
    network=model,
    w_lr=config.w_lr, w_mom=config.w_mom, w_wd=config.w_wd,
    t_lr=config.t_lr, t_wd=config.t_wd, t_beta=config.t_beta,
    init_temperature=config.init_temperature, temperature_decay=config.temperature_decay,
    logger=_logger, lr_scheduler={'T_max':400, 'logger':_logger, 'alpha':1e-4,
                                  'warmup_step':100, 't_mul':1.5, 'lr_mul':0.98},
    gpus=args.gpus, save_tb_log=args.tb_log, save_theta_prefix=args.tb_log,
    dtype=args.dtype
)

trainer.search(
    train_queue, val_queue,
    total_epoch=config.total_epoch,
    start_w_epoch=args.warmup,
    log_frequence=args.log_frequence,
)

# -------------------------
# خروجی معماری نهایی
# -------------------------
out_json = "./final_arch.json"
try:
    trainer.export_final_architecture(out_json=out_json, print_table=True)
except Exception:
    net = trainer._mod.module if hasattr(trainer._mod, "module") else trainer._mod
    selected_ops = [int(torch.argmax(t.detach().cpu()).item()) for t in net.theta]
    payload = {"selected_ops": selected_ops}
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print("\n=== FBNet Final Architecture (fallback) ===")
    for i, op in enumerate(selected_ops):
        print(f"Layer {i:02d}: op={op}")
    print(f"Saved final architecture → {out_json}")
