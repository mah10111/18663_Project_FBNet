# train_cifar10.py  (speed-only, AMP-ready Trainer/FBNet)
import os, time, logging, argparse
import numpy as np
import torch
from torch import nn
import torchvision.datasets as dset
import torchvision.transforms as transforms

from supernet import Trainer, FBNet     # ← نسخه‌ی اصلاح‌شده‌ی قبلی
from candblks import get_blocks
from utils import _logger, _set_file

# -------------------------
# پیکربندی پایه
# -------------------------
class Config(object):
    num_cls_used = 0
    init_theta = 1.0
    alpha = 0.2
    beta = 0.6
    speed_f = './speed_cpu.txt'
    w_lr = 0.1
    w_mom = 0.9
    w_wd = 1e-4
    t_lr = 0.01
    t_wd = 5e-4
    t_beta = (0.9, 0.999)
    init_temperature = 5.0
    temperature_decay = 0.956
    model_save_path = './term_output'
    total_epoch = 10
    start_w_epoch = 1
    train_portion = 0.8

lr_scheduler_params = {
    'logger' : _logger,
    'T_max' : 400,
    'alpha' : 1e-4,
    'warmup_step' : 100,
    't_mul' : 1.5,
    'lr_mul' : 0.98,
}

config = Config()
logging.basicConfig(level=logging.INFO)

# -------------------------
# آرگومان‌ها
# -------------------------
parser = argparse.ArgumentParser(description="Train FBNet supernet on CIFAR-10 (speed-only loss).")
parser.add_argument('--batch-size', type=int, default=256, help='global batch size')
parser.add_argument('--epochs', type=int, default=200, help='(ignored; see total_epoch in Config)')
parser.add_argument('--log-frequence', type=int, default=100, help='log frequency (steps)')
parser.add_argument('--gpus', type=str, default='0', help='GPU ids, e.g. "0" or "0,1"')
parser.add_argument('--load-model-path', type=str, default=None, help='(unused)')
parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers (train)')
parser.add_argument('--tb-log', type=str, default='run_fbnet', help='TensorBoard log folder name')
parser.add_argument('--warmup', type=int, default=2, help='warmup epochs (train_w only)')
# loss scaling terms (latency only is used)
parser.add_argument('--alpha', type=float, default=1e-2, help='latency loss scale (alpha)')
parser.add_argument('--beta', type=float, default=1.0, help='latency loss power (beta)')
# files
parser.add_argument('--latency-file', type=str, default='rpi_speed.txt', help='target device latency file')
parser.add_argument('--energy-file', type=str, default=None, help='(unused now)')
args = parser.parse_args()

# paths & logs
args.model_save_path = f"{config.model_save_path}/{args.tb_log}/"
if not os.path.exists(args.model_save_path):
    _logger.warn(f"{args.model_save_path} not exists, create it")
    os.makedirs(args.model_save_path)
_set_file(os.path.join(args.model_save_path, 'log.log'))

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

# توجه: برای سادگی، همان train_data را برای val هم استفاده می‌کنیم (مثل کد قبلی)
train_queue = torch.utils.data.DataLoader(
    train_data,
    batch_size=args.batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=max(0, min(args.num_workers, 4)),  # Colab-friendly
    persistent_workers=True if args.num_workers > 0 else False,
)

val_queue = torch.utils.data.DataLoader(
    train_data,
    batch_size=args.batch_size,
    shuffle=False,
    pin_memory=True,
    num_workers=max(0, min(args.num_workers // 2, 2)),
    persistent_workers=True if (args.num_workers // 2) > 0 else False,
)

# -------------------------
# مدل: فقط latency (speed) در لا‌س
# -------------------------
blocks = get_blocks(cifar10=True)
num_classes = config.num_cls_used if config.num_cls_used > 0 else 10

model = FBNet(
    num_classes=num_classes,
    blocks=blocks,
    init_theta=config.init_theta,
    # فقط latency را روشن می‌کنیم
    use_latency=True,
    use_energy=False,
    use_flops=False,
    # فایل‌ها
    speed_f=args.latency_file,
    energy_f=None,          # خاموش
    flops_f=None,           # خاموش
    # ضرایب لا‌س (latency-only)
    alpha=args.alpha,
    beta=args.beta,
    gamma=0.0,
    delta=0.0,
    eta=0.0,                # بدون penalty rounds
    dim_feature=1984,
)

# -------------------------
# ترینر (AMP داخلش فعاله)
# -------------------------
trainer = Trainer(
    network=model,
    w_lr=config.w_lr,
    w_mom=config.w_mom,
    w_wd=config.w_wd,
    t_lr=config.t_lr,
    t_wd=config.t_wd,
    t_beta=config.t_beta,
    init_temperature=config.init_temperature,
    temperature_decay=config.temperature_decay,
    logger=_logger,
    lr_scheduler=lr_scheduler_params,
    gpus=args.gpus,
    save_tb_log=args.tb_log,
    save_theta_prefix=args.tb_log,
)

# -------------------------
# آموزش: warmup (w) → هر ایپاک: t سپس w
# -------------------------
trainer.search(
    train_queue,
    val_queue,
    total_epoch=config.total_epoch,
    start_w_epoch=args.warmup,
    log_frequence=args.log_frequence,
)

# خروجی معماری نهایی (argmax θ)
trainer.export_final_architecture(out_json="./final_arch.json", print_table=True)
