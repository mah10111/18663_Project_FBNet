# train_cifar10.py  (FLOPs-only)

import os, sys, time, logging, argparse, json
import numpy as np
import torch
from torch import nn
import torchvision.datasets as dset
import torchvision.transforms as transforms

# مسیر سورس را قبل از ایمپورت‌ها اضافه کن
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
    # آپتیمایزرها
    w_lr = 0.1; w_mom = 0.9; w_wd = 1e-4
    t_lr = 0.01; t_wd = 5e-4; t_beta = (0.9, 0.999)
    # دما
    init_temperature = 5.0; temperature_decay = 0.956
    # حلقه‌ها
    model_save_path = './term_output'
    total_epoch = 10
    start_w_epoch = 2
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
parser = argparse.ArgumentParser(description="Train FBNet supernet on CIFAR-10 (FLOPs-only loss).")
parser.add_argument('--batch-size', type=int, default=256, help='global batch size')
parser.add_argument('--log-frequence', type=int, default=100, help='log frequency (steps)')
parser.add_argument('--gpus', type=str, default='0', help='GPU ids, e.g. \"0\" or \"0,1\"')
parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers (train)')
parser.add_argument('--tb-log', type=str, default='run_fbnet_flops', help='TensorBoard log folder name')
parser.add_argument('--warmup', type=int, default=config.start_w_epoch, help='warmup epochs (train_w only)')

# --- FLOPs-only params
parser.add_argument('--flops-file', type=str, default='flops.txt', help='LUT of FLOPs (per layer, per op)')
parser.add_argument('--lambda-flops', type=float, default=1e-2, help='scale for FLOPs loss term (alpha)')
parser.add_argument('--beta', type=float, default=1.0, help='power for FLOPs loss term (beta)')
args = parser.parse_args()

# paths & logs
args.model_save_path = f"{config.model_save_path}/{args.tb_log}/"
os.makedirs(args.model_save_path, exist_ok=True)
_set_file(os.path.join(args.model_save_path, 'log.log'))

# اطمینان از وجود LUT
assert os.path.exists(args.flops_file), f"Missing LUT: {args.flops_file}"

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

# Colab-friendly DataLoaders
train_queue = torch.utils.data.DataLoader(
    train_data,
    batch_size=args.batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=max(0, min(args.num_workers, 4)),
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
# مدل: فقط FLOPs در لا‌س
# -------------------------
blocks = get_blocks(cifar10=True)
num_classes = config.num_cls_used if config.num_cls_used > 0 else 10

model = FBNet(
    num_classes=num_classes,
    blocks=blocks,
    init_theta=config.init_theta,
    flops_f=args.flops_file,        # ← فقط LUT-FLOPs
    alpha=args.lambda_flops,        # ← وزن ترم FLOPs در loss
    beta=args.beta,                 # ← توان ترم FLOPs
    dim_feature=1984,
)

# -------------------------
# ترینر
# -------------------------
trainer = Trainer(
    network=model,
    w_lr=config.w_lr, w_mom=config.w_mom, w_wd=config.w_wd,
    t_lr=config.t_lr, t_wd=config.t_wd, t_beta=config.t_beta,
    init_temperature=config.init_temperature, temperature_decay=config.temperature_decay,
    logger=_logger, lr_scheduler=lr_scheduler_params,
    gpus=args.gpus, save_tb_log=args.tb_log, save_theta_prefix=args.tb_log,
)

# -------------------------
# آموزش: warmup (w) → هر ایپاک: t سپس w
# -------------------------
trainer.search(
    train_queue, val_queue,
    total_epoch=config.total_epoch,
    start_w_epoch=args.warmup,
    log_frequence=args.log_frequence,
)

# -------------------------
# خروجی معماری نهایی (argmax θ) — fallback اگر متد نبود
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
