# train_cifar10_fp.py — CIFAR-10 + AMP (fp32/fp16/bf16) + robust FLOPs/Latency build
import os, sys, time, logging, argparse, json, importlib, inspect, math
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

# =========================
# ---------- NEW: LoRA Conv Impl (lightweight, optional)
# =========================
class LoRAConv2d(nn.Module):
    """
    Lightweight LoRA wrapper for nn.Conv2d.
    Adds two 1x1 convs: A (in->r) and B (r->out) and scales by alpha/r.
    Freezes base conv weights.
    """
    def __init__(self, base_conv: nn.Conv2d, r: int = 4, alpha: int = 16):
        super().__init__()
        assert isinstance(base_conv, nn.Conv2d)
        self.base_conv = base_conv
        # freeze base weights
        for p in self.base_conv.parameters():
            p.requires_grad = False

        in_ch = base_conv.in_channels
        out_ch = base_conv.out_channels
        self.r = r
        self.alpha = alpha
        self.scaling = float(alpha) / max(1, r)

        # A: reduce channels -> r (1x1 conv)
        self.lora_A = nn.Conv2d(in_ch, r, kernel_size=1, stride=1, padding=0, bias=False)
        # B: expand r -> out_ch (1x1 conv)
        self.lora_B = nn.Conv2d(r, out_ch, kernel_size=1, stride=1, padding=0, bias=False)

        # init
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        base = self.base_conv(x)
        delta = self.lora_B(self.lora_A(x)) * self.scaling
        return base + delta

def inject_lora_conv(module: nn.Module, r=4, alpha=16, target_names=None):
    """
    Recursively replace nn.Conv2d with LoRAConv2d in module.
    If target_names is provided (list of substrings), replacement only occurs for modules
    whose full name contains one of these substrings.
    """
    for name, child in list(module.named_children()):
        # If it's Conv2d -> replace
        if isinstance(child, nn.Conv2d):
            # optional name filter
            if target_names:
                # find parent path check by name substring
                full_name = name
                # try to get qualified name by tracing (best-effort)
                # Here we simply apply replacement regardless of name if target_names None
                matched = any(t in name for t in target_names)
                if not matched:
                    # recurse into child (though conv has no children)
                    continue
            lora = LoRAConv2d(child, r=r, alpha=alpha)
            setattr(module, name, lora)
        else:
            inject_lora_conv(child, r=r, alpha=alpha, target_names=target_names)
    return module

def lora_state_dict(model: nn.Module):
    """Return state_dict items that belong to LoRA adapters (lora_A or lora_B)."""
    sd = {}
    for k, v in model.state_dict().items():
        if 'lora_A' in k or 'lora_B' in k:
            sd[k] = v.cpu()
    return sd

def save_lora(model: nn.Module, path: str):
    sd = lora_state_dict(model)
    dirn = os.path.dirname(path)
    if dirn:
        os.makedirs(dirn, exist_ok=True)
    torch.save(sd, path)

def load_lora_into_model(model: nn.Module, path: str, strict=False):
    sd = torch.load(path, map_location='cpu')
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    return missing, unexpected

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

def resolve_here(p):
    if not p: return p
    return p if os.path.isabs(p) else os.path.join(here, p)

def build_dataloaders(args):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD  = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_data = dset.CIFAR10(root=resolve_here('./data'), train=True, download=True, transform=train_transform)

    # persistent_workers فقط وقتی >0 معنا دارد
    nw = max(0, min(args.num_workers, 4))
    pw = True if nw > 0 else False
    val_nw = max(0, min(args.num_workers // 2, 2))
    val_pw = True if val_nw > 0 else False

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True,
        num_workers=nw, persistent_workers=pw,
    )

    val_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=False, pin_memory=True,
        num_workers=val_nw, persistent_workers=val_pw,
    )
    return train_queue, val_queue

def build_model(args):
    import supernet
    import importlib, inspect
    importlib.reload(supernet)
    print("Using supernet from:", supernet.__file__)

    FBNet = supernet.FBNet
    sig = inspect.signature(FBNet.__init__)
    params = set(sig.parameters.keys())

    blocks = get_blocks(cifar10=True)
    num_classes = config.num_cls_used if config.num_cls_used > 0 else 10

    common_kwargs = dict(
        num_classes=num_classes,
        blocks=blocks,
        init_theta=config.init_theta,
        dim_feature=1984,
        # knobs for rounds/hw
        eta=args.eta,
        round_factor=args.round_factor,
        pe_capacity=args.pe_capacity,
        num_pe=args.num_pe,
        capacity_scale=args.capacity_scale,
        hard_choice=args.hard_choice,
    )

    flops_path   = resolve_here(args.flops_file) if getattr(args, 'flops_file', None) else None
    latency_path = resolve_here(args.latency_file) if getattr(args, 'latency_file', None) else None

    has_flops = ('flops_f' in params) and flops_path and os.path.exists(flops_path)
    has_lat   = ('speed_f' in params) and latency_path and os.path.exists(latency_path)

    print(f"[PATH] cwd={os.getcwd()}")
    print(f"[PATH] flops_file   = {flops_path}   exists={os.path.exists(flops_path) if flops_path else False}")
    print(f"[PATH] latency_file = {latency_path} exists={os.path.exists(latency_path) if latency_path else False}")

    if has_flops and has_lat:
        # اگر هر دو LUT موجودند، اینجا latency را ارجح می‌دهیم
        model = FBNet(**common_kwargs,
                      speed_f=latency_path,
                      alpha=args.alpha, beta=args.beta, gamma=0.0, delta=0.0)
        print(">> Built FBNet (latency-only; both LUTs present).")
    elif has_flops:
        model = FBNet(**common_kwargs,
                      flops_f=flops_path,
                      alpha=args.lambda_flops, beta=args.beta)
        print(">> Built FBNet (FLOPs-only).")
    elif has_lat:
        model = FBNet(**common_kwargs,
                      speed_f=latency_path,
                      alpha=args.alpha, beta=args.beta, gamma=0.0, delta=0.0)
        print(">> Built FBNet (latency-only).")
    else:
        print(">> WARNING: Neither FLOPs nor Latency LUT found. Building minimal FBNet.")
        model = FBNet(**common_kwargs)
    return model

if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)  # شفاف‌سازی روی ویندوز
    except RuntimeError:
        pass

    # -------------------------
    # آرگومان‌ها
    # -------------------------
    parser = argparse.ArgumentParser(description="FBNet supernet on CIFAR-10 with AMP (fp32/fp16/bf16).")
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--log-frequence', type=int, default=100)
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--tb-log', type=str, default='run_fbnet_amp')
    parser.add_argument('--warmup', type=int, default=config.start_w_epoch)

    # دقت شناور
    parser.add_argument('--dtype', type=str, default='fp16', choices=['fp32','fp16','bf16'],
                        help='Numerics for AMP: fp32 (off), fp16, or bf16')

    # FLOPs (اختیاری)
    parser.add_argument('--flops-file', type=str, default='flops.txt',
                        help='FLOPs LUT (per layer, per op)')
    parser.add_argument('--lambda-flops', type=float, default=1e-2,
                        help='scale for FLOPs loss term (alpha)')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='power for FLOPs/latency term (beta)')

    # Latency/Energy (اختیاری)
    parser.add_argument('--latency-file', type=str, default='rpi_speed.txt',
                        help='Latency LUT (per layer, per op)')
    parser.add_argument('--alpha', type=float, default=1e-2,
                        help='scale for latency loss term (alpha)')

    # پنالتی روندها و پارامترهای سخت‌افزار
    parser.add_argument('--eta', type=float, default=0.0, help='rounds penalty weight')
    parser.add_argument('--round-factor', type=float, default=1.0, help='multiplier on computed rounds')
    parser.add_argument('--pe-capacity', type=float, default=50000, help='capacity per PE (ops)')
    parser.add_argument('--num-pe', type=int, default=20, help='number of PEs')
    parser.add_argument('--capacity-scale', type=float, default=1.0, help='scales total capacity')
    parser.add_argument('--hard-choice', action='store_true',
                        help='use hard argmax choice per MixedOp instead of soft gumbel')

    # -------------------------
    # <<< NEW: post-search (LoRA) args
    parser.add_argument('--post-lora', action='store_true',
                        help='After NAS search, perform LoRA fine-tune on top-k architectures and save adapters.')
    parser.add_argument('--top-k', type=int, default=3, help='How many top architectures to derive from theta (k).')
    parser.add_argument('--lora-r', type=int, default=8, help='LoRA rank (r).')
    parser.add_argument('--lora-alpha', type=int, default=16, help='LoRA alpha scaling.')
    parser.add_argument('--lora-epochs', type=int, default=8, help='Epochs for LoRA fine-tune per arch.')
    parser.add_argument('--lora-lr', type=float, default=5e-4, help='LR for adapter fine-tuning.')
    parser.add_argument('--use_peft', action='store_true', help='If set, try to use peft.get_peft_model (if installed).')
    # -------------------------

    args = parser.parse_args()

    # -------------------------
    # paths & logs (فقط در پروسهٔ اصلی)
    # -------------------------
    args.model_save_path = os.path.join(config.model_save_path, args.tb_log)
    os.makedirs(args.model_save_path, exist_ok=True)

    # utils._set_file الآن خودش نام یکتا با timestamp می‌سازد (نسخه‌ی جدید utils.py)
    _set_file(os.path.join(args.model_save_path, 'log.log'))

    # -------------------------
    # دقت شناور و بهینه‌سازی matmul
    # -------------------------
    torch.backends.cuda.matmul.allow_tf32 = (args.dtype in ['fp32','bf16'])
    torch.backends.cudnn.allow_tf32 = (args.dtype in ['fp32','bf16'])

    # -------------------------
    # دیتاست و دیتالودر
    # -------------------------
    train_queue, val_queue = build_dataloaders(args)

    # -------------------------
    # سازندهٔ داینامیک FBNet
    # -------------------------
    model = build_model(args)

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
    # خروجی معماری نهایی (single best) — تابع موجود
    # -------------------------
    out_json = resolve_here("./final_arch.json")
    try:
        trainer.export_final_architecture(out_json=out_json, print_table=True)
    except Exception:
        net = trainer._mod.module if hasattr(trainer._mod, "module") else trainer._mod
        selected_ops = [int(torch.argmax(t.detach().cpu()).item()) for t in net.theta]
        payload = {"selected_ops": selected_ops}
        # ensure output dir exists (if any)
        out_dir = os.path.dirname(out_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(payload, f, indent=2)
        print("\n=== FBNet Final Architecture (fallback) ===")
        for i, op in enumerate(selected_ops):
            print(f"Layer {i:02d}: op={op}")
        print(f"Saved final architecture → {out_json}")

    # -------------------------
    # <<< NEW: derive top-k architectures and optionally fine-tune adapters (LoRA)
    # -------------------------
    def derive_topk_architectures(trainer_obj, k=3):
        net = trainer_obj._mod.module if hasattr(trainer_obj._mod, "module") else trainer_obj._mod
        theta_list = [t.detach().cpu().numpy() for t in net.theta]
        # for each layer, get sorted indices descending
        top_indices = [np.argsort(-t) for t in theta_list]
        archs = []
        for a in range(k):
            arch = []
            for layer_inds in top_indices:
                # if this layer has < a+1 choices, pick top-1
                idx = layer_inds[a] if a < len(layer_inds) else layer_inds[0]
                arch.append(int(idx))
            archs.append(arch)
        return archs

    final_archs_path = resolve_here("./final_archs.json")
    top_k = max(1, int(args.top_k))

    # Try to derive and save top-k; if that fails, fallback to using final_arch.json (single best)
    try:
        archs = derive_topk_architectures(trainer, k=top_k)
        # ensure dir exists
        archs_dir = os.path.dirname(final_archs_path)
        if archs_dir:
            os.makedirs(archs_dir, exist_ok=True)
        with open(final_archs_path, "w") as f:
            json.dump({"selected_ops_list": archs}, f, indent=2)
        print(f"[POST-NAS] Saved top-{top_k} architectures → {final_archs_path}")

    except Exception as e:
        print(f"[WARNING] derive_topk_architectures failed: {e}. Trying fallback from final_arch.json ...")
        fallback_arch = resolve_here("./final_arch.json")
        if os.path.exists(fallback_arch):
            with open(fallback_arch, "r") as f:
                data = json.load(f)
            selected_ops = data.get("selected_ops", [])
            archs = [selected_ops] if selected_ops else []
            # ensure dir exists
            archs_dir = os.path.dirname(final_archs_path)
            if archs_dir:
                os.makedirs(archs_dir, exist_ok=True)
            with open(final_archs_path, "w") as f:
                json.dump({"selected_ops_list": archs}, f, indent=2)
            print(f"[POST-NAS] Fallback: created {final_archs_path} from final_arch.json")
        else:
            print("[ERROR] No final_arch.json to fallback to. Please run NAS first and ensure final_arch.json exists.")
            archs = []

    # If user requested LoRA post-processing, run fine-tune for each arch
    if args.post_lora:
        if not archs:
            print("[POST-NAS] No architectures available for LoRA post-processing. Exiting post-lora step.")
        else:
            print("[POST-NAS] Starting LoRA fine-tune for selected architectures...")
            adapters_dir = resolve_here("./adapters")
            os.makedirs(adapters_dir, exist_ok=True)

            # Attempt to import peft if user asked; fallback to our lightweight LoRAConv2d
            use_peft = False
            if args.use_peft:
                try:
                    from peft import get_peft_model, LoraConfig, TaskType
                    use_peft = True
                    print("[POST-NAS] peft imported successfully; will use get_peft_model if compatible.")
                except Exception as e:
                    print("[POST-NAS] peft import failed, falling back to internal LoRAConv2d. Error:", e)
                    use_peft = False

            # build fresh base model (same builder as earlier)
            # We'll instantiate a new FBNet for each arch so weights are consistent with base initialization
            for i, arch in enumerate(archs):
                arch_name = f"arch_top{i+1}"
                print(f"\n[POST-NAS] Fine-tuning {arch_name} with LoRA: arch len={len(arch)}")

                # instantiate base model (same as build_model)
                sub_model = build_model(args)
                sub_model = sub_model.cuda()
                sub_model.train()

                # create theta_list one-hot tensors for forward override (lengths must align)
                theta_onehots = []
                for layer_idx, t in enumerate(sub_model.theta):
                    num_ops = t.shape[0]
                    chosen = arch[layer_idx] if layer_idx < len(arch) else int(torch.argmax(t).item())
                    onehot = torch.zeros((num_ops,), dtype=torch.float32)
                    onehot[chosen] = 1.0
                    theta_onehots.append(onehot)

                # Apply LoRA: either via peft (if supports arbitrary modules) or via inject_lora_conv
                if use_peft:
                    try:
                        # Try to configure peft for feature-extraction style (vision)
                        lora_cfg = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION,
                                              r=args.lora_r, lora_alpha=args.lora_alpha,
                                              target_modules=["conv", "Conv2d", "Conv"], lora_dropout=0.1)
                        lora_model = get_peft_model(sub_model, lora_cfg)
                        print("[POST-NAS] Applied PEFT LoRA to model.")
                    except Exception as e:
                        print("[POST-NAS] PEFT application failed; falling back. Error:", e)
                        print("[POST-NAS] Using internal LoRAConv2d injection.")
                        lora_model = inject_lora_conv(sub_model, r=args.lora_r, alpha=args.lora_alpha, target_names=None)
                else:
                    lora_model = inject_lora_conv(sub_model, r=args.lora_r, alpha=args.lora_alpha, target_names=None)
                    print("[POST-NAS] Applied internal LoRAConv2d injection.")

                lora_model = lora_model.cuda()
                # freeze non-lora params
                for n, p in lora_model.named_parameters():
                    if 'lora_A' in n or 'lora_B' in n or (args.use_peft and 'lora' in n):
                        p.requires_grad = True
                    else:
                        p.requires_grad = False

                # prepare optimizer for adapter params
                adapter_params = [p for p in lora_model.parameters() if p.requires_grad]
                opt = torch.optim.AdamW(adapter_params, lr=args.lora_lr)

                # AMP scaler if fp16
                scaler = GradScaler(enabled=(args.dtype == 'fp16'))

                # small training loop using theta_onehots as override
                for epoch in range(args.lora_epochs):
                    lora_model.train()
                    tot_loss = 0.0
                    ncnt = 0
                    for step, (x, y) in enumerate(train_queue):
                        x = x.cuda(non_blocking=True)
                        y = y.cuda(non_blocking=True)
                        opt.zero_grad(set_to_none=True)
                        # forward with theta_list override; FBNet.forward signature supports theta_list param
                        with (autocast(dtype=torch.float16) if args.dtype == 'fp16' else torch.cuda.amp.autocast(enabled=False)):
                            loss, ce, lat, acc, ener = lora_model(x, y, temperature=1.0, theta_list=theta_onehots)
                        if args.dtype == 'fp16':
                            scaler.scale(loss).backward()
                            scaler.step(opt)
                            scaler.update()
                        else:
                            loss.backward()
                            opt.step()
                        tot_loss += float(loss.detach().cpu().item())
                        ncnt += x.size(0)
                    avg_loss = tot_loss / (max(1, ncnt))
                    print(f"[POST-NAS][{arch_name}] Epoch {epoch+1}/{args.lora_epochs}  avg_loss={avg_loss:.4f}")

                # save adapter-only state
                adapter_path = os.path.join(adapters_dir, f"{arch_name}_lora.pt")
                try:
                    # if peft used and has save_pretrained, use it
                    if use_peft and hasattr(lora_model, "save_pretrained"):
                        # save full peft adapter dir
                        save_dir = os.path.join(adapters_dir, f"{arch_name}_peft")
                        lora_model.save_pretrained(save_dir)
                        print(f"[POST-NAS] Saved PEFT adapters to {save_dir}")
                    else:
                        save_lora(lora_model, adapter_path)
                        print(f"[POST-NAS] Saved LoRA adapter state_dict → {adapter_path}")
                except Exception as e:
                    print("[POST-NAS] Adapter save failed:", e)

            print("\n[POST-NAS] LoRA fine-tune completed for all selected architectures.")
