# fbnet_supernet_trainer.py
# --------------------------------------------
# FBNet Supernet + Trainer (AMP-enabled), speed-only loss
# θ updates in train_t, weights update in train_w
# --------------------------------------------

import os, time, json, logging
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn import DataParallel
from torch.cuda.amp import autocast, GradScaler

# از utils پروژه‌ی شما:
from utils import Tensorboard, weights_init, AvgrageMeter, CosineDecayLR


# -----------------------------
# MixedOp
# -----------------------------
class MixedOp(nn.Module):
    """Weighted sum of candidate blocks."""
    def __init__(self, blocks):
        super().__init__()
        self._ops = nn.ModuleList(blocks)

    def forward(self, x, weights_bxO):
        # weights_bxO: [B, O]
        outs = []
        for i, op in enumerate(self._ops):
            y = op(x)
            w = weights_bxO[..., i].reshape((-1, 1, 1, 1))
            outs.append(w * y)
        return sum(outs)


# -----------------------------
# FBNet Supernet
# -----------------------------
class FBNet(nn.Module):
    def __init__(
        self,
        num_classes,
        blocks,
        init_theta=1.0,
        speed_f="./speed.txt",
        energy_f=None,   # اختیاری (فعلاً خاموش)
        flops_f=None,    # اختیاری (فعلاً خاموش)
        # ضرایب لأس
        alpha=1.0, beta=1.0,     # ← وزن و توان ترم latency
        gamma=0.0, delta=0.0,    # ← انرژی خاموش
        eta=0.0,                 # ← penalty داوطلبانه برای rounds (خاموش)
        # سوییچ‌ها
        use_latency=True,
        use_energy=False,
        use_flops=False,
        lambda_flops=1.0,
        flops_pow=1.0,
        criterion=None,
        dim_feature=1984,
    ):
        super().__init__()

        # هایپرپارامترها/سوییچ‌ها
        self._blocks = blocks
        self._alpha, self._beta = alpha, beta
        self._gamma, self._delta = gamma, delta
        self._eta = eta
        self._use_latency = use_latency
        self._use_energy = use_energy
        self._use_flops = use_flops
        self._lambda_flops = lambda_flops
        self._flops_pow = flops_pow

        self._criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            self._criterion = self._criterion.cuda()

        # ---- ساخت ورودی/خروجی ثابت و Mixedها
        self.theta = []
        self._ops = nn.ModuleList()

        # لایه‌های ثابت قبل از Mixedها
        tmp, input_conv_count = [], 0
        for b in blocks:
            if isinstance(b, nn.Module):
                tmp.append(b)
                input_conv_count += 1
            else:
                break
        self._input_conv = nn.Sequential(*tmp)
        self._input_conv_count = input_conv_count

        # Mixed layers + thetas
        init_const = lambda x: nn.init.constant_(x, init_theta)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for b in blocks:
            if isinstance(b, list):
                num_ops = len(b)
                t = nn.Parameter(torch.ones((num_ops,), device=device), requires_grad=True)
                init_const(t)
                self.theta.append(t)
                self._ops.append(MixedOp(b))
                input_conv_count += 1

        # لایه‌های ثابت بعد از Mixedها
        tmp = []
        for b in blocks[input_conv_count:]:
            if isinstance(b, nn.Module):
                tmp.append(b)
                input_conv_count += 1
            else:
                break
        self._output_conv = nn.Sequential(*tmp)

        # ---- LUT loaderها (فقط speed ضروری است)
        def _maybe_load_txt(path):
            if path and os.path.exists(path):
                with open(path, "r") as f:
                    rows = [[float(t) for t in s.strip().split()] for s in f]
                return rows
            return None

        _speed = _maybe_load_txt(speed_f)
        _energy = _maybe_load_txt(energy_f) if energy_f else None
        _flops = _maybe_load_txt(flops_f) if flops_f else None

        # هماهنگ‌سازی طول سطرها (برای لایه‌هایی که skip ندارند)
        def _pad_last_col(mat_list):
            if not mat_list:
                return mat_list
            max_len = max(len(r) for r in mat_list)
            full_last_vals = [r[max_len - 1] for r in mat_list if len(r) == max_len]
            if len(full_last_vals) == 0:
                return mat_list
            iden = float(np.mean(full_last_vals))
            for i in range(len(mat_list)):
                if len(mat_list[i]) == max_len - 1:
                    mat_list[i].append(iden)
            return mat_list

        _speed = _pad_last_col(_speed) if _speed is not None else None
        _energy = _pad_last_col(_energy) if _energy is not None else None
        _flops = _pad_last_col(_flops) if _flops is not None else None

        self._speed = torch.tensor(_speed, requires_grad=False, device=device) if _speed is not None else None
        self._energy = torch.tensor(_energy, requires_grad=False, device=device) if _energy is not None else None
        self._flops = torch.tensor(_flops, requires_grad=False, device=device) if _flops is not None else None

        self.classifier = nn.Linear(dim_feature, num_classes)

    def forward(self, input, target, temperature=5.0, theta_list=None):
        batch_size = input.size(0)
        data = self._input_conv(input)

        lat_terms, ener_terms, flops_terms = [], [], []
        self.rounds_per_layer = []
        theta_idx = 0

        for l_idx in range(self._input_conv_count, len(self._blocks)):
            block = self._blocks[l_idx]
            if not isinstance(block, list):
                break

            theta = self.theta[theta_idx] if theta_list is None else theta_list[theta_idx]
            weights_O = F.gumbel_softmax(theta, tau=temperature, hard=False)  # [O]
            w_batch = weights_O.unsqueeze(0).expand(batch_size, -1)           # [B,O] بدون کپی حافظه
            blk_len = len(block)

            # --- latency term (speed) ---
            if self._use_latency and (self._speed is not None):
                sp = self._speed[theta_idx][:blk_len]  # [O] همان device
                lat_terms.append(torch.dot(weights_O, sp))  # اسکالر

            # --- (اختیاری) energy / flops اگر خواستی بعداً روشن کنی ---
            if self._use_energy and (self._energy is not None):
                en = self._energy[theta_idx][:blk_len]
                ener_terms.append(torch.dot(weights_O, en))

            if self._use_flops and (self._flops is not None):
                fl = self._flops[theta_idx][:blk_len]
                flops_terms.append(torch.dot(weights_O, fl))

                # مثال «rounds»
                ops_this_layer = torch.dot(weights_O, fl).item() * 1e9  # اگر fl بر حسب GFLOPs باشد
                pe_capacity = 50000
                num_pe = 20
                total_capacity = num_pe * pe_capacity
                rounds = int((ops_this_layer + total_capacity - 1) // total_capacity)
                self.rounds_per_layer.append(rounds)

            # MixedOp
            data = self._ops[theta_idx](data, w_batch)
            theta_idx += 1

        data = self._output_conv(data)
        data = F.avg_pool2d(data, data.size()[2:])
        data = data.reshape((batch_size, -1))
        logits = self.classifier(data)

        # losses
        self.lat_loss = sum(lat_terms) if len(lat_terms) else torch.tensor(0.0, device=input.device)
        self.ener_loss = sum(ener_terms) if len(ener_terms) else torch.tensor(0.0, device=input.device)
        self.flops_loss = sum(flops_terms) if len(flops_terms) else torch.tensor(0.0, device=input.device)

        self.ce = self._criterion(logits, target).mean()  # mean پایدارتر

        max_rounds = max(self.rounds_per_layer) if len(self.rounds_per_layer) else 0
        rounds_loss = torch.tensor(float(max_rounds), device=input.device)

        self.loss = self.ce \
            + (self._alpha * (self.lat_loss.clamp_min(0) ** self._beta) if self._use_latency else 0.0) \
            + (self._gamma * (self.ener_loss.clamp_min(0) ** self._delta) if self._use_energy else 0.0) \
            + (self._lambda_flops * (self.flops_loss.clamp_min(0) ** self._flops_pow) if self._use_flops else 0.0) \
            + (self._eta * rounds_loss if self._eta > 0 else 0.0)

        pred = torch.argmax(logits, dim=1)
        self.acc = (pred == target).float().mean()

        return self.loss, self.ce, self.lat_loss, self.acc, self.ener_loss

    # ابزارهای کمکی برای خروجی معماری
    def export_final_architecture(self, out_json="final_arch.json", print_table=True):
        """Export argmax op per mixed layer + readable specs."""
        mixed_candidates = [b for b in self._blocks if isinstance(b, list)]

        final_ops, final_names, layer_rows = [], [], []
        for layer_idx, t in enumerate(self.theta):
            t_cpu = t.detach().cpu()
            if t_cpu.ndim != 1:
                raise RuntimeError(f"theta at layer {layer_idx} has unexpected shape: {tuple(t_cpu.shape)}")
            num_real_ops = len(mixed_candidates[layer_idx])
            t_use = t_cpu[:num_real_ops]
            best_op = int(torch.argmax(t_use).item())
            final_ops.append(best_op)

            op_mod = mixed_candidates[layer_idx][best_op]
            op_name = type(op_mod).__name__
            spec = OrderedDict(name=op_name)
            for attr in ["kernel_size", "stride", "groups", "expand", "expansion", "in_channels", "out_channels"]:
                if hasattr(op_mod, attr):
                    v = getattr(op_mod, attr)
                    try:
                        spec[attr] = int(v) if isinstance(v, (int, np.integer)) else (
                            tuple(v) if isinstance(v, (list, tuple)) else v
                        )
                    except Exception:
                        spec[attr] = str(v)
            final_names.append(spec)

            if print_table:
                probs_row = [float(x) for x in t_use.tolist()]
                layer_rows.append({"layer": layer_idx, "chosen_idx": best_op, "scores": probs_row, "op_name": op_name})

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

    def print_architecture(self, lut_ops=None):
        """Quick print of current argmax choices."""
        print("=== Selected Architecture (argmax θ) ===")
        mixed_layers = [b for b in self._blocks if isinstance(b, list)]
        for i, t in enumerate(self.theta):
            num_real_ops = len(mixed_layers[i])
            idx = int(torch.argmax(t[:num_real_ops]).item())
            name = lut_ops[idx] if (lut_ops and idx < len(lut_ops)) else f"op_{idx}"
            print(f"Layer {i}: {name}")
        print("========================================")


# -----------------------------
# Trainer
# -----------------------------
class Trainer(object):
    """Training network parameters and theta separately (AMP-enabled)."""
    def __init__(
        self,
        network: FBNet,
        w_lr=0.01, w_mom=0.9, w_wd=1e-4,
        t_lr=0.001, t_wd=3e-3, t_beta=(0.5, 0.999),
        init_temperature=5.0, temperature_decay=0.965,
        logger=logging,
        lr_scheduler={"T_max": 200},
        gpus=[0],
        save_theta_prefix="",
        save_tb_log="",
    ):
        assert isinstance(network, FBNet)
        network.apply(weights_init)
        network = network.train().cuda()
        if isinstance(gpus, str):
            gpus = [int(i) for i in gpus.strip().split(",")]
        network = DataParallel(network, gpus)

        self.gpus = gpus
        self._mod = network
        self._tem_decay = temperature_decay
        self.temp = init_temperature
        self.logger = logger

        self.tensorboard = Tensorboard("logs/" + (save_tb_log if save_tb_log else "default_log"))
        self.save_theta_prefix = save_theta_prefix

        self._acc_avg = AvgrageMeter("acc")
        self._ce_avg = AvgrageMeter("ce")
        self._lat_avg = AvgrageMeter("lat")
        self._loss_avg = AvgrageMeter("loss")
        self._ener_avg = AvgrageMeter("ener")

        # جداسازی پارامترهای θ از سایر وزن‌ها:
        theta_params = list(network.module.theta)  # فقط θها
        theta_ids = {id(p) for p in theta_params}
        w_params = [p for p in network.parameters() if id(p) not in theta_ids]

        self.theta = theta_params
        self.w = w_params

        self.w_opt = torch.optim.SGD(self.w, w_lr, momentum=w_mom, weight_decay=w_wd)
        self.w_sche = CosineDecayLR(self.w_opt, **lr_scheduler)

        self.t_opt = torch.optim.Adam(self.theta, lr=t_lr, betas=t_beta, weight_decay=t_wd)

        # AMP scalers
        self.scaler_w = GradScaler()
        self.scaler_t = GradScaler()

    # -------------------------
    # AMP-enabled update steps
    # -------------------------
    def train_w(self, input, target, decay_temperature=False):
        self.w_opt.zero_grad(set_to_none=True)
        with autocast():
            loss, ce, lat, acc, ener = self._mod(input, target, self.temp)
        self.scaler_w.scale(loss).backward()
        self.scaler_w.step(self.w_opt)
        self.scaler_w.update()
        if decay_temperature:
            tmp = self.temp
            self.temp *= self._tem_decay
            self.logger.info("Change temperature from %.5f to %.5f" % (tmp, self.temp))
        return loss.item(), ce.item(), lat.item(), acc.item(), ener.item()

    def train_t(self, input, target, decay_temperature=False):
        self.t_opt.zero_grad(set_to_none=True)
        with autocast():
            loss, ce, lat, acc, ener = self._mod(input, target, self.temp)
        self.scaler_t.scale(loss).backward()
        self.scaler_t.step(self.t_opt)
        self.scaler_t.update()
        if decay_temperature:
            tmp = self.temp
            self.temp *= self._tem_decay
            self.logger.info("Change temperature from %.5f to %.5f" % (tmp, self.temp))
        return loss.item(), ce.item(), lat.item(), acc.item(), ener.item()

    # -------------------------
    def decay_temperature(self, decay_ratio=None):
        tmp = self.temp
        self.temp *= (self._tem_decay if decay_ratio is None else decay_ratio)
        self.logger.info("Change temperature from %.5f to %.5f" % (tmp, self.temp))

    def _step(self, input, target, epoch, step, log_frequence, func):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        loss, ce, lat, acc, ener = func(input, target)

        # batch_size از خود مدل
        try:
            batch_size = self._mod.module.batch_size
        except AttributeError:
            batch_size = getattr(self._mod, "batch_size", input.size(0))

        self._acc_avg.update(acc)
        self._ce_avg.update(ce)
        self._lat_avg.update(lat)
        self._loss_avg.update(loss)
        self._ener_avg.update(ener)

        if step > 1 and (step % log_frequence == 0):
            self.toc = time.time()
            speed = 1.0 * (batch_size * log_frequence) / (self.toc - self.tic)
            self.tensorboard.log_scalar("Total Loss", self._loss_avg.getValue(), step)
            self.tensorboard.log_scalar("Accuracy", self._acc_avg.getValue(), step)
            self.tensorboard.log_scalar("Latency", self._lat_avg.getValue(), step)
            self.tensorboard.log_scalar("Energy", self._ener_avg.getValue(), step)
            self.logger.info(
                "Epoch[%d] Batch[%d] Speed: %.6f samples/sec %s %s %s %s %s"
                % (epoch, step, speed, self._loss_avg, self._acc_avg, self._ce_avg, self._lat_avg, self._ener_avg)
            )
            for avg in [self._loss_avg, self._acc_avg, self._ce_avg, self._lat_avg, self._ener_avg]:
                avg.reset()
            self.tic = time.time()

    # حلقه‌ی جستجو (warm-up w → هر ایپاک: t سپس w)
    def search(self, train_w_ds, train_t_ds, total_epoch=10, start_w_epoch=5, log_frequence=100):
        assert start_w_epoch >= 1, "Start to train w"
        self.tic = time.time()
        # warm-up: فقط w
        for epoch in range(start_w_epoch):
            self.logger.info("Start to train w for epoch %d" % epoch)
            for step, (input, target) in enumerate(train_w_ds, 1):
                self._step(input, target, epoch, step, log_frequence, lambda x, y: self.train_w(x, y, False))
                self.w_sche.step()
                self.tensorboard.log_scalar(
                    "Learning rate curve", self.w_sche.last_epoch, self.w_opt.param_groups[0]["lr"]
                )

        # alternating t and w
        self.tic = time.time()
        for epoch in range(total_epoch):
            E = epoch + start_w_epoch
            self.logger.info("Start to train theta for epoch %d" % E)
            for step, (input, target) in enumerate(train_t_ds, 1):
                self._step(input, target, E, step, log_frequence, lambda x, y: self.train_t(x, y, False))

            # ذخیره θ یک‌بار در پایان ایپاک
            prefix = self.save_theta_prefix if self.save_theta_prefix else "run"
            self.save_theta(f"./theta-result/{prefix}_theta_epoch_{E}.txt", epoch=E, plot=False, annot=False)

            self.decay_temperature()

            self.logger.info("Start to train w for epoch %d" % E)
            for step, (input, target) in enumerate(train_w_ds, 1):
                self._step(input, target, E, step, log_frequence, lambda x, y: self.train_w(x, y, False))
                self.w_sche.step()

        self.tensorboard.close()

    def save_theta(self, save_path="theta.txt", epoch=0, plot=False, annot=False):
        """Save theta values. Ensures the parent directory exists."""
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        res = []
        with p.open("w") as f:
            for i, t in enumerate(self.theta):
                t_list = list(t.detach().cpu().numpy())
                res.append(t_list)
                f.write(" ".join(str(v) for v in t_list) + "\n")
                # لاگ ایندکس برنده برای هر لایه
                max_index = int(np.argmax(t_list))
                self.tensorboard.log_scalar(f"Layer {i}", max_index, epoch)

        if plot:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                import seaborn as sns

                val = np.array(res, dtype=np.float32)
                ax = sns.heatmap(val, cbar=True, annot=annot)
                ax.figure.savefig(p.with_suffix(".png"))
                plt.close(ax.figure)
            except Exception as e:
                self.logger.warning(f"save_theta heatmap failed: {e}")
        return res
