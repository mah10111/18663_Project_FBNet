import torch
from supernet import FBNet, build_flops_lut
from blocks import build_blocks  # مطمئن شو این مسیر درست است

if __name__ == "__main__":
    # این تابع باید دقیقا مثل ساخت مدل در train.py باشد
    blocks = build_blocks()  # باید با پروژه تو هماهنگ باشد
    model = FBNet(
        num_classes=10,
        blocks=blocks,
        speed_f="./speed.txt",
        energy_f="./energy.txt"
    ).cuda()

    # ورودی تست (مثلا برای CIFAR-10: 32x32)
    dummy_input = torch.randn(1, 3, 32, 32).cuda()

    # ساخت و ذخیره LUT
    build_flops_lut(model, dummy_input, save_path="./flops.txt")
