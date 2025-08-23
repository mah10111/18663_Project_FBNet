import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis
from candblks import get_blocks

def build_flops_lut(save_path="flops_lut.txt", input_shape=(1, 3, 224, 224)):
    blocks = get_blocks()
    flops_dict = {}

    dummy_input = torch.randn(input_shape)

    for i, block_choices in enumerate(blocks):
        flops_dict[i] = {}
        for j, block in enumerate(block_choices):
            model = nn.Sequential(block)
            flops = FlopCountAnalysis(model, dummy_input).total()
            flops_dict[i][j] = flops
            print(f"Block {i}-{j}: FLOPs = {flops/1e6:.2f} MFLOPs")

    # ذخیره در فایل
    with open(save_path, "w") as f:
        for i in flops_dict:
            for j in flops_dict[i]:
                f.write(f"{i} {j} {flops_dict[i][j]}\n")

    print(f"FLOPs LUT saved to {save_path}")

if __name__ == "__main__":
    build_flops_lut()
