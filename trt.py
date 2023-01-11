import argparse
import glob
import numpy as np
import os
import time

import cv2
import torch
import tensorrt as trt
from torch2trt import torch2trt, tensorrt_converter

from utils.args_parser import args_parser
from utils.frontend import SuperPointFrontend
from utils.net import SuperPointNet


# @tensorrt_converter('torch.nn.ReLU.forward')
# def convert_ReLU(ctx):
#     input = ctx.method_args[1]
#     output = ctx.method_return
#     layer = ctx.network.add_activation(input=input._trt, type=trt.ActivationType.RELU)
#     output._trt = layer.get_output(0)


@torch.no_grad()
def main():
    args = args_parser()

    front_end = SuperPointFrontend(args.weights_path, 0, 0, 0, cuda=True)

    model = front_end.net

    x = torch.ones(1, 1, args.H, args.W).cuda()
    model_trt = torch2trt(
        model,
        [x],
        fp16_mode=True,
        log_level=trt.Logger.INFO,
        # max_workspace_size=(1 << args.workspace),
        max_batch_size=1,
        use_onnx=True
    )
    save_dir = os.path.join(args.write_dir, f"H{args.H}-W{args.W}")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model_trt.state_dict(), os.path.join(save_dir, "model_trt.pth"))
    print("Converted TensorRT model done.")
    engine_file = os.path.join(save_dir, "model_trt.engine")
    # engine_file_demo = os.path.join("demo", "TensorRT", "cpp", "model_trt.engine")
    with open(engine_file, "wb") as f:
        f.write(model_trt.engine.serialize())

    # shutil.copyfile(engine_file, engine_file_demo)

    print("Converted TensorRT model engine file is saved for C++ inference.")


if __name__ == "__main__":
    main()
