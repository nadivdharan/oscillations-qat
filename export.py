import torch
import argparse
import os
import onnx
from onnxsim import simplify

from torchvision.models import resnet18

from models.mobilenet_v2 import MobileNetV2

SUPPORTED_ARCHS = {
    "mobilenet_v2": MobileNetV2,
    "resnet18": resnet18
}
# import torch.nn as nn
# class CustomReLU6(nn.Module):
#     def __init__(self, **kwargs):
#         super(CustomReLU6, self).__init__()
    
#     def forward(self, x):
#         # return torch.clamp(x, min=0, max=6)
#         return torch.max(torch.zeros_like(x), torch.min(x, torch.full_like(x, 6)))
    
# def custom_relu6_symbolic(g, x):
#     # Define your custom ONNX node logic here
#     return g.op("Max", g.op("Constant", value_t=torch.tensor(0.0)), 
#                 g.op("Min", x, g.op("Constant", value_t=torch.tensor(6.0))))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("arch", help="Architecture to export", choices=SUPPORTED_ARCHS)
    parser.add_argument("--model-dir", default=None, help="Checkpoint path")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version")
    parser.add_argument("--out", default=None, help="Output onnx path")
    args = parser.parse_args()
    return args
    

def main(arch, model_dir, onnx_path, opset):
    if arch not in SUPPORTED_ARCHS:
        raise ValueError(f"Arch. {arch} is not supported")
    fp_model = SUPPORTED_ARCHS[arch](pretrained=True)
    if arch != 'resnet18':
        # Load model from pretrained FP32 weights
        assert os.path.exists(model_dir)
        print(f"Loading pretrained weights from {model_dir}")
        state_dict = torch.load(model_dir)
        fp_model.load_state_dict(state_dict)
    fp_model.eval()
    
    # from torch.onnx import register_custom_op_symbolic
    # register_custom_op_symbolic("::CustomReLU6", custom_relu6_symbolic, opset_version=opset)

    
    # export ONNX
    img = torch.rand(1, 3, 224, 224)     
    with torch.no_grad():
        torch.onnx.export(fp_model,
                          img,
                          onnx_path,
                          do_constant_folding=False,
                        #   keep_initializers_as_inputs=False,
                          opset_version=opset)
        model_onnx = onnx.load(onnx_path)
        try:
            assert 0
            print('Simplifying model..')
            model_simp, check = simplify(model_onnx)
            assert check, "Error in onnx simplifier..."
            onnx.save(model_simp, args.out)
        except onnx.onnx_cpp2py_export.shape_inference.InferenceError as err:
            print('Simplifying failed... Skipping...')
            onnx.save(model_onnx, args.out)
        print(f"ONNX model saved at {onnx_path}")

if __name__ == "__main__":
    args = parse_args()
    main(args.arch, args.model_dir, args.out, args.opset)
