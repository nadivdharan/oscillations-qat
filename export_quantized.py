
import logging
import click
import os
import torch
import onnx
from onnxsim import simplify

from utils import DotDict
from quantization.utils import pass_data_for_range_estimation
from utils.qat_utils import get_dataloaders_and_model
from utils.click_options import (
    quantization_options,
    quant_params_dict,
    base_options,
)

# setup stuff
class Config(DotDict):
    pass


@click.group()
def export():
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
    
pass_config = click.make_pass_decorator(Config, ensure=True)


@export.command()
@pass_config
@base_options
@quantization_options
@click.option("--load-type", type=click.Choice(["fp32", "quantized"]),
              default="quantized",
              help='Either "fp32", or "quantized". Specify weather to load a quantized or a FP ' "model.")
@click.option("--onnx-path", type=str,
              default=None,
              help="Output onnx path")
@click.option("--opset", type=int,
              default=13,
              help="ONNX opset version")
# @click.option("--out", type=str,
#               default=None,
#               help="Output ONNX path")
# def onnx_export(config, load_type, onnx_path, opset, out):
def onnx_export(config, load_type, onnx_path, opset):
    print("Setting up network and data loaders")
    qparams = quant_params_dict(config)
    
    dataloaders, model = get_dataloaders_and_model(config=config, load_type=load_type, **qparams)
    if load_type == "fp32":
        # Estimate ranges using training data
        pass_data_for_range_estimation(
            loader=dataloaders.train_loader,
            model=model,
            act_quant=config.quant.act_quant,
            weight_quant=config.quant.weight_quant,
            max_num_batches=config.quant.num_est_batches,
        )
        # Ensure we have the desired quant state
        model.set_quant_state(config.quant.weight_quant, config.quant.act_quant)

    # Fix ranges
    model.fix_ranges()
    from models.mobilenet_v2 import MobileNetV2
    fp_model = MobileNetV2(pretrained=True)
    fp_model.load_state_dict(torch.load('/home/nadivd/workspace/QAT/repos/oscillations-qat/pretrained/mobilenet_v2.pth.tar'))
    # from models.dummy_net import DummyNet
    # fp_model = DummyNet()
    sd_fp = fp_model.state_dict()
    sd_q = model.state_dict()
    shared_keys = [k for k in sd_q if k in sd_fp]
    import ipdb; ipdb.set_trace()
    
    print("Loaded model:\n{}".format(model))

    device = next(model.parameters()).device
    dummy_input = torch.rand(*model.input_size, device=device)
    
    import ipdb; ipdb.set_trace()
    # export ONNX
    img = torch.rand(1, 3, 224, 224)     
    with torch.no_grad():
        torch.onnx.export(model,
                          img,
                          onnx_path,
                          do_constant_folding=False,
                          opset_version=opset)
        model_onnx = onnx.load(onnx_path)
        # try:
        print('Simplifying model..')
        model_simp, check = simplify(model_onnx)
        assert check, "Error in onnx simplifier..."
        onnx_path = onnx_path.split('.onnx')[0] + '_simplified.onnx'
        onnx.save(model_simp, onnx_path)
        # except onnx.onnx_cpp2py_export.shape_inference.InferenceError as err:
        #     print('Simplifying failed... Skipping...')
        #     onnx.save(model_onnx, onnx_path)
        print(f"ONNX model saved at {onnx_path}")

if __name__ == "__main__":
    export()
"""
time CUDA_VISIBLE_DEVICES=0 python export_quantized.py onnx-export --architecture mobilenet_v2_quantized --load-type fp32 --out mobilenet_v2_qat_check_v1.onnx --model-dir ~/workspace/QAT/repos/oscillations-qat/pretrained/mobilenet_v2.pth.tar --images-dir /data/data/imagenet/
"""