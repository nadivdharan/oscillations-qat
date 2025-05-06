
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

from models.mobilenet_v2 import MobileNetV2
MB_V2_PRETRAINED_PATH = '/home/nadivd/workspace/QAT/repos/oscillations-qat/pretrained/mobilenet_v2.pth.tar'
    
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
    
    print("Loaded quantized model:\n{}".format(model))
    import ipdb; ipdb.set_trace()

    # Load FP32 model
    fp_model = MobileNetV2(pretrained=True)
    fp_model.load_state_dict(torch.load(MB_V2_PRETRAINED_PATH))
    # from models.dummy_net import DummyNet
    # fp_model = DummyNet()
    sd_fp = fp_model.state_dict()
    sd_q = model.state_dict()
    # shared_keys1 = [k for k in sd_q if k in sd_fp]
    # non_shared_keys1 = [k for k in sd_q if k not in sd_fp]
    shared_keys = [k for k in sd_fp if k in sd_q] # convs + BN scales (gamma) + BN running stats (mean, var, num_batch_tracked)
    non_shared_keys = [k for k in sd_fp if k not in sd_q] # BN biases (beta)

    sd_fp_keys = list(sd_fp.keys())
    sd_q_keys = list(sd_q.keys())
    num_keys_fp = len(sd_fp)
    num_keys_q = len(sd_q)
    i_fp = 0
    i_q = 0
    layer_pairs = []
    print("Assiging weights from quantized model to fp32 model")
    while i_fp < num_keys_fp:
        fp_key = sd_fp_keys[i_fp]
        if not fp_key.endswith('weight'):
            i_fp += 1
            continue
        else:
            print("==> Setting {}".format(fp_key))
            if sd_fp_keys[i_fp+1].endswith('bias'):
                # Classifier
                bias_fp_key = sd_fp_keys[i_fp+1]
            else:
                gamma_fp_key = sd_fp_keys[i_fp+1]
                beta_fp_key = sd_fp_keys[i_fp+2]
                running_mean_fp_key = sd_fp_keys[i_fp+3]
                running_var_fp_key = sd_fp_keys[i_fp+4]
                assert gamma_fp_key.endswith('weight')
                assert beta_fp_key.endswith('bias')
                assert running_mean_fp_key.endswith('running_mean')
                assert running_var_fp_key.endswith('running_var')
            while i_q < num_keys_q:
                weight_q_key = sd_q_keys[i_q]
                if not weight_q_key.endswith('weight'):
                    i_q += 1
                    continue
                else:
                    sd_fp[fp_key] = sd_q[weight_q_key] # assign weights
                    layer_pairs.append((fp_key, weight_q_key))
                    if sd_q_keys[i_q+1].endswith('bias'):
                        # Classifier
                        bias_q_key = sd_q_keys[i_q+1]
                        assert bias_fp_key == bias_q_key
                        assert fp_key == weight_q_key
                        sd_fp[bias_fp_key] = sd_q[bias_q_key] # assign classifier bias
                        layer_pairs.append((bias_fp_key, bias_q_key))
                        i_q += 2
                        i_fp += 2
                        break
                    else:
                        gamma_q_key = sd_q_keys[i_q+1]
                        beta_q_key = sd_q_keys[i_q+2]
                        running_mean_q_key = sd_q_keys[i_q+5]
                        running_var_q_key = sd_q_keys[i_q+6]
                        assert gamma_q_key.endswith('gamma')
                        assert beta_q_key.endswith('beta')
                        assert running_mean_q_key.endswith('running_mean')
                        assert running_var_q_key.endswith('running_var')
                        
                        # assign BN parameters
                        sd_fp[gamma_fp_key] = sd_q[gamma_q_key]
                        sd_fp[beta_fp_key] = sd_q[beta_q_key]
                        sd_fp[running_mean_fp_key] = sd_q[running_mean_q_key]
                        sd_fp[running_var_fp_key] = sd_q[running_var_q_key]
                        layer_pairs.append((gamma_fp_key, gamma_q_key))
                        layer_pairs.append((beta_fp_key, beta_q_key))
                        layer_pairs.append((running_mean_fp_key, running_mean_q_key))
                        layer_pairs.append((running_var_fp_key, running_var_q_key))
                        
                        i_q += 7
                        i_fp += 5
                        break

    print('Finished assigning quantized weights to FP32 model')
    print('Re-Loading FP32 model with quantized weights')
    fp_model.load_state_dict(sd_fp)

    #
    # for layer_pair in layer_pairs:
    #     assert torch.equal(sd_fp[layer_pair[0]], sd_q[layer_pair[1]])
    
    # for key in shared_keys:
    #     sd_fp[key] = sd_q[key]
    # for key in non_shared_keys:
    #     import ipdb; ipdb.set_trace()
    #     if 'bias' in key:
    #         bn_key = key.split('.bias')[0]+'.beta'
    #     elif 'weight' in key:
    #         bn_key = key.split('.weight')[0]+'.gamma'
    #     elif 'running_mean' in key:
    #         bn_key = key.split('.running_mean')[0]+'.running_mean'
    #     elif 'running_var' in key:
    #         bn_key = key.split('.running_var')[0]+'.running_var'
    #     else:
    #         continue
    #     print("==> Setting {} to {}".format(key, bn_key))
    #     assert key in sd_fp, f"{key} not in sd_fp" 
    #     assert bn_key in sd_q, f"{bn_key} not in sd_q"
    #     sd_fp[key] = sd_q[bn_key]
    # fp_model.load_state_dict(sd_fp)

    device = next(model.parameters()).device
    fp_model.to(device)
    dummy_input = torch.rand(*model.input_size, device=device)
    
    from ignite.engine import create_supervised_evaluator
    from ignite.metrics import Accuracy, TopKCategoricalAccuracy
    from ignite.contrib.handlers import ProgressBar
    
    # Validation
    print('Evaluating model')
    metrics = {"top_1_accuracy": Accuracy(), "top_5_accuracy": TopKCategoricalAccuracy()}
    evaluator = create_supervised_evaluator(model=model, metrics=metrics, device=device)
    pbar = ProgressBar()
    pbar.attach(evaluator)
    print("Running evaluation of quantized model")
    evaluator.run(dataloaders.val_loader)
    q_metrics = evaluator.state.metrics
    
    metrics = {"top_1_accuracy": Accuracy(), "top_5_accuracy": TopKCategoricalAccuracy()}
    evaluator = create_supervised_evaluator(model=fp_model, metrics=metrics, device=device)
    pbar = ProgressBar()
    pbar.attach(evaluator)
    print("Running evaluation of re-assigned FP32 model")
    evaluator.run(dataloaders.val_loader)
    fp_metrics = evaluator.state.metrics
    
    print("FP32 model evaluation metrics:")
    print(fp_metrics)
    print("Quantized model evaluation metrics:")
    print(q_metrics)
    
    # export ONNX
    print("Exporting to ONNX...")
    with torch.no_grad():
        torch.onnx.export(
            fp_model,
            dummy_input,
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
        print(f"ONNX model saved at {onnx_path}")

if __name__ == "__main__":
    export()

#time CUDA_VISIBLE_DEVICES=0 python export_quantized.py onnx-export --architecture mobilenet_v2_quantized --load-type fp32 --out mobilenet_v2_qat_check_v1.onnx --model-dir ~/workspace/QAT/repos/oscillations-qat/pretrained/mobilenet_v2.pth.tar --images-dir /data/data/imagenet/
"""
time CUDA_VISIBLE_DEVICES=7 python export_quantized.py onnx-export 
    --architecture mobilenet_v2_quantized 
    --load-type quantized 
    --onnx-path ~/workspace/QAT/repos/oscillation/export/mb_v2_4bit_oscillations_2.onnx 
    --model-dir ~/workspace/QAT/repos/oscillation/runs/2025-03-10/reg_loss/oscillations/4bits/mobilenet_v2/exp2/final.pth 
    --images-dir /data/data/imagenet/ 
    --act-quant-method MSE 
    --weight-quant-method MSE 
    --n-bits 4
"""