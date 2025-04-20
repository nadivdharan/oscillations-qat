import argparse
import os
from pathlib import Path
from time import time

cmd = "CUDA_VISIBLE_DEVICES={cuda} python main.py train-quantized  --architecture {arch}_quantized --images-dir /data/data/imagenet/ --act-quant-method MSE  --weight-quant-method MSE --optimizer SGD --weight-decay 2.5e-05 --sep-quant-optimizer --quant-optimizer Adam --quant-learning-rate 1e-5 --quant-weight-decay 0.0 --learning-rate-schedule cosine:0 --n-bits {nbits} --learning-rate {lr} --progress-bar --save-checkpoint-dir {save_dir}  --max-epochs {epochs} --no-reestimate-bn-stats"
NUM_EXPS = 4
BASE_DIR = "/home/nadivd/workspace/QAT/repos/oscillation/runs/2025-03-10/"
# EXP_TYPES = ["regularization", "knowledge_distillation", "batchnorm_folding"]
LEARNING_RATES = {
    2: 0.01,
    3: 0.01,
    4: 0.0033,
    8: 0.0033,
}
EXP_TYPES = {
    "vanilla":
        [
            "reg_loss/none",
        ],
    "regularization":
        [
            "reg_loss/oscillations",
            "reg_loss/bin_reg",
            "reg_loss/smoothing",
        ],
    "knowledge_distillation":
        [
            "reg_loss/distillation/kl_loss",
            "reg_loss/distillation/kl_no_ce_loss",
            "reg_loss/distillation/qfd/with_hooks",
        ],                
    "batchnorm_folding":
        [
            "bn_fold/fold_quant/",
            "bn_fold/fold_quant_update_bn/",
        ]
}

def parse_args():
    parser = argparse.ArgumentParser(description="Run all tests")
    parser.add_argument(
        "--cuda",
        type=int,
        default=None,
        help="GPU ID to use",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        choices=["mobilenet_v2", "resnet18", "resnet50"],
        help="Model architecture",
    )
    parser.add_argument(
        "--nbits",
        type=int,
        default=None,
        help="Number of bits for quantization",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=12,
        help="Number of training epochs",
    )
    # parser.add_argument(
    #     "--save-dir",
    #     type=str,
    #     default=None,
    #     help="Directory to save the results",
    # )
    parser.add_argument(
        "--exp-type",
        type=str,
        default=None,
        choices=[exp for exp in EXP_TYPES],
        help="Type of QAT experiment",
    )
    return parser.parse_args()

def get_more_args(cmd2run, exp_type, run_type):
    if exp_type == "regularization":
        if "oscillations" in run_type:
            cmd2run += " --oscillations-dampen-weight 0 --oscillations-dampen-weight-final 0.1"
        elif "bin_reg" in run_type:
            cmd2run += " --bin-regularization-weight 0.5"
        elif "smoothing" in run_type:
            cmd2run += " --smoothing-factor 0.9999"
        else:
            raise ValueError(f"Unknown regularization type: {run_type}")
    elif exp_type == "knowledge_distillation":
        if "kl_loss" in run_type:
            cmd2run += " --distillation-weight 0.5 --distillation-temperature  1.0  --distillation-target logits --distillation-loss-type kl_div"
        elif "kl_no_ce_loss" in run_type:
            cmd2run += " --distillation-weight 1.0 --distillation-temperature  1.0  --distillation-target logits --distillation-loss-type kl_div"
        elif "qfd" in run_type:
            cmd2run += " --distillation-weight 0.5 --distillation-temperature  1.0  --distillation-target qfd --distillation-loss-type mse"
        else:
            raise ValueError(f"Unknown knowledge distillation type: {run_type}")
    elif exp_type == "batchnorm_folding":
        if "update" in run_type:
            cmd2run += " --bn-folding update"
        else:
            cmd2run += " --bn-folding naive"
    elif exp_type == "vanilla":
        pass
    else:
        raise ValueError(f"Unknown experiment type: {exp_type}")
    
    return cmd2run
            
def main(args):
    problems = []
    
    assert args.cuda is not None
    assert args.nbits is not None
    assert args.arch is not None
    # assert args.save_dir is not None
    assert args.exp_type is not None
    
    cuda = args.cuda
    nbits = args.nbits
    arch = args.arch
    # save_dir = args.save_dir
    exp_type = args.exp_type
    epochs = args.epochs
    lr = LEARNING_RATES[nbits]
    
    for run_type in EXP_TYPES[exp_type]:
        # Create the directory for each experiment
        run_dir = os.path.join(BASE_DIR, run_type, f'{nbits}bits', arch)
        os.makedirs(run_dir, exist_ok=True)
        for exp_id in range(NUM_EXPS):
        # for exp_id in range(2, NUM_EXPS):
            run_dir_exp = os.path.join(run_dir, f'exp{exp_id}/')
            os.makedirs(run_dir_exp, exist_ok=True)
            
            cmd2run = cmd.format(cuda=cuda,
                                 arch=arch,
                                 nbits=nbits,
                                 epochs=epochs,
                                 lr=lr,
                                 save_dir=run_dir_exp)
            cmd2run = get_more_args(cmd2run, exp_type, run_type)
            cmd2run += " > {}/log.txt".format(run_dir_exp)

            print(f"Running command: {cmd2run}")
            os.system(f"echo {cmd2run} > {run_dir_exp}/cmd")
            # try:
            if True:
                t0 = time()
                res = os.system(cmd2run)
                # os.system(f"rm {run_dir_exp}/*.pth")
                # os.system(f"rm {run_dir_exp}/*.pt")
                t1 = time()
                print(f"Experiment run time: {((t1 - t0)/60):.2f} minutes")
            # except RuntimeError:
            if res != 0:
                print(f"Error running command: {cmd2run}")
                problems.append((run_type, exp_id))
                continue
    if problems:
        print("Problems encountered during the following experiments:")
        for run_type, exp_id in problems:
            print(f"Run type: {run_type}, Experiment ID: {exp_id}")
            

if __name__ == '__main__':
    args = parse_args()
    main(args)
