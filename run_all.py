import argparse
import os
from pathlib import Path
from time import time
# import subprocess
# import torch
# import glob

cmd = "CUDA_VISIBLE_DEVICES={cuda} python main.py train-quantized  --architecture {arch}_quantized --images-dir /data/data/imagenet/ --act-quant-method MSE  --weight-quant-method MSE --optimizer SGD --weight-decay 2.5e-05 --sep-quant-optimizer --quant-optimizer Adam --quant-learning-rate 1e-5 --quant-weight-decay 0.0 --learning-rate-schedule {lr_schedule} --n-bits {nbits} --learning-rate {lr} --progress-bar --save-checkpoint-dir {save_dir}  --max-epochs {epochs} --no-reestimate-bn-stats --tb-logging-dir {tb_dir}"
# cmd = "python main.py train-quantized  --architecture {arch}_quantized --images-dir /data/data/imagenet/ --act-quant-method MSE  --weight-quant-method MSE --optimizer SGD --weight-decay 2.5e-05 --sep-quant-optimizer --quant-optimizer Adam --quant-learning-rate 1e-5 --quant-weight-decay 0.0 --learning-rate-schedule cosine:0 --n-bits {nbits} --learning-rate {lr} --progress-bar --save-checkpoint-dir {save_dir}  --max-epochs {epochs} --no-reestimate-bn-stats"
NUM_EXPS = 2
BASE_DIR = "/home/nadivd/workspace/QAT/repos/oscillation/runs/2025-03-10/"
# EXP_TYPES = ["regularization", "knowledge_distillation", "batchnorm_folding"]
LEARNING_RATES = {
    2: 0.01,
    3: 0.01,
    4: 0.0033,
    8: 0.0033,
}

SCHEDULES = {
    "multistep": {
        "milestones": "4:8:12"
        # "milestones": "13"
    },
    "cosine": {
        "eta_min": "0"
    },
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
    # "batchnorm_folding":
    #     [
    #         # "bn_fold/fold_quant/",
    #         # "bn_fold/fold_quant_update_bn/",
    #         "bn_fold/fold_quant_update_and_train_bn_again/",
    #         # "bn_fold/fold_quant_krishnamoorthi_train_bn/",
    #         # "bn_fold/fold_fp32_update_and_train_bn/",
    #     ]
    "batchnorm_folding":
        [
            # "bn_fold/fold_quant_update_and_train_bn_again/cosine/lr_1e-6",
            # "bn_fold/fold_quant_update_and_train_bn_again/cosine/lr_1e-5",
            # "bn_fold/fold_quant_update_and_train_bn_again/cosine_no_sep_opt/lr_1e-6",
            # "bn_fold/fold_quant_update_and_train_bn_again/multi_step/lr_1e-6",
            # "bn_fold/fold_quant_update_and_train_bn_again/multi_step/lr_1e-5",
            "bn_fold/fold_quant_update_and_train_bn_again/multi_step_no_sep_opt/lr_1e-6",
            # "bn_fold/fold_quant_update_and_train_bn_again/const_lr/lr_1e-6",
            # "bn_fold/fold_quant_update_and_train_bn_again/const_lr/lr_1e-5",
            # "bn_fold/fold_quant_again/multi_step_again/lr_1e-6",
            # "bn_fold/fold_quant_again/cosine/lr_1e-6",
            # "bn_fold/fold_quant_krishnamoorthi_train_bn/multi_step_again/lr_1e-6",
            # "bn_fold/fold_quant_krishnamoorthi_train_bn/cosine/lr_1e-6",
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
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default="cosine",
        choices=[sch for sch in SCHEDULES],
        help="Learning rate schedule to use",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate",
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
            if "fp32" in run_type or "no_sep_opt" in run_type:
                deletions = [" --sep-quant-optimizer",
                             " --quant-optimizer Adam",
                             " --quant-optimizer Adam",
                             " --quant-learning-rate 1e-5",
                             " --quant-weight-decay 0.0"]
                for deletion in deletions:
                    cmd2run = cmd2run.replace(deletion, "")
                if "fp232" in run_type:
                    cmd2run += " --no-weight-quant --no-act-quant"
        elif "krishnamoorthi" in run_type:
            cmd2run += " --bn-folding krishnamoorthi"
        else:
            cmd2run += " --bn-folding naive"
    elif exp_type == "vanilla":
        pass
    else:
        raise ValueError(f"Unknown experiment type: {exp_type}")
    
    return cmd2run
            
def get_lr_schedule(schedule_name):
    if schedule_name not in SCHEDULES:
        raise ValueError(f"Unknown optimizer: {schedule_name}")
    lr_schedule_key = list(SCHEDULES[schedule_name].keys())
    assert len(lr_schedule_key) == 1, f"Expected only one schedule key for {schedule_name}, got {len(lr_schedule_key)}"
    lr_schedule_key = lr_schedule_key[0]
    lr_schedule_val = SCHEDULES[schedule_name][lr_schedule_key]
    schedule = f"{schedule_name}:{lr_schedule_val}"
    return schedule

def main(args):
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
    lr = args.lr if args.lr is not None else LEARNING_RATES[nbits]
    lr_schedule = get_lr_schedule(args.lr_schedule)
    
    problems = []
    for run_type in EXP_TYPES[exp_type]:
        # Create the directory for each experiment
        run_dir = os.path.join(BASE_DIR, run_type, f'{nbits}bits', arch)
        os.makedirs(run_dir, exist_ok=True)
        for exp_id in range(NUM_EXPS):
        # for exp_id in range(2, NUM_EXPS):
            run_dir_exp = os.path.join(run_dir, f'exp{exp_id}/')
            os.makedirs(run_dir_exp, exist_ok=True)
            tb_dir = os.path.join(run_dir_exp, "tb_logs")
            cmd2run = cmd.format(cuda=cuda,
                                 arch=arch,
                                 nbits=nbits,
                                 epochs=epochs,
                                 lr=lr,
                                 save_dir=run_dir_exp,
                                 lr_schedule=lr_schedule,
                                 tb_dir=tb_dir)
            cmd2run = get_more_args(cmd2run, exp_type, run_type)
            cmd2run += " > {}/log.txt".format(run_dir_exp)

            print(f"Running command: {cmd2run}")
            os.system(f"echo {cmd2run} > {run_dir_exp}/cmd")

            if True:
                t0 = time()
                res = os.system(cmd2run)
                # os.system(f"rm {run_dir_exp}/*.pth")
                # os.system(f"rm {run_dir_exp}/*.pt")
                t1 = time()
                print(f"Experiment run time: {((t1 - t0)/60):.2f} minutes")
            if res != 0:
                print(f"Error running command: {cmd2run}")
                problems.append((run_type, exp_id))
                continue

            # logfile = "{}/log2.txt".format(run_dir_exp)
            # try:
            #     t0 = time()
            #     torch.cuda.synchronize()
            #     env = os.environ.copy()
            #     env["CUDA_VISIBLE_DEVICES"] = str(cuda)  
            #     # res = subprocess.run(cmd2run, shell=True, check=True, capture_output=True, text=True, env=env)
            #     # # os.system(f"rm {run_dir_exp}/*.pth")
            #     # # os.system(f"rm {run_dir_exp}/*.pt")
            #     # for pth_paths in glob.glob(f"{run_dir_exp}/*.pth") + glob.glob(f"{run_dir_exp}/*.pt"):
            #     #     try:
            #     #         os.remove(pth_paths)
            #     #         print(f"Removed files {pth_paths}")
            #     #     except Exception as e:
            #     #         print(f"Error removing file {pth_paths}: {e}")
            #     with open(logfile, "w") as flog:
            #         process = subprocess.Popen( 
            #             cmd2run,
            #             shell=True,
            #             stdout=subprocess.PIPE,
            #             stderr=subprocess.STDOUT,
            #             universal_newlines=True,
            #             env=env,
            #             bufsize=1  # Line buffered
            #         )
            #          # Stream output line-by-line
            #         for line in process.stdout:
            #             print(line, end="\r")       # Also print to console
            #             flog.write(line)       # Write to log
            #         process.wait()
            #     torch.cuda.synchronize()
            #     t1 = time()
            #     print(f"Experiment run time: {((t1 - t0)/60):.2f} minutes")
            #     # with open(logfile, 'a') as f:
            #     #     f.write(res.stdout)
            #     #     if not res.stderr:
            #     #         f.write("\n=== STDERR ===\n")
            #     #         f.write(res.stderr)
            #     #     f.write(f"Experiment run time: {((t1 - t0)/60):.2f} minutes\n")
            #     # print(f"Experiment run time: {((t1 - t0)/60):.2f} minutes")
            # except subprocess.CalledProcessError as e:
            #     print(f"Error running command: {cmd2run}")
            #     # print(f"Stdout: {e.stdout}")
            #     print(f"Stderr: {e.stderr}")
            #     print("-" * 80)
            #     errfile = "{}/error.txt".format(run_dir_exp)
            #     with open(errfile, 'a') as f:
            #         f.write(e.stdout)
            #         if not e.stderr:
            #             f.write("\n=== STDERR ===\n")
            #             f.write(e.stderr)
            #     problems.append((run_dir, exp_id, e.stdout, e.stderr))
            # finally:
            #     torch.cuda.empty_cache()
    if problems:
        print("-" * 80)
        print("\n\nProblems encountered during the following experiments:")
        # for run_dir, exp_id, stdout, stderr in problems:
        for run_dir, exp_id in problems:
            # print("-" * 80)
            print(f"Run type: {run_dir}, Experiment ID: {exp_id}")
            # print(f"Stdout: {stdout}")
            # print(f"Stderr: {stderr}")
        print("-" * 80)
            

if __name__ == '__main__':
    args = parse_args()
    main(args)
