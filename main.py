# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.
import logging
import os

import click
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events, create_supervised_evaluator
from ignite.metrics import Accuracy, TopKCategoricalAccuracy, Loss
from torch.nn import CrossEntropyLoss
import torch

from quantization.utils import (
    pass_data_for_range_estimation,
    separate_quantized_model_params,
    set_range_estimators,
)
from utils import DotDict, CosineTempDecay
from utils.click_options import (
    qat_options,
    quantization_options,
    quant_params_dict,
    base_options,
    multi_optimizer_options,
)
from utils.optimizer_utils import optimizer_lr_factory
from utils.oscillation_tracking_utils import add_oscillation_trackers
from utils.qat_utils import (
    get_dataloaders_and_model,
    MethodPropagator,
    DampeningLoss,
    CompositeLoss,
    UpdateDampeningLossWeighting,
    UpdateFreezingThreshold,
    ReestimateBNStats,
    BinRegularizationLoss,
    QuantSmoother,
    ModelChecker
)
from utils.supervised_driver import create_trainer_engine, setup_tensorboard_logger, log_metrics


# setup stuff
class Config(DotDict):
    pass


@click.group()
def oscillations():
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


pass_config = click.make_pass_decorator(Config, ensure=True)


@oscillations.command()
@pass_config
@base_options
@multi_optimizer_options()
@quantization_options
@qat_options
def train_quantized(config):
    """
    Main QAT function
    """

    print("\n==========================")
    print(f'Seed: {config.base.seed}')
    print("==========================\n")

    print("Setting up network and data loaders")
    qparams = quant_params_dict(config)

    dataloaders, model = get_dataloaders_and_model(config, **qparams)

    # Estimate ranges using training data
    pass_data_for_range_estimation(
        loader=dataloaders.train_loader,
        model=model,
        act_quant=config.quant.act_quant,
        weight_quant=config.quant.weight_quant,
        max_num_batches=config.quant.num_est_batches,
    )

    # Put quantizers in desirable state
    if config.qat.sep_quant_optimizer and not config.qat.learn_ranges:
        raise ValueError("Seperate optimizer for quantized params is supported only when "
                         f"ranges are learnable, but got learn_ranges={config.qat.learn_ranges}")
    set_range_estimators(config, model, learn_ranges=config.qat.learn_ranges)

    print("Loaded model:\n{}".format(model))
    # import ipdb; ipdb.set_trace()

    # Get all models parameters in  subcategories
    quantizer_params, model_params, grad_params = separate_quantized_model_params(model)
    model_optimizer, quant_optimizer = None, None
    if config.qat.sep_quant_optimizer:
        # Separate optimizer for model and quantization parameters
        model_optimizer, model_lr_scheduler = optimizer_lr_factory(
            config.optimizer, model_params, config.base.max_epochs
        )
        quant_optimizer, quant_lr_scheduler = optimizer_lr_factory(
            config.quant_optimizer, quantizer_params, config.base.max_epochs
        )

        optimizer = MethodPropagator([model_optimizer, quant_optimizer])
        lr_schedulers = [s for s in [model_lr_scheduler, quant_lr_scheduler] if s is not None]
        lr_scheduler = MethodPropagator(lr_schedulers) if len(lr_schedulers) else None
    else:
        optimizer, lr_scheduler = optimizer_lr_factory(
            config.optimizer, quantizer_params + model_params, config.base.max_epochs
        )

    print("Optimizer:\n{}".format(optimizer))
    print(f"LR scheduler\n{lr_scheduler}")

    # Define metrics for ingite engine
    metrics = {"top_1_accuracy": Accuracy(), "top_5_accuracy": TopKCategoricalAccuracy()}

    # Set-up losses
    task_loss_fn = CrossEntropyLoss()
    dampening_loss = None
    bin_reg_loss = None
    if config.osc_damp.weight is not None:
        # Add dampening loss to task loss
        dampening_loss = DampeningLoss(model, config.osc_damp.weight, config.osc_damp.aggregation)
        loss_dict = {"task_loss": task_loss_fn, "dampening_loss": dampening_loss}
        loss_func = CompositeLoss(loss_dict)
        loss_metrics = {
            "task_loss": Loss(task_loss_fn),
            "dampening_loss": Loss(dampening_loss),
            "loss": Loss(loss_func),
        }
    elif config.bin_reg.weight is not None:
        print("Info: Building Bin-Regularization-Loss ")
        bin_reg_loss = BinRegularizationLoss(model, config.bin_reg.weight)
        loss_dict = {"task_loss": task_loss_fn, "bin_reg_loss": bin_reg_loss}
        loss_func = CompositeLoss(loss_dict)
        loss_metrics = {
            "task_loss": Loss(task_loss_fn),
            "bin_reg_loss": Loss(bin_reg_loss),
            "loss": Loss(loss_func),
        }
    else:
        loss_func = task_loss_fn
        loss_metrics = {"loss": Loss(loss_func)}

    metrics.update(loss_metrics)

    # Set up ignite trainer and evaluator
    trainer, evaluator = create_trainer_engine(
        model=model,
        optimizer=optimizer,
        criterion=loss_func,
        data_loaders=dataloaders,
        metrics=metrics,
        lr_scheduler=lr_scheduler,
        save_checkpoint_dir=config.base.save_checkpoint_dir,
        device="cuda" if config.base.cuda else "cpu",
        distill=config.distillation.weight is not None,
        config=config,
    )

    if config.base.progress_bar:
        pbar = ProgressBar()
        pbar.attach(trainer)
        pbar.attach(evaluator)

    # Create TensorboardLogger
    if config.base.tb_logging_dir:
        if config.qat.sep_quant_optimizer:
            optimizers_dict = {"model": model_optimizer, "quant_params": quant_optimizer}
        else:
            optimizers_dict = optimizer
        tb_logger = setup_tensorboard_logger(
            trainer, evaluator, config.base.tb_logging_dir, optimizers_dict
        )

    if config.osc_damp.weight_final:
        # Apply cosine annealing of dampening loss
        total_iterations = len(dataloaders.train_loader) * config.base.max_epochs
        annealing_schedule = CosineTempDecay(
            t_max=total_iterations,
            temp_range=(config.osc_damp.weight, config.osc_damp.weight_final),
            rel_decay_start=config.osc_damp.anneal_start,
        )
        print(f"Weight gradient parameter cosine annealing schedule:\n{annealing_schedule}")
        trainer.add_event_handler(
            Events.ITERATION_STARTED,
            UpdateDampeningLossWeighting(dampening_loss, annealing_schedule),
        )

    # Evaluate model
    print("Running evaluation before training")
    evaluator.run(dataloaders.val_loader)
    log_metrics(evaluator.state.metrics, "Evaluation", trainer.state.epoch)

    # import ipdb; ipdb.set_trace()
    # BN Re-estimation
    if config.qat.reestimate_bn_stats:
        evaluator.add_event_handler(
            Events.EPOCH_STARTED, ReestimateBNStats(model, dataloaders.train_loader)
        )

    # Add oscillation trackers to the model and set up oscillation freezing
    if config.osc_freeze.threshold:
        oscillation_tracker_dict = add_oscillation_trackers(
            model,
            max_bits=config.osc_freeze.max_bits,
            momentum=config.osc_freeze.ema_momentum,
            freeze_threshold=config.osc_freeze.threshold,
            use_ema_x_int=config.osc_freeze.use_ema,
        )

        if config.osc_freeze.threshold_final:
            # Apply cosine annealing schedule to the freezing threshdold
            total_iterations = len(dataloaders.train_loader) * config.base.max_epochs
            annealing_schedule = CosineTempDecay(
                t_max=total_iterations,
                temp_range=(config.osc_freeze.threshold, config.osc_freeze.threshold_final),
                rel_decay_start=config.osc_freeze.anneal_start,
            )
            print(f"Oscillation freezing annealing schedule:\n{annealing_schedule}")
            trainer.add_event_handler(
                Events.ITERATION_STARTED,
                UpdateFreezingThreshold(oscillation_tracker_dict, annealing_schedule),
            )
    elif config.smoothing.factor:
        # Apply smoothing to the model
        smoothing_factor = config.smoothing.factor
        print(f"Applying smoothing factor {smoothing_factor} to the model")
        # trainer.add_event_handler(
        #         Events.ITERATION_STARTED,
        #         ModelChecker(model),
        # )
        trainer.add_event_handler(
                Events.ITERATION_COMPLETED,
                QuantSmoother(model, alpha=smoothing_factor),
        )

    print("Starting training")

    trainer.run(dataloaders.train_loader, max_epochs=config.base.max_epochs)

    print("Finished training")

    sd = model.state_dict()
    final_ckpt_path = str(config.base.save_checkpoint_dir) + '/final.pth'
    torch.save(sd, final_ckpt_path)
    print(f"Final checkpoint saved at {final_ckpt_path}")

@oscillations.command()
@pass_config
@base_options
@quantization_options
@click.option(
    "--load-type",
    type=click.Choice(["fp32", "quantized"]),
    default="quantized",
    help='Either "fp32", or "quantized". Specify weather to load a quantized or a FP ' "model.",
)
def validate_quantized(config, load_type):
    """
    function for running validation on pre-trained quantized models
    """
    print("Setting up network and data loaders")

    # validate_fp32_model = True
    validate_fp32_model = False

    if not validate_fp32_model:
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
        # from utils.qat_utils import reestimate_BN_stats
        # # reestimate_BN_stats(model, dataloaders.train_loader, num_batches=50, store_ema_stats=False)
        # # import ipdb; ipdb.set_trace()
    
    else:
        # # Validate FP32 model
        from torchvision.models import resnet18
        from models.mobilenet_v2 import MobileNetV2
        from utils.imagenet_dataloaders import ImageNetDataLoaders

        dataloaders = ImageNetDataLoaders(
            config.base.images_dir,
            224,
            config.base.batch_size,
            config.base.num_workers,
            config.base.interpolation,
        )
        
        test_model = 'resnet_18'
        # test_model = 'mobilenet_v2'

        if test_model == 'resnet_18':
            model = resnet18(pretrained=True)
        elif test_model == 'mobilenet_v2':
            model = MobileNetV2()
            # Load model from pretrained FP32 weights
            model_dir = config.base.model_dir
            assert os.path.exists(model_dir)
            print(f"Loading pretrained weights from {model_dir}")
            state_dict = torch.load(model_dir)
            model.load_state_dict(state_dict)
        model.to("cuda:0" if config.base.cuda else "cpu")
    print("Loaded model:\n{}".format(model))

    # Create evaluator
    loss_func = CrossEntropyLoss()
    metrics = {
        "top_1_accuracy": Accuracy(),
        "top_5_accuracy": TopKCategoricalAccuracy(),
        "loss": Loss(loss_func),
    }

    pbar = ProgressBar()
    evaluator = create_supervised_evaluator(
        model=model, metrics=metrics, device="cuda" if config.base.cuda else "cpu"
    )
    pbar.attach(evaluator)
    print("Start quantized validation")
    evaluator.run(dataloaders.val_loader)
    final_metrics = evaluator.state.metrics
    print(final_metrics)


if __name__ == "__main__":
    oscillations()
