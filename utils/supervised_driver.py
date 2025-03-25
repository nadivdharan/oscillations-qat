#!/usr/bin/env python
# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

from ignite.contrib.handlers import TensorboardLogger
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import Checkpoint, global_step_from_engine
from torch.optim import Optimizer

import torch
import torch.nn.functional as F

from ignite.engine import Engine
from typing import Any, Callable, Optional, Union, Sequence, Tuple
from ignite.engine import _prepare_batch, _check_arg
from ignite.engine.deterministic import DeterministicEngine

from utils.qat_utils import get_fp32_model


def kd_loss_fn(student_logits, teacher_logits, temperature=1.0):            
    kd_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean"
    ) * (temperature ** 2)
    return kd_loss

def distillation_supervised_step(
    model_fp32: torch.nn.Module,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Union[Callable[[Any, Any], torch.Tensor], torch.nn.Module],
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
    prepare_batch: Callable = _prepare_batch,
    model_transform: Callable[[Any], Any] = lambda output: output,
    output_transform: Callable[[Any, Any, Any, torch.Tensor], Any] = lambda x, y, y_pred, loss: loss.item(),
    gradient_accumulation_steps: int = 1,
    model_fn: Callable[[torch.nn.Module, Any], Any] = lambda model, x: model(x),
    config: Optional[dict] = None,

) -> Callable:
    if gradient_accumulation_steps <= 0:
        raise ValueError(
            "Gradient_accumulation_steps must be strictly positive. "
            "No gradient accumulation if the value set to one (default)."
        )

    def update(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        if (engine.state.iteration - 1) % gradient_accumulation_steps == 0:
            optimizer.zero_grad()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        with torch.no_grad():
            output_fp32 = model_fn(model_fp32, x)
            y_pred_fp32 = model_transform(output_fp32)
        output = model_fn(model, x)
        y_pred = model_transform(output)

        loss_kd = config.distillation.weight * kd_loss_fn(y_pred, y_pred_fp32, temperature=config.distillation.temperature)
        loss = (1. - config.distillation.weight) * loss_fn(y_pred, y)
        loss += loss_kd

        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
        loss.backward()
        if engine.state.iteration % gradient_accumulation_steps == 0:
            optimizer.step()
        return output_transform(x, y, y_pred, loss * gradient_accumulation_steps)

    return update

def create_distilling_supervised_trainer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Union[Callable[[Any, Any], torch.Tensor], torch.nn.Module],
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
    prepare_batch: Callable = _prepare_batch,
    model_transform: Callable[[Any], Any] = lambda output: output,
    output_transform: Callable[[Any, Any, Any, torch.Tensor], Any] = lambda x, y, y_pred, loss: loss.item(),
    deterministic: bool = False,
    amp_mode: Optional[str] = None,
    scaler: Union[bool, "torch.cuda.amp.GradScaler"] = False,
    gradient_accumulation_steps: int = 1,
    model_fn: Callable[[torch.nn.Module, Any], Any] = lambda model, x: model(x),
    config: Optional[dict] = None,
) -> Engine:
    
    device_type = device.type if isinstance(device, torch.device) else device
    on_tpu = "xla" in device_type if device_type is not None else False
    on_mps = "mps" in device_type if device_type is not None else False
    mode, _scaler = _check_arg(on_tpu, on_mps, amp_mode, scaler)
    
    model_fp32 = get_fp32_model(config)
    # Freeze 
    for param in model_fp32.parameters():
        param.requires_grad = False
    model_fp32.to(device).eval()

    _update = distillation_supervised_step(
        model_fp32,
        model,
        optimizer,
        loss_fn,
        device,
        non_blocking,
        prepare_batch,
        model_transform,
        output_transform,
        gradient_accumulation_steps,
        model_fn,
        config,
    )
    trainer = Engine(_update) if not deterministic else DeterministicEngine(_update)
    if _scaler and scaler and isinstance(scaler, bool):
        trainer.state.scaler = _scaler  # type: ignore[attr-defined]

    return trainer
    
def create_trainer_engine(
    model,
    optimizer,
    criterion,
    metrics,
    data_loaders,
    lr_scheduler=None,
    save_checkpoint_dir=None,
    device="cuda",
    distill=False,
    config=None,
):
    # Create trainer
    if not distill:
        trainer = create_supervised_trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=criterion,
            device=device,
            output_transform=custom_output_transform,
        )
    else:
        print("Info: Creating Supervised Trainer with Knowledge-Distillation-Loss ")
        trainer = create_distilling_supervised_trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=criterion,
            device=device,
            output_transform=custom_output_transform,
            config=config,
        )
        

    for name, metric in metrics.items():
        metric.attach(trainer, name)

    # Add lr_scheduler
    if lr_scheduler:
        trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: lr_scheduler.step())

    # Create evaluator
    evaluator = create_supervised_evaluator(model=model, metrics=metrics, device=device)

    # Save model checkpoint
    if save_checkpoint_dir:
        to_save = {"model": model, "optimizer": optimizer}
        if lr_scheduler:
            to_save["lr_scheduler"] = lr_scheduler
        checkpoint = Checkpoint(
            to_save,
            save_checkpoint_dir,
            n_saved=1,
            global_step_transform=global_step_from_engine(trainer),
        )
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint)

    # Add hooks for logging metrics
    trainer.add_event_handler(Events.EPOCH_COMPLETED, log_training_results, optimizer)

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED, run_evaluation_for_training, evaluator, data_loaders.val_loader
    )

    return trainer, evaluator


def custom_output_transform(x, y, y_pred, loss):
    return y_pred, y


def log_training_results(trainer, optimizer):
    learning_rate = optimizer.param_groups[0]["lr"]
    log_metrics(trainer.state.metrics, "Training", trainer.state.epoch, learning_rate)


def run_evaluation_for_training(trainer, evaluator, val_loader):
    evaluator.run(val_loader)
    log_metrics(evaluator.state.metrics, "Evaluation", trainer.state.epoch)


def log_metrics(metrics, stage: str = "", training_epoch=None, learning_rate=None):
    log_text = "  {}".format(metrics) if metrics else ""
    if training_epoch is not None:
        log_text = "Epoch: {}".format(training_epoch) + log_text
    if learning_rate and learning_rate > 0.0:
        log_text += "  Learning rate: {:.2E}".format(learning_rate)
    log_text = "Results - " + log_text
    if stage:
        log_text = "{} ".format(stage) + log_text
    print(log_text, flush=True)


def setup_tensorboard_logger(trainer, evaluator, output_path, optimizers=None):
    logger = TensorboardLogger(logdir=output_path)

    # Attach the logger to log loss and accuracy for both training and validation
    for tag, cur_evaluator in [("train", trainer), ("validation", evaluator)]:
        logger.attach_output_handler(
            cur_evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag=tag,
            metric_names="all",
            global_step_transform=global_step_from_engine(trainer),
        )

    # Log optimizer parameters
    if isinstance(optimizers, Optimizer):
        optimizers = {None: optimizers}

    for k, optimizer in optimizers.items():
        logger.attach_opt_params_handler(
            trainer, Events.EPOCH_COMPLETED, optimizer, param_name="lr", tag=k
        )

    return logger
