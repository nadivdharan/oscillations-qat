#!/usr/bin/env python
# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

from ignite.contrib.handlers import TensorboardLogger
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import Checkpoint, global_step_from_engine
from torch.optim import Optimizer

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from ignite.engine import Engine
from typing import Any, Callable, Optional, Union, Sequence, Tuple
from ignite.engine import _prepare_batch, _check_arg
from ignite.engine.deterministic import DeterministicEngine

from utils.qat_utils import get_fp32_model
from quantization.autoquant_utils import QuantLinear
from quantization.quantization_manager import QuantizationManager


def kd_loss_fn(student_output, teacher_output, temperature=1.0, loss_type="kl_div"):            
    if loss_type == "kl_div":
        kd_loss = F.kl_div(
            F.log_softmax(student_output / temperature, dim=1),
            F.softmax(teacher_output / temperature, dim=1),
            reduction="batchmean"
        ) * (temperature ** 2)
    elif loss_type == "mse":
        kd_loss = F.mse_loss(student_output, teacher_output, reduction="mean")
    else:
        raise ValueError(f"Unknown KD-loss type: {loss_type}")
    return kd_loss
            
def get_classifier_activation_quantizer(linear_layer, mode='eval'):
    if mode not in ['eval', 'train']:
        raise ValueError(f"Unknown mode: {mode}")    
    quantizer = None
    # ResNet
    if isinstance(linear_layer, QuantLinear) and (
        hasattr(linear_layer, "quantize_input") and
        linear_layer.quantize_input and
        linear_layer._quant_a
    ):
        if mode =='eval':
            # Freeze the quantizer parameters
            quantizer = copy.deepcopy(linear_layer.activation_quantizer)
            for param in quantizer.parameters():
                param.requires_grad = False
            return quantizer
        else:
            return linear_layer.activation_quantizer
    
    # MobileNetV2, EfficientNet
    elif isinstance(linear_layer, nn.Sequential):
        for module in linear_layer:
            if isinstance(module, QuantLinear):
                if (hasattr(module, "quantize_input") and
                    module.quantize_input and
                    module._quant_a
                ):
                    break
        if mode == 'eval':
            # Freeze the quantizer parameters
            quantizer = copy.deepcopy(module.activation_quantizer)
            for param in quantizer.parameters():
                param.requires_grad = False
            return quantizer
        else:
            return module.activation_quantizer
    else:
        raise ValueError(f"Expected classifier with type QuantLinear or nn.Sequential but got {type(fc)}")

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
    activations_dict: Optional[dict] = None,

) -> Callable:
    if gradient_accumulation_steps <= 0:
        raise ValueError(
            "Gradient_accumulation_steps must be strictly positive. "
            "No gradient accumulation if the value set to one (default)."
        )
    def update(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        if (engine.state.iteration - 1) % gradient_accumulation_steps == 0:
            optimizer.zero_grad()
        model.train()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        with torch.no_grad():
            output_fp32 = model_fn(model_fp32, x)
            features_fp32 = activations_dict.get("model_fp32_avgpool", None) # For QFD
            y_pred_fp32 = model_transform(output_fp32)
        output = model_fn(model, x)
        features = activations_dict.get("model_avgpool", None) # For QFD
        y_pred = model_transform(output)
        
        # For QFD
        quantizer = None
        if 'qfd' in config.distillation.target:
            # Get the classifier layer
            if hasattr(model, 'fc'):
                assert hasattr(model_fp32, 'fc')
                linear = model.fc
            elif hasattr(model, 'classifier'):
                assert hasattr(model_fp32, 'classifier')
                linear = model.classifier
            else:
                raise ValueError("Model does not have a recognizable classifier layer")

            # Get the classifier quantizer for the FP32 teacher (frozen)
            quantizer_fp32 = get_classifier_activation_quantizer(linear, mode='eval')
            if not (isinstance(quantizer_fp32, QuantizationManager) or quantizer is None):
                raise ValueError("Activation quantizer not found in the classifier")
            # Quantize the teacher's features for distillation
            assert features_fp32 is not None
            with torch.no_grad():   
                features_fp32 = quantizer_fp32(features_fp32.detach())

            # Get the classifier quantizer for the Quantized student (trainable)
            quantizer = get_classifier_activation_quantizer(linear, mode='train')
            if not (isinstance(quantizer, QuantizationManager) or quantizer is None):
                raise ValueError("Activation quantizer not found in the classifier")
            
            # Quantize the student's features for distillation
            assert features is not None
            features = quantizer(features)
                
            # Calculate the feature-based distillation loss
            loss_kd = kd_loss_fn(features,
                                 features_fp32,
                                 temperature=config.distillation.temperature,
                                 loss_type=config.distillation.loss_type)
        else:
            # Calculate the logits-based distillation loss
            loss_kd = kd_loss_fn(y_pred,
                                 y_pred_fp32,
                                 temperature=config.distillation.temperature,
                                 loss_type=config.distillation.loss_type)

        loss_kd = config.distillation.weight * loss_kd
        # Calculate the standard loss (non-KD loss such as CE loss)
        if config.distillation.weight == 1.:
            loss = 0.
        else:
            loss = (1. - config.distillation.weight) * loss_fn(y_pred, y)
        loss += loss_kd
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
        loss.backward()
        if engine.state.iteration % gradient_accumulation_steps == 0:
            optimizer.step()
        return output_transform(x, y, y_pred, loss * gradient_accumulation_steps)

    return update

def get_activations_hook(activations_dict, model_name, layer_name):
    def hook(module, input, output):
        activations_dict[f"{model_name}_{layer_name}"] = output.detach()
    return hook


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

    activations_dict = {}
    if config.distillation.target == 'qfd':
        # Register hooks to get the (un-quantized) activations of the last layer
        if hasattr(model, 'avgpool'):
            assert hasattr(model_fp32, 'avgpool')
            model.avgpool.register_forward_hook(get_activations_hook(activations_dict, "model", "avgpool"))
            model_fp32.avgpool.register_forward_hook(get_activations_hook(activations_dict, "model_fp32", "avgpool"))
        else:
            model.features[-1].register_forward_hook(get_activations_hook(activations_dict, "model", "avgpool"))
            model_fp32.features[-1].register_forward_hook(get_activations_hook(activations_dict, "model_fp32", "avgpool"))

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
        activations_dict,
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
