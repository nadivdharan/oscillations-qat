#!/usr/bin/env python
# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.conv import _ConvNd

from quantization.hijacker import QuantizationHijacker


class BNFusedKrishnamoorthiHijacker(QuantizationHijacker):
    """Extension to the QuantizationHijacker that fuses batch normalization (BN) after a weight
    layer into a joined module. The parameters and the statistics of the BN layer remain in
    full-precision.
    """

    def __init__(self, *args, **kwargs):
        kwargs.pop("bias", None) # Bias will be learned by BN params
        super().__init__(*args, **kwargs, bias=False)
        bn_dim = self.get_bn_dim()
        self.bn_dim = bn_dim
        self.register_buffer("running_mean", torch.zeros(bn_dim))
        self.register_buffer("running_var", torch.ones(bn_dim))
        self.momentum = kwargs.pop("momentum", 0.1)
        self.gamma = nn.Parameter(torch.ones(bn_dim), requires_grad=False)
        self.beta = nn.Parameter(torch.zeros(bn_dim), requires_grad=False)
        self.epsilon = kwargs.get("eps", 1e-5)
        self.bias = None
        self.use_running_stats = False
        
    def forward(self, x):
        # Quantize input
        if self.quantize_input and self._quant_a:
            x = self.activation_quantizer(x)
            
        # Get batch stats
        if self.training:
            with torch.no_grad():
                res_for_stats = self.run_forward(x,
                                    self.weight.detach(),
                                    self.bias.detach() if self.bias is not None else None)
                if len(res_for_stats.shape) == 4:
                    # For 2D conv
                    batch_mean = torch.mean(res_for_stats, axis=(0, 2, 3))
                    batch_var = torch.var(res_for_stats, axis=(0, 2, 3))
                    batch_std = torch.sqrt(batch_var + self.epsilon)
                elif len(res_for_stats.shape) == 2:
                    # For 1D conv / linear
                    batch_mean = torch.mean(res_for_stats, axis=(0))
                    batch_var = torch.var(res_for_stats, axis=(0))
                    batch_std = torch.sqrt(batch_var + self.epsilon)
                else:
                    raise ValueError(f"Unsupported input shape: {res_for_stats.shape}")

        with torch.no_grad():
            gamma = self.gamma.detach()
            beta = self.beta.detach()
            running_mean = self.running_mean.detach()
            std = torch.sqrt(self.running_var.detach() + self.epsilon)
            if len(self.weight.shape) == 4:
                # For 2D conv
                weight = self.weight.detach() * (gamma / std).view(-1, 1, 1, 1)
            elif len(self.weight.shape) == 2:
                # For 1D conv / linear
                weight = self.weight.detach() * (gamma / std).view(-1, 1)
            else:
                raise ValueError(f"Unsupported input shape: {self.weight.shape}")              
            bias = (self.bias.detach() if self.bias is not None else torch.zeros_like(running_mean)) - (gamma * running_mean / std) + beta
        
        # Get quantized weight
        # weight, bias = self.get_params()
        if self._quant_w:
            weight = self.quantize_weights(weight)

        res = self.run_forward(x, weight, bias=None)

        if self.training and not self.use_running_stats:
            correction_factor = (std / batch_std).view(1, self.bn_dim, 1, 1)
            bias += gamma * (running_mean / std - batch_mean / batch_std)
        else:
            correction_factor = torch.ones_like(std).view(1, self.bn_dim, 1, 1)
        res *= correction_factor
        res += bias.view(1, self.bn_dim, 1, 1)

        
        # Apply fused activation function
        if self.activation_function is not None:
            res = self.activation_function(res)

        # Quantize output
        if not self.quantize_input and self._quant_a:
            res = self.activation_quantizer(res)
            
        # TODO Check if this is indeed necessary
        # Update running stats
        if self.training and not self.use_running_stats:
            with torch.no_grad():
                self.running_mean = (1. - self.momentum) * self.running_mean + self.momentum * batch_mean
                self.running_var  = (1. - self.momentum) * self.running_var  + self.momentum * batch_var

        return res
    
    def get_bn_dim(self):
        if isinstance(self, nn.Linear):
            return self.out_features
        elif isinstance(self, _ConvNd):
            return self.out_channels
        else:
            msg = (
                f"Unsupported type used: {self}. Must be a linear or (transpose)-convolutional "
                f"nn.Module"
            )
            raise NotImplementedError(msg)

class BNFusedUpdateHijacker(QuantizationHijacker):
    """Extension to the QuantizationHijacker that fuses batch normalization (BN) after a weight
    layer into a joined module. The parameters and the statistics of the BN layer remain in
    full-precision.
    """

    def __init__(self, *args, **kwargs):
        kwargs.pop("bias", None) # Bias will be learned by BN params
        super().__init__(*args, **kwargs, bias=False)
        bn_dim = self.get_bn_dim()
        self.register_buffer("running_mean", torch.zeros(bn_dim))
        self.register_buffer("running_var", torch.ones(bn_dim))
        self.momentum = kwargs.pop("momentum", 0.1)
        self.gamma = nn.Parameter(torch.ones(bn_dim), requires_grad=False)
        self.beta = nn.Parameter(torch.zeros(bn_dim), requires_grad=False)
        self.epsilon = kwargs.get("eps", 1e-5)
        self.bias = None

    def forward(self, x):
        # Quantize input
        if self.quantize_input and self._quant_a:
            x = self.activation_quantizer(x)
            
        # Update running stats
        if self.training:
            with torch.no_grad():
                res_for_stats = self.run_forward(x,
                                    self.weight.detach(),
                                    self.bias.detach() if self.bias is not None else None)
        
        with torch.no_grad():
            gamma = self.gamma.detach()
            beta = self.beta.detach()
            running_mean = self.running_mean.detach()
            std = torch.sqrt(self.running_var.detach() + self.epsilon)
            if len(self.weight.shape) == 4:
                # For 2D conv
                weight = self.weight.detach() * (gamma / std).view(-1, 1, 1, 1)
            elif len(self.weight.shape) == 2:
                # For 1D conv / linear
                weight = self.weight.detach() * (gamma / std).view(-1, 1)
            else:
                raise ValueError(f"Unsupported input shape: {self.weight.shape}")              
            bias = (self.bias.detach() if self.bias is not None else torch.zeros_like(running_mean)) - (gamma * running_mean / std) + beta

        # Get quantized weight
        # weight, bias = self.get_params()
        if self._quant_w:
            weight = self.quantize_weights(weight)

        res = self.run_forward(x, weight, bias)
        
        # Apply fused activation function
        if self.activation_function is not None:
            res = self.activation_function(res)

        # Quantize output
        if not self.quantize_input and self._quant_a:
            res = self.activation_quantizer(res)
        
        if self.training:
            with torch.no_grad():
                # Get batch stats
                if len(res_for_stats.shape) == 4:
                    # For 2D conv
                    batch_mean = torch.mean(res_for_stats, axis=(0, 2, 3))
                    batch_var = torch.var(res_for_stats, axis=(0, 2, 3))
                elif len(res_for_stats.shape) == 2:
                    # For 1D conv / linear
                    batch_mean = torch.mean(res_for_stats, axis=(0))
                    batch_var = torch.var(res_for_stats, axis=(0))
                else:
                    raise ValueError(f"Unsupported input shape: {res_for_stats.shape}")
                # Update running stats
                self.running_mean = (1. - self.momentum) * self.running_mean + self.momentum * batch_mean
                self.running_var  = (1. - self.momentum) * self.running_var  + self.momentum * batch_var

        return res

    def get_bn_dim(self):
        if isinstance(self, nn.Linear):
            return self.out_features
        elif isinstance(self, _ConvNd):
            return self.out_channels
        else:
            msg = (
                f"Unsupported type used: {self}. Must be a linear or (transpose)-convolutional "
                f"nn.Module"
            )
            raise NotImplementedError(msg)

class BNFusedHijacker(QuantizationHijacker):
    """Extension to the QuantizationHijacker that fuses batch normalization (BN) after a weight
    layer into a joined module. The parameters and the statistics of the BN layer remain in
    full-precision.
    """

    def __init__(self, *args, **kwargs):
        kwargs.pop("bias", None)  # Bias will be learned by BN params
        super().__init__(*args, **kwargs, bias=False)
        bn_dim = self.get_bn_dim()
        self.register_buffer("running_mean", torch.zeros(bn_dim))
        self.register_buffer("running_var", torch.ones(bn_dim))
        self.momentum = kwargs.pop("momentum", 0.1)
        self.gamma = nn.Parameter(torch.ones(bn_dim))
        self.beta = nn.Parameter(torch.zeros(bn_dim))
        self.epsilon = kwargs.get("eps", 1e-5)
        self.bias = None

    def forward(self, x):
        # Quantize input
        if self.quantize_input and self._quant_a:
            x = self.activation_quantizer(x)

        # Get quantized weight
        weight, bias = self.get_params()
        res = self.run_forward(x, weight, bias)

        res = F.batch_norm(
            res,
            self.running_mean,
            self.running_var,
            self.gamma,
            self.beta,
            self.training,
            self.momentum,
            self.epsilon,
        )
        # Apply fused activation function
        if self.activation_function is not None:
            res = self.activation_function(res)

        # Quantize output
        if not self.quantize_input and self._quant_a:
            res = self.activation_quantizer(res)
        return res

    def get_bn_dim(self):
        if isinstance(self, nn.Linear):
            return self.out_features
        elif isinstance(self, _ConvNd):
            return self.out_channels
        else:
            msg = (
                f"Unsupported type used: {self}. Must be a linear or (transpose)-convolutional "
                f"nn.Module"
            )
            raise NotImplementedError(msg)

class BNHijacker(QuantizationHijacker):
    """Extension to the QuantizationHijacker that fuses batch normalization (BN) after a weight
    layer into a joined module. The parameters and the statistics of the BN layer remain in
    full-precision.
    """

    def __init__(self, *args, **kwargs):
        kwargs.pop("bias", None)  # Bias will be learned by BN params
        # super().__init__(*args, **kwargs, bias=False)
        super().__init__(*args, **kwargs)
        bn_dim = self.get_bn_dim()
        self.register_buffer("running_mean", torch.zeros(bn_dim))
        self.register_buffer("running_var", torch.ones(bn_dim))
        self.momentum = kwargs.pop("momentum", 0.1)
        self.gamma = nn.Parameter(torch.ones(bn_dim))
        self.beta = nn.Parameter(torch.zeros(bn_dim))
        self.epsilon = kwargs.get("eps", 1e-5)
        self.bias = None

    def forward(self, x):
        # Quantize input
        if self.quantize_input and self._quant_a:
            x = self.activation_quantizer(x)

        # Get quantized weight
        weight, bias = self.get_params()
        # res = self.run_forward(x, weight, bias)
        res = x

        res = F.batch_norm(
            res,
            self.running_mean,
            self.running_var,
            self.gamma,
            self.beta,
            self.training,
            self.momentum,
            self.epsilon,
        )
        # Apply fused activation function
        if self.activation_function is not None:
            res = self.activation_function(res)

        # Quantize output
        if not self.quantize_input and self._quant_a:
            res = self.activation_quantizer(res)
        return res

    def get_bn_dim(self):
        # if isinstance(self, nn.Linear):
        #     return self.out_features
        # elif isinstance(self, _ConvNd):
        #     return self.out_channels
        if isinstance(self, nn.BatchNorm2d):
            return self.num_features
        else:
            msg = (
                f"Unsupported type used: {self}. Must be a batch norm nn.Module"
            )
            raise NotImplementedError(msg)
