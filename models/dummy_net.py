import torch
import torch .nn as nn
import math

from quantization.base_quantized_model import QuantizedModel
from quantization.autoquant_utils import quantize_sequential

__all__ = ["DummyNet"]


class DummyNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=6):
        super().__init__()

        batch_norm = True
        # batch_norm = False

        self.batch_norm = batch_norm
        features = [
            nn.Conv2d(in_channels=in_ch,
                      out_channels=out_ch,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.Conv2d(in_channels=out_ch,
                      out_channels=2*out_ch,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False)
        ]
        if batch_norm:
            features.append(nn.BatchNorm2d(2*out_ch))
        
        self.features = nn.Sequential(*features)
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    
    def forward(self, x):
        out = self.features(x) 
        return out
