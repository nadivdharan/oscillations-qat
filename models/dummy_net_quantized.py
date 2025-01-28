from models.dummy_net import DummyNet
from quantization.base_quantized_model import QuantizedModel
from quantization.autoquant_utils import quantize_sequential, quantize_model

class QuantizedDummyNet(QuantizedModel):
    def __init__(self, model_fp, quant_setup=None, **quant_params):
        super().__init__()
        # quantize_input = quant_setup and quant_setup == "LSQ_paper"
        self.model_fp = model_fp
        # self.features = quantize_sequential(
        self.features = quantize_model(
            model_fp.features,
            # tie_activation_quantizers=not quantize_input,
            **quant_params,
        )
        
    def forward(self, x):
        out = self.features(x)
        return out
    
def dummynet_quantized(pretrained=False, load_type=None, model_dir=None, **qparams):
    fp_model = DummyNet()
    quant_model = QuantizedDummyNet(fp_model, **qparams)
    return quant_model