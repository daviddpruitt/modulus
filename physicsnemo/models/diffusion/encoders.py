import modulus.models.diffusion as diffusion
from dataclasses import dataclass
from typing import Union
import nvtx
from modulus.models.meta import ModelMetaData
from modulus.models.module import Module


@dataclass
class MetaData(ModelMetaData):
    name: str = "Conv2dSerializable"
    # Optimization
    jit: bool = True
    cuda_graphs: bool = True
    amp_cpu: bool = True
    amp_gpu: bool = True
    torch_fx: bool = False
    # Data type
    bf16: bool = True
    # Inference
    onnx: bool = False
    # Physics informed
    func_torch: bool = False
    auto_grad: bool = True


class Conv2dSerializable(Module):
    """
    A serializable version of a 2d convolution

    Parameters
    ----------
    in_channels: int
        Number of input channels
    out_channels: int
        Number of output channels
    kernel_size: int
        Size of the convolution kernel
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
    ):
        super().__init__(meta=MetaData())
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.net = diffusion.Conv2d(
            in_channels,
            out_channels,
            kernel_size,)

    @nvtx.annotate(message="Conv2dSerializable", color="blue")
    def forward(self, x):
        """ forward pass """
        return self.net(x)
