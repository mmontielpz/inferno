# 🔧 inferno/hooks.py
"""
Hook registration and profiling logic for PyTorch modules.
Supports FLOPs, MACs, and memory tracking for standard CNN layers.
"""
import torch
import torch.nn as nn
import math

# Constants
SIZE_MAP = {
    torch.float32: 4,
    torch.float64: 8,
    torch.float16: 2,
    torch.int8: 1,
    torch.int16: 2,
    torch.int32: 4,
    torch.int64: 8,
    torch.uint8: 1
}

PYTORCH_MIN_ALLOCATE = 2 ** 9


def get_data_type_size(data_type):
    if data_type in SIZE_MAP:
        return SIZE_MAP[data_type]
    raise ValueError(f"Unsupported data type: {data_type}")


class ModuleProfiler:
    def __init__(self, module, input, output):
        self.module = module
        self.input = input
        self.output = output
        self.name = self._short_name(module.__class__.__name__.lower())
        self.module_params = list(module.parameters())
        self.module_buffers = list(module.buffers())
        self.bit_depth = get_data_type_size(input.dtype)

    def _short_name(self, name):
        return {
            "conv2d": "conv",
            "adaptiveavgpool2d": "adap_avg_pool",
            "avgpool2d": "avg_pool",
            "maxpool2d": "max_pool",
            "adaptivemaxpool2d": "adap_max_pool",
            "batchnorm2d": "bn",
        }.get(name, name)

    def _element_mem(self, tensor):
        element_size = tensor.element_size()
        fact_numel = tensor.storage().size()
        raw_size = fact_numel * element_size
        return math.ceil(raw_size / PYTORCH_MIN_ALLOCATE) * PYTORCH_MIN_ALLOCATE

    def get_mem(self):
        return sum(self._element_mem(x) for x in self.module_params + self.module_buffers + [self.input, self.output])

    def get_flops(self):
        name = self.name
        if name == "conv":
            Cin, H, W = self.input.shape[1:]
            Cout, Kh, Kw = self.module.weight.shape[:3]
            groups = self.module.groups
            flops = (Cin * Cout * Kh * Kw * H * W) / groups
            return flops

        elif name == "linear":
            In, Out = self.module.in_features, self.module.out_features
            flops = In * Out
            return flops

        elif name in {"avg_pool", "max_pool"}:
            C, H, W = self.input.shape[1:]
            Kh, Kw = self.module.kernel_size
            return C * H * W * Kh * Kw

        elif name == "bn":
            C, H, W = self.input.shape[1:]
            return 4 * C * H * W

        else:
            return 0

    def get_macs(self):
        # TODO: This is a simplification only for classic CNN layers.
        # In the future, implement true MAC calculation that considers:
        # - Input access (MAC_in)
        # - Weight access (MAC_weight)
        # - Output write (MAC_out)
        # For now, we return FLOPs as a rough approximation for MACs.
        return self.get_flops()  # For simplicity in standard ops

    def get_info(self):
        return {
            "op_name": self.name,
            "input_shape": tuple(self.input.shape),
            "output_shape": tuple(self.output.shape),
            "mem": self.get_mem(),
            "flops": self.get_flops(),
            "macs": self.get_macs()
        }


def profiler_hook(module, input, output):
    profiler = ModuleProfiler(module, input[0], output)
    info = profiler.get_info()
    module.__flops__ = info["flops"]
    module.__mac__ = info["macs"]
    module.__tot_mem__ = info["mem"]
    module.__profile_info__ = info


# Hook registry for standard CNN components
HOOK_MAP = {
    nn.Conv2d: profiler_hook,
    nn.Linear: profiler_hook,
    nn.BatchNorm2d: profiler_hook,
    nn.AdaptiveAvgPool2d: profiler_hook,
    nn.AvgPool2d: profiler_hook,
    nn.MaxPool2d: profiler_hook,
}


def get_default_hooks():
    return HOOK_MAP
