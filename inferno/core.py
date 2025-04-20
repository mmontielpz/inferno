# 🔥 inferno/core.py
"""
Performance benchmarking for PyTorch models.
Tracks FLOPs, MACs, memory, and inference time using hooks.
"""
import torch
import torch.cuda as cuda
from tqdm import tqdm
from decimal import Decimal, getcontext
from inferno.hooks import get_default_hooks

getcontext().prec = 4  


def compute_avg(list_values):
    return float(round(sum(list_values) / len(list_values), 2))


def compute_fps(infer_time_ms):
    sec = infer_time_ms / 1000
    return round(1 / sec, 1)


class Performance:
    def __init__(self):
        self.hooks = get_default_hooks()
        self.handles = []
        self.tot_flops = 0
        self.tot_mac = 0
        self.tot_mem = 0

    def reset(self):
        self.tot_flops = 0
        self.tot_mac = 0
        self.tot_mem = 0
        self.handles = []

    def register_hooks(self, model):
        for module in model.modules():
            if type(module) in self.hooks:
                self.handles.append(module.register_forward_hook(self.hooks[type(module)]))

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def collect_metrics(self, model):
        for module in model.modules():
            if hasattr(module, "__flops__"):
                self.tot_flops += module.__flops__
            if hasattr(module, "__mac__"):
                self.tot_mac += module.__mac__
            if hasattr(module, "__tot_mem__"):
                self.tot_mem += module.__tot_mem__

    def get_totals(self):
        return {
            "flops": float(Decimal(self.tot_flops) / Decimal(1e9)),
            "macs": float(Decimal(self.tot_mac) / Decimal(1e9)),
            "memory": float(Decimal(self.tot_mem) / Decimal(1e6)),
        }

    def run_warmup(self, model, input_data, n=50):
        with torch.no_grad():
            for _ in tqdm(range(n), desc="[🔥] Warmup Runs"):
                _ = model(input_data)
        cuda.synchronize()

    def measure_inference_time(self, model, input_data, n=100):
        times = []
        with torch.no_grad():
            for _ in tqdm(range(n), desc="[⏱️ ] Measuring Inference Time"):
                cuda.synchronize()
                start = cuda.Event(enable_timing=True)
                end = cuda.Event(enable_timing=True)

                start.record()
                _ = model(input_data)
                end.record()
                cuda.synchronize()

                times.append(start.elapsed_time(end))

        return compute_avg(times)

    def benchmark(self, model, input_shape=(32, 3, 224, 224), warmups=50):
        input_data = torch.randn(input_shape).to("cuda")
        model.to("cuda")
        model.eval()

        self.reset()
        self.register_hooks(model)

        _ = model(input_data)
        self.remove_hooks()
        self.collect_metrics(model)

        print("[INFO] Warmup...")
        self.run_warmup(model, input_data, n=warmups)

        print("[INFO] Measuring Inference...")
        infer_time = self.measure_inference_time(model, input_data)

        totals = self.get_totals()
        totals.update({
            "inference_time_ms": round(infer_time, 2),
            "fps": compute_fps(infer_time)
        })

        print("\n[🔥 Inferno Results]")
        for k, v in totals.items():
            print(f"{k}: {v}")

        return totals
