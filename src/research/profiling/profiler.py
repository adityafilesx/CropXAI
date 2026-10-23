"""
CropXAI Profiler
==================

Calculates hardware metrics: FLOPs, Latency, Peak VRAM, and RAM usage.
Utilizes fvcore for accurate FLOP counting across CNNs and Transformers.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import psutil
import torch
import torch.nn as nn

try:
    from fvcore.nn import FlopCountAnalysis, parameter_count
except ImportError:
    FlopCountAnalysis = None
    parameter_count = None

logger = logging.getLogger(__name__)


class ResourceProfiler:
    """
    Profiles model performance and resource utilization.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device

    def profile(self, model: nn.Module, input_shape: tuple[int, ...] = (1, 1, 3, 224, 224)) -> dict[str, Any]:
        """Run all profiling steps."""
        model.eval()
        dummy_input = torch.randn(input_shape, device=self.device)
        
        results = {
            "parameters": self._count_parameters(model),
            "flops": self._count_flops(model, dummy_input),
            "vram_mb": self._measure_vram(model, dummy_input),
            "ram_mb": self._measure_ram(),
            "latency_ms": self._measure_latency(model, dummy_input),
        }
        
        if results["latency_ms"] > 0:
            results["throughput_ips"] = 1000.0 / results["latency_ms"]
        else:
            results["throughput_ips"] = 0.0
            
        return results

    def _count_parameters(self, model: nn.Module) -> int:
        if parameter_count is not None:
            return sum(parameter_count(model).values())
        return sum(p.numel() for p in model.parameters())

    def _count_flops(self, model: nn.Module, inputs: torch.Tensor) -> float:
        if FlopCountAnalysis is not None:
            flops = FlopCountAnalysis(model, (inputs,))
            return flops.total()
        logger.warning("fvcore not installed. FLOP calculation skipped.")
        return 0.0

    def _measure_latency(self, model: nn.Module, inputs: torch.Tensor, runs: int = 50) -> float:
        for _ in range(10):
            with torch.no_grad():
                _ = model(inputs)
                
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            
        start_time = time.time()
        for _ in range(runs):
            with torch.no_grad():
                _ = model(inputs)
                
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            
        end_time = time.time()
        return ((end_time - start_time) / runs) * 1000.0

    def _measure_vram(self, model: nn.Module, inputs: torch.Tensor) -> float:
        if self.device.type != "cuda":
            return 0.0
            
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(inputs)
        peak_bytes = torch.cuda.max_memory_allocated()
        return peak_bytes / (1024 ** 2)

    def _measure_ram(self) -> float:
        process = psutil.Process()
        return process.memory_info().rss / (1024 ** 2)


__all__ = ["ResourceProfiler"]
