"""NVIDIA Jetson hardware profiles used for memory/latency budgeting.

Numbers are taken from public NVIDIA datasheets/marketing material and are
meant for *planning*. Always confirm against the datasheet for your exact
module + JetPack version (e.g. "Super" modes raise clocks on Orin Nano/NX).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class HardwareProfile:
    key: str
    name: str
    memory_gb: float  # unified LPDDR memory shared by CPU *and* GPU
    memory_bandwidth_gbs: float  # the main limiter for LLM token generation
    ai_perf: str  # headline AI throughput as marketed (sparse)
    gpu: str
    power_modes_w: Tuple[int, ...]
    # Memory you should NOT plan to give the model: OS, ROS 2, camera
    # drivers, navigation stack, CUDA context, etc.
    reserved_gb: float = 2.5

    @property
    def usable_gb(self) -> float:
        return max(self.memory_gb - self.reserved_gb, 0.0)


PROFILES: Dict[str, HardwareProfile] = {
    p.key: p
    for p in [
        HardwareProfile(
            key="orin_nano_8gb",
            name="Jetson Orin Nano 8GB (Super mode)",
            memory_gb=8,
            memory_bandwidth_gbs=102,
            ai_perf="67 INT8 TOPS",
            gpu="Ampere, 1024 CUDA cores, 32 Tensor Cores",
            power_modes_w=(7, 15, 25),
            reserved_gb=2.5,
        ),
        HardwareProfile(
            key="orin_nx_16gb",
            name="Jetson Orin NX 16GB (Super mode)",
            memory_gb=16,
            memory_bandwidth_gbs=102.4,
            ai_perf="157 INT8 TOPS",
            gpu="Ampere, 1024 CUDA cores, 32 Tensor Cores",
            power_modes_w=(10, 15, 25, 40),
            reserved_gb=3.0,
        ),
        HardwareProfile(
            key="agx_orin_64gb",
            name="Jetson AGX Orin 64GB",
            memory_gb=64,
            memory_bandwidth_gbs=204.8,
            ai_perf="275 INT8 TOPS",
            gpu="Ampere, 2048 CUDA cores, 64 Tensor Cores",
            power_modes_w=(15, 30, 50, 60),
            reserved_gb=4.0,
        ),
        HardwareProfile(
            key="thor_t5000",
            name="Jetson AGX Thor (T5000)",
            memory_gb=128,
            memory_bandwidth_gbs=273,
            ai_perf="2070 FP4 TFLOPS",
            gpu="Blackwell, 2560 CUDA cores, 96 Tensor Cores",
            power_modes_w=(40, 70, 130),
            reserved_gb=6.0,
        ),
    ]
}

DEFAULT_TARGET = "orin_nano_8gb"


def get_profile(key: str = DEFAULT_TARGET) -> HardwareProfile:
    try:
        return PROFILES[key]
    except KeyError as exc:
        raise KeyError(f"Unknown hardware '{key}'. Options: {sorted(PROFILES)}") from exc
