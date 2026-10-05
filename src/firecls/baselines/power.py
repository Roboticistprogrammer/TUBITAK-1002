"""Board power sampling on NVIDIA Jetson through ``tegrastats``.

Supported rails, in order of preference (first one present in the output is used):
``VDD_IN`` (Orin, Xavier NX), ``POM_5V_IN`` (Nano, TX2), ``VDD_CPU_GPU_CV`` (Orin fallback).
Values are reported as instantaneous milliwatts. On machines without ``tegrastats`` the
monitor is a no-op and returns ``None`` so desktop benchmarks still run.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import threading
from pathlib import Path

RAILS = ("VDD_IN", "POM_5V_IN", "VDD_CPU_GPU_CV")
_PATTERN = re.compile(r"(?P<rail>[A-Z0-9_]+) (?P<now>\d+)(?:mW)?/(?P<avg>\d+)(?:mW)?")
_RAM = re.compile(r"RAM (?P<used>\d+)/(?P<total>\d+)MB")


def parse_power_mw(line: str) -> tuple[str, float] | None:
    readings = {m.group("rail"): float(m.group("now")) for m in _PATTERN.finditer(line)}
    for rail in RAILS:
        if rail in readings:
            return rail, readings[rail]
    return None


def parse_ram_mb(line: str) -> tuple[float, float] | None:
    """(used, total) system RAM in MB. On Jetson the GPU shares this memory, so it is the runtime footprint."""
    match = _RAM.search(line)
    return (float(match.group("used")), float(match.group("total"))) if match else None


def system_ram_used_mb() -> float | None:
    """RAM in use now (same notion as tegrastats/free: total - free - buffers - cache). Linux only."""
    try:
        info = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            info[key] = float(value.split()[0]) / 1024.0
        return info["MemTotal"] - info["MemFree"] - info["Buffers"] - info["Cached"] - info.get("SReclaimable", 0.0)
    except (OSError, KeyError, ValueError):
        return None


class TegrastatsMonitor:
    def __init__(self, interval_ms: int = 100, log_path: Path | None = None) -> None:
        self.interval_ms = interval_ms
        self.log_path = log_path
        self.available = shutil.which("tegrastats") is not None
        self.samples: list[float] = []
        self.ram_samples: list[float] = []
        self.rail: str | None = None
        self._process = None
        self._thread = None

    def __enter__(self):
        if self.available:
            self._process = subprocess.Popen(
                ["tegrastats", "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            self._thread = threading.Thread(target=self._read, daemon=True)
            self._thread.start()
        return self

    def _read(self) -> None:
        log = self.log_path.open("w", encoding="utf-8") if self.log_path else None
        try:
            for line in self._process.stdout:
                if log:
                    log.write(line)
                ram = parse_ram_mb(line)
                if ram:
                    self.ram_samples.append(ram[0])
                parsed = parse_power_mw(line)
                if parsed:
                    self.rail, value = parsed
                    self.samples.append(value)
        finally:
            if log:
                log.close()

    def __exit__(self, *exc) -> None:
        if self._process:
            self._process.terminate()
            self._process.wait(timeout=5)
        if self._thread:
            self._thread.join(timeout=5)

    def summary(self) -> dict | None:
        if not self.samples:
            return None
        return {
            "rail": self.rail,
            "samples": len(self.samples),
            "mean_mw": sum(self.samples) / len(self.samples),
            "max_mw": max(self.samples),
        }

    def memory_summary(self) -> dict | None:
        if not self.ram_samples:
            return None
        return {
            "samples": len(self.ram_samples),
            "first_ram_mb": self.ram_samples[0],
            "peak_ram_mb": max(self.ram_samples),
            "mean_ram_mb": sum(self.ram_samples) / len(self.ram_samples),
        }
