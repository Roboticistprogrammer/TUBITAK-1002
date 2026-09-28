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


def parse_power_mw(line: str) -> tuple[str, float] | None:
    readings = {m.group("rail"): float(m.group("now")) for m in _PATTERN.finditer(line)}
    for rail in RAILS:
        if rail in readings:
            return rail, readings[rail]
    return None


class TegrastatsMonitor:
    def __init__(self, interval_ms: int = 100, log_path: Path | None = None) -> None:
        self.interval_ms = interval_ms
        self.log_path = log_path
        self.available = shutil.which("tegrastats") is not None
        self.samples: list[float] = []
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
