from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class CommitStats:
    committed_bytes: int
    limit_bytes: int

    @property
    def headroom_bytes(self) -> int:
        return max(self.limit_bytes - self.committed_bytes, 0)

    @property
    def headroom_percent(self) -> float:
        if self.limit_bytes <= 0:
            return 0.0
        return 100.0 * self.headroom_bytes / self.limit_bytes


@dataclass(frozen=True)
class CommitGuardResult:
    stats: CommitStats
    minimum_headroom_percent: float
    passed: bool

    @property
    def headroom_percent(self) -> float:
        return self.stats.headroom_percent


def check_commit_headroom(
    stats: CommitStats, minimum_headroom_percent: float = 20.0
) -> CommitGuardResult:
    if minimum_headroom_percent < 0 or minimum_headroom_percent > 100:
        raise ValueError("minimum_headroom_percent must be between 0 and 100")
    return CommitGuardResult(
        stats=stats,
        minimum_headroom_percent=minimum_headroom_percent,
        passed=stats.headroom_percent >= minimum_headroom_percent,
    )


def get_system_commit() -> CommitStats:
    """Read Windows system committed pages and commit limit from PSAPI."""
    if sys.platform != "win32":
        raise OSError("System commit guard is only available on Windows")

    class PerformanceInformation(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_uint32),
            ("CommitTotal", ctypes.c_size_t),
            ("CommitLimit", ctypes.c_size_t),
            ("CommitPeak", ctypes.c_size_t),
            ("PhysicalTotal", ctypes.c_size_t),
            ("PhysicalAvailable", ctypes.c_size_t),
            ("SystemCache", ctypes.c_size_t),
            ("KernelTotal", ctypes.c_size_t),
            ("KernelPaged", ctypes.c_size_t),
            ("KernelNonpaged", ctypes.c_size_t),
            ("PageSize", ctypes.c_size_t),
            ("HandleCount", ctypes.c_uint32),
            ("ProcessCount", ctypes.c_uint32),
            ("ThreadCount", ctypes.c_uint32),
        ]

    info = PerformanceInformation()
    info.cb = ctypes.sizeof(info)
    get_performance_info = ctypes.WinDLL("Psapi", use_last_error=True).GetPerformanceInfo
    get_performance_info.argtypes = [ctypes.POINTER(PerformanceInformation), ctypes.c_uint32]
    get_performance_info.restype = ctypes.c_int
    if not get_performance_info(ctypes.byref(info), info.cb):
        raise OSError(ctypes.get_last_error(), "GetPerformanceInfo failed")
    return CommitStats(
        committed_bytes=int(info.CommitTotal * info.PageSize),
        limit_bytes=int(info.CommitLimit * info.PageSize),
    )
