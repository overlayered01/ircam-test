"""카메라 소스 공통 인터페이스.

수명: open() → [start(mode) → read()… → stop()]… → close()
모드 전환은 stop() + start(new_mode). 한 번에 한 소스만 열어 쓴다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from femto_fov.fov_math import Intrinsics


class SourceUnavailable(RuntimeError):
    """SDK 미설치, 장치 없음, 장치 점유 등으로 소스를 열 수 없음. 메시지는 사용자 안내문."""


@dataclass(frozen=True)
class Mode:
    width: int
    height: int
    fps: float
    pixel_format: str
    label: str = ""
    # 소스별 추가 설정 (예: (("ir_mode", 1),)). frozen 이라 tuple 로 둔다.
    params: tuple[tuple[str, object], ...] = ()

    def param(self, key: str, default=None):
        return dict(self.params).get(key, default)

    @property
    def res(self) -> str:
        return f"{self.width}x{self.height}"

    def __str__(self) -> str:
        fps = f"{self.fps:g}"
        s = f"{self.res}@{fps} {self.pixel_format}"
        return f"{s} [{self.label}]" if self.label else s


@dataclass
class Frame:
    raw: np.ndarray  # 센서 원본 (uint16 / uint8)
    image: np.ndarray  # 8비트 흑백 (표시·인식용)
    mode: Mode
    intrinsics: Intrinsics | None = None
    timestamp: float = 0.0  # 장치 타임스탬프 (s)
    meta: dict = field(default_factory=dict)


class CameraSource(ABC):
    kind = ""

    def __init__(self, name: str):
        self.name = name
        self.mode: Mode | None = None  # start() 된 모드, 정지 상태면 None

    @classmethod
    def available(cls) -> tuple[bool, str]:
        """SDK 가 설치되어 있는지. (False, 이유) 면 open() 은 SourceUnavailable."""
        return True, ""

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def modes(self) -> list[Mode]: ...

    @abstractmethod
    def start(self, mode: Mode) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def read(self, timeout_ms: int = 1000) -> Frame | None:
        """다음 프레임. 시간 안에 안 오면 None."""

    def info(self) -> dict[str, str]:
        return {"name": self.name}

    def intrinsics(self, mode: Mode) -> Intrinsics | None:
        """모드의 내부 파라미터 (공장 캘리브레이션 또는 calib 파일). 없으면 None."""
        return None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()
