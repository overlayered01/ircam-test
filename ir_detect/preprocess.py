"""센서 원본 → 8비트 흑백 변환."""

from __future__ import annotations

import cv2
import numpy as np


def to_u8(raw: np.ndarray) -> np.ndarray:
    """기본 변환. 8비트는 그대로, 16비트는 제곱근 압축 후 99 백분위로 정규화한다.
    ToF active IR 은 가까운 물체가 포화(65535)되어 최대값 기준이면 나머지가 거의 검게 나온다."""
    if raw.dtype == np.uint8:
        return raw
    v = np.sqrt(raw.astype(np.float32))
    hi = float(np.percentile(v[::4, ::4], 99))
    return cv2.convertScaleAbs(v, alpha=255.0 / max(hi, 1.0))
