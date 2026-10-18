# -----------------------------------------------------------------------------
# HybridRT | C++20 Model Inference Framework
# File: bindings/python/hybridrt/_ocr.py
# Description: Wraps sequential OCR models and owned UTF-8 text results.
# -----------------------------------------------------------------------------

import ctypes
import threading
from dataclasses import dataclass

import numpy as np

from ._capi import lib
from ._model import Package, _check, _image


@dataclass
class OcrText:
    rect: np.ndarray  # (4,) float32: x, y, width, height in source pixels.
    text: str
    detection_score: float
    recognition_score: float


class OcrModel:
    """Two algos of a package; calls on this instance are serialized without pools."""

    def __init__(self, package, detection="ppocrv6_tiny_det", recognition="ppocrv6_tiny_rec"):
        self._lock = threading.Lock()
        self._handle = ctypes.c_void_p()
        self._result = ctypes.c_void_p()
        owned = not isinstance(package, Package)
        source = Package(package) if owned else package
        try:
            _check(lib.hrt_ocr_model_create(source._handle, detection.encode(), recognition.encode(),
                                          ctypes.byref(self._handle)))
            _check(lib.hrt_ocr_result_create(ctypes.byref(self._result)))
        finally:
            if owned:
                source.close()

    def process(self, image, format="inter_bgr"):
        with self._lock:
            if not self._handle:
                raise ValueError("the OCR model is closed")
            native, image = _image(image, format)
            _check(lib.hrt_ocr_model_process(self._handle, ctypes.byref(native), self._result))
            results = []
            for index in range(lib.hrt_ocr_result_count(self._result)):
                item = lib.hrt_ocr_result_text(self._result, index).contents
                results.append(OcrText(
                    np.array([item.x, item.y, item.width, item.height], dtype=np.float32),
                    ctypes.string_at(item.text, item.text_size).decode("utf-8"),
                    item.detection_score, item.recognition_score,
                ))
            return results

    __call__ = process

    def close(self):
        with self._lock:
            if self._result:
                lib.hrt_ocr_result_destroy(self._result)
                self._result = ctypes.c_void_p()
            if self._handle:
                lib.hrt_ocr_model_destroy(self._handle)
                self._handle = ctypes.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()
