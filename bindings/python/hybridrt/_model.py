# -----------------------------------------------------------------------------
# HybridRT | C++20 Model Inference Framework
# File: bindings/python/hybridrt/_model.py
# Description: Wraps packages, image models and results of the C API with
#              numpy inputs and outputs.
# -----------------------------------------------------------------------------

import atexit
import ctypes
import enum
import logging
import os
from dataclasses import dataclass, field

import numpy as np

from . import _capi
from ._capi import lib


class Status(enum.IntEnum):
    OK = 0
    INVALID_ARGUMENT = 1
    NOT_FOUND = 2
    INVALID_CONFIG = 3
    UNAVAILABLE = 4
    PROCESS_FAILED = 5


class Task(enum.IntEnum):
    DETECT = 0
    ROTATED = 1
    CLASSIFY = 2
    KEYPOINT = 3
    SEGMENT = 4


class LogLevel(enum.IntEnum):
    TRACE = 0
    DEBUG = 1
    INFO = 2
    WARN = 3
    ERROR = 4
    FATAL = 5
    OFF = 6


class HybridRTError(RuntimeError):
    def __init__(self, status, message):
        super().__init__(f"{Status(status).name}: {message}")
        self.status = Status(status)


def _check(status):
    if status != Status.OK:
        raise HybridRTError(status, lib.hrt_last_error().decode("utf-8", "replace"))


# Format names as in the preprocess config, with each layout's planes as in hrt.h.
_FORMATS = {
    "inter_bgr": 1,
    "inter_rgb": 2,
    "planar_bgr": 3,
    "planar_rgb": 4,
    "nv12": 5,
    "nv21": 6,
    "yu12": 7,
    "yv12": 8,
    "inter_rgba": 9,
    "gray": 10,
}
_CHANNELS = {"inter_bgr": 3, "inter_rgb": 3, "inter_rgba": 4}


def version():
    return lib.hrt_version().decode()


def engine_available(engine_type):
    """Whether engine_type (e.g. "ort", "trt") is compiled into the library."""
    return bool(lib.hrt_engine_available(engine_type.encode()))


def set_log_level(level):
    lib.hrt_set_log_level(LogLevel(level) if not isinstance(level, str) else LogLevel[level.upper()])


_log_callback = None


def set_log_callback(callback):
    """Sends log records to callback(level: LogLevel, message: str) instead of
    the console; None restores the console. The callback must not call
    set_log_callback."""
    global _log_callback
    if callback is None:
        lib.hrt_set_log_callback(_capi.LOG_CALLBACK(), None)
        # The library no longer calls the old callback once the call returns.
        _log_callback = None
        return

    def forward(level, message, length, _user_data):
        callback(LogLevel(level), ctypes.string_at(message, length).decode("utf-8", "replace"))

    native = _capi.LOG_CALLBACK(forward)
    lib.hrt_set_log_callback(native, None)
    _log_callback = native


_PYTHON_LEVELS = {
    LogLevel.TRACE: logging.DEBUG,
    LogLevel.DEBUG: logging.DEBUG,
    LogLevel.INFO: logging.INFO,
    LogLevel.WARN: logging.WARNING,
    LogLevel.ERROR: logging.ERROR,
    LogLevel.FATAL: logging.CRITICAL,
}


def use_python_logging(logger="hybridrt"):
    """Forwards library logs to a logging.Logger or the logger of that name."""
    target = logging.getLogger(logger) if isinstance(logger, str) else logger
    set_log_callback(lambda level, message: target.log(_PYTHON_LEVELS.get(level, logging.INFO), message))


@atexit.register
def _restore_console_log():
    # Library threads may still log while the interpreter shuts down.
    if _log_callback is not None:
        lib.hrt_set_log_callback(_capi.LOG_CALLBACK(), None)


@dataclass
class DetectResult:
    boxes: np.ndarray  # (N, 4) float32: x, y, width, height in source pixels.
    scores: np.ndarray  # (N,) float32.
    class_ids: np.ndarray  # (N,) int64.
    track_ids: np.ndarray  # (N,) int64; -1 means unassigned.

    def __len__(self):
        return len(self.scores)


@dataclass
class RotatedResult:
    boxes: np.ndarray  # (N, 5) float32: center x, center y, width, height, clockwise angle in degrees.
    scores: np.ndarray
    class_ids: np.ndarray
    track_ids: np.ndarray

    def __len__(self):
        return len(self.scores)


@dataclass
class KeypointResult(DetectResult):
    # (N, K, 3) float32 x, y, score when every object has K keypoints, else a
    # list of (K_i, 3) arrays.
    keypoints: object = None
    # Visibility as the model outputs it: 0 unlabeled, 1 invisible, 2 visible;
    # shaped like keypoints without the last axis.
    visible: object = None


@dataclass
class Mask:
    rect: tuple  # x, y, width, height in source pixels.
    data: np.ndarray  # (height, width) uint8; 1 marks the class.


@dataclass
class SegmentResult(DetectResult):
    # One per object; semantic results hold one per class over the input region.
    masks: list = field(default_factory=list)


@dataclass
class ClassifyResult:
    scores: list  # One float32 array per engine output, indexed by class ID.

    def __len__(self):
        return len(self.scores)


def _array(pointer, count):
    if not count:
        return None
    # ctypes' PEP3118 format omits the trailing padding of C structs on Python 3.11.
    element = pointer._type_
    buffer = (ctypes.c_ubyte * (count * ctypes.sizeof(element))).from_address(ctypes.addressof(pointer.contents))
    return np.frombuffer(buffer, dtype=np.dtype(element), count=count)


def _boxes(data):
    count = data.count
    boxes = _array(data.boxes, count)
    if boxes is None:
        empty = np.zeros(0, np.float32)
        return dict(boxes=np.zeros((0, 4), np.float32), scores=empty, class_ids=np.zeros(0, np.int64),
                    track_ids=np.zeros(0, np.int64))
    return dict(
        boxes=np.stack([boxes["x"], boxes["y"], boxes["width"], boxes["height"]], axis=1).astype(np.float32),
        scores=boxes["score"].copy(),
        class_ids=boxes["class_id"].copy(),
        track_ids=boxes["track_id"].copy(),
    )


def _rotated(data):
    boxes = _array(data.rotated_boxes, data.count)
    if boxes is None:
        return RotatedResult(np.zeros((0, 5), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64),
                             np.zeros(0, np.int64))
    columns = [boxes[name] for name in ("center_x", "center_y", "width", "height", "angle")]
    return RotatedResult(np.stack(columns, axis=1).astype(np.float32), boxes["score"].copy(),
                         boxes["class_id"].copy(), boxes["track_id"].copy())


def _keypoints(data):
    offsets = np.ctypeslib.as_array(data.keypoint_offsets, (data.count + 1,)).copy()
    total = int(offsets[-1])
    points = _array(data.keypoints, total)
    values = (np.stack([points["x"], points["y"], points["score"]], axis=1).astype(np.float32)
              if points is not None else np.zeros((0, 3), np.float32))
    visible = points["visible"].copy() if points is not None else np.zeros(0, np.int32)
    counts = np.diff(offsets)
    if data.count == 0 or np.all(counts == counts[0]):
        per_object = int(counts[0]) if data.count else 0
        return values.reshape(data.count, per_object, 3), visible.reshape(data.count, per_object)
    keypoints = [values[begin:end] for begin, end in zip(offsets[:-1], offsets[1:])]
    return keypoints, [visible[begin:end] for begin, end in zip(offsets[:-1], offsets[1:])]


def _masks(data):
    masks = []
    for i in range(data.count):
        mask = data.masks[i]
        rect = mask.rect
        size = rect.width * rect.height
        pixels = (np.ctypeslib.as_array(mask.data, (size,)).reshape(rect.height, rect.width).copy()
                  if size > 0 else np.zeros((max(rect.height, 0), max(rect.width, 0)), np.uint8))
        masks.append(Mask((rect.x, rect.y, rect.width, rect.height), pixels))
    return masks


def _classify(data):
    if data.count == 0:
        return ClassifyResult([])
    offsets = np.ctypeslib.as_array(data.score_offsets, (data.count + 1,))
    total = int(offsets[-1])
    scores = np.ctypeslib.as_array(data.scores, (total,)) if total else np.zeros(0, np.float32)
    return ClassifyResult([scores[begin:end].copy() for begin, end in zip(offsets[:-1], offsets[1:])])


def _convert(data):
    task = Task(data.task)
    if task == Task.ROTATED:
        return _rotated(data)
    if task == Task.CLASSIFY:
        return _classify(data)
    boxes = _boxes(data)
    if task == Task.KEYPOINT:
        keypoints, visible = _keypoints(data)
        return KeypointResult(**boxes, keypoints=keypoints, visible=visible)
    if task == Task.SEGMENT:
        return SegmentResult(**boxes, masks=_masks(data))
    return DetectResult(**boxes)


def _image(array, format):
    """An HrtImage over `array` (kept alive by the caller) and that array."""
    if format not in _FORMATS:
        raise ValueError(f"unknown image format {format!r}; expected one of {', '.join(_FORMATS)}")
    image = np.asarray(array)
    if image.dtype != np.uint8:
        raise TypeError(f"expected a uint8 image, got {image.dtype}")
    if format == "gray" and image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    if format in _CHANNELS:
        channels = _CHANNELS[format]
        if image.ndim != 3 or image.shape[2] != channels:
            raise ValueError(f"{format} expects an (H, W, {channels}) array, got {image.shape}")
        if image.strides[2] != 1 or image.strides[1] != channels or image.strides[0] <= 0:
            image = np.ascontiguousarray(image)
        height, width = image.shape[:2]
        row_pitch = image.strides[0]
    elif format.startswith("planar"):
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"{format} expects a (3, H, W) array, got {image.shape}")
        if (image.strides[2] != 1 or image.strides[1] <= 0
                or image.strides[0] != image.shape[1] * image.strides[1]):
            image = np.ascontiguousarray(image)
        height, width = image.shape[1:]
        row_pitch = image.strides[1]
    else:
        if image.ndim != 2:
            raise ValueError(f"{format} expects a 2-D array, got {image.shape}")
        if image.strides[1] != 1 or image.strides[0] <= 0:
            image = np.ascontiguousarray(image)
        height, width = image.shape
        if format != "gray":
            # YUV 4:2:0 stacks the chroma rows under the luma rows.
            if height % 3 != 0:
                raise ValueError(f"{format} expects (H * 3 / 2, W) rows, got {image.shape}")
            height = height * 2 // 3
        row_pitch = image.strides[0]
    native = _capi.HrtImage(image.ctypes.data, _FORMATS[format], width, height, row_pitch)
    return native, image


class Package:
    """An .hrtc package (path or bytes) or a .json manifest whose relative
    paths resolve against the working directory."""

    def __init__(self, source):
        self._handle = ctypes.c_void_p()
        if isinstance(source, (bytes, bytearray, memoryview)):
            buffer = (ctypes.c_uint8 * len(source)).from_buffer_copy(source)
            _check(lib.hrt_package_load_memory(buffer, len(source), ctypes.byref(self._handle)))
        else:
            _check(lib.hrt_package_load(os.fsencode(os.fspath(source)), ctypes.byref(self._handle)))

    @property
    def algos(self):
        count = lib.hrt_package_algo_count(self._handle)
        return [lib.hrt_package_algo_name(self._handle, i).decode() for i in range(count)]

    def model(self, algo=None, **options):
        return Model(self, algo, **options)

    def close(self):
        if self._handle:
            lib.hrt_package_destroy(self._handle)
            self._handle = ctypes.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()


class Model:
    """One algo of a package, run by the first of its engines that works on
    this machine. Calls are thread-safe and release the GIL; preprocess, engine
    and postprocess set how many calls each stage serves at once (1 to 64)."""

    def __init__(self, package, algo=None, *, preprocess=1, engine=1, postprocess=1):
        self._handle = ctypes.c_void_p()
        owned = not isinstance(package, Package)
        source = Package(package) if owned else package
        try:
            options = _capi.HrtModelOptions(preprocess, engine, postprocess)
            name = algo.encode() if algo is not None else None
            _check(lib.hrt_model_create(source._handle, name, ctypes.byref(options), ctypes.byref(self._handle)))
        finally:
            if owned:
                source.close()
        self.task = Task(lib.hrt_model_task(self._handle))
        self.algo = lib.hrt_model_algo_name(self._handle).decode()

    def process(self, image, format="inter_bgr", roi=None):
        """Runs on a uint8 host image laid out as `format` (OpenCV's BGR by
        default); roi is (x, y, width, height) in image pixels. Results are in
        image coordinates."""
        if not self._handle:
            raise ValueError("the model is closed")
        # `image` keeps a converted copy alive through the call.
        native, image = _image(image, format)
        rect = ctypes.byref(_capi.HrtRect(*roi)) if roi is not None else None
        result = ctypes.c_void_p()
        _check(lib.hrt_result_create(ctypes.byref(result)))
        try:
            _check(lib.hrt_model_process(self._handle, ctypes.byref(native), rect, result))
            return _convert(lib.hrt_result_data(result).contents)
        finally:
            lib.hrt_result_destroy(result)

    __call__ = process

    def close(self):
        if self._handle:
            lib.hrt_model_destroy(self._handle)
            self._handle = ctypes.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()
