# -----------------------------------------------------------------------------
# HybridRT | C++20 Model Inference Framework
# File: bindings/python/hybridrt/_capi.py
# Description: Loads hybridrt_c and declares the include/hrt.h types and
#              functions for ctypes.
# -----------------------------------------------------------------------------

import ctypes
import os
import sys
from pathlib import Path

_c_size = ctypes.c_size_t
_c_int = ctypes.c_int


class HrtModelOptions(ctypes.Structure):
    _fields_ = [("preprocess", _c_size), ("engine", _c_size), ("postprocess", _c_size)]


class HrtRect(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_int32),
        ("y", ctypes.c_int32),
        ("width", ctypes.c_int32),
        ("height", ctypes.c_int32),
    ]


class HrtImage(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("format", _c_int),
        ("width", ctypes.c_int32),
        ("height", ctypes.c_int32),
        ("row_pitch", _c_size),
    ]


class HrtBox(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("width", ctypes.c_float),
        ("height", ctypes.c_float),
        ("score", ctypes.c_float),
        ("class_id", ctypes.c_int64),
        ("track_id", ctypes.c_int64),
    ]


class HrtRotatedBox(ctypes.Structure):
    _fields_ = [
        ("center_x", ctypes.c_float),
        ("center_y", ctypes.c_float),
        ("width", ctypes.c_float),
        ("height", ctypes.c_float),
        ("angle", ctypes.c_float),
        ("score", ctypes.c_float),
        ("class_id", ctypes.c_int64),
        ("track_id", ctypes.c_int64),
    ]


class HrtKeypoint(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("score", ctypes.c_float),
        ("visible", ctypes.c_int32),
    ]


class HrtMask(ctypes.Structure):
    _fields_ = [("rect", HrtRect), ("data", ctypes.POINTER(ctypes.c_uint8))]


class HrtResultData(ctypes.Structure):
    _fields_ = [
        ("task", _c_int),
        ("count", _c_size),
        ("boxes", ctypes.POINTER(HrtBox)),
        ("rotated_boxes", ctypes.POINTER(HrtRotatedBox)),
        ("keypoint_offsets", ctypes.POINTER(_c_size)),
        ("keypoints", ctypes.POINTER(HrtKeypoint)),
        ("masks", ctypes.POINTER(HrtMask)),
        ("score_offsets", ctypes.POINTER(_c_size)),
        ("scores", ctypes.POINTER(ctypes.c_float)),
    ]


class HrtOcrText(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("width", ctypes.c_float),
        ("height", ctypes.c_float),
        ("text", ctypes.c_void_p),
        ("text_size", _c_size),
        ("detection_score", ctypes.c_float),
        ("recognition_score", ctypes.c_float),
    ]


LOG_CALLBACK = ctypes.CFUNCTYPE(None, _c_int, ctypes.c_char_p, _c_size, ctypes.c_void_p)


def _library_path():
    """HYBRIDRT_LIBRARY, else the library installed beside this package."""
    override = os.environ.get("HYBRIDRT_LIBRARY")
    if override:
        return Path(override)
    if sys.platform == "win32":
        name = "hybridrt_c.dll"
    elif sys.platform == "darwin":
        name = "libhybridrt_c.dylib"
    else:
        name = "libhybridrt_c.so"
    return Path(__file__).resolve().parent / name


def _load():
    path = _library_path()
    if not path.is_file():
        raise ImportError(f"hybridrt: cannot find the C library at {path}; build it or set HYBRIDRT_LIBRARY")
    if sys.platform == "win32":
        # Backend DLLs are deployed beside the library.
        os.add_dll_directory(str(path.parent))
    lib = ctypes.CDLL(str(path))
    handle = ctypes.c_void_p
    out_handle = ctypes.POINTER(ctypes.c_void_p)
    signatures = {
        "hrt_ocr_model_create": (_c_int, [handle, ctypes.c_char_p, ctypes.c_char_p, out_handle]),
        "hrt_ocr_model_destroy": (None, [handle]),
        "hrt_ocr_result_create": (_c_int, [out_handle]),
        "hrt_ocr_result_destroy": (None, [handle]),
        "hrt_ocr_result_count": (_c_size, [handle]),
        "hrt_ocr_result_text": (ctypes.POINTER(HrtOcrText), [handle, _c_size]),
        "hrt_ocr_model_process": (_c_int, [handle, ctypes.POINTER(HrtImage), handle]),
        "hrt_version": (ctypes.c_char_p, []),
        "hrt_last_error": (ctypes.c_char_p, []),
        "hrt_set_log_level": (None, [_c_int]),
        "hrt_set_log_callback": (None, [LOG_CALLBACK, ctypes.c_void_p]),
        "hrt_engine_available": (_c_int, [ctypes.c_char_p]),
        "hrt_package_load": (_c_int, [ctypes.c_char_p, out_handle]),
        "hrt_package_load_memory": (_c_int, [ctypes.c_void_p, _c_size, out_handle]),
        "hrt_package_destroy": (None, [handle]),
        "hrt_package_algo_count": (_c_size, [handle]),
        "hrt_package_algo_name": (ctypes.c_char_p, [handle, _c_size]),
        "hrt_model_options_default": (HrtModelOptions, []),
        "hrt_model_create": (_c_int, [handle, ctypes.c_char_p, ctypes.POINTER(HrtModelOptions), out_handle]),
        "hrt_model_destroy": (None, [handle]),
        "hrt_model_task": (_c_int, [handle]),
        "hrt_model_algo_name": (ctypes.c_char_p, [handle]),
        "hrt_result_create": (_c_int, [out_handle]),
        "hrt_result_destroy": (None, [handle]),
        "hrt_result_data": (ctypes.POINTER(HrtResultData), [handle]),
        "hrt_model_process": (_c_int, [handle, ctypes.POINTER(HrtImage), ctypes.POINTER(HrtRect), handle]),
    }
    for name, (restype, argtypes) in signatures.items():
        function = getattr(lib, name)
        function.restype = restype
        function.argtypes = argtypes
    return lib


lib = _load()
