"""Insert native HIP event nodes into a PyTorch stream-captured graph.

Events are created outside capture. All preceding stream work precedes the
marker, and all following stream work depends on it. Read results only after
replay completes. The owner must outlive graph execution.
"""
import ctypes as C
from functools import cache


@cache
def runtime():
    lib = C.CDLL('libamdhip64.so')
    P = C.c_void_p
    signatures = {
        'hipEventCreateWithFlags': [C.POINTER(P), C.c_uint],
        'hipEventDestroy': [P],
        'hipEventElapsedTime': [C.POINTER(C.c_float), P, P],
        'hipStreamGetCaptureInfo_v2': [P, C.POINTER(C.c_int), C.POINTER(C.c_ulonglong),
                                     C.POINTER(P), C.POINTER(C.POINTER(P)), C.POINTER(C.c_size_t)],
        'hipGraphAddEventRecordNode': [C.POINTER(P), P, C.POINTER(P), C.c_size_t, P],
        'hipStreamUpdateCaptureDependencies': [P, C.POINTER(P), C.c_size_t, C.c_uint],
        'hipGraphDebugDotPrint': [P, C.c_char_p, C.c_uint],
    }
    for name, args in signatures.items():
        getattr(lib, name).argtypes = args
        getattr(lib, name).restype = C.c_int
    lib.hipGetErrorString.argtypes = [C.c_int]
    lib.hipGetErrorString.restype = C.c_char_p
    return lib


def call(name, *args):
    lib = runtime()
    rc = getattr(lib, name)(*args)
    if rc:
        raise RuntimeError(f'{name}: {rc} {lib.hipGetErrorString(rc).decode()}')


class Event:
    def __init__(self, flags=0x20000000):
        # hipEventDisableSystemFence: timing only, never inter-device synchronization.
        # Callers synchronize the completed graph before reading timestamps.
        self.handle = C.c_void_p()
        call('hipEventCreateWithFlags', C.byref(self.handle), flags)

    def record(self):
        import torch
        stream = C.c_void_p(torch.cuda.current_stream().cuda_stream)
        status, capture_id = C.c_int(), C.c_ulonglong()
        graph, node = C.c_void_p(), C.c_void_p()
        deps, count = C.POINTER(C.c_void_p)(), C.c_size_t()
        call('hipStreamGetCaptureInfo_v2', stream, C.byref(status), C.byref(capture_id),
             C.byref(graph), C.byref(deps), C.byref(count))
        if status.value != 1:
            raise RuntimeError('Native marker requires active stream capture')
        call('hipGraphAddEventRecordNode', C.byref(node), graph, deps, count, self.handle)
        # hipStreamSetCaptureDependencies = 1; replace the capture frontier.
        call('hipStreamUpdateCaptureDependencies', stream, C.byref(node), 1, 1)
        self.graph = graph

    def elapsed_us(self, end):
        ms = C.c_float()
        call('hipEventElapsedTime', C.byref(ms), self.handle, end.handle)
        return ms.value * 1000

    def close(self):
        if self.handle.value:
            call('hipEventDestroy', self.handle)
            self.handle = C.c_void_p()

    def __del__(self):
        if getattr(self, 'handle', None) and self.handle.value:
            runtime().hipEventDestroy(self.handle)
