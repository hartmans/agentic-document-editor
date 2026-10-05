"""Incremental Ogg Opus encoding through libopusenc (no FFmpeg dependency).

The native library owns resampling, codec delay, granule positions, and end
trimming. Its pull API avoids Python callbacks from C and lets HTTP consumers
apply backpressure one small PCM block at a time.
API: https://opus-codec.org/docs/libopusenc_api-0.2/group__encoding.html
"""
from __future__ import annotations

import asyncio
from contextlib import aclosing
import ctypes as C
from ctypes.util import find_library
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def opus_library():
    name = find_library("opusenc")
    if not name:
        raise RuntimeError("libopusenc is required for audio playback; install libopusenc0")
    lib = C.CDLL(name)
    signatures = {
        "ope_comments_create": ([], C.c_void_p),
        "ope_comments_destroy": ([C.c_void_p], None),
        "ope_encoder_create_pull": ([C.c_void_p, C.c_int32, C.c_int, C.c_int, C.POINTER(C.c_int)], C.c_void_p),
        "ope_encoder_write_float": ([C.c_void_p, C.POINTER(C.c_float), C.c_int], C.c_int),
        "ope_encoder_get_page": ([C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_int32), C.c_int], C.c_int),
        "ope_encoder_flush_header": ([C.c_void_p], C.c_int),
        "ope_encoder_drain": ([C.c_void_p], C.c_int),
        "ope_encoder_destroy": ([C.c_void_p], None),
        "ope_strerror": ([C.c_int], C.c_char_p),
        # Only the fixed parameters of this variadic function belong in argtypes.
        "ope_encoder_ctl": ([C.c_void_p, C.c_int], C.c_int),
    }
    for function, (arguments, result) in signatures.items():
        getattr(lib, function).argtypes = arguments
        getattr(lib, function).restype = result
    return lib


class OpusEncoder:
    """A single mono/stereo logical stream; write blocks, finish, then close."""

    def __init__(self, sample_rate: int, channels: int):
        if sample_rate <= 0 or channels not in (1, 2):
            raise ValueError("Opus playback requires a positive sample rate and mono/stereo PCM")
        self.channels = channels
        self._lib = opus_library()
        self._encoder = None
        self._finished = False
        comments = self._lib.ope_comments_create()
        if not comments:
            raise MemoryError("Could not allocate Opus comments")
        try:
            error = C.c_int()
            self._encoder = self._lib.ope_encoder_create_pull(
                comments, sample_rate, channels, 0, C.byref(error),
            )
            self._check(error.value)
            if not self._encoder:
                raise MemoryError("Could not allocate Opus encoder")
            # libopusenc defaults to substantial decision/muxing buffering for
            # file encoding. Disable decision delay and use 100 ms Ogg pages.
            self._check(self._lib.ope_encoder_ctl(self._encoder, 14000, C.c_int(0)))
            self._check(self._lib.ope_encoder_ctl(self._encoder, 14002, C.c_int(4800)))
            self._check(self._lib.ope_encoder_ctl(self._encoder, 4002, C.c_int(64000 * channels)))
        except BaseException:
            self.close()
            raise
        finally:
            self._lib.ope_comments_destroy(comments)

    def _check(self, result: int) -> None:
        if result < 0:
            raise RuntimeError(self._lib.ope_strerror(result).decode("utf-8"))

    def _pages(self) -> list[bytes]:
        output = []
        page, length = C.c_void_p(), C.c_int32()
        while True:
            result = self._lib.ope_encoder_get_page(self._encoder, C.byref(page), C.byref(length), 1)
            self._check(result)
            if not result:
                return output
            output.append(C.string_at(page, length.value))

    def headers(self) -> list[bytes]:
        self._check(self._lib.ope_encoder_flush_header(self._encoder))
        return self._pages()

    def write(self, samples: np.ndarray) -> list[bytes]:
        if self._finished or self._encoder is None:
            raise RuntimeError("Opus encoder is closed or finished")
        pcm = np.asarray(samples, dtype=np.float32)
        if pcm.ndim == 1 and self.channels == 1:
            pcm = pcm.reshape(-1, 1)
        if pcm.ndim != 2 or pcm.shape[1] != self.channels:
            raise ValueError("PCM channel layout does not match the Opus stream")
        pcm = np.ascontiguousarray(pcm)
        if len(pcm):
            self._check(self._lib.ope_encoder_write_float(
                self._encoder, pcm.ctypes.data_as(C.POINTER(C.c_float)), len(pcm),
            ))
        return self._pages()

    def finish(self) -> list[bytes]:
        if self._finished or self._encoder is None:
            raise RuntimeError("Opus encoder is closed or finished")
        self._check(self._lib.ope_encoder_drain(self._encoder))
        self._finished = True
        return self._pages()

    def close(self) -> None:
        if self._encoder is not None:
            self._lib.ope_encoder_destroy(self._encoder)
            self._encoder = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


async def encode_opus(results, *, sample_rate: int, channels: int):
    """Encode RenderResult PCM incrementally; failure never emits a clean EOS.

    Cap each native call at 100 ms of PCM, including when a cache hit supplies
    an entire chapter. No producer queue or full encoded-file buffer is needed.
    The caller owns the source; closing this iterator also closes its reader.
    """
    async with aclosing(results):
        with OpusEncoder(sample_rate, channels) as encoder:
            for page in encoder.headers():
                yield page
            async for result in results:
                block_size = max(1, sample_rate // 10)
                for start in range(0, result.frame_count, block_size):
                    for page in encoder.write(result.audio[start:start + block_size]):
                        yield page
                    await asyncio.sleep(0)
            for page in encoder.finish():
                yield page
