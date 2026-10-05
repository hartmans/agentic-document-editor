"""Test the editor's encoding and stream lifecycle with real libopusenc."""
import asyncio
import io

import numpy as np
import pytest
import soundfile as sf
from radio_drama import RenderResult

from agentic_editor.opus import OpusEncoder, encode_opus


@pytest.mark.parametrize('rate', [24000, 44100, 48000])
@pytest.mark.parametrize('channels', [1, 2])
def test_streamed_opus_preserves_duration_and_channel_layout(rate, channels):
    count = rate + 173  # Includes a partial codec frame.
    signal = (0.2 * np.sin(2 * np.pi * 440 * np.arange(count) / rate)).astype('float32')
    pcm = signal if channels == 1 else np.column_stack((signal, -signal))
    with OpusEncoder(rate, channels) as encoder:
        pages = encoder.headers()
        for start in range(0, count, 777):
            pages.extend(encoder.write(pcm[start:start + 777]))
        pages.extend(encoder.finish())
    decoded, decoded_rate = sf.read(io.BytesIO(b''.join(pages)), always_2d=True)
    assert decoded.shape[1] == channels
    assert abs(len(decoded) / decoded_rate - count / rate) <= 1 / decoded_rate
    expected = 0.2 * np.sin(2 * np.pi * 440 * np.arange(len(decoded)) / decoded_rate)
    # Detect an uncompensated codec lookahead offset, not just valid decoding.
    assert np.corrcoef(decoded[:, 0], expected)[0, 1] > 0.99
    if channels == 2:
        assert np.corrcoef(decoded[:, 0], decoded[:, 1])[0, 1] < -0.99


def test_empty_opus_and_closed_encoder():
    with OpusEncoder(48000, 1) as encoder:
        data = b''.join(encoder.headers() + encoder.finish())
        with pytest.raises(RuntimeError):
            encoder.write(np.zeros(10))
    assert sf.info(io.BytesIO(data)).frames == 0
    assert encoder._encoder is None


def test_audio_pages_arrive_before_source_completion_and_close_source_on_disconnect():
    async def exercise():
        released = asyncio.Event()
        closed = asyncio.Event()
        async def source():
            try:
                yield RenderResult(audio=np.zeros(4800, dtype='float32'))
                await released.wait()
                yield RenderResult(audio=np.zeros(4800, dtype='float32'))
            finally:
                closed.set()
        encoded = encode_opus(source(), sample_rate=48000, channels=1)
        try:
            while True:
                page = await asyncio.wait_for(anext(encoded), 2)
                # Ogg page sequence: 0 identification, 1 comments, then audio.
                if int.from_bytes(page[18:22], 'little') >= 2:
                    break
            assert not released.is_set()
        finally:
            await encoded.aclose()
        assert closed.is_set()
    asyncio.run(exercise())


def test_failed_source_never_gets_a_successful_ogg_end_marker():
    async def source():
        yield RenderResult(audio=np.zeros(4800, dtype='float32'))
        raise RuntimeError('source failed')
    async def exercise():
        pages = []
        with pytest.raises(RuntimeError, match='source failed'):
            async for page in encode_opus(source(), sample_rate=48000, channels=1):
                pages.append(page)
        assert pages
        assert not any(page[5] & 4 for page in pages)
    asyncio.run(exercise())
