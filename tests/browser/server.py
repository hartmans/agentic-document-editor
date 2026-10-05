"""Real HTTP/Opus fixtures for browser tests; synthetic PCM needs no GPU."""
import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from radio_drama import RenderResult
from agentic_editor.opus import encode_opus

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["http://127.0.0.1:5179"], allow_methods=["GET", "POST"], allow_headers=["*"])


def tone(seconds):
    t = np.arange(round(seconds * 48000)) / 48000
    return RenderResult(audio=(0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32))


@app.get('/audio')
async def audio(mode: str = 'complete'):
    async def pcm():
        if mode == 'pending':
            await asyncio.sleep(5)
        if mode in ('slow', 'broken', 'truncated'):
            yield tone(4)
            await asyncio.sleep(3)
            if mode in ('broken', 'truncated'):
                raise RuntimeError('Intentional interrupted HTTP response')
            yield tone(2)
        else:
            yield tone(4)
    async def encoded():
        try:
            async for page in encode_opus(pcm(), sample_rate=48000, channels=1):
                yield page
        except RuntimeError:
            if mode != 'truncated':
                raise
            # Clean HTTP EOF, but missing the Ogg EOS page.
    return StreamingResponse(encoded(), media_type='audio/ogg; codecs=opus')


@app.get('/chunks/{document_id}/{chunk_id}/audio')
async def chunk_audio(document_id: str, chunk_id: int):
    return await audio('slow' if document_id == 'delayed' else 'complete')


@app.get('/chunks/{document_id}/{chunk_id}/marks')
async def marks(document_id: str, chunk_id: int):
    if document_id == 'delayed':
        await asyncio.sleep(3)
    return {'documentId': document_id, 'chunkId': chunk_id,
            'marks': [{'timeMs': 0, 'docCharStart': 0, 'docCharEnd': 5},
                      {'timeMs': 600, 'docCharStart': 6, 'docCharEnd': 11}]}


@app.get('/health')
def health():
    return {'ok': True}


DOCUMENT = {'document': 'Hello world', 'voice_id': 'fixture',
            'chunking_strategy': 'paragraph',
            'chunks': [{'doc_char_start': 0, 'doc_char_end': 11, 'has_audio': True}]}


@app.get('/status')
def status():
    return {'documentId': 'complete', 'document': DOCUMENT,
            'documentLength': 11, 'chunkCount': 1, 'voiceId': 'fixture',
            'chunkingStrategy': 'paragraph'}


@app.post('/instructions')
def instructions(request: dict):
    return {'documentId': 'complete', 'document': DOCUMENT,
            'responseDocumentId': 'reply', 'responseDocument': DOCUMENT,
            'resumePoint': request['resumePoint'],
            'disposition': 'pause' if request['instructions'] == 'pause' else 'play'}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8019)
