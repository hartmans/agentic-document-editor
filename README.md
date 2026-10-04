# agentic-document-editor

TTS uses `radio_drama`, defaulting to `voxcpm2_longform`. Copy `config.example.yml` to
`config.yml` and set `voice_sample` to a reference recording. Select another
backend with `tts` or `--tts`; named proxy engines are configured by radio_drama
in `~/.config/radio-drama/tts.toml`. Its built-in engines are also selectable.
The old Polly TTS path and `speech_driver` setting have been removed.

Install the local radio_drama checkout into the Python environment used to run
the backend (`python -m pip install -e ../radio-drama-app`), or include that
checkout on `PYTHONPATH` alongside `~/ai/ai-tools`. The environment also needs
the editor's LangChain providers, including `langchain-ollama` for the default
editing model. On this system `~/venv` contains the required dependencies.

```sh
PYTHONPATH=../ai-tools:../radio-drama-app ~/venv/bin/python backend.py \
  --document-path path/to/chapter.rst --config config.yml
yarnpkg dev
```

`cache_dir` is a common base (default `.tts_cache` under cwd). The editor passes
radio_drama a subdirectory named from the document stem and a hash of its
absolute path. Edits retain that directory; moving the document selects another.
Assistant responses share the source document's directory. Delete the document
cache when changing voices. No caches are placed beside the source document.

Document construction establishes local chunk boundaries and filters each chunk
into a list of dialogue lines (currently one line, or none for empty text).
`build_graph_state()` forks a shallow document draft for each instruction round;
`DocumentState.update(text, start, end)` copies affected chunks on their first
edit. Absolute positions belong to the revision, so shifted chunks can share
speech resources with older revisions. Empty slots remain in document responses;
their array index is their chunk ID and playback skips slots without audio.

At round completion, identical document text keeps the published state and ID.
After text changes, an interior resume point splits only its containing chunk,
and only if that chunk has no completed audio. Registration first checks cache
availability without starting rendering; new split chunks are then prepared and
planned. Other boundaries and resources survive. Reposition-only rounds do not
split chunks. Each chunk owns its filtered text, dialogue lines, AudioPlan, and
WAV task. radio_drama owns concurrent timing memoization. Published revisions
stay available for playback.

Startup eagerly renders all chunks and requests timing as each finishes.
Post-edit registration does not eagerly render new chunks. The backend PCM
playback iterator streams unavailable audio, starts rendering the next nonempty
chunk after its first audio arrives, and pregenerates timing independently.

Audio and word marks are independently available. Playback starts at a chunk
boundary or a known audio offset without awaiting marks; a character seek inside
a chunk waits for timing. Instruction submission waits for marks to resolve an
interruption to a character position; audio offsets remain local to playback.
Late marks enable highlighting at the current playback time. Missing marks
retain the existing preceding-word approximation.
Phase 1 exposes this behavior through `stream_document_audio()`; the current
WAV route still waits for a complete result. Incremental Opus delivery and
frontend playback are phase 2; see [the saved plan](docs/streaming_phase_2.md).

Voice input is unchanged: Gemma interprets audio using document context, and
setting `region_name` selects AWS Transcribe. radio_drama alignment is used only
to obtain TTS word marks.

Run editor tests with:

```sh
PYTHONPATH=../ai-tools:../radio-drama-app ~/venv/bin/python -m pytest tests
```

The integration test needs a configured VoxCPM2 engine, model
access, and a reference recording (`RADIO_DRAMA_TEST_VOICE`, default `chandra.wav`
here). Frontend checks are `yarnpkg build`, `node tests/test_speech_marks.mjs`, and
`node tests/test_playback_readiness.mjs`. Suggested library-owned tests are in
[the radio_drama handoff](docs/radio_drama_test_suggestions.md).

Document revision API:

- `/status` returns the current immutable `documentId` and document metadata.
- `/instructions` requires `baseDocumentId`. Its result contains the updated
  `documentId`, `document`, and revision-bound `resumePoint`, plus the separate
  `responseDocumentId` and `responseDocument` for the assistant's spoken reply,
  and a `disposition` of `play`, `pause`, or `stop` controlling what follows it.
  Concurrent instructions are serialized; a stale base revision returns HTTP 409.
- Old revisions remain available for audio until server restart.
- Stop cancels pending playback and automatic continuation, but an instruction
  already submitted can still finish editing the document.
