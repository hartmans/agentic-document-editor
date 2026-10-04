# Streaming playback: phase 2 implementation plan

Agreed with the user, October 4, 2026. Phase 1 implements backend/document
behavior only. Do not change frontend playback as part of phase 1.

## Phase 2 scope

- Use Opus in Ogg for both completed and incrementally generated chunk audio.
  Feed production-format PCM from radio-drama to one encoder per chunk stream.
  PyAV and FFmpeg/libopus are available locally. Use short container flushing
  intervals; encoder throughput has been verified, browser startup latency has
  not. Preserve the PCM timeline through codec pre-skip and end trimming.
- Expose the phase 1 PCM iterator through an incremental HTTP audio response.
  Keep the revision/chunk URL model; no temporary playback IDs or suffix plans.
- Replace complete-WAV fetch/decodeAudioData/AudioBufferSource playback with an
  HTML audio element. Preserve the existing React 19 generation/intent guards,
  revision checks, autoplay behavior, empty-chunk skipping, and natural-end flow.
  Review AGENTS.md before editing DocumentPlayback.tsx.
- On interruption call pause(), then read currentTime immediately before any
  source reset. Keep this chunk-relative audio position locally; resolve it
  through word marks before submitting character-only instructions. Do not use
  the last timeupdate event or a wall clock to infer the stopped position.
- A mid-chunk seek into already rendered audio may wait for timing. Repositioning
  alone must not split chunks. Handle seeking and buffered resumes explicitly.
- Cover buffering/stalls, interrupted startup, cancellation, stream errors versus
  successful EOF, and stale events after switching chunks/revisions. Keep audio
  flowing while marks are pending at a chunk boundary.
- A disconnected playback consumer must not cancel radio-drama's cache filling.
  Bound encoder/transport buffers and clean up the encoder on disconnect.

## Settled backend behavior

- Initially prepare/register all chunks, eagerly render the entire document,
  and call ensure_timing as each render completes.
- After a text-changing round, if the final resume point lies inside a chunk
  without available audio, split that chunk at the resume position. Preserve
  all other chunk boundaries and resources. Prepare and plan changed chunks.
- If final text is unchanged, retain the original state/ID and boundaries.
- New playback chunks start streaming before new background batch rendering.
  Once the first nonempty audio arrives, render the next chunk in that document.
  Existing background renders continue; radio-drama owns batching.
- For an unavailable chunk, render_stream may race an existing batch render.
  The user accepts approximate marks when the batch's different audio wins the
  cache. Do not add per-playback timing identities or corrective machinery.
- radio-drama owns concurrent ensure_timing memoization. Background timing and
  marks requests call it directly; the editor only projects source-word marks.

## Validation boundaries

Editor tests cover our splitting, revision isolation, trigger ordering, source
mark projection, HTTP output, playback state, and stop positions. Library cache,
stream sharing, and timing memoization tests belong in radio-drama; suggestions
are in docs/radio_drama_test_suggestions.md. Use real radio-drama integration,
not replacement engines to bypass dependencies. Run frontend commands with
`yarnpkg`. Device buffering means stop positions have no sample-accurate audible
output guarantee; that limitation is accepted, as with the previous player.

## Phase 1 handoff

- `agentic_editor.tts.prepare_document_round(previous, draft, position)` selects
  the surviving state, probes availability by registration, and performs a local
  resume split when required. It never starts rendering.
- `submit_document_audio(..., eager=True)` registers plans and starts full startup
  rendering/timing; instructions use `eager=False`.
- `stream_document_audio(state, chunk_id)` yields radio-drama RenderResults with
  production-rate/channel PCM and triggers the next nonempty chunk after first
  audio. Connect this iterator to the Opus encoder in phase 2. Timing starts in
  the background and is not a prerequisite for yielding audio.
- `SpeechService.stream_audio(chunk)` supplies the single-chunk primitive.
  Cold marks requests also join this streaming path so they cannot start a batch
  ahead of the new chunk's playback. They then call ensure_timing directly.
- The existing WAV route remains compatible and waits for a complete result;
  it does not yet use the document iterator's successor trigger.
- GPU-dependent streaming/rendering checks were deferred at the user's request
  because GPU memory was scarce. Run the complete backend suite when available,
  including `test_streaming_playback_starts_next_chunk_and_background_timing`.
