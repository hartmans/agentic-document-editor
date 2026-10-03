# Suggested radio_drama regression tests

The agentic-document-editor migration consumes prepared ScriptPlans and relies
on the contracts below. Please place tests of these library guarantees in
radio_drama, rather than duplicating them in the editor. This is a handoff for
an agent working in that repository; no changes there have been made here.

## Word markers with native dialogue-line timing

Register a request whose TTS backend returns native DialogueLineTiming with no
word marks. Supply nonempty DialogueLine.mark_offsets. Calling ensure_timing
must request word alignment and return marks parallel to the supplied offsets,
while preserving the native line bounds. Include offsets zero, inside the line,
and len(spoken_text), as well as duplicates and unknown interior boundaries.

The current CachedTtsRequest.ensure_timing implementation appears to enrich
native line timing correctly. If an actual backend bypasses this, treat it as a
radio_drama bug; the editor will not synthesize substitute word timings.

## Independent rendering and alignment

A prepared ScriptPlan with word offsets must render audio without loading or
waiting for forced alignment. A simultaneous ensure_timing operation must not
hold up another render reader. An alignment failure must leave rendered audio
available, and an alignment retry should be supported without regenerating it.

## Cache identities and timing reuse

Changing mark offsets must reuse audio and refresh marker timing. A process
restart must reload suitable audio/timing. Identical rendered speech with a
changed original-document location must not require new audio: document source
positions belong to the editor, not the speech request. Explicit
CACHE_DIRECTORY_KEY must override document/output-derived cache locations.
Voice fingerprinting is intentionally out of scope; users delete their cache
when changing reference voices.

## Prepared plans and batching

Preparing multiple ScriptPlans with one DialogueLine each, before triggering
rendering, must make those requests available to backend batching. Check proxy
and built-in engines using their intended batch sizes. Preparing should not
start rendering. A shared registration must support concurrent render readers,
including one canceled reader, without duplicate production or stranded waits.

## Sparse alignment and empty speech

Preserve unknown marker sides as None rather than fabricating coverage. Timing
must remain in the documented audio timebase after output-rate conversion.
A ScriptPlan with no dialogue should render an empty result without requesting
TTS or alignment. Exercise shutdown while a batch or alignment is in progress,
with no abandoned subprocesses or unobserved task exceptions.

## Future recording-backed chunk slices

When this feature is developed, add coverage for a shared recording-backed
script sliced into chunks: each slice needs local audio/timing, stable source
correspondences, and reuse of the document-level alignment. This migration does
not yet implement that feature.

## Observed VoxCPM2 proxy failure on this system

The editor's real integration test registered a ScriptPlan successfully, then
received `JSONDecodeError: Expecting value: line 1 column 1 (char 0)` during
VoxCPM2 initialization with the installed
`localhost/radio-drama-voxcpm2:latest` image. The output showed vLLM Omni loading
`openbmb/VoxCPM2`. This suggests non-protocol stdout, but the exact offending
line has not yet been captured. Please investigate and add a protocol-cleanliness
regression test around actual engine startup, including worker-process output
that may bypass Python's redirect_stdout. The editor does not skip malformed
protocol lines or replace the real engine with a test double.

This failure was reported fixed upstream on October 3, 2026. The editor now
selects `voxcpm2_longform` (image `localhost/radio-drama-voxcpm2-longform:latest`)
and uses its installed configuration directly. The earlier network override
was removed. The full editor suite passed with that configuration on October 3:
66 tests, including real rendering, word marks, and revision-relative projection.
Retain startup protocol-cleanliness coverage upstream to catch regressions.

## What stays in editor tests

Filter execution and preparation reuse, original/filtered text matching,
word-marker projection to document offsets, revision-bound HTTP responses,
chunk grouping at registration, WAV encoding, and playback before marks are
ready. Gemma's contextual voice interpretation and AWS Transcribe are unchanged.
