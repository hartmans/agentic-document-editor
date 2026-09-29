# agentic-document-editor

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
