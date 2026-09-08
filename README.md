# agentic-document-editor

Document revision API:

- `/status` returns the current immutable `documentId` and document metadata.
- `/instructions` requires `baseDocumentId`. Its result contains the updated
  `documentId`, `document`, and revision-bound `resumePoint`, plus the separate
  `responseDocumentId` and `responseDocument` for the assistant's spoken reply.
  Concurrent instructions are serialized; a stale base revision returns HTTP 409.
- `/interruptions` requires the `documentId` of the audio being heard. Old
  revisions remain available for audio and position mapping until server restart.
- Stop cancels pending playback and automatic continuation, but an instruction
  already submitted can still finish editing the document.
