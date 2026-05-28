"""Video-to-text transcription pipeline.

Modules:
    models       - Data models (TranscriptDocument, Segment, PipelineContext)
    pipeline     - Pipeline orchestration framework and concrete steps
    ingestion    - File discovery and media extraction
    transcription - ASR engines (Whisper)
    processing   - Post-transcription processing (cleanup, future: topics, summarization)
    output       - Formatting and writing results
    utils        - Configuration, CLI, device selection, logging
"""
