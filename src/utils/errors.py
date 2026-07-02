class ProcessingError(RuntimeError):
    """Raised when processing cannot continue safely."""


class MediaDecodeError(ProcessingError):
    """Raised when an input media file cannot be read or decoded.

    Covers corrupt/truncated files, a missing audio track, and unsupported
    codecs. Distinct from the generic error bucket so the UI can show a
    specific, actionable message.
    """
