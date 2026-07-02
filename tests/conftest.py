"""Pytest session setup shared by the whole test suite.

speechbrain lazy-import guard
-----------------------------
``pyannote.audio`` pulls in ``speechbrain``, which registers many of its
optional integrations (k2-FSA, several huggingface helpers, ...) as
``speechbrain.utils.importutils.LazyModule`` objects living in ``sys.modules``.
A ``LazyModule`` only imports its real target on first *attribute* access — and
its ``__getattr__`` forces that import for **any** attribute, including dunders,
re-raising as ``ImportError`` when the optional dependency is absent.

The trap: stdlib introspection walks ``sys.modules`` and probes dunders. In
particular ``linecache.getlines`` does ``getattr(mod, '__file__', None)`` over
every module while formatting a traceback or warning. ``getattr(obj, name,
default)`` only swallows ``AttributeError`` — the ``ImportError`` raised by
``LazyModule.__getattr__`` propagates instead, from a finalizer / ``sys.excepthook``
context that is *outside* pytest's per-test guard. On a machine missing any of
those optional deps (e.g. ``k2`` here) this interrupts the whole in-process
session: pytest prints a partial "N passed" with a non-zero exit, and tests
collected after the interrupt never run. Every test file still passes in
isolation; only the single full-suite ``pytest`` invocation is truncated.

speechbrain's own code already side-steps this for one caller — ``ensure_module``
raises ``AttributeError`` when the importing frame is ``inspect.py`` (so PyTorch's
op-registration machinery can't trigger spurious imports). We extend that same
intent: a not-yet-loaded ``LazyModule`` answers *dunder* introspection probes
with ``AttributeError`` (the normal "absent" signal) instead of forcing the
import. Real attribute access (e.g. ``sbk2.graph_compiler``) is unchanged, and
``__path__`` is deliberately left alone so genuine submodule imports still work.

This lives in the test harness only; it changes no production code path. The
patch is process-wide for the test session (a fixture would apply too late to
catch session-level crashers): a test that deliberately asserts a missing
optional dep raises ``ImportError`` on *dunder* access would not see it — no
such test exists, and that is an unusual thing to assert.
"""

try:
    from speechbrain.utils import importutils as _sb_importutils
except Exception:  # speechbrain not installed / import failed — nothing to guard
    _sb_importutils = None

if _sb_importutils is not None:
    _LazyModule = _sb_importutils.LazyModule
    _orig_getattr = _LazyModule.__getattr__

    def _lazy_getattr_no_force_on_dunder(self, attr):
        # Don't let stdlib introspection (linecache, inspect, pickle, copy, ...)
        # force a lazy — and possibly dependency-missing — import just by probing
        # a dunder. __path__ is excluded so real submodule imports still resolve.
        if attr != "__path__" and attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(attr)
        return _orig_getattr(self, attr)

    _LazyModule.__getattr__ = _lazy_getattr_no_force_on_dunder
