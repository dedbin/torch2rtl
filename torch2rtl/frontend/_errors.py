from __future__ import annotations


class UnsupportedOpError(RuntimeError):
    """Raised when the FX graph contains an operation outside the MVP subset."""


# Preserve the established public import/pickle contract while keeping the
# physical definition in the dependency-leaf component.
UnsupportedOpError.__module__ = "torch2rtl.frontend.pytorch_fx"


__all__ = ["UnsupportedOpError"]
