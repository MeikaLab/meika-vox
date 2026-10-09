"""Redact common secrets from logs while keeping actionable error details."""

from __future__ import annotations

import re
from collections.abc import Iterable

_HF_TOKEN = re.compile(r"hf_[A-Za-z0-9_-]{8,}")
_CREDENTIAL = re.compile(
    r"(?i)\\b(authorization|bearer|api[_-]?key|token|password|secret)"
    r"(\\s*[:=]\\s*|\\s+)([^\\s,\"';]{8,})"
)


def redact(value: object, secrets: Iterable[str | None] = ()) -> str:
    """Hide supplied tokens and common credential patterns before logging."""
    result = str(value)
    for secret in secrets:
        if secret and len(secret) >= 6:
            result = result.replace(secret, "[clave oculta]")
    result = _HF_TOKEN.sub("[clave oculta]", result)
    return _CREDENTIAL.sub(
        lambda match: match.group(1) + match.group(2) + "[clave oculta]", result
    )


def tail(value: object, limit: int = 3500) -> str:
    """Keep the end of an error log (usually the useful final exception)."""
    text = str(value)
    return text if len(text) <= limit else "..." + text[-limit:]
