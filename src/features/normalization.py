"""Conservative, configurable text normalization for blocking experiments."""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping


DEFAULT_OPTIONS = {
    "casefold": True,
    "unicode_nfkc": True,
    "punctuation_to_space": True,
    "collapse_whitespace": True,
}


def normalize_text(value: object, options: Mapping[str, bool] | None = None) -> str:
    settings = DEFAULT_OPTIONS | dict(options or {})
    unknown = set(settings) - set(DEFAULT_OPTIONS)
    if unknown:
        raise ValueError(f"Unknown normalization options: {sorted(unknown)}")
    text = "" if value is None else str(value)
    if settings["unicode_nfkc"]:
        text = unicodedata.normalize("NFKC", text)
    if settings["casefold"]:
        text = text.casefold()
    if settings["punctuation_to_space"]:
        text = "".join(char if char.isalnum() or char.isspace() else " " for char in text)
    if settings["collapse_whitespace"]:
        text = " ".join(text.split())
    return text


def normalize_business_name(value: object, options: Mapping[str, bool] | None = None) -> str:
    return normalize_text(value, options)


def normalize_business_address(value: object, options: Mapping[str, bool] | None = None) -> str:
    return normalize_text(value, options)
