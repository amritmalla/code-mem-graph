from __future__ import annotations

from importlib.metadata import version

import tiktoken

from .serialization import serialize


class PayloadCounter:
    def __init__(self, encoding: str) -> None:
        self.encoding = encoding
        self._encoder = tiktoken.get_encoding(encoding)

    def __call__(self, value: object) -> int:
        return len(self._encoder.encode(serialize(value), disallowed_special=()))

    @property
    def description(self) -> dict:
        return {"library": "tiktoken", "version": version("tiktoken"), "encoding": self.encoding,
                "basis": "canonical_serialized_payload_estimate"}
