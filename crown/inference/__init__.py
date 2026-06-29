"""API-specific inference extensions."""

from __future__ import annotations

from typing import Optional

from surya.inference import SuryaInferenceManager, _autodetect_backend, _build_backend

from crown.inference.vllm import VllmPersistentBackend


class ApiInferenceManager(SuryaInferenceManager):
    """SuryaInferenceManager that uses persistent vLLM Docker containers."""

    def __init__(self, method: Optional[str] = None, lazy: bool = True):
        self.method = method or _autodetect_backend()
        if self.method.lower() == "vllm":
            self.backend = VllmPersistentBackend()
        else:
            self.backend = _build_backend(self.method)
        if not lazy:
            self.backend.start()
