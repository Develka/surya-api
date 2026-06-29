"""vLLM backend extension: exactly one persistent Docker container (no --rm).

Lifecycle on start():
  - container missing            -> create it
  - container exists but stopped -> start it; recreate if it stays unhealthy
  - container running + healthy   -> reuse as-is

Container (re)creation is capped at MAX_RECREATE_ATTEMPTS to avoid an infinite
crash/recreate loop; after the cap a SpawnError is raised.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from time import perf_counter
from typing import Callable, Optional

from openai import OpenAI

from surya.inference.backends.base import ServerHandle
from surya.inference.backends.spawn import (
    SpawnError,
    _lock_path,
    _write_sentinel,
    find_free_port,
    probe_model_id,
    wait_for_health,
)
from surya.inference.backends.vllm import (
    VllmBackend,
    _gpu_settings,
    _health_url,
    _openai_url,
    _resolve_docker_binary,
)
from surya.logging import get_logger
from surya.settings import settings

try:
    from filelock import FileLock
except ImportError:  # pragma: no cover - filelock is a hard dependency in practice
    FileLock = None  # type: ignore[assignment]

logger = get_logger()

# Exactly one inference container ever exists, under this fixed name.
CANONICAL_CONTAINER = "surya-vllm"

# Hard cap on create/recreate attempts before giving up.
MAX_RECREATE_ATTEMPTS = 3


def _startup_timeout() -> float:
    return settings.SURYA_INFERENCE_STARTUP_TIMEOUT


# --------------------------------------------------------------------------- #
# Cache storage (cross-platform)
#
# Two model/compile caches matter for startup time:
#   - HuggingFace weights   -> /root/.cache/huggingface
#   - vLLM torch.compile     -> /root/.cache/vllm   (lost on container recreate
#                                                     unless persisted)
#
# Mode is controlled by SURYA_DOCKER_CACHE_MODE:
#   - "volume" (default): Docker *named volumes*. These live inside the Docker
#     storage backend (ext4 in the WSL2/Docker-Desktop VM on Windows/macOS,
#     /var/lib/docker on native Linux), avoiding the slow 9P bind-mount path on
#     WSL2. The `name:/container/path` syntax is identical on every OS.
#   - "bind": host directory bind mounts (handy on native Linux or to inspect
#     files from the host). Paths are resolved per-OS via pathlib.
# --------------------------------------------------------------------------- #
def _cache_mode() -> str:
    return os.environ.get("SURYA_DOCKER_CACHE_MODE", "volume").strip().lower()


def _hf_cache_volume() -> str:
    return os.environ.get("SURYA_HF_CACHE_VOLUME", "surya-hf-cache")


def _vllm_cache_volume() -> str:
    return os.environ.get("SURYA_VLLM_CACHE_VOLUME", "surya-vllm-cache")


def _vllm_cache_bind_path() -> str:
    raw = os.environ.get("DOCKER_VLLM_CACHE_PATH", "~/.cache/surya-vllm")
    return str(Path(raw).expanduser().resolve())


def _hf_cache_bind_path() -> str:
    return str(Path(settings.DOCKER_HF_CACHE_PATH).expanduser().resolve())


def _hf_cache_mount() -> str:
    if _cache_mode() == "bind":
        return f"{_hf_cache_bind_path()}:/root/.cache/huggingface"
    return f"{_hf_cache_volume()}:/root/.cache/huggingface"


def _vllm_cache_mount() -> str:
    if _cache_mode() == "bind":
        return f"{_vllm_cache_bind_path()}:/root/.cache/vllm"
    return f"{_vllm_cache_volume()}:/root/.cache/vllm"


def _ensure_volume(name: str) -> None:
    """Create a Docker named volume if missing (idempotent, cross-platform)."""
    subprocess.run(
        ["docker", "volume", "create", name],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def _prepare_cache_storage() -> None:
    """Ensure cache backing stores exist before the container is created."""
    if _cache_mode() == "bind":
        Path(_hf_cache_bind_path()).mkdir(parents=True, exist_ok=True)
        Path(_vllm_cache_bind_path()).mkdir(parents=True, exist_ok=True)
        return
    _ensure_volume(_hf_cache_volume())
    _ensure_volume(_vllm_cache_volume())


def _container_running(container_name: str) -> Optional[bool]:
    """True/False if the container exists (running or not), None if it does not."""
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", container_name],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() == "true"


def _resolve_container_port(container_name: str) -> Optional[int]:
    """Host port mapped to the container's 8000/tcp, or None if unavailable.

    Reads HostConfig.PortBindings via `docker inspect`, which is persisted in the
    container config and therefore works for *stopped* containers too. `docker
    port` only reports published ports for running containers, so relying on it
    would mis-resolve a stopped container's port and make health checks probe the
    wrong port (wasting the full startup timeout before a needless recreate).
    """
    fmt = '{{with index .HostConfig.PortBindings "8000/tcp"}}{{(index . 0).HostPort}}{{end}}'
    result = subprocess.run(
        ["docker", "inspect", "-f", fmt, container_name],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if result.returncode == 0 and result.stdout.strip():
        try:
            return int(result.stdout.strip())
        except ValueError:
            pass

    # Fallback: docker port (running containers only).
    result = subprocess.run(
        ["docker", "port", container_name, "8000/tcp"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    line = result.stdout.strip().splitlines()[0]
    try:
        return int(line.rsplit(":", 1)[-1])
    except ValueError:
        return None


def _choose_port() -> int:
    return settings.SURYA_INFERENCE_PORT or find_free_port()


def _start_container(container_name: str) -> None:
    result = subprocess.run(
        ["docker", "start", container_name],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        raise SpawnError(
            f"docker start failed for {container_name}: "
            f"{result.stderr or result.stdout}"
        )
    logger.info("Started existing container %s", container_name)


def _remove_container(container_name: str) -> None:
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    logger.info("Removed container %s", container_name)


def _run_container(port: int) -> None:
    """Create and start the canonical container on the given host port."""
    docker = _resolve_docker_binary()
    max_batched_tokens, max_num_seqs = _gpu_settings(settings.VLLM_GPU_TYPE)
    _prepare_cache_storage()
    cmd = [
        docker,
        "run",
        "-d",
        "--name",
        CANONICAL_CONTAINER,
        "--runtime",
        "nvidia",
        "--gpus",
        f"device={settings.VLLM_GPUS}",
        "-v",
        _hf_cache_mount(),
        "-v",
        _vllm_cache_mount(),
        "-p",
        f"{port}:8000",
        "--ipc=host",
        settings.VLLM_DOCKER_IMAGE,
        "--model",
        settings.SURYA_MODEL_CHECKPOINT,
        "--no-enforce-eager",
        "--max-num-seqs",
        str(max_num_seqs),
        "--dtype",
        settings.VLLM_DTYPE,
        "--max-model-len",
        str(settings.VLLM_MAX_MODEL_LEN),
        "--max-num-batched-tokens",
        str(max_batched_tokens),
        "--gpu-memory-utilization",
        str(settings.VLLM_GPU_MEMORY_UTILIZATION),
        "--enable-prefix-caching",
        "--mm-processor-kwargs",
        json.dumps({"min_pixels": 3136, "max_pixels": 6291456}),
        "--served-model-name",
        settings.SURYA_MODEL_CHECKPOINT,
    ]
    if settings.VLLM_ENABLE_MTP:
        cmd.extend(
            [
                "--speculative-config",
                json.dumps(
                    {
                        "method": "mtp",
                        "num_speculative_tokens": settings.VLLM_MTP_TOKENS,
                    }
                ),
            ]
        )
    for extra in (settings.VLLM_EXTRA_ARGS or "").split():
        cmd.append(extra)

    logger.info("Spawning: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise SpawnError(f"docker run failed: {result.stderr or result.stdout}")


def _write_container_sentinel(backend: str, port: int, last_updated: float) -> None:
    _write_sentinel(
        backend,
        {
            "port": port,
            "pid": None,
            "model": settings.SURYA_MODEL_CHECKPOINT,
            "backend": backend,
            "cleanup_id": CANONICAL_CONTAINER,
            "cleanup_kind": "docker",
            "request_count": 0,
            "last_updated": last_updated,
        },
    )


def _probe_health(url: str, timeout: float = 5.0) -> bool:
    import httpx

    try:
        with httpx.Client(timeout=timeout, trust_env=False) as client:
            return client.get(f"{url.rstrip('/')}/health").status_code == 200
    except Exception:
        return False


def ensure_single_container(probe_health: Callable[..., bool] = _probe_health) -> int:
    """Guarantee exactly one healthy canonical container; return its host port.

    Raises SpawnError if the container cannot be made healthy within
    MAX_RECREATE_ATTEMPTS create/recreate attempts.
    """
    timeout = _startup_timeout()
    last_error = "unknown error"

    for attempt in range(1, MAX_RECREATE_ATTEMPTS + 1):
        state = _container_running(CANONICAL_CONTAINER)

        # Already running and healthy -> reuse without touching it.
        if state is True:
            port = _resolve_container_port(CANONICAL_CONTAINER)
            if port and probe_health(_health_url(port)):
                logger.info(
                    "Reusing healthy container %s on port %s",
                    CANONICAL_CONTAINER,
                    port,
                )
                return port

        if state is None:
            # Missing -> create fresh.
            port = _choose_port()
            logger.info(
                "Creating container %s on port %s (attempt %d/%d)",
                CANONICAL_CONTAINER,
                port,
                attempt,
                MAX_RECREATE_ATTEMPTS,
            )
            try:
                _run_container(port)
            except SpawnError as exc:
                last_error = str(exc)
                logger.warning("Create failed: %s", last_error)
                _remove_container(CANONICAL_CONTAINER)
                continue
        else:
            # Exists. The port MUST come from the container itself; never fall
            # back to a random port here, or we would health-check a port the
            # container isn't bound to and waste the full startup timeout.
            port = _resolve_container_port(CANONICAL_CONTAINER)
            if port is None:
                last_error = "could not resolve container port; recreating"
                logger.warning(
                    "Container %s exists but its port is unresolvable; recreating "
                    "(attempt %d/%d)",
                    CANONICAL_CONTAINER,
                    attempt,
                    MAX_RECREATE_ATTEMPTS,
                )
                _remove_container(CANONICAL_CONTAINER)
                continue
            if state is False:
                logger.info(
                    "Starting stopped container %s on port %s (attempt %d/%d)",
                    CANONICAL_CONTAINER,
                    port,
                    attempt,
                    MAX_RECREATE_ATTEMPTS,
                )
                try:
                    _start_container(CANONICAL_CONTAINER)
                except SpawnError as exc:
                    last_error = str(exc)
                    logger.warning("Start failed: %s; recreating", last_error)
                    _remove_container(CANONICAL_CONTAINER)
                    continue

        # Wait for health after create/start (or running-but-still-warming-up).
        if wait_for_health(_health_url(port), total_timeout=timeout):
            logger.info("Container %s healthy on port %s", CANONICAL_CONTAINER, port)
            return port

        last_error = f"did not become healthy within {int(timeout)}s"
        logger.warning(
            "Container %s %s; removing and recreating (attempt %d/%d)",
            CANONICAL_CONTAINER,
            last_error,
            attempt,
            MAX_RECREATE_ATTEMPTS,
        )
        _remove_container(CANONICAL_CONTAINER)

    raise SpawnError(
        f"vLLM container '{CANONICAL_CONTAINER}' failed to become healthy after "
        f"{MAX_RECREATE_ATTEMPTS} attempts: {last_error}"
    )


class VllmPersistentBackend(VllmBackend):
    """vLLM backed by exactly one persistent Docker container."""

    def start(self):
        if self.handle is not None:
            return self.handle

        if settings.SURYA_INFERENCE_URL:
            return super().start()

        lock_ctx = (
            FileLock(str(_lock_path(self.name)), timeout=timeout_for_lock())
            if FileLock is not None
            else _NullLock()
        )
        with lock_ctx:
            # Re-check under the lock: another worker may have started it.
            if self.handle is not None:
                return self.handle

            port = ensure_single_container()
            base_url = _openai_url(port)
            model_name = probe_model_id(base_url) or settings.SURYA_MODEL_CHECKPOINT
            _write_container_sentinel(self.name, port, perf_counter())

            self.handle = ServerHandle(
                base_url=base_url,
                model_name=model_name,
                spawned_by_us=True,
            )
            self._client = OpenAI(
                api_key=settings.VLLM_API_KEY,
                base_url=base_url,
            )
            return self.handle


def timeout_for_lock() -> float:
    # Allow enough time to acquire the spawn lock even if another caller is in
    # the middle of a full cold start.
    return MAX_RECREATE_ATTEMPTS * _startup_timeout() + 120


class _NullLock:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
