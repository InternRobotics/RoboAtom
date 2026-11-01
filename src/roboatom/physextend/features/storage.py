"""Read rclone-style remote paths without persisting benchmark media."""

from __future__ import annotations

import os
import re
import subprocess


RCLONE_REMOTE_RE = re.compile(r"^[^/:\s]+:.+")
DEFAULT_RCLONE_TIMEOUT_SECONDS = 300.0
PROXY_VARIABLES = (
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "all_proxy",
)


def is_rclone_remote(path: str) -> bool:
    """Return whether a path uses an rclone remote rather than an S3 URI.

    Args:
        path: Candidate local or remote path.

    Returns:
        True for values such as ``remote:bucket/file``.
    """
    return RCLONE_REMOTE_RE.match(path) is not None and "s3://" not in path


def _network_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for variable in PROXY_VARIABLES:
        environment.pop(variable, None)
    return environment


def _rclone_binary() -> str:
    return os.environ.get("RCLONE_BIN", "rclone")


def _rclone_timeout_seconds() -> float:
    raw_timeout = os.environ.get(
        "ROBOATOM_RCLONE_TIMEOUT_SECONDS", str(DEFAULT_RCLONE_TIMEOUT_SECONDS)
    )
    try:
        timeout_seconds = float(raw_timeout)
    except ValueError as error:
        raise ValueError(
            "ROBOATOM_RCLONE_TIMEOUT_SECONDS must be a positive number"
        ) from error
    if timeout_seconds <= 0:
        raise ValueError("ROBOATOM_RCLONE_TIMEOUT_SECONDS must be positive")
    return timeout_seconds


def read_rclone_bytes(path: str) -> bytes:
    """Read one rclone object into memory with proxy variables removed.

    Args:
        path: Fully resolved rclone remote object.

    Returns:
        Exact object bytes.

    Raises:
        ValueError: If ``path`` is not an rclone remote.
        RuntimeError: If rclone cannot read the object.
    """
    if not is_rclone_remote(path):
        raise ValueError(f"Not an rclone path: {path}")
    try:
        result = subprocess.run(
            [_rclone_binary(), "cat", path],
            check=True,
            env=_network_environment(),
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            timeout=_rclone_timeout_seconds(),
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"Failed to read rclone object: {path}") from error
    return bytes(result.stdout)
