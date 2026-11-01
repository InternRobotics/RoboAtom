"""Configured local and remote sensor source aliases."""

from __future__ import annotations

from pathlib import Path

from roboatom.physextend.storage_config import load_storage_config


_STORAGE_CONFIG = load_storage_config()
ROOT_URIS = _STORAGE_CONFIG["remote_roots"]
LOCAL_ROOTS = {dataset: Path(value) for dataset, value in _STORAGE_CONFIG["local_roots"].items()}
URI_MIGRATIONS = _STORAGE_CONFIG["uri_migrations"]


def _join_uri(root: str, relative: str) -> str:
    if not relative.strip("/"):
        return root.rstrip("/")
    return f"{root.rstrip('/')}/{relative.lstrip('/')}"


def _rewrite_current_uri(source_dataset: str, uri: str) -> str:
    """Normalize explicitly configured source aliases without private defaults."""
    value = uri.strip()
    for old, new in URI_MIGRATIONS.items():
        if value.startswith(old):
            value = new + value[len(old):]
            break
    if value.startswith("cluster_hoss:s3://"):
        return "hoss:" + value.removeprefix("cluster_hoss:s3://")
    if value.startswith("hoss:s3://"):
        return "hoss:" + value.removeprefix("hoss:s3://")
    root_uri = ROOT_URIS.get(source_dataset)
    local_root = LOCAL_ROOTS.get(source_dataset)
    if root_uri is not None and local_root is not None:
        local_text = str(local_root)
        if value == local_text or value.startswith(local_text + "/"):
            return _join_uri(root_uri, value.removeprefix(local_text))
    return value
