"""Optional user-owned storage aliases for portable PhysExtend deployments."""
from __future__ import annotations

import json
import os
from pathlib import Path

SECTIONS = ("remote_roots", "local_roots", "uri_migrations")

def load_storage_config(path: Path | None = None) -> dict[str, dict[str, str]]:
    """Load storage mappings without assuming any private deployment roots.

    Args:
        path: JSON configuration; defaults to ROBOATOM_STORAGE_CONFIG.

    Returns:
        Remote roots, local roots, and URI migrations. Missing sections are empty.
        Relative local roots resolve against the configuration directory.

    Raises:
        ValueError: If the configuration contains unknown sections or non-string mappings.
    """
    configured = os.environ.get("ROBOATOM_STORAGE_CONFIG")
    if path is None and not configured:
        return {section: {} for section in SECTIONS}
    source = Path(path or configured or "").expanduser().resolve()
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) - set(SECTIONS):
        raise ValueError(
            "Storage configuration must contain only remote_roots, "
            "local_roots, and uri_migrations"
        )
    result: dict[str, dict[str, str]] = {}
    for section in SECTIONS:
        mapping = data.get(section, {})
        if not isinstance(mapping, dict) or any(
            not isinstance(key, str) or not isinstance(value, str) or not key or not value
            for key, value in mapping.items()
        ):
            raise ValueError(f"{section} must map nonempty strings to nonempty strings")
        result[section] = dict(mapping)
    for dataset, value in result["local_roots"].items():
        local = Path(value).expanduser()
        result["local_roots"][dataset] = str(
            (local if local.is_absolute() else source.parent / local).resolve()
        )
    return result
