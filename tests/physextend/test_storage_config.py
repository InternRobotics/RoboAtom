"""Storage aliases are optional, explicit, and independent of the checkout."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from roboatom.physextend.storage_config import load_storage_config


def test_missing_config_uses_no_private_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ROBOATOM_STORAGE_CONFIG", raising=False)

    assert load_storage_config() == {
        "remote_roots": {}, "local_roots": {}, "uri_migrations": {}
    }


def test_relative_roots_resolve_from_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "storage.json"
    config.write_text(json.dumps({
        "remote_roots": {"custom": "remote:bucket/dataset"},
        "local_roots": {"custom": "data"},
        "uri_migrations": {"legacy:": "remote:"},
    }))
    monkeypatch.setenv("ROBOATOM_STORAGE_CONFIG", str(config))

    mappings = load_storage_config()

    assert mappings["local_roots"]["custom"] == str(tmp_path / "data")
    assert mappings["remote_roots"]["custom"] == "remote:bucket/dataset"
    assert mappings["uri_migrations"] == {"legacy:": "remote:"}


@pytest.mark.parametrize("payload", [
    [], {"secret": {}}, {"remote_roots": []}, {"local_roots": {"custom": 123}},
    {"uri_migrations": {"": "remote:"}},
])
def test_invalid_config_fails_early(tmp_path: Path, payload: object) -> None:
    config = tmp_path / "storage.json"
    config.write_text(json.dumps(payload))

    with pytest.raises(ValueError):
        load_storage_config(config)


def test_uri_alias_only_rewrites_configured_roots(monkeypatch: pytest.MonkeyPatch) -> None:
    from roboatom.physextend.features import sources

    monkeypatch.setattr(sources, "ROOT_URIS", {"custom": "remote:bucket/data"})
    monkeypatch.setattr(sources, "LOCAL_ROOTS", {"custom": Path("/data/custom")})
    monkeypatch.setattr(sources, "URI_MIGRATIONS", {"old:": "remote:"})

    assert sources._rewrite_current_uri("custom", "/data/custom/episode.parquet") == (
        "remote:bucket/data/episode.parquet"
    )
    assert sources._rewrite_current_uri("custom", "/data/custom_other/x") == (
        "/data/custom_other/x"
    )
    assert sources._rewrite_current_uri("custom", "old:bucket/x") == "remote:bucket/x"
    assert sources._rewrite_current_uri("unknown", "/data/unknown/x") == "/data/unknown/x"
