#!/usr/bin/env python3
import argparse
from pathlib import Path

from roboatom.annotation.app import MultiUserManifest, RuntimeConfig, ValidationError, merge_multi_user_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge per-annotator outputs into a full merged output")
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to collaboration.json",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to annotation_ui config.ini",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = MultiUserManifest.load(Path(args.manifest).resolve())
    for warning in manifest.load_warnings:
        print(f"[WARNING] {warning}")
    runtime_config = RuntimeConfig.load(Path(args.config).resolve())
    merged_path = merge_multi_user_outputs(manifest, runtime_config=runtime_config)
    print(f"Merged output written to {merged_path}")


if __name__ == "__main__":
    try:
        main()
    except ValidationError as exc:
        raise SystemExit(f"Failed to merge multi-user outputs: {exc}") from exc
    except OSError as exc:
        raise SystemExit(f"Failed to write merged output: {exc}") from exc
