#!/usr/bin/env python3
import argparse
from pathlib import Path

from roboatom.annotation.app import MultiUserManifest, ValidationError, prepare_multi_user_seed_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare per-annotator seed inputs for multi-user mode")
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to collaboration.json",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing seed_input_json files instead of skipping them",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = MultiUserManifest.load(Path(args.manifest).resolve())
    for warning in manifest.load_warnings:
        print(f"[WARNING] {warning}")
    written_paths = prepare_multi_user_seed_files(manifest, force=bool(args.force))
    print(f"Wrote {len(written_paths)} seed input files from {manifest.master_input_path}")
    for path in written_paths:
        print(path)
    if not args.force:
        skipped_paths = [
            annotator.seed_input_path
            for annotator in manifest.annotators
            if not annotator.is_reviewer
            and annotator.seed_input_path.exists()
            and annotator.seed_input_path not in written_paths
        ]
        for path in skipped_paths:
            print(f"[SKIPPED existing seed] {path}")


if __name__ == "__main__":
    try:
        main()
    except ValidationError as exc:
        raise SystemExit(f"Failed to prepare multi-user seed files: {exc}") from exc
    except OSError as exc:
        raise SystemExit(f"Failed to write seed files: {exc}") from exc
