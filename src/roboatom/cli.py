"""Unified public RoboAtom command line."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any
import yaml


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('Configuration must be a mapping')
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='roboatom')
    sub = parser.add_subparsers(dest='command', required=True)
    annotate = sub.add_parser('annotate', help='First annotation and human/model-assisted review UI')
    group = annotate.add_mutually_exclusive_group(required=True)
    group.add_argument('--input', type=Path)
    group.add_argument('--manifest', type=Path)
    annotate.add_argument('--output', type=Path)
    annotate.add_argument('--media-root', type=Path, required=True)
    annotate.add_argument('--config', type=Path, required=True)
    annotate.add_argument('--mode', choices=['annotation', 'model-review'], default='annotation')
    annotate.add_argument('--review', type=Path)
    annotate.add_argument('--host', default='127.0.0.1')
    annotate.add_argument('--port', type=int, default=7860)
    annotate.add_argument('--prepare', action='store_true', help='Prepare multi-user seed files and exit')
    annotate.add_argument('--merge', action='store_true', help='Merge multi-user outputs and exit')
    review = sub.add_parser('review', help='Generate offline model review suggestions')
    for flag in ['input', 'output', 'config', 'media-root']:
        review.add_argument('--'+flag, type=Path, required=True)
    review.add_argument('--prompt-file', type=Path)
    extend = sub.add_parser('extend', help='PhysExtend same-task boundary extension')
    for flag in ['input', 'references', 'data-root', 'output']:
        extend.add_argument('--'+flag, type=Path, required=True)
    extend.add_argument('--route', choices=['single','multi'], default='multi')
    evaluate = sub.add_parser('evaluate', help='Score saved predictions; does not perform inference')
    for flag in ['gt','prediction','protocol','output-dir']:
        evaluate.add_argument('--'+flag, type=Path, required=True)
    evaluate.add_argument('--name', default='prediction')
    evaluate.add_argument('--benchmark-input', type=Path, help='Frozen input manifest supplying GT FPS and frame counts')
    validate = sub.add_parser('validate', help='Validate records or frozen benchmark')
    vg = validate.add_mutually_exclusive_group(required=True)
    vg.add_argument('--input', type=Path)
    vg.add_argument('--benchmark-root', type=Path)
    validate.add_argument('--media-root', type=Path)
    validate.add_argument('--prediction', type=Path)
    validate.add_argument('--expected-count', type=int)
    serve_parser = sub.add_parser('serve', help='Run the RoboAtom-Anno inference service')
    serve_parser.add_argument('--model', type=Path, required=True)
    serve_parser.add_argument('--media-root', type=Path, required=True)
    serve_parser.add_argument('--host', default='127.0.0.1')
    serve_parser.add_argument('--port', type=int, default=3000)
    infer_parser = sub.add_parser('infer', help='Generate predictions through the inference service')
    infer_parser.add_argument('--input', type=Path, required=True)
    infer_parser.add_argument('--output', type=Path, required=True)
    infer_parser.add_argument('--base-url', default='http://127.0.0.1:3000')
    args = parser.parse_args(argv)
    try:
        if args.command == 'serve':
            from .inference.server import serve
            serve(args.model.resolve(), args.media_root.resolve(), args.host, args.port)
        elif args.command == 'infer':
            from .inference.client import infer
            infer(args.input.resolve(), args.output.resolve(), args.base_url)
        elif args.command == 'annotate':
            from .annotation import app
            os.environ['ROBOATOM_MEDIA_ROOT'] = str(args.media_root.resolve())
            os.environ['ROBOATOM_ANNOTATION_CONFIG'] = str(args.config.resolve())
            if args.prepare or args.merge:
                if not args.manifest:
                    raise ValueError('--manifest is required for --prepare/--merge')
                manifest = app.MultiUserManifest.load(args.manifest.resolve())
                if args.prepare:
                    app.prepare_multi_user_seed_files(manifest, force=False)
                else:
                    print(app.merge_multi_user_outputs(manifest, runtime_config=app.RuntimeConfig.load(args.config.resolve())))
                return 0
            if args.input and not args.output:
                raise ValueError('--output is required with --input')
            old_argv = sys.argv
            flags = ['--config', str(args.config.resolve()), '--host', args.host, '--port', str(args.port), '--mode', args.mode, '--no-debug']
            if args.manifest:
                flags += ['--multi-user-manifest', str(args.manifest.resolve())]
            else:
                flags += ['--input-json', str(args.input.resolve()), '--output-json', str(args.output.resolve())]
            if args.review:
                flags += ['--model-review-json', str(args.review.resolve())]
            try:
                sys.argv = ['roboatom annotate', *flags]
                app.main()
            finally:
                sys.argv = old_argv
        elif args.command == 'review':
            from .annotation import review
            os.environ['ROBOATOM_MEDIA_ROOT'] = str(args.media_root.resolve())
            config = load_config(args.config)
            annotation_config = (args.config.resolve().parent / config['annotation_config']).resolve()
            output = args.output.resolve()
            flags = ['--input-jsonl', str(args.input.resolve()), '--output-jsonl', str(output),
                     '--config', str(annotation_config), '--base-url', config['base_url'], '--model', config['model'],
                     '--assets-dir', str(output.parent / (output.stem+'_assets')),
                     '--summary-json', str(output.with_suffix('.summary.json')), '--no-prompt-log']
            if args.prompt_file:
                flags += ['--prompt-file', str(args.prompt_file.resolve())]
            old_argv = sys.argv
            try:
                sys.argv = ['roboatom review', *flags]
                review.main()
            finally:
                sys.argv = old_argv
        elif args.command == 'extend':
            from .physextend.extend import run
            run(args)
        elif args.command == 'evaluate':
            from .validation.data import read_rows, check_evaluator_coverage
            from .evaluation import action
            protocol = load_config(args.protocol)
            output = args.output_dir.resolve()
            output.mkdir(parents=True, exist_ok=True)
            gold_path = args.gt.resolve()
            if args.benchmark_input:
                from .validation.data import canonical_row, write_rows
                metadata = {canonical_row(row)['episode_id']: row for row in read_rows(args.benchmark_input)}
                enriched = []
                for row in read_rows(args.gt):
                    key = canonical_row(row)['episode_id']
                    if key not in metadata:
                        raise ValueError(f'GT episode absent from benchmark input: {key}')
                    enriched.append({**metadata[key], **row})
                gold_path = output / 'evaluation_gt.jsonl'
                write_rows(gold_path, enriched)
            gold_index, samples, _ = action.base_evaluator.read_gt_input(gold_path)
            pred_index, parse_errors, _ = action.base_evaluator.read_prediction_input(path=args.prediction, llamafactory_samples=samples)
            if protocol.get('strict_coverage', True):
                if parse_errors:
                    raise ValueError(f'Malformed model outputs: {parse_errors}')
                check_evaluator_coverage(gold_index, pred_index, protocol.get('expected_count'))
            flags = ['--gt', str(gold_path), '--prediction', args.name+'='+str(args.prediction.resolve()),
                     '--evaluation-window', protocol['evaluation_window'],
                     '--f1-overlaps', ','.join(map(str,protocol['f1_overlaps'])),
                     '--skill-f1-overlap', str(protocol['skill_f1_overlap']),
                     '--skill-match-tiou', str(protocol['skill_match_tiou']),
                     '--output-json', str(output/'metrics.json'), '--output-jsonl', str(output/'episodes.jsonl'),
                     '--output-tsv', str(output/'summary.tsv')]
            for label in protocol.get('background_labels', []):
                flags += ['--background-label', label]
            if protocol.get('skip_skill_metrics', False):
                flags.append('--skip-skill-metrics')
            status = action.main(flags)
            if status:
                return status
            report = json.loads((output/'metrics.json').read_text())
            report['release_protocol'] = protocol
            (output/'metrics.json').write_text(json.dumps(report, indent=2)+'\n')
        elif args.command == 'validate':
            from .validation.data import read_rows, validate_rows, check_coverage
            if args.benchmark_root:
                from .validation.benchmark import validate_benchmark
                print(json.dumps(validate_benchmark(args.benchmark_root, args.media_root), indent=2))
            elif args.prediction:
                print(json.dumps(check_coverage(read_rows(args.input), read_rows(args.prediction), args.expected_count)))
            else:
                ids = validate_rows(read_rows(args.input))
                if args.expected_count is not None and len(ids) != args.expected_count:
                    raise ValueError('Unexpected episode count')
                print(json.dumps({'valid_episodes': len(ids)}))
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(2, f'error: {error}\n')
    return 0
