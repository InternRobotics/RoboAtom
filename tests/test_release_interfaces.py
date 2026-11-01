"""Integration checks for portable release inputs and strict scoring."""
import json
from pathlib import Path
import pytest
from roboatom.validation.data import check_coverage, canonical_row
from roboatom.cli import main
from roboatom.annotation import app
from roboatom.physextend.selection import select_references, _validate_reference_coverage
from roboatom.physextend.extend import _source_row

ROOT = Path(__file__).resolve().parents[1]

def row(identifier='a'):
    return {'episode_id': identifier, 'num_frames': 10, 'fps': 5,
            'frame_interval': 'closed', 'segments': [{'start_frame': 0, 'end_frame': 9, 'skill': 'Pick', 'subtask': 'Pick object.'}]}

def test_duplicate_missing_extra_and_malformed_are_rejected():
    for predictions in ([row(), row()], [row('b')], [{**row(), 'segments': []}]):
        with pytest.raises(ValueError):
            check_coverage([row()], predictions)
    with pytest.raises(ValueError):
        check_coverage([row()], [], expected=500)

def test_closed_half_open_equivalence():
    closed = row()
    half = {**closed, 'frame_interval': 'half_open', 'segments': [{**closed['segments'][0], 'end_frame': 10}]}
    assert canonical_row(closed)['segments'] == canonical_row(half)['segments']

def test_public_annotation_roundtrip_and_media_root(tmp_path, monkeypatch):
    monkeypatch.setenv('ROBOATOM_MEDIA_ROOT', str(ROOT / 'examples'))
    value = {**row(), 'task_id': 'pick_object', 'views': {'front': 'demo.mp4'}}
    path = tmp_path / 'input.jsonl'
    path.write_text(json.dumps(value)+'\n')
    config = app.RuntimeConfig.load(ROOT / 'configs/annotation.ini')
    store = app.AnnotationStore.load(input_path=path, output_path=tmp_path/'output.jsonl', runtime_config=config)
    client = app.create_app(store=store).test_client()
    response = client.post('/api/save')
    assert response.status_code == 200
    saved = json.loads((tmp_path/'output.jsonl').read_text())
    assert saved['segments'] == value['segments']
    assert saved['num_frames'] == 10
    assert saved['views']['front'] == 'demo.mp4'
    assert 'action_config' not in saved
    assert store.resolve_video_source(0)[2] == ROOT / 'examples/demo.mp4'

def test_cli_rejects_incomplete_formal_benchmark(tmp_path, capsys):
    gt = tmp_path / 'gt.jsonl'
    gt.write_text(json.dumps(row())+'\n')
    prediction = tmp_path/'prediction.jsonl'
    prediction.write_text(json.dumps(row())+'\n')
    with pytest.raises(SystemExit) as exc:
        main(['evaluate','--gt',str(gt),
              '--prediction',str(prediction),'--protocol',str(ROOT/'configs/roboatom_bench.yaml'),
              '--output-dir',str(tmp_path/'evaluation')])
    assert exc.value.code == 2
    assert not (tmp_path/'evaluation/metrics.json').exists()
    assert 'Expected 500 GT episodes; got 1' in capsys.readouterr().err

def test_reference_and_target_disjoint(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(['extend','--references',str(ROOT/'examples/references.jsonl'),
              '--input',str(ROOT/'examples/references.jsonl'), '--data-root',str(ROOT/'examples'),
              '--output',str(tmp_path/'prediction.jsonl')])
    assert exc.value.code == 2

def test_public_extend_uses_final_recipe(tmp_path):
    output = tmp_path / 'prediction.jsonl'
    assert main(['extend', '--references', str(ROOT/'examples/references.jsonl'),
                 '--input', str(ROOT/'examples/trajectories.jsonl'),
                 '--data-root', str(ROOT/'examples'), '--output', str(output)]) == 0
    prediction = json.loads(output.read_text())
    diagnostic = json.loads(output.with_suffix('.diagnostics.jsonl').read_text())
    assert diagnostic['recipe'] == 'agilex_short_segment_guard'
    assert 'short_segment_guard' in diagnostic['evaluated_steps']
    assert prediction['segments'][0]['start_frame'] == 0
    assert prediction['segments'][-1]['end_frame'] == prediction['num_frames'] - 1


def test_repeated_extend_reloads_metadata_without_patching_modules(tmp_path: Path) -> None:
    from roboatom.physextend.features.extraction import feature_paths
    from roboatom.physextend.refine import gripper, methods
    from roboatom.physextend.runtime import current_run

    target_path = tmp_path / 'target.jsonl'
    output = tmp_path / 'prediction.jsonl'
    target = json.loads((ROOT / 'examples/trajectories.jsonl').read_text())
    original_helpers = (gripper.load_channels, gripper.alignment, methods.reference_predictions)
    flags = [
        'extend', '--references', str(ROOT / 'examples/references.jsonl'),
        '--input', str(target_path), '--data-root', str(ROOT / 'examples'),
        '--output', str(output),
    ]

    for robot, route in [('franka', 'multi'), ('ur', 'single')]:
        target_path.write_text(json.dumps({**target, 'embodiment': robot}) + '\n', encoding='utf-8')
        assert main([*flags, '--route', route]) == 0
        workdir = tmp_path / 'prediction_physextend_work'
        profiles = json.loads((workdir / 'audit/embodiments.json').read_text())
        target_profile = next(value for value in profiles.values() if value['benchmark_id'] == target['episode_id'])
        assert target_profile['robot_type'] == robot
        assert (gripper.load_channels, gripper.alignment, methods.reference_predictions) == original_helpers
        from roboatom.physextend.features import extraction
        assert extraction.feature_paths is feature_paths
        with pytest.raises(RuntimeError, match='No active'):
            current_run()

    assert not (tmp_path / 'experiments').exists()

def test_source_adapter_preserves_remote_sensor_uri(tmp_path):
    source = {**row('remote'), 'task_id': 'move', 'signals_path': 'hoss:bucket/path.parquet'}
    adapted = _source_row(canonical_row(source), tmp_path, True)
    assert adapted['data_uri'] == source['signals_path']
    assert adapted['segments'][0]['end'] == source['num_frames']

def test_extend_rejects_ambiguous_reference_sequences_and_meanings():
    target = {**row('target'), 'task_id': 'move'}
    first = {**row('seed-1'), 'task_id': 'move'}
    different_skill = {**row('seed-2'), 'task_id': 'move'}
    different_skill['segments'] = [{**different_skill['segments'][0], 'skill': 'Place'}]
    with pytest.raises(ValueError, match='Ambiguous same-task references'):
        select_references(target, [first, different_skill])
    different_text = {**row('seed-3'), 'task_id': 'move'}
    different_text['segments'] = [{**different_text['segments'][0], 'subtask': 'Pick a second object.'}]
    with pytest.raises(ValueError, match='Ambiguous same-task references'):
        select_references(target, [first, different_text])
    assert [seed['episode_id'] for seed in select_references(target, [first])] == ['seed-1']

def test_extend_only_uses_explicit_reviewed_variant():
    target = {**row('target'), 'task_id': 'move', 'task_variant': 'one_object'}
    first = {**row('seed-1'), 'task_id': 'move', 'task_variant': 'one_object'}
    second = {**row('seed-2'), 'task_id': 'move', 'task_variant': 'two_objects'}
    second['segments'] = [{**second['segments'][0], 'skill': 'Place'}]
    assert select_references(target, [first, second]) == [first]
    with pytest.raises(ValueError, match='needs task_variant'):
        select_references({key: value for key, value in target.items() if key != 'task_variant'}, [first, second])
    with pytest.raises(ValueError, match='No reference for task_variant'):
        select_references({**target, 'task_variant': 'other'}, [first, second])

def test_extend_reference_must_cover_all_frames():
    complete = canonical_row(row('seed'))
    _validate_reference_coverage(complete)
    for start, end in [(1, 9), (0, 8)]:
        incomplete = {**complete, 'segments': [{**complete['segments'][0], 'start_frame': start, 'end_frame': end}]}
        with pytest.raises(ValueError, match='unannotated frame gap|final frame'):
            _validate_reference_coverage(incomplete)

def test_seconds_predictions_and_malformed_segments():
    from roboatom.validation.data import check_evaluator_coverage
    gold = {'a': {**row(), 'segments': [{'start_sec': 0, 'end_sec': 2, 'skill': 'Pick', 'subtask': 'Pick object.'}]}}
    assert check_evaluator_coverage(gold, gold)['coverage'] == 1
    invalid = {'a': {**row(), 'segments': [{'start_sec': 0, 'end_sec': 0, 'skill': 'Pick'}]}}
    with pytest.raises(ValueError):
        check_evaluator_coverage(gold, invalid)

def test_review_requires_environment_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    from roboatom.annotation import review

    for name in ('NEWAPI_API_KEY', 'DASHSCOPE_API_KEY', 'VLLM_API_KEY', 'OPENAI_API_KEY'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('OPENAI_API_KEY','test_key_for_mock_only')
    assert review.resolve_api_key('') == 'test_key_for_mock_only'
    with pytest.raises(ValueError):
        review.resolve_api_key('test_key_for_mock_only')

def test_model_review_overlay_and_edit_save(tmp_path, monkeypatch):
    monkeypatch.setenv('ROBOATOM_MEDIA_ROOT', str(ROOT/'examples'))
    config = app.RuntimeConfig.load(ROOT/'configs/annotation.ini')
    store = app.AnnotationStore.load(input_path=ROOT/'examples/annotation.jsonl', output_path=tmp_path/'reviewed.jsonl', runtime_config=config)
    path=tmp_path/'review.jsonl'
    path.write_text(json.dumps({'line_no':1,'sample_id':'target','verdict':'incorrect','issues':[{'segment_index':0,'severity':'incorrect','reason':'Check the boundary.','suggested_fix':'Review with the original video.'}]})+'\n')
    overlay=app.ModelReviewStore.load(path)
    client=app.create_app(store=store,model_review_store=overlay,app_mode='model-review').test_client()
    assert client.get('/').status_code == 200
    payload=client.get('/api/video/0').get_json()
    assert payload['model_review'] is not None
    assert len(payload['views']) == 2
    assert client.post('/api/save').status_code == 200
    reloaded=app.AnnotationStore.load(input_path=tmp_path/'reviewed.jsonl', output_path=tmp_path/'second.jsonl', runtime_config=config)
    assert reloaded.session_data[0]['action_config'] == store.session_data[0]['action_config']
