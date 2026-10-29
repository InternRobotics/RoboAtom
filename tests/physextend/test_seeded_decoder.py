"""Tests for the production-solver seeded cascade."""
from types import SimpleNamespace

import numpy as np
import pytest

from roboatom.physextend.decoders import seeded_decoder as decoder
from roboatom.physextend.decoders.decoder import OursDecodeResult


def test_monotone_repair_never_moves_high_confidence_anchor() -> None:
    assert decoder.project_with_anchors([170, 60], {1: 60}, 200)[1] == 60
    points = decoder.project_with_anchors([170, 60], {1: 60}, 200)
    assert 0 < points[0] < points[1] < 200


def test_obvious_anchors_survive_conflicting_sensor_path() -> None:
    actual = decoder.project_with_anchors([80, 10, 120], {0: 80, 2: 120}, 200)
    assert actual[0] == 80
    assert actual[2] == 120
    assert 80 < actual[1] < 120


def test_impossible_anchor_constraints_fail_explicitly() -> None:
    with pytest.raises(ValueError, match='Conflicting anchors'):
        decoder.project_with_anchors([10, 20], {0: 1, 1: 1}, 30)


def test_cascade_preserves_anchor_and_cuts_unanchored_boundary(monkeypatch) -> None:
    ours = OursDecodeResult((40, 100), (), 0.0, ({'anchor_applied': True}, {'anchor_applied': False}), {})
    monkeypatch.setattr(decoder, 'decode_ours', lambda *args: ours)
    monkeypatch.setattr(decoder, 'sensor_cut', lambda *args: ((20, 140), {'used_fallback': False}))
    row = {'transcript': [{'skill': 'pick'}, {'skill': 'move'}, {'skill': 'place'}]}
    result = decoder.decode_seeded(row, SimpleNamespace(length=200), None, {'cascade': True})
    assert result.boundaries == (40, 140)
    assert [item['source'] for item in result.decisions] == ['ours_anchor', 'sensor']
    assert result.segments[0]['start'] == 0 and result.segments[-1]['end'] == 200


def test_sensor_confidence_controls_only_unanchored_correction(monkeypatch) -> None:
    ours = OursDecodeResult((40, 100), (), 0.0, ({'anchor_applied': True}, {'anchor_applied': False}), {})
    monkeypatch.setattr(decoder, 'decode_ours', lambda *args: ours)
    monkeypatch.setattr(decoder, 'sensor_cut', lambda *args: ((20, 140), {'used_fallback': False, 'confidence': [0.5, 0.5]}))
    row = {'transcript': [{'skill': 'pick'}, {'skill': 'move'}, {'skill': 'place'}]}
    result = decoder.decode_seeded(row, SimpleNamespace(length=200), None, {'cascade': True, 'sensor_confidence_weight': True})
    assert result.boundaries == (40, 110)
    assert result.decisions[1]['sensor_weight'] == 0.25


def test_sensor_features_do_not_read_unselected_columns(monkeypatch) -> None:
    monkeypatch.setattr(decoder, 'feature_paths', lambda target: ('features', 'raw', 'metadata'))
    monkeypatch.setattr(decoder, 'raw_sensor_cache', lambda *args: {'state': np.arange(50).reshape(25, 2).astype(np.float32), 'annotation': np.ones((25, 2))*9999})
    result = decoder.sensor_episode({'episode_key': 'example', 'length': 50}, ['state'])
    assert result.length == 50
    assert result.state_features.shape == (10, 42)
    assert np.isfinite(result.state_features).all()
    assert result.visual_features.shape == (10, 1)
    assert not result.visual_features.any()


def test_centered_sensor_features_ignore_constant_pose_offset(monkeypatch) -> None:
    monkeypatch.setattr(decoder, 'feature_paths', lambda target: ('features', 'raw', 'metadata'))
    raw = {'state': np.arange(50).reshape(25, 2).astype(np.float32)}
    monkeypatch.setattr(decoder, 'raw_sensor_cache', lambda *args: raw)
    row = {'episode_key': 'example', 'length': 25}
    first = decoder.sensor_episode(row, ['state'], center_state=True)
    raw['state'] = raw['state']+1000
    second = decoder.sensor_episode(row, ['state'], center_state=True)
    assert np.array_equal(first.state_features, second.state_features)
    assert np.array_equal(first.state_motion_full, second.state_motion_full)


def test_strong_event_survives_template_candidate_truncation() -> None:
    from roboatom.physextend.decoders.coarse_decoder import DecoderConfig, _candidate_frames
    from roboatom.physextend.features.signals import SignalEvent
    features = SimpleNamespace(length=500, events=(SignalEvent(420, 'gripper_increase', 0.95, 'gripper'),))
    config = DecoderConfig(name='test', hypothesis='keep evidence', use_gripper_events=True, candidate_keep=8, retain_strong_events=True)
    candidates = _candidate_frames(100, features, 1, 499, config)
    assert 420 in candidates


def test_metadata_role_mapping_uses_names_not_vector_shape() -> None:
    assert decoder.metadata_gripper_indices({'names': {'motors': ['left_gripper', 'right_gripper']}}, 2) == (0, 1)
    assert decoder.metadata_gripper_indices({'names': {'motors': ['x', 'y']}}, 2) == ()
    assert decoder.metadata_gripper_indices({'field_descriptions': {'state/left_effector/position': {'dimensions': 1, 'indices': [0]}, 'state/end/position': {'dimensions': 6, 'indices': [2,3,4,5,6,7]}}}, 8) == (0,)


def test_metadata_gripper_indices_reads_nested_names_and_ee_openings() -> None:
    names = [f"joint_{index}" for index in range(30)]
    names[21] = "ee.l_g"
    names[29] = "ee.r_g"
    assert decoder.metadata_gripper_indices({"names": [names]}, 30) == (21, 29)
    assert decoder.metadata_gripper_indices({"names": [["joint_0", "ee.l_x", "dim_0"]]}, 3) == ()


@pytest.mark.parametrize('source', ['state', 'action'])
def test_gripper_event_timing_uses_selected_physical_source(monkeypatch, source) -> None:
    from roboatom.physextend.features.signals import FeatureBundle
    zeros = np.zeros(100, dtype=np.float32)
    bundle = FeatureBundle(length=100, activity=zeros, pause=zeros, turn=zeros, gripper_delta=zeros, gripper_level=zeros, visual_motion=zeros, source_columns=('state',), events=())
    measured = np.r_[np.zeros(50), np.ones(50)][:, None]
    commanded = np.r_[np.zeros(30), np.ones(70)][:, None]
    monkeypatch.setattr(decoder, 'feature_paths', lambda target: ('features', 'raw', 'metadata'))
    monkeypatch.setattr(decoder, 'raw_sensor_cache', lambda *args: {'state': measured, 'action': commanded})
    monkeypatch.setattr(decoder, 'signal_uri', lambda target: 'source')
    monkeypatch.setattr(decoder, 'sensor_schema', lambda uri: {name: {'names': ['gripper']} for name in ['state', 'action']})
    result = decoder.with_metadata_grippers(bundle, {}, event_source=source)
    expected = result.state_gripper_delta if source == 'state' else result.action_gripper_delta
    assert np.array_equal(result.gripper_delta, expected)
    assert not np.array_equal(result.state_gripper_delta, result.action_gripper_delta)


def test_zero_gripper_events_removed_without_dropping_real_transition(monkeypatch) -> None:
    from roboatom.physextend.features.signals import FeatureBundle, SignalEvent
    from roboatom.physextend.decoders.reference_decoder import SeedTrajectory
    zeros = np.zeros(100, dtype=np.float32)
    delta = zeros.copy()
    delta[40] = 0.2
    bundle = FeatureBundle(length=100, activity=zeros, pause=zeros, turn=zeros, gripper_delta=delta, gripper_level=zeros, visual_motion=zeros, source_columns=('state',), events=(SignalEvent(20, 'gripper_increase', 0.0, 'gripper'), SignalEvent(30, 'activity', 0.8, 'activity'), SignalEvent(40, 'gripper_increase', 1.0, 'gripper')))
    seed = SeedTrajectory(episode_key='seed', segments=({'start': 0, 'end': 100, 'skill': 'move'},), features=bundle)
    observed = []
    def capture(transcript, features, seeds):
        observed.extend(event.frame for event in features.events)
        return OursDecodeResult((), (), 0.0, (), {})
    monkeypatch.setattr(decoder, 'decode_ours', capture)
    decoder.decode_seeded({'transcript': [{'skill': 'move'}]}, bundle, seed, {'drop_zero_gripper_events': True})
    assert observed == [30, 40]


@pytest.mark.parametrize('allow_obvious,quality,expected', [(False, 1.0, False), (True, 1.0, True), (True, 0.2, False)])
def test_obvious_ambiguity_exception_requires_real_quality(allow_obvious, quality, expected) -> None:
    from roboatom.physextend.decoders.anchor_decoder import EventCluster, AnchorDecoderConfig, _AnchorTemplate, _match_target_template
    template = _AnchorTemplate(segment_index=0, skill='pick', kind='gripper_decrease', event_fraction=0.5, confidence=1.0, seed_event_frame=100, has_seed_retry=False)
    clusters = [EventCluster(start=frame-2, end=frame+2, frame=frame, kind='gripper_decrease', quality=quality, persistence=1.0, purity=1.0, after_level=0.0, event_count=1) for frame in [95, 105]]
    config = AnchorDecoderConfig(name='test', hypothesis='obvious', anchor_weight=0.75, uniqueness_margin=0.12, allow_obvious_ambiguous=allow_obvious)
    match, detail = _match_target_template(template, 100.0, 1, 199, clusters, config, 100)
    assert (match is not None) == expected


def test_conflicting_semantic_proposals_enter_global_path() -> None:
    from roboatom.physextend.decoders.anchor_decoder import EventCluster, _AnchorTemplate, _TargetMatch, _boundary_proposals
    def match(index, frame):
        template = _AnchorTemplate(index, 'pick', 'gripper_decrease', 0.5, 1.0, 50, False)
        event = EventCluster(frame, frame, frame, 'gripper_decrease', 1.0, 1.0, 1.0, 0.0, 1)
        return _TargetMatch(template, event, event, 1.0, 1, False)
    baseline = SimpleNamespace(boundaries=(100, 200))
    matches = [match(0, 20), match(1, 170), match(2, 220)]
    proposals, diagnostics = _boundary_proposals(matches, baseline, 3, 300, True)
    assert set(proposals) == {0, 1}
    assert 0 < proposals[0] < proposals[1] < 300
    assert len(diagnostics[0]) == 2


def test_seed_activity_cost_uses_energy_not_time_fraction() -> None:
    from roboatom.physextend.features.signals import FeatureBundle
    from roboatom.physextend.decoders.coarse_decoder import DecoderConfig, decode_trajectory
    zeros = np.zeros(100, dtype=np.float32)
    bundle = FeatureBundle(length=100, activity=np.r_[np.zeros(50), np.ones(50)], pause=zeros, turn=zeros, gripper_delta=zeros, gripper_level=zeros, visual_motion=zeros, source_columns=('state',), events=())
    segments = ({'start': 0, 'end': 50, 'skill': 'move'}, {'start': 50, 'end': 100, 'skill': 'place'})
    result = decode_trajectory(segments, bundle, DecoderConfig(name='test', hypothesis='unit-consistent activity', seed_activity_progress=True), segments, bundle)
    assert result.boundaries == (50,)
    assert result.decisions[0]['activity_cost'] == 0.0


def test_other_arm_does_not_hide_clear_local_gripper_change() -> None:
    from dataclasses import replace
    from roboatom.physextend.features.signals import FeatureBundle, SignalEvent
    from roboatom.physextend.decoders.anchor_decoder import _cluster_from_events
    zeros = np.zeros(100, dtype=np.float32)
    event = SignalEvent(50, 'gripper_increase', 1.0, 'gripper')
    bundle = FeatureBundle(length=100, activity=zeros, pause=zeros, turn=zeros, gripper_delta=zeros, gripper_level=np.ones(100), visual_motion=zeros, source_columns=('state',), events=(event,))
    hidden = _cluster_from_events((event,), bundle, 6)
    tracks = np.stack([np.r_[np.zeros(50), np.ones(50)], np.ones(100)], axis=1)
    recovered = _cluster_from_events((event,), replace(bundle, embodiment={'gripper_tracks': tracks}), 6)
    assert recovered.persistence > 0.9
    assert recovered.quality > hidden.quality


def test_seed_arm_identity_excludes_opposite_arm_event() -> None:
    from roboatom.physextend.decoders.anchor_decoder import EventCluster, AnchorDecoderConfig, _AnchorTemplate, _match_target_template
    template = _AnchorTemplate(0, 'pick', 'gripper_decrease', 0.5, 1.0, 50, False, 'state:left')
    clusters = [EventCluster(frame, frame, frame, 'gripper_decrease', 1.0, 1.0, 1.0, 0.0, 1, channel) for frame, channel in [(95, 'state:left'), (100, 'state:right')]]
    config = AnchorDecoderConfig(name='test', hypothesis='same-arm correspondence', anchor_weight=.75, match_gripper_channel=True)
    match, _ = _match_target_template(template, 100, 0, 199, clusters, config, 100)
    assert match.selected.frame == 95


def test_seed_event_warp_uses_ordered_correspondences_and_seed_boundaries(monkeypatch) -> None:
    from roboatom.physextend.decoders import anchor_decoder as anchor
    from roboatom.physextend.decoders.coarse_decoder import DecodeResult
    segments = ({'start': 0, 'end': 60, 'skill': 'pick'}, {'start': 60, 'end': 150, 'skill': 'place'}, {'start': 150, 'end': 200, 'skill': 'move'})
    seed = SimpleNamespace(features=SimpleNamespace(length=200), segments=segments, episode_key='seed')
    baseline = DecodeResult((100, 200), (), 0., ({}, {}))
    matches = []
    for index, seed_frame, target_frame in [(0, 20, 30), (1, 100, 240)]:
        template = anchor._AnchorTemplate(index, segments[index]['skill'], 'gripper_increase', .5, 1., seed_frame, False)
        event = anchor.EventCluster(target_frame, target_frame, target_frame, 'gripper_increase', 1., 1., 1., 1., 1)
        matches.append(anchor._TargetMatch(template, event, event, 1., 1, False))
    monkeypatch.setattr(anchor, 'decode_trajectory', lambda *a, **kw: baseline)
    monkeypatch.setattr(anchor, '_seed_templates', lambda *a: ([m.template for m in matches], []))
    monkeypatch.setattr(anchor, '_target_matches', lambda *a: (matches, []))
    result = anchor.decode_anchor(segments, SimpleNamespace(length=300), seed, anchor.AnchorDecoderConfig(name='test', hypothesis='event warp', anchor_weight=1., seed_event_warp=True))
    assert result.boundaries == (135, 270)
    assert result.diagnostics['event_seed_warp_applied']
