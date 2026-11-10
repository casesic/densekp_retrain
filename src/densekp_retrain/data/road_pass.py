"""Model-free seeded road-pass recipes. Geometry acceptance is a separate gate.

Assets are prepared sequence manifests, not source-specific adapters. Exact
recipe replay requires the same configuration, ordered asset pool and version.
"""
import random

VERSION = "road_pass_v1"


def longest_true_run(values):
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def plan(config):
    if config.get("schema")=="road_pass_mixed_sources_v1":
        from .mixed_road_pass import plan_mixed
        return plan_mixed(config)
    if config.get('schema') == 'road_pass_multiperson_scene_owned_v1':
        from .multiperson_road_pass import plan_multiperson
        return plan_multiperson(config)
    if config.get('schema') == 'road_pass_scene_owned_v1':
        from .dynamic_preflight import plan_scene_owned
        return plan_scene_owned(config)
    if config['mode'] != 'road_pass':
        raise ValueError('Only road_pass is implemented')
    counts = config['person_counts']
    if len(counts) != config['episode_count'] or any(type(n) is not int or not 1 <= n <= 5 for n in counts):
        raise ValueError('One person_count in [1,5] per episode required')
    assets = config['asset_pool']
    if not assets or any(not a['available_combinations_verified'] for a in assets):
        raise ValueError('Verified bound combinations required')
    rng = random.Random(config['master_seed'])
    episodes = []
    for i, count in enumerate(counts):
        seed = rng.getrandbits(63); r = random.Random(seed)
        rig = dict(config['rig'], origin_m=[r.uniform(-.04,.04), r.uniform(-.03,.03),0.0], yaw_deg=config['rig'].get('road_yaw_deg',0.0)+r.uniform(-1,1))
        people=[]
        for j in range(count):
            asset=r.choice(assets)
            people.append(dict(person_id=j+1, role='target' if j==0 else 'occluder',
                asset_id=asset['asset_id'], asset_manifest=asset['manifest'], character_family=asset['character_family'],
                source_start_frame=r.choice(asset['allowed_start_frames']),
                # Parallel source-root paths; rear target, nearer occluder. No added trajectory.
                road_start_m=[2.1-j*.80, r.uniform(-.025,.025) if j==0 else .20+.65*(j-1),0.0],
                direction_yaw_deg=90.0, entry_time_tick=0,
                root_mode='source_root_relative_to_start_plus_fixed_scene_matrix',
                speed_policy='source_rate_unchanged', outfit_policy='fixed_bound_combination_entire_episode'))
        episodes.append(dict(episode_id=f'road_pass_{i:02d}', sequence_id=f'road_pass_{i:02d}',
            child_seed=seed, integration_only=True, diagnostic_split='shared_source_diagnostic' if i==0 else 'training_inputs',
            people=people, rig=rig, frame_count=31, time_base=[1,10], source_stride=3,
            source_rate_hz=30, time_mapping='time_tick=frame_index; source_frame=start+3*frame_index',
            sync_status='synthetic_same_state', scene=config['scene'], render=config['render'],
            crop_policy='deployment', crop_expansion=1.0,
            geometry_acceptance='pending', geometry_attempt_limit=8))
    return dict(algorithm_version=VERSION, config=config, episodes=episodes,
        source_independence=False, seed_is_only_generation_provenance=True,
        future_modes=['opposing_crossing','three_way_junction'])
