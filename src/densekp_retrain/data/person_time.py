"""Exact synthetic clock to per-person original observations (no interpolation)."""
from fractions import Fraction

SCHEMA='per_person_nearest_v1'
POLICY='nearest_original_record_ties_earlier'


def rational(value):
    if not isinstance(value,(list,tuple)) or len(value)!=2 or any(type(v)is not int for v in value) or value[1]<=0:
        raise ValueError('invalid_rational_time')
    return Fraction(*value)


def pair(value):return [value.numerator,value.denominator]


def map_timeline(source,episode,*,origin_record):
    if type(origin_record)is not int or not 0<=origin_record<len(source):raise ValueError('source_origin_out_of_range')
    times=[];previous=None;asset_frames=set()
    for row in source:
        if any(type(row.get(k))is not int for k in ['asset_frame','source_frame','source_time_tick']):raise ValueError('missing_integer_source_record')
        base=rational(row['time_base'])
        if base<=0 or row['asset_frame']<0 or row['asset_frame'] in asset_frames:raise ValueError('invalid_source_frame_or_base')
        value=row['source_time_tick']*base
        if previous is not None and value<=previous:raise ValueError('nonmonotonic_source_time')
        asset_frames.add(row['asset_frame']);times.append(value);previous=value
    origin=times[origin_record];output=[];previous=None;last_time=None
    for k,clock in enumerate(episode):
        if clock['frame_index']!=k or type(clock['episode_time_tick'])is not int:raise ValueError('invalid_episode_clock')
        elapsed=clock['episode_time_tick']*rational(clock['time_base'])
        if elapsed<0 or (last_time is not None and elapsed<=last_time):raise ValueError('nonmonotonic_episode_clock')
        desired=origin+elapsed
        if not times[0]<=desired<=times[-1]:raise ValueError('desired_time_outside_source')
        i=min(range(len(source)),key=lambda j:(abs(times[j]-desired),times[j]))
        row=source[i];error=times[i]-desired
        if previous is not None and (row['asset_frame']<=output[-1]['asset_frame'] or row['source_frame']<=output[-1]['source_frame'] or i<=previous):raise ValueError('source_record_repeated_or_reversed')
        # The adjacent interval on the desired side defines this Voronoi cell.
        neighbor=i-1 if error>0 else i+1
        if error==0:neighbor=i+1 if i+1<len(times) else i-1
        if not 0<=neighbor<len(times):raise ValueError('quantization_neighbor_missing')
        interval=abs(times[neighbor]-times[i])
        if abs(error)>interval/2:raise ValueError('quantization_error_exceeds_half_local_interval')
        output.append(dict(clock,asset_frame=row['asset_frame'],source_frame=row['source_frame'],source_time_tick=row['source_time_tick'],
            source_time_base=row['time_base'],source_record_index=i,source_origin_record=origin_record,
            source_time_origin_tick=source[origin_record]['source_time_tick'],source_time_origin_base=source[origin_record]['time_base'],
            mapping_policy=POLICY,quantization_error_seconds=pair(error),local_interval_seconds=pair(interval)))
        previous=i;last_time=elapsed
    return output


def validate_record(row,clock,person):
    if any(row[k]!=clock[k] for k in ['frame_index','episode_time_tick','time_base']):raise ValueError('person_episode_time_conflict')
    k=clock['frame_index']
    if any(type(row.get(key))is not int for key in ['asset_frame','source_frame','source_time_tick','source_time_origin_tick','source_record_index','source_origin_record']):raise ValueError('missing_person_source_time')
    if row['mapping_policy']!=POLICY:raise ValueError('unknown_time_mapping_policy')
    if person['asset_frames'][k]!=row['asset_frame'] or person['source_frames'][k]!=row['source_frame']:raise ValueError('person_selected_record_conflict')
    desired=row['source_time_origin_tick']*rational(row['source_time_origin_base'])+clock['episode_time_tick']*rational(clock['time_base'])
    error=row['source_time_tick']*rational(row['source_time_base'])-desired
    interval=rational(row['local_interval_seconds'])
    if error!=rational(row['quantization_error_seconds']) or interval<=0 or abs(error)>interval/2:raise ValueError('person_quantization_conflict')
    return row
