"""
PathSnap — Feature Engineering Pipeline
src/features.py

Converts raw GPS track + road network into ML-ready feature rows.
Each row = one (GPS fix, candidate arc) pair.
"""

import numpy as np
import pandas as pd
import math


# ─────────────────────────────────────────────
#  Geometry helpers
# ─────────────────────────────────────────────

def haversine(lon1, lat1, lon2, lat2):
    """Return distance in metres between two (lon, lat) points."""
    R = 6_371_000  # Earth radius in metres
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def bearing(lon1, lat1, lon2, lat2):
    """Compass bearing in degrees [0, 360) from point 1 → point 2."""
    lat1, lat2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def point_to_segment_distance(px, py, ax, ay, bx, by):
    """
    Perpendicular distance (in degrees, converted later) from point P
    to line segment A→B.  Returns (distance_deg, t) where t ∈ [0,1].
    """
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay), 0.0
    t = max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    nearest_x = ax + t * dx
    nearest_y = ay + t * dy
    return math.hypot(px - nearest_x, py - nearest_y), t


def dist_metres(px, py, ax, ay, bx, by):
    """Perpendicular distance in metres from GPS point to arc segment."""
    # Midpoint of arc to convert degree-distance to metres
    mid_lat = (ay + by) / 2
    deg_per_metre_lat = 1 / 111_111
    deg_per_metre_lon = 1 / (111_111 * math.cos(math.radians(mid_lat)))

    # Normalise to metres
    px_m = px / deg_per_metre_lon
    py_m = py / deg_per_metre_lat
    ax_m = ax / deg_per_metre_lon
    ay_m = ay / deg_per_metre_lat
    bx_m = bx / deg_per_metre_lon
    by_m = by / deg_per_metre_lat
    dist, _ = point_to_segment_distance(px_m, py_m, ax_m, ay_m, bx_m, by_m)
    return dist


# ─────────────────────────────────────────────
#  Data loaders
# ─────────────────────────────────────────────

def load_track(path):
    """Load .track file → DataFrame[longitude, latitude, timestamp_sec]."""
    df = pd.read_csv(path, sep=r'\s+', header=None,
                     names=['longitude', 'latitude', 'timestamp_sec'])
    df = df.sort_values('timestamp_sec').reset_index(drop=True)
    return df


def load_nodes(path):
    """Load .nodes file → DataFrame[node_id, longitude, latitude]."""
    df = pd.read_csv(path, sep=r'\s+', header=None,
                     names=['longitude', 'latitude'])
    df.index.name = 'node_id'
    df = df.reset_index()
    return df


def load_arcs(path):
    """Load .arcs file → DataFrame[arc_id, from_node_id, to_node_id]."""
    df = pd.read_csv(path, sep=r'\s+', header=None,
                     names=['from_node_id', 'to_node_id'])
    df.index.name = 'arc_id'
    df = df.reset_index()
    return df


def load_route(path):
    """Load .route file → list of arc_ids in traversal order."""
    with open(path, 'r') as f:
        return [int(line.strip()) for line in f if line.strip()]


# ─────────────────────────────────────────────
#  Candidate arc generation
# ─────────────────────────────────────────────

def build_arc_geometry(nodes_df, arcs_df):
    """
    Returns list of dicts: {arc_id, from_node, to_node,
                             lon_a, lat_a, lon_b, lat_b, arc_length_m, arc_bearing}
    """
    node_lookup = nodes_df.set_index('node_id')[['longitude', 'latitude']].to_dict('index')
    arcs = []
    for _, row in arcs_df.iterrows():
        a = node_lookup[row['from_node_id']]
        b = node_lookup[row['to_node_id']]
        length = haversine(a['longitude'], a['latitude'], b['longitude'], b['latitude'])
        brg = bearing(a['longitude'], a['latitude'], b['longitude'], b['latitude'])
        arcs.append({
            'arc_id': row['arc_id'],
            'from_node_id': row['from_node_id'],
            'to_node_id': row['to_node_id'],
            'lon_a': a['longitude'], 'lat_a': a['latitude'],
            'lon_b': b['longitude'], 'lat_b': b['latitude'],
            'arc_length_m': length,
            'arc_bearing': brg,
        })
    return arcs


def get_candidate_arcs(gps_lon, gps_lat, arc_geom_list, radius_m=50, top_k=10):
    """Return up to top_k arcs within radius_m metres of the GPS point."""
    candidates = []
    for arc in arc_geom_list:
        d = dist_metres(gps_lon, gps_lat,
                        arc['lon_a'], arc['lat_a'],
                        arc['lon_b'], arc['lat_b'])
        if d <= radius_m:
            candidates.append({**arc, 'dist_to_arc_m': d})

    # Sort by distance, keep top_k
    candidates.sort(key=lambda x: x['dist_to_arc_m'])
    return candidates[:top_k]


# ─────────────────────────────────────────────
#  Feature computation
# ─────────────────────────────────────────────

FEATURE_COLS = [
    'dist_to_arc_m',
    'bearing_diff',
    'cos_bearing_diff',
    'sin_bearing_diff',
    'speed_mps',
    'time_delta_sec',
    'arc_length_m',
    'candidate_rank',
]


def compute_features_for_trip(track_df, arc_geom_list, route_arc_ids=None,
                               radius_m=50, top_k=10):
    """
    Build the feature matrix for one trip.

    Returns:
        features_df  — DataFrame with FEATURE_COLS + arc_id + fix_index
        labels       — Series of binary labels (1 = correct arc)
                       Only populated when route_arc_ids is provided (training).
    """
    rows = []
    n = len(track_df)

    for i, gps in track_df.iterrows():
        # Speed & time delta (from previous fix)
        if i == 0:
            speed = 0.0
            dt = 0.0
        else:
            prev = track_df.iloc[i - 1]
            dt = gps['timestamp_sec'] - prev['timestamp_sec']
            dist = haversine(prev['longitude'], prev['latitude'],
                             gps['longitude'], gps['latitude'])
            speed = dist / dt if dt > 0 else 0.0

        # GPS bearing from previous fix
        if i == 0:
            gps_bearing = 0.0
        else:
            prev = track_df.iloc[i - 1]
            gps_bearing = bearing(prev['longitude'], prev['latitude'],
                                  gps['longitude'], gps['latitude'])

        candidates = get_candidate_arcs(
            gps['longitude'], gps['latitude'], arc_geom_list,
            radius_m=radius_m, top_k=top_k
        )

        # Ground-truth arc for this fix (training only)
        gt_arc = route_arc_ids[i] if (route_arc_ids is not None and i < len(route_arc_ids)) else None

        for rank, cand in enumerate(candidates):
            bdiff = (gps_bearing - cand['arc_bearing'] + 360) % 360
            if bdiff > 180:
                bdiff = 360 - bdiff  # [0, 180]

            rows.append({
                'fix_index': i,
                'arc_id': cand['arc_id'],
                'dist_to_arc_m': cand['dist_to_arc_m'],
                'bearing_diff': bdiff,
                'cos_bearing_diff': math.cos(math.radians(bdiff)),
                'sin_bearing_diff': math.sin(math.radians(bdiff)),
                'speed_mps': speed,
                'time_delta_sec': dt,
                'arc_length_m': cand['arc_length_m'],
                'candidate_rank': rank + 1,
                'label': 1 if (gt_arc is not None and cand['arc_id'] == gt_arc) else 0,
            })

    df = pd.DataFrame(rows)
    return df


# ─────────────────────────────────────────────
#  Build full training dataset from all trips
# ─────────────────────────────────────────────

import os
from glob import glob


def build_dataset(trips_dir, split='training'):
    """
    Scan trips_dir for all trips, build feature rows.
    Returns X (features), y (labels), groups (trip_id per row).

    Supports two layouts:
      (a) data/training/00000001/00000001.track   ← actual dataset layout
      (b) data/training/00000001/*.track          ← generic layout
    """
    all_rows = []
    trip_dirs = sorted([d for d in os.listdir(trips_dir)
                        if os.path.isdir(os.path.join(trips_dir, d))])

    for trip_id, trip_name in enumerate(trip_dirs):
        trip_path = os.path.join(trips_dir, trip_name)
        # Support named files: e.g. 00000001.track inside 00000001/
        track_files = glob(os.path.join(trip_path, '*.track'))
        node_files  = glob(os.path.join(trip_path, '*.nodes'))
        arc_files   = glob(os.path.join(trip_path, '*.arcs'))
        route_files = glob(os.path.join(trip_path, '*.route'))

        if not (track_files and node_files and arc_files):
            continue

        track_df = load_track(track_files[0])
        nodes_df = load_nodes(node_files[0])
        arcs_df = load_arcs(arc_files[0])
        arc_geom = build_arc_geometry(nodes_df, arcs_df)

        route_arcs = None
        if route_files and split == 'training':
            route_arcs = load_route(route_files[0])

        trip_features = compute_features_for_trip(
            track_df, arc_geom, route_arcs
        )
        trip_features['trip_id'] = trip_id
        trip_features['trip_name'] = trip_name
        all_rows.append(trip_features)
        print(f"  ✓ {trip_name}: {len(track_df)} fixes, {len(trip_features)} candidate rows")

    full_df = pd.concat(all_rows, ignore_index=True)
    X = full_df[FEATURE_COLS]
    y = full_df['label']
    groups = full_df['trip_id']
    return X, y, groups, full_df
