"""
PathSnap — Feature Engineering Pipeline
src/features.py

Converts raw GPS track + road network into ML-ready feature rows.
Each row = one (GPS fix, candidate arc) pair.
Uses vectorized NumPy + scipy KDTree for fast (<1s/trip) inference.
"""

import os
import math
from glob import glob
import numpy as np
import pandas as pd
from scipy.spatial import KDTree


# ─────────────────────────────────────────────
#  Geometry helpers
# ─────────────────────────────────────────────

def haversine(lon1, lat1, lon2, lat2):
    """Return distance in metres between two (lon, lat) points."""
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2.0 * math.asin(math.sqrt(min(1.0, a)))


def bearing(lon1, lat1, lon2, lat2):
    """Compass bearing in degrees [0, 360) from point 1 -> point 2."""
    lat1, lat2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def point_to_segment_distance(px, py, ax, ay, bx, by):
    """Perpendicular distance from point P to line segment A->B."""
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay), 0.0
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    nearest_x = ax + t * dx
    nearest_y = ay + t * dy
    return math.hypot(px - nearest_x, py - nearest_y), t


def dist_metres(px, py, ax, ay, bx, by):
    """Perpendicular distance in metres from GPS point to arc segment."""
    mid_lat = (ay + by) / 2.0
    deg_per_metre_lat = 1.0 / 111_111.0
    deg_per_metre_lon = 1.0 / (111_111.0 * math.cos(math.radians(mid_lat)))
    dist, _ = point_to_segment_distance(
        px / deg_per_metre_lon, py / deg_per_metre_lat,
        ax / deg_per_metre_lon, ay / deg_per_metre_lat,
        bx / deg_per_metre_lon, by / deg_per_metre_lat,
    )
    return dist


# ─────────────────────────────────────────────
#  Data loaders
# ─────────────────────────────────────────────

def load_track(path):
    """Load .track file -> DataFrame[longitude, latitude, timestamp_sec]."""
    df = pd.read_csv(path, sep=r'\s+', header=None,
                     names=['longitude', 'latitude', 'timestamp_sec'])
    return df.sort_values('timestamp_sec').reset_index(drop=True)


def load_nodes(path):
    """Load .nodes file -> DataFrame[node_id, longitude, latitude]."""
    df = pd.read_csv(path, sep=r'\s+', header=None,
                     names=['longitude', 'latitude'])
    df.index.name = 'node_id'
    return df.reset_index()


def load_arcs(path):
    """Load .arcs file -> DataFrame[arc_id, from_node_id, to_node_id]."""
    df = pd.read_csv(path, sep=r'\s+', header=None,
                     names=['from_node_id', 'to_node_id'])
    df.index.name = 'arc_id'
    return df.reset_index()


def load_route(path):
    """Load .route file -> list of arc_ids in traversal order."""
    with open(path, 'r') as f:
        return [int(line.strip()) for line in f if line.strip()]


# ─────────────────────────────────────────────
#  Vectorized Arc Geometry + KDTree Index
# ─────────────────────────────────────────────

def build_arc_geometry(nodes_df, arcs_df):
    """
    Build vectorized arc geometry & KDTree index for fast spatial queries.
    Returns a dict holding numpy arrays and the KDTree.
    """
    node_lon = nodes_df['longitude'].to_numpy(dtype=np.float64)
    node_lat = nodes_df['latitude'].to_numpy(dtype=np.float64)
    arc_ids  = arcs_df['arc_id'].to_numpy(dtype=np.int64)
    from_ids = arcs_df['from_node_id'].to_numpy(dtype=np.int64)
    to_ids   = arcs_df['to_node_id'].to_numpy(dtype=np.int64)

    lon_a, lat_a = node_lon[from_ids], node_lat[from_ids]
    lon_b, lat_b = node_lon[to_ids],   node_lat[to_ids]

    # Vectorized haversine length
    phi1, phi2 = np.radians(lat_a), np.radians(lat_b)
    dphi = phi2 - phi1
    dlam = np.radians(lon_b - lon_a)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    arc_len = 6_371_000.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

    # Vectorized compass bearing
    bx_term = np.sin(dlam) * np.cos(phi2)
    by_term = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlam)
    arc_brg = (np.degrees(np.arctan2(bx_term, by_term)) + 360.0) % 360.0

    # Local metre projection + KDTree on arc midpoints
    lat_scale = 111_111.0
    lon_scale = 111_111.0 * math.cos(math.radians(float(np.mean(lat_a))))
    ax_m, ay_m = lon_a * lon_scale, lat_a * lat_scale
    bx_m, by_m = lon_b * lon_scale, lat_b * lat_scale
    mid_xy = np.column_stack([(ax_m + bx_m) * 0.5, (ay_m + by_m) * 0.5])
    tree = KDTree(mid_xy)

    return {
        'arc_ids': arc_ids,
        'from_ids': from_ids,
        'to_ids': to_ids,
        'lon_a': lon_a, 'lat_a': lat_a,
        'lon_b': lon_b, 'lat_b': lat_b,
        'ax_m': ax_m, 'ay_m': ay_m,
        'bx_m': bx_m, 'by_m': by_m,
        'arc_len': arc_len,
        'arc_brg': arc_brg,
        'lat_scale': lat_scale,
        'lon_scale': lon_scale,
        'tree': tree,
    }


def get_candidate_arcs(gps_lon, gps_lat, arc_geom, radius_m=50.0, top_k=10):
    """Return up to top_k candidate arcs within radius_m metres using KDTree."""
    px = gps_lon * arc_geom['lon_scale']
    py = gps_lat * arc_geom['lat_scale']
    idxs = arc_geom['tree'].query_ball_point([px, py], r=300.0)
    if not idxs:
        return []
    idxs = np.asarray(idxs, dtype=np.int64)

    ax = arc_geom['ax_m'][idxs]
    ay = arc_geom['ay_m'][idxs]
    dx = arc_geom['bx_m'][idxs] - ax
    dy = arc_geom['by_m'][idxs] - ay
    denom = dx * dx + dy * dy
    safe_denom = np.where(denom > 0, denom, 1.0)
    proj = np.where(
        denom > 0,
        np.clip(((px - ax) * dx + (py - ay) * dy) / safe_denom, 0.0, 1.0),
        0.0,
    )
    dists = np.hypot(px - (ax + proj * dx), py - (ay + proj * dy))

    mask = dists <= radius_m
    if not np.any(mask):
        return []
    cand_idxs = idxs[mask]
    cand_dists = dists[mask]
    order = np.argsort(cand_dists)[:top_k]

    return [
        {
            'arc_id': int(arc_geom['arc_ids'][cand_idxs[o]]),
            'dist_to_arc_m': float(cand_dists[o]),
            'arc_length_m': float(arc_geom['arc_len'][cand_idxs[o]]),
            'arc_bearing': float(arc_geom['arc_brg'][cand_idxs[o]]),
        }
        for o in order
    ]


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


def compute_features_for_trip(track_df, arc_geom, route_arc_ids=None,
                              radius_m=50.0, top_k=10):
    """
    Build the feature matrix for one trip using vectorized KDTree lookup.
    Exact feature parity with Kaggle training notebook.
    """
    route_set = set(route_arc_ids) if route_arc_ids is not None else None

    t_lon = track_df['longitude'].to_numpy(dtype=np.float64)
    t_lat = track_df['latitude'].to_numpy(dtype=np.float64)
    t_sec = track_df['timestamp_sec'].to_numpy(dtype=np.float64)

    lon_scale = arc_geom['lon_scale']
    lat_scale = arc_geom['lat_scale']
    ax_m, ay_m = arc_geom['ax_m'], arc_geom['ay_m']
    bx_m, by_m = arc_geom['bx_m'], arc_geom['by_m']
    arc_ids_arr = arc_geom['arc_ids']
    arc_len = arc_geom['arc_len']
    arc_brg = arc_geom['arc_brg']
    tree = arc_geom['tree']

    rows = []
    for i in range(len(track_df)):
        if i == 0:
            speed, dt, gps_brg = 0.0, 0.0, 0.0
        else:
            dt = float(t_sec[i] - t_sec[i - 1])
            step_d = haversine(t_lon[i - 1], t_lat[i - 1], t_lon[i], t_lat[i])
            speed = step_d / dt if dt > 0 else 0.0
            gps_brg = bearing(t_lon[i - 1], t_lat[i - 1], t_lon[i], t_lat[i])

        px, py = t_lon[i] * lon_scale, t_lat[i] * lat_scale
        idxs = tree.query_ball_point([px, py], r=300.0)
        if not idxs:
            # Fallback to nearest single arc so fix always gets a candidate
            _, nn_idx = tree.query([px, py])
            idxs = [int(nn_idx)]
        idxs = np.asarray(idxs, dtype=np.int64)

        dx = bx_m[idxs] - ax_m[idxs]
        dy = by_m[idxs] - ay_m[idxs]
        denom = dx * dx + dy * dy
        safe_denom = np.where(denom > 0, denom, 1.0)
        proj = np.where(
            denom > 0,
            np.clip(((px - ax_m[idxs]) * dx + (py - ay_m[idxs]) * dy) / safe_denom, 0.0, 1.0),
            0.0,
        )
        dists = np.hypot(px - (ax_m[idxs] + proj * dx), py - (ay_m[idxs] + proj * dy))

        mask = dists <= radius_m
        if not np.any(mask):
            # Keep closest arc even if > radius_m during noisy GPS spikes
            best_idx = int(np.argmin(dists))
            cand_idxs = idxs[best_idx:best_idx + 1]
            cand_dists = dists[best_idx:best_idx + 1]
        else:
            cand_idxs = idxs[mask]
            cand_dists = dists[mask]

        order = np.argsort(cand_dists)[:top_k]
        for rank, o in enumerate(order):
            pos = int(cand_idxs[o])
            aid = int(arc_ids_arr[pos])
            bdiff = (gps_brg - float(arc_brg[pos]) + 360.0) % 360.0
            if bdiff > 180.0:
                bdiff = 360.0 - bdiff
            rows.append({
                'fix_index': i,
                'arc_id': aid,
                'dist_to_arc_m': float(cand_dists[o]),
                'bearing_diff': bdiff,
                'cos_bearing_diff': math.cos(math.radians(bdiff)),
                'sin_bearing_diff': math.sin(math.radians(bdiff)),
                'speed_mps': speed,
                'time_delta_sec': dt,
                'arc_length_m': float(arc_len[pos]),
                'candidate_rank': rank + 1,
                'label': 1 if (route_set is not None and aid in route_set) else 0,
            })

    return pd.DataFrame(rows)


def build_dataset(trips_dir, split='training'):
    """Scan trips_dir for all trips and build the combined feature matrix."""
    all_rows = []
    trip_dirs = sorted([d for d in os.listdir(trips_dir)
                        if os.path.isdir(os.path.join(trips_dir, d))])

    for trip_id, trip_name in enumerate(trip_dirs):
        trip_path = os.path.join(trips_dir, trip_name)
        track_files = glob(os.path.join(trip_path, '*.track'))
        node_files  = glob(os.path.join(trip_path, '*.nodes'))
        arc_files   = glob(os.path.join(trip_path, '*.arcs'))
        route_files = glob(os.path.join(trip_path, '*.route'))

        if not (track_files and node_files and arc_files):
            continue

        track_df = load_track(track_files[0])
        nodes_df = load_nodes(node_files[0])
        arcs_df  = load_arcs(arc_files[0])
        arc_geom = build_arc_geometry(nodes_df, arcs_df)

        route_arcs = None
        if route_files and split == 'training':
            route_arcs = load_route(route_files[0])

        trip_features = compute_features_for_trip(track_df, arc_geom, route_arcs)
        trip_features['trip_id'] = trip_id
        trip_features['trip_name'] = trip_name
        all_rows.append(trip_features)
        print(f"  ✓ {trip_name}: {len(track_df)} fixes, {len(trip_features)} candidate rows")

    full_df = pd.concat(all_rows, ignore_index=True)
    return full_df[FEATURE_COLS], full_df['label'], full_df['trip_id'], full_df
