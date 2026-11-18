"""
PathSnap — Unit Tests for src/features.py
tests/test_features.py

Tests the core geometry helpers and the feature engineering pipeline
that is the foundation of every model in PathSnap.

Run with:
    python -m pytest tests/test_features.py -v
    # or directly
    python tests/test_features.py

No external data files are needed: all tests are self-contained and
use synthetic inputs that mirror the structure of the real GPS/road data.
"""

import sys
import math
import tempfile
import os
from pathlib import Path

import pandas as pd
import numpy as np

# Allow imports from src/
_SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(_SRC))

from features import (
    haversine,
    bearing,
    point_to_segment_distance,
    dist_metres,
    load_track,
    load_nodes,
    load_arcs,
    build_arc_geometry,
    get_candidate_arcs,
    compute_features_for_trip,
    FEATURE_COLS,
)


# ═════════════════════════════════════════════════════════════════════
#  Geometry helpers
# ═════════════════════════════════════════════════════════════════════

class TestHaversine:
    """Tests for the haversine distance function."""

    def test_same_point_is_zero(self):
        assert haversine(77.5, 12.9, 77.5, 12.9) == 0.0

    def test_known_distance(self):
        # Distance between two points ~1 km apart in Bangalore
        # Approximate: 1 degree lat ≈ 111,111 m
        # Moving 0.009° north ≈ 1000 m
        d = haversine(77.5946, 12.9716, 77.5946, 12.9816)
        assert 1050 < d < 1150, f"Expected ~1100 m, got {d:.1f} m"

    def test_symmetry(self):
        d1 = haversine(77.5, 12.9, 77.6, 13.0)
        d2 = haversine(77.6, 13.0, 77.5, 12.9)
        assert abs(d1 - d2) < 1e-6

    def test_positive_result(self):
        d = haversine(0.0, 0.0, 1.0, 1.0)
        assert d > 0


class TestBearing:
    """Tests for the bearing (compass direction) function."""

    def test_due_north(self):
        b = bearing(77.5, 12.9, 77.5, 13.0)  # Same lon, increasing lat → north
        assert abs(b - 0.0) < 1.0 or abs(b - 360.0) < 1.0, f"Expected ~0°, got {b:.2f}°"

    def test_due_east(self):
        b = bearing(77.5, 12.9, 78.5, 12.9)  # Same lat, increasing lon → east
        assert abs(b - 90.0) < 2.0, f"Expected ~90°, got {b:.2f}°"

    def test_due_south(self):
        b = bearing(77.5, 13.0, 77.5, 12.9)  # Same lon, decreasing lat → south
        assert abs(b - 180.0) < 1.0, f"Expected ~180°, got {b:.2f}°"

    def test_due_west(self):
        b = bearing(78.5, 12.9, 77.5, 12.9)  # Decreasing lon → west
        assert abs(b - 270.0) < 2.0, f"Expected ~270°, got {b:.2f}°"

    def test_range_0_to_360(self):
        for lon2 in [77.4, 77.5, 77.6]:
            for lat2 in [12.8, 12.9, 13.0]:
                b = bearing(77.5, 12.9, lon2, lat2)
                assert 0.0 <= b < 360.0, f"Bearing out of range: {b}"


class TestPointToSegmentDistance:
    """Tests for point-to-segment distance in degree space."""

    def test_point_on_segment(self):
        # Midpoint of a horizontal segment
        d, t = point_to_segment_distance(1.0, 0.0, 0.0, 0.0, 2.0, 0.0)
        assert d < 1e-9
        assert abs(t - 0.5) < 1e-9

    def test_point_at_vertex_a(self):
        d, t = point_to_segment_distance(0.0, 0.0, 0.0, 0.0, 2.0, 0.0)
        assert d < 1e-9
        assert abs(t - 0.0) < 1e-9

    def test_point_at_vertex_b(self):
        d, t = point_to_segment_distance(2.0, 0.0, 0.0, 0.0, 2.0, 0.0)
        assert d < 1e-9
        assert abs(t - 1.0) < 1e-9

    def test_perpendicular_above_midpoint(self):
        # Point is directly above the midpoint of a horizontal unit segment
        d, t = point_to_segment_distance(0.5, 1.0, 0.0, 0.0, 1.0, 0.0)
        assert abs(d - 1.0) < 1e-9
        assert abs(t - 0.5) < 1e-9

    def test_zero_length_segment(self):
        # Segment is a single point
        d, t = point_to_segment_distance(3.0, 4.0, 1.0, 1.0, 1.0, 1.0)
        assert abs(d - math.hypot(2.0, 3.0)) < 1e-9

    def test_t_clamped_before_a(self):
        # Point is "behind" the start of the segment
        d, t = point_to_segment_distance(-1.0, 0.0, 0.0, 0.0, 2.0, 0.0)
        assert abs(t - 0.0) < 1e-9  # Clamped to start

    def test_t_clamped_after_b(self):
        # Point is "past" the end of the segment
        d, t = point_to_segment_distance(5.0, 0.0, 0.0, 0.0, 2.0, 0.0)
        assert abs(t - 1.0) < 1e-9  # Clamped to end


class TestDistMetres:
    """Tests for the metre-space distance from a GPS point to an arc segment."""

    def test_point_on_arc_is_zero(self):
        # GPS fix is exactly on the midpoint of the arc
        lon_a, lat_a = 77.5000, 12.9000
        lon_b, lat_b = 77.5020, 12.9000  # Horizontal arc (~200 m)
        mid_lon = (lon_a + lon_b) / 2
        d = dist_metres(mid_lon, lat_a, lon_a, lat_a, lon_b, lat_b)
        assert d < 1.0, f"Expected ~0 m, got {d:.2f} m"

    def test_distance_is_positive(self):
        d = dist_metres(77.5010, 12.9010, 77.5000, 12.9000, 77.5020, 12.9000)
        assert d > 0

    def test_farther_point_has_larger_distance(self):
        lon_a, lat_a = 77.5000, 12.9000
        lon_b, lat_b = 77.5020, 12.9000
        mid_lon = 77.5010
        d_near = dist_metres(mid_lon, lat_a + 0.0001, lon_a, lat_a, lon_b, lat_b)
        d_far  = dist_metres(mid_lon, lat_a + 0.001,  lon_a, lat_a, lon_b, lat_b)
        assert d_far > d_near


# ═════════════════════════════════════════════════════════════════════
#  Data loader round-trips (via temp files)
# ═════════════════════════════════════════════════════════════════════

class TestDataLoaders:

    def _write_temp(self, content: str, suffix: str) -> str:
        fd, path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "w") as f:
            f.write(content)
        return path

    def test_load_track(self):
        content = "77.5946 12.9716 1000.0\n77.5950 12.9720 1001.0\n"
        path = self._write_temp(content, ".track")
        try:
            df = load_track(path)
            assert list(df.columns) == ["longitude", "latitude", "timestamp_sec"]
            assert len(df) == 2
            assert df["longitude"].iloc[0] == pytest.approx(77.5946)
        finally:
            os.unlink(path)

    def test_load_track_sorted_by_timestamp(self):
        # Rows out of order: second row has earlier timestamp
        content = "77.5950 12.9720 1002.0\n77.5946 12.9716 1000.0\n"
        path = self._write_temp(content, ".track")
        try:
            df = load_track(path)
            assert df["timestamp_sec"].iloc[0] < df["timestamp_sec"].iloc[1]
        finally:
            os.unlink(path)

    def test_load_nodes(self):
        content = "77.5946 12.9716\n77.5950 12.9720\n"
        path = self._write_temp(content, ".nodes")
        try:
            df = load_nodes(path)
            assert "node_id" in df.columns
            assert "longitude" in df.columns
            assert "latitude" in df.columns
            assert df["node_id"].iloc[0] == 0
            assert df["node_id"].iloc[1] == 1
        finally:
            os.unlink(path)

    def test_load_arcs(self):
        content = "0 1\n1 2\n"
        path = self._write_temp(content, ".arcs")
        try:
            df = load_arcs(path)
            assert "arc_id" in df.columns
            assert "from_node_id" in df.columns
            assert "to_node_id" in df.columns
            assert df["arc_id"].iloc[0] == 0
            assert df["from_node_id"].iloc[0] == 0
            assert df["to_node_id"].iloc[0] == 1
        finally:
            os.unlink(path)


# ═════════════════════════════════════════════════════════════════════
#  Arc geometry builder
# ═════════════════════════════════════════════════════════════════════

class TestBuildArcGeometry:

    def _make_nodes(self):
        return pd.DataFrame({
            "node_id": [0, 1, 2],
            "longitude": [77.5000, 77.5020, 77.5040],
            "latitude":  [12.9000, 12.9000, 12.9000],
        })

    def _make_arcs(self):
        return pd.DataFrame({
            "arc_id":      [0, 1],
            "from_node_id":[0, 1],
            "to_node_id":  [1, 2],
        })

    def test_returns_list_of_dicts(self):
        arc_geom = build_arc_geometry(self._make_nodes(), self._make_arcs())
        assert isinstance(arc_geom, list)
        assert len(arc_geom) == 2

    def test_required_keys_present(self):
        arc_geom = build_arc_geometry(self._make_nodes(), self._make_arcs())
        required = {"arc_id", "from_node_id", "to_node_id",
                    "lon_a", "lat_a", "lon_b", "lat_b",
                    "arc_length_m", "arc_bearing"}
        for arc in arc_geom:
            assert required.issubset(arc.keys()), f"Missing keys: {required - arc.keys()}"

    def test_arc_length_positive(self):
        arc_geom = build_arc_geometry(self._make_nodes(), self._make_arcs())
        for arc in arc_geom:
            assert arc["arc_length_m"] > 0

    def test_arc_bearing_range(self):
        arc_geom = build_arc_geometry(self._make_nodes(), self._make_arcs())
        for arc in arc_geom:
            assert 0.0 <= arc["arc_bearing"] < 360.0

    def test_eastbound_arc_bearing_near_90(self):
        # Nodes are at the same latitude, increasing longitude → east → ~90°
        arc_geom = build_arc_geometry(self._make_nodes(), self._make_arcs())
        for arc in arc_geom:
            assert abs(arc["arc_bearing"] - 90.0) < 2.0, (
                f"Expected ~90° for eastbound arc, got {arc['arc_bearing']:.2f}°"
            )


# ═════════════════════════════════════════════════════════════════════
#  Candidate arc generation
# ═════════════════════════════════════════════════════════════════════

class TestGetCandidateArcs:

    def _make_arc_geom(self):
        nodes = pd.DataFrame({
            "node_id": [0, 1, 2, 3],
            "longitude": [77.5000, 77.5010, 77.5100, 77.5110],
            "latitude":  [12.9000, 12.9000, 12.9000, 12.9000],
        })
        arcs = pd.DataFrame({
            "arc_id":      [0, 1],
            "from_node_id":[0, 2],
            "to_node_id":  [1, 3],
        })
        return build_arc_geometry(nodes, arcs)

    def test_returns_list(self):
        arc_geom = self._make_arc_geom()
        result = get_candidate_arcs(77.5005, 12.9000, arc_geom, radius_m=50)
        assert isinstance(result, list)

    def test_nearby_arc_included(self):
        arc_geom = self._make_arc_geom()
        # GPS fix is on arc 0 (77.5000–77.5010, 12.9000)
        result = get_candidate_arcs(77.5005, 12.9000, arc_geom, radius_m=50)
        arc_ids = [c["arc_id"] for c in result]
        assert 0 in arc_ids, "Arc 0 should be a candidate for a point on it"

    def test_distant_arc_excluded(self):
        arc_geom = self._make_arc_geom()
        # GPS fix is at arc 0; arc 1 is ~1 km away
        result = get_candidate_arcs(77.5005, 12.9000, arc_geom, radius_m=50)
        arc_ids = [c["arc_id"] for c in result]
        assert 1 not in arc_ids, "Arc 1 is too far and should not be a candidate"

    def test_sorted_by_distance(self):
        arc_geom = self._make_arc_geom()
        result = get_candidate_arcs(77.5005, 12.9000, arc_geom, radius_m=200)
        distances = [c["dist_to_arc_m"] for c in result]
        assert distances == sorted(distances), "Candidates should be sorted by distance"

    def test_top_k_respected(self):
        arc_geom = self._make_arc_geom()
        result = get_candidate_arcs(77.5005, 12.9000, arc_geom, radius_m=5000, top_k=1)
        assert len(result) <= 1


# ═════════════════════════════════════════════════════════════════════
#  compute_features_for_trip (integration-level)
# ═════════════════════════════════════════════════════════════════════

class TestComputeFeaturesForTrip:

    def _make_inputs(self, n_fixes: int = 5):
        """Generate a minimal synthetic trip."""
        # A simple east-west road: 3 nodes, 2 arcs
        nodes = pd.DataFrame({
            "node_id": [0, 1, 2],
            "longitude": [77.5000, 77.5020, 77.5040],
            "latitude":  [12.9000, 12.9000, 12.9000],
        })
        arcs = pd.DataFrame({
            "arc_id":      [0, 1],
            "from_node_id":[0, 1],
            "to_node_id":  [1, 2],
        })
        arc_geom = build_arc_geometry(nodes, arcs)

        # GPS track that runs along the road
        track = pd.DataFrame({
            "longitude":     [77.5000 + i * 0.004 for i in range(n_fixes)],
            "latitude":      [12.9000] * n_fixes,
            "timestamp_sec": [1000.0 + i for i in range(n_fixes)],
        })

        # Route: alternate between arc 0 and arc 1
        route = [0, 1, 0, 1, 0][:n_fixes]

        return track, arc_geom, route

    def test_returns_dataframe(self):
        track, arc_geom, route = self._make_inputs()
        df = compute_features_for_trip(track, arc_geom, route)
        assert isinstance(df, pd.DataFrame)

    def test_all_feature_cols_present(self):
        track, arc_geom, route = self._make_inputs()
        df = compute_features_for_trip(track, arc_geom, route)
        for col in FEATURE_COLS:
            assert col in df.columns, f"Missing feature column: {col}"

    def test_label_column_present_with_route(self):
        track, arc_geom, route = self._make_inputs()
        df = compute_features_for_trip(track, arc_geom, route)
        assert "label" in df.columns

    def test_no_label_without_route(self):
        track, arc_geom, _ = self._make_inputs()
        df = compute_features_for_trip(track, arc_geom, route_arc_ids=None)
        # label column is still present, but all 0 (no ground truth)
        assert "label" in df.columns
        assert df["label"].sum() == 0

    def test_no_nan_in_feature_cols(self):
        track, arc_geom, route = self._make_inputs()
        df = compute_features_for_trip(track, arc_geom, route)
        for col in FEATURE_COLS:
            n_nan = df[col].isna().sum()
            assert n_nan == 0, f"Column '{col}' has {n_nan} NaN(s)"

    def test_candidate_rank_starts_at_1(self):
        track, arc_geom, route = self._make_inputs()
        df = compute_features_for_trip(track, arc_geom, route)
        assert df["candidate_rank"].min() == 1

    def test_first_fix_speed_zero(self):
        track, arc_geom, route = self._make_inputs()
        df = compute_features_for_trip(track, arc_geom, route)
        first_fix_rows = df[df["fix_index"] == 0]
        assert (first_fix_rows["speed_mps"] == 0.0).all()

    def test_first_fix_time_delta_zero(self):
        track, arc_geom, route = self._make_inputs()
        df = compute_features_for_trip(track, arc_geom, route)
        first_fix_rows = df[df["fix_index"] == 0]
        assert (first_fix_rows["time_delta_sec"] == 0.0).all()

    def test_dist_to_arc_non_negative(self):
        track, arc_geom, route = self._make_inputs()
        df = compute_features_for_trip(track, arc_geom, route)
        assert (df["dist_to_arc_m"] >= 0).all()

    def test_bearing_diff_in_range(self):
        track, arc_geom, route = self._make_inputs(n_fixes=10)
        df = compute_features_for_trip(track, arc_geom, route)
        assert (df["bearing_diff"] >= 0).all()
        assert (df["bearing_diff"] <= 180).all()

    def test_cos_bearing_diff_in_range(self):
        track, arc_geom, route = self._make_inputs(n_fixes=10)
        df = compute_features_for_trip(track, arc_geom, route)
        assert (df["cos_bearing_diff"] >= -1.0001).all()
        assert (df["cos_bearing_diff"] <=  1.0001).all()

    def test_empty_track_returns_empty_df(self):
        _, arc_geom, route = self._make_inputs()
        empty_track = pd.DataFrame(
            columns=["longitude", "latitude", "timestamp_sec"]
        )
        df = compute_features_for_trip(empty_track, arc_geom, route)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


# ═════════════════════════════════════════════════════════════════════
#  FEATURE_COLS contract
# ═════════════════════════════════════════════════════════════════════

class TestFeatureCols:

    def test_feature_cols_is_list(self):
        assert isinstance(FEATURE_COLS, list)

    def test_feature_cols_has_expected_length(self):
        assert len(FEATURE_COLS) == 8

    def test_expected_names_present(self):
        expected = {
            "dist_to_arc_m", "bearing_diff", "cos_bearing_diff",
            "sin_bearing_diff", "speed_mps", "time_delta_sec",
            "arc_length_m", "candidate_rank",
        }
        assert set(FEATURE_COLS) == expected

    def test_no_duplicates(self):
        assert len(FEATURE_COLS) == len(set(FEATURE_COLS))


# ═════════════════════════════════════════════════════════════════════
#  Simple test runner (no pytest needed)
# ═════════════════════════════════════════════════════════════════════

def _run_all_tests():
    """
    Minimal test runner used when pytest is not installed.
    Discovers all Test* classes and test_* methods and runs them.
    """
    import traceback

    test_classes = [
        TestHaversine, TestBearing, TestPointToSegmentDistance,
        TestDistMetres, TestDataLoaders, TestBuildArcGeometry,
        TestGetCandidateArcs, TestComputeFeaturesForTrip,
        TestFeatureCols,
    ]

    passed = failed = 0
    failures = []

    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith("test_")]
        for method in methods:
            try:
                getattr(instance, method)()
                passed += 1
                print(f"  ✓  {cls.__name__}.{method}")
            except Exception as e:
                failed += 1
                failures.append((cls.__name__, method, traceback.format_exc()))
                print(f"  ✗  {cls.__name__}.{method}  →  {e}")

    print(f"\n{'─'*55}")
    print(f"  {passed} passed  |  {failed} failed")
    if failures:
        print(f"\n{'─'*55}  FAILURES  {'─'*5}")
        for cls_name, mname, tb in failures:
            print(f"\n{cls_name}.{mname}\n{tb}")
    return failed == 0


# Support both pytest and direct execution
try:
    import pytest
    # pytest.approx is available globally when running under pytest
except ImportError:
    # Provide a minimal approx shim so tests can run without pytest
    class _Approx:
        def __init__(self, expected, rel=1e-6):
            self.expected = expected
            self.rel = rel
        def __eq__(self, other):
            return abs(other - self.expected) <= self.rel * max(abs(self.expected), 1e-12)
        def __repr__(self):
            return f"approx({self.expected})"

    class _PytestShim:
        @staticmethod
        def approx(value, rel=1e-6):
            return _Approx(value, rel)

    pytest = _PytestShim()


if __name__ == "__main__":
    print(f"\n{'='*55}")
    print("  PathSnap — Feature Unit Tests")
    print(f"{'='*55}\n")
    success = _run_all_tests()
    sys.exit(0 if success else 1)
