"""Tests with a tiny hand-made road network (a test fixture, not real Kumasi roads)."""
import json

import pandas as pd
import pytest

from atmosnofear import segments as seg


def way(way_id, name, pts, **tags):
    """pts: list of (node_id, lat, lon)."""
    return {"id": way_id, "nodes": [p[0] for p in pts],
            "geometry": [{"lat": p[1], "lon": p[2]} for p in pts],
            "tags": {"name": name, "highway": "primary", **tags}}


# Roads on a 0.01 degree grid (~1.1 km per step along latitude).
EAST = way(1, "East Road", [(1, 6.70, -1.60), (2, 6.70, -1.59), (3, 6.70, -1.58)])
ONEWAY = way(2, "One Way", [(3, 6.70, -1.58), (4, 6.71, -1.58)], oneway="yes")
ROUND = way(3, "Test Roundabout", [(10, 6.72, -1.60), (11, 6.721, -1.60), (12, 6.72, -1.601)], junction="roundabout")
NETWORK = [EAST, ONEWAY, ROUND]


def test_haversine_one_degree_latitude():
    assert seg.haversine_m(0, 0, 1, 0) == pytest.approx(111195, rel=0.01)


def test_path_length_follows_the_road():
    row, line = seg.build_segment(NETWORK, "T1", "t", "east", (6.70, -1.60), (6.70, -1.58))
    assert row["length_m"] == pytest.approx(seg.haversine_m(6.70, -1.60, 6.70, -1.58), rel=0.001)
    assert len(line) == 3 and "osm_ways=1" in row["notes"]


def test_oneway_blocks_the_wrong_direction():
    row, _ = seg.build_segment(NETWORK, "T2", "t", "north", (6.70, -1.58), (6.71, -1.58))
    assert row["length_m"] > 0
    with pytest.raises(ValueError, match="no drivable path"):
        seg.build_segment(NETWORK, "T3", "t", "south", (6.71, -1.58), (6.70, -1.58))


def test_roads_filter_restricts_the_network():
    with pytest.raises(ValueError):  # One Way is excluded, so (6.71,-1.58) is far from East Road
        seg.build_segment(NETWORK, "T4", "t", "x", (6.70, -1.60), (6.71, -1.58), roads={"East Road"})


def test_point_far_from_any_road_is_rejected():
    with pytest.raises(ValueError, match="nearest selected road"):
        seg.build_segment(NETWORK, "T5", "t", "x", (6.70, -1.60), (7.50, -1.58))


def test_same_snap_point_is_rejected():
    with pytest.raises(ValueError, match="same point"):
        seg.build_segment(NETWORK, "T6", "t", "x", (6.70, -1.60), (6.7001, -1.6001))


def test_landmark_is_way_centre_and_case_insensitive():
    lat, lon = seg.find_landmark(NETWORK, "test roundabout")
    assert lat == pytest.approx(6.72033, abs=1e-4) and lon == pytest.approx(-1.60033, abs=1e-4)
    with pytest.raises(ValueError):
        seg.find_landmark(NETWORK, "Nowhere")


def test_roundabout_is_treated_as_one_way():
    adj, _ = seg.build_graph([ROUND])
    assert [v for v, _, _ in adj[10]] == [11] and adj[11][0][0] == 12


def test_append_segment_writes_and_rejects_duplicates(tmp_path):
    row, line = seg.build_segment(NETWORK, "T1", "t", "east", (6.70, -1.60), (6.70, -1.58))
    path = tmp_path / "segments.csv"
    seg.append_segment(path, row)
    assert list(pd.read_csv(path).columns) == seg.SEGMENT_COLUMNS
    with pytest.raises(ValueError, match="already exists"):
        seg.append_segment(path, row)
    seg.write_geojson(tmp_path / "g.geojson", "T1", line)
    assert json.loads((tmp_path / "g.geojson").read_text())["features"][0]["geometry"]["type"] == "LineString"


def test_road_summary_totals_per_name():
    out = seg.road_summary(NETWORK)
    assert set(out["road"]) == {"East Road", "One Way", "Test Roundabout"}
