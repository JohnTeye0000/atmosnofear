"""fetch_roads cache behaviour, with the network call replaced by a stub."""
import json

from atmosnofear import segments as seg

GOOD = [{"id": 1, "nodes": [1, 2], "geometry": [{"lat": 6.7, "lon": -1.6}, {"lat": 6.7, "lon": -1.59}],
         "tags": {"name": "X", "highway": "primary"}}]


class FakeResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return {"elements": GOOD}


def test_cache_without_node_ids_is_refetched(tmp_path, monkeypatch):
    cache = tmp_path / "osm.json"
    cache.write_text(json.dumps([{"id": 1, "tags": {"name": "X"}}]))   # old, unusable format
    calls = []
    monkeypatch.setattr(seg.requests, "post", lambda *a, **k: calls.append(k["data"]["data"]) or FakeResponse())
    assert seg.fetch_roads(cache) == GOOD
    assert "out body geom" in calls[0]                                  # asks for node ids this time


def test_valid_cache_is_used_without_network(tmp_path, monkeypatch):
    cache = tmp_path / "osm.json"
    cache.write_text(json.dumps(GOOD))
    monkeypatch.setattr(seg.requests, "post", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network used")))
    assert seg.fetch_roads(cache) == GOOD
