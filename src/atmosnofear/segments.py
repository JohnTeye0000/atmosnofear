"""Define road segments from OpenStreetMap geometry.

A segment is a directed drivable path between two points, found on the real road
network (one-way roads respected) and measured along the road, not in a straight line.
Data: (c) OpenStreetMap contributors, ODbL. Requests are cached; Overpass is a shared
public service, so fetch once and reuse.
"""
from __future__ import annotations

import heapq
import json
import math
from pathlib import Path

import pandas as pd
import requests

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
USER_AGENT = "atmosnofear-research/0.1 (student research project)"
KUMASI_BBOX = (6.62, -1.70, 6.78, -1.53)          # south, west, north, east
ROAD_CLASSES = "trunk|primary|secondary"
MAX_SNAP_M = 300.0                                  # refuse points farther than this from a road
SEGMENT_COLUMNS = ["segment_id", "name", "direction", "start_lat", "start_lon",
                   "end_lat", "end_lon", "length_m", "notes"]


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371000.0 * math.asin(math.sqrt(h))


def fetch_roads(cache_file: Path, refresh: bool = False, bbox=KUMASI_BBOX) -> list[dict]:
    """Named major-road ways with geometry, cached on disk."""
    cache_file = Path(cache_file)
    if cache_file.exists() and not refresh:
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        if all("nodes" in w for w in cached):   # an older cache saved without node ids is unusable
            return cached
    query = (f'[out:json][timeout:60];way["highway"~"^({ROAD_CLASSES})$"]["name"]'
             f"({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});out body geom;")
    resp = requests.post(OVERPASS_URL, data={"data": query}, headers={"User-Agent": USER_AGENT}, timeout=90)
    resp.raise_for_status()
    elements = resp.json()["elements"]
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(elements), encoding="utf-8")
    return elements


def road_summary(elements: list[dict]) -> pd.DataFrame:
    """One row per road name: total length, way count, classes."""
    rows = {}
    for w in elements:
        g = w["geometry"]
        length = sum(haversine_m(g[i]["lat"], g[i]["lon"], g[i + 1]["lat"], g[i + 1]["lon"]) for i in range(len(g) - 1))
        r = rows.setdefault(w["tags"]["name"], {"road": w["tags"]["name"], "km": 0.0, "ways": 0, "classes": set()})
        r["km"] += length / 1000
        r["ways"] += 1
        r["classes"].add(w["tags"]["highway"])
    out = pd.DataFrame(rows.values())
    out["classes"] = out["classes"].map(lambda s: ",".join(sorted(s)))
    return out.sort_values("km", ascending=False).reset_index(drop=True)


def _direction_flags(tags: dict) -> tuple[bool, bool]:
    """(can_go_forward, can_go_backward) along the way's node order."""
    oneway = tags.get("oneway")
    if oneway in ("yes", "true", "1") or tags.get("junction") in ("roundabout", "circular"):
        return True, False
    if oneway == "-1":
        return False, True
    return True, True


def build_graph(elements: list[dict], roads: set[str] | None = None):
    """Directed graph: node -> [(neighbour, metres, way_id)], plus node coordinates."""
    adj: dict[int, list] = {}
    coords: dict[int, tuple[float, float]] = {}
    for w in elements:
        if roads is not None and w["tags"]["name"] not in roads:
            continue
        fwd, bwd = _direction_flags(w["tags"])
        nodes, geom = w["nodes"], w["geometry"]
        for n, p in zip(nodes, geom):
            coords[n] = (p["lat"], p["lon"])
        for i in range(len(nodes) - 1):
            a, b = nodes[i], nodes[i + 1]
            d = haversine_m(*coords[a], *coords[b])
            adj.setdefault(a, [])
            adj.setdefault(b, [])
            if fwd:
                adj[a].append((b, d, w["id"]))
            if bwd:
                adj[b].append((a, d, w["id"]))
    return adj, coords


def nearest_node(coords: dict, lat: float, lon: float, max_m: float = MAX_SNAP_M) -> int:
    if not coords:
        raise ValueError("no roads loaded for the requested road names")
    node, dist = min(((n, haversine_m(lat, lon, *c)) for n, c in coords.items()), key=lambda x: x[1])
    if dist > max_m:
        raise ValueError(f"point ({lat}, {lon}) is {dist:.0f} m from the nearest selected road (limit {max_m:.0f} m)")
    return node


def shortest_path(adj: dict, src: int, dst: int) -> tuple[float, list[int], list[int]]:
    """Dijkstra. Returns (metres, node path, ordered unique way ids). Raises if unreachable."""
    best = {src: 0.0}
    prev: dict[int, tuple[int, int]] = {}
    heap = [(0.0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if u == dst:
            break
        if d > best.get(u, math.inf):
            continue
        for v, w, way in adj.get(u, []):
            nd = d + w
            if nd < best.get(v, math.inf):
                best[v], prev[v] = nd, (u, way)
                heapq.heappush(heap, (nd, v))
    if dst not in best:
        raise ValueError("no drivable path between the points on the selected roads (check one-way directions)")
    path, ways, node = [dst], [], dst
    while node != src:
        node, way = prev[node]
        path.append(node)
        ways.append(way)
    path.reverse()
    ways.reverse()
    return best[dst], path, list(dict.fromkeys(ways))


def find_landmark(elements: list[dict], name: str) -> tuple[float, float]:
    """Centre of the first OSM way whose name matches (case-insensitive), e.g. a roundabout."""
    for w in elements:
        if w["tags"]["name"].lower() == name.lower():
            g = w["geometry"]
            return sum(p["lat"] for p in g) / len(g), sum(p["lon"] for p in g) / len(g)
    raise ValueError(f"no OSM way named {name!r} in the loaded roads")


def build_segment(elements: list[dict], segment_id: str, name: str, direction: str,
                  start: tuple[float, float], end: tuple[float, float],
                  roads: set[str] | None = None) -> tuple[dict, list[tuple[float, float]]]:
    """Return (segments.csv row, path as [(lon, lat), ...]) for a directed path."""
    adj, coords = build_graph(elements, roads)
    src, dst = nearest_node(coords, *start), nearest_node(coords, *end)
    if src == dst:
        raise ValueError("start and end snap to the same point")
    metres, nodes, ways = shortest_path(adj, src, dst)
    row = {"segment_id": segment_id, "name": name, "direction": direction,
           "start_lat": round(coords[src][0], 6), "start_lon": round(coords[src][1], 6),
           "end_lat": round(coords[dst][0], 6), "end_lon": round(coords[dst][1], 6),
           "length_m": round(metres, 1),
           "notes": f"osm_ways={';'.join(map(str, ways))}; roads={','.join(sorted(roads)) if roads else 'any'}; OSM ODbL"}
    return row, [(coords[n][1], coords[n][0]) for n in nodes]


def append_segment(path: Path, row: dict) -> None:
    """Add a segment to segments.csv; refuses a duplicate segment_id."""
    path = Path(path)
    existing = pd.read_csv(path) if path.exists() and path.stat().st_size > 0 else pd.DataFrame(columns=SEGMENT_COLUMNS)
    if row["segment_id"] in set(existing["segment_id"]):
        raise ValueError(f"segment_id {row['segment_id']!r} already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([existing, pd.DataFrame([row])], ignore_index=True)[SEGMENT_COLUMNS].to_csv(path, index=False)


def write_geojson(path: Path, segment_id: str, line: list[tuple[float, float]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    feature = {"type": "Feature", "properties": {"segment_id": segment_id},
               "geometry": {"type": "LineString", "coordinates": [list(p) for p in line]}}
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [feature]}), encoding="utf-8")
