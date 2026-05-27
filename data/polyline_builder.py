"""Reconstruct polylines from a flat set of 2-point chord annotations.

Why this exists
---------------
Labellers click each lateral segment as an isolated chord (two endpoints).
Long laterals are therefore represented as many short chords whose shared
endpoints almost coincide. Two consecutive clicks that *should* land on the
same vertex usually disagree by a few pixels. To get continuous polylines
back, we:

1. Treat every chord endpoint as a candidate vertex.
2. Fuse endpoints within ``merge_radius`` pixels into a single vertex placed
   at the centroid of the cluster (union-find).
3. Build an undirected graph: each chord becomes one edge between the two
   merged vertex IDs (self-loops dropped, duplicates deduplicated).
4. Walk every maximal degree-2 chain between *topology vertices* (degree
   ``!= 2``) — each such chain becomes one polyline.
5. Any edges still unvisited form pure degree-2 cycles, which we emit as
   closed polylines.

The output is a list of polylines, each a numpy array of shape ``(N, 2)`` in
``(x_col, y_row)`` pixel coordinates.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .coco_loader import Chord


@dataclass(frozen=True)
class Polyline:
    """A single polyline: ordered (x, y) points."""

    points: np.ndarray  # shape (N, 2)


# ---------------------------------------------------------------------------
# Union-Find for the endpoint merging step
# ---------------------------------------------------------------------------


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------


def build_polylines(
    chords: list[Chord],
    merge_radius: float = 10.0,
) -> list[Polyline]:
    """Reconstruct polylines from chord annotations.

    Parameters
    ----------
    chords
        List of :class:`Chord` annotations for one image.
    merge_radius
        Pixel distance threshold for fusing nearby endpoints. ``0`` disables
        merging.

    Returns
    -------
    list[Polyline]
    """

    if not chords:
        return []

    # ── Step 1: collect raw endpoints ─────────────────────────────────────
    endpoints: list[np.ndarray] = []
    chord_endpoint_ids: list[tuple[int, int]] = []
    for c in chords:
        i0 = len(endpoints)
        endpoints.append(c.p1)
        i1 = len(endpoints)
        endpoints.append(c.p2)
        chord_endpoint_ids.append((i0, i1))

    n = len(endpoints)
    pts = np.stack(endpoints, axis=0)  # (n, 2)

    # ── Step 2: union-find merge by pairwise distance ─────────────────────
    uf = _UnionFind(n)
    if merge_radius > 0 and n > 1:
        # O(n²); fine since chord counts per image are at most a few thousand.
        # If this becomes a bottleneck, swap in scipy.spatial.cKDTree.
        diffs = pts[:, None, :] - pts[None, :, :]   # (n, n, 2)
        d2 = (diffs ** 2).sum(-1)
        triu = np.triu(np.ones_like(d2, dtype=bool), k=1)
        i_idx, j_idx = np.where((d2 <= merge_radius ** 2) & triu)
        for i, j in zip(i_idx, j_idx):
            uf.union(int(i), int(j))

    # Group endpoint indices by union-find root, compute centroid per cluster.
    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        clusters[uf.find(i)].append(i)

    root_to_vid: dict[int, int] = {}
    vertex_pos: list[np.ndarray] = []
    for vid, (root, members) in enumerate(clusters.items()):
        root_to_vid[root] = vid
        vertex_pos.append(pts[members].mean(axis=0))
    V = np.stack(vertex_pos, axis=0)  # (n_vertices, 2)

    # ── Step 3: build undirected adjacency ────────────────────────────────
    adj: dict[int, set[int]] = defaultdict(set)
    edges: set[tuple[int, int]] = set()
    for i0, i1 in chord_endpoint_ids:
        u = root_to_vid[uf.find(i0)]
        v = root_to_vid[uf.find(i1)]
        if u == v:
            continue  # collapsed to a self-loop after merging
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in edges:
            continue
        edges.add((a, b))
        adj[u].add(v)
        adj[v].add(u)

    if not edges:
        return []

    # ── Step 4: walk chains rooted at topology vertices ───────────────────
    topo: set[int] = {v for v in adj if len(adj[v]) != 2}
    visited: set[tuple[int, int]] = set()
    polylines: list[Polyline] = []

    def _ek(a: int, b: int) -> tuple[int, int]:
        return (a, b) if a < b else (b, a)

    def _walk(start: int, first: int) -> list[int]:
        """Walk degree-2 from ``start`` through ``first`` until reaching a
        topology vertex, closing back on ``start``, or running out of edges.
        """
        path = [start, first]
        visited.add(_ek(start, first))
        prev, curr = start, first
        while True:
            # Stop at the next topology vertex (which may equal start in a
            # topology-loop, in which case the loop is already complete).
            if curr in topo and curr != start:
                break
            nxt = next((m for m in adj[curr] if m != prev), None)
            if nxt is None:
                break
            key = _ek(curr, nxt)
            if key in visited:
                # The next edge is already taken. If it's the closing edge of
                # a pure cycle, include it before stopping.
                if nxt == start:
                    path.append(nxt)
                break
            visited.add(key)
            path.append(nxt)
            if nxt == start:
                break  # cycle closed
            prev, curr = curr, nxt
        return path

    # Chains rooted at topology vertices.
    for u in sorted(topo):
        for v in list(adj[u]):
            if _ek(u, v) in visited:
                continue
            path = _walk(u, v)
            polylines.append(Polyline(points=V[path].copy()))

    # ── Step 5: pure degree-2 cycles (no topology vertex) ─────────────────
    while True:
        remaining = edges - visited
        if not remaining:
            break
        u, v = next(iter(remaining))
        path = _walk(u, v)
        polylines.append(Polyline(points=V[path].copy()))

    return polylines
