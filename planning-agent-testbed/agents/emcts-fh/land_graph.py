"""Shortest land routes on the map, computed once per game.

WeakAi asks GameMap.getRoute / Utils.findNearest for land routes that avoid impassable territories.
Ownership never changes that, so the distances are static and are precomputed here with one BFS
per territory. Passable neutral land counts as land (WeakAi's capital routes would avoid it; the
thesis map has none).
"""

from __future__ import annotations

from collections import deque
from typing import List, Optional

from triplea_fm import NULL_PLAYER

UNREACHABLE = 10**9


class LandGraph:
    def __init__(self, d):
        self.d = d
        n = len(d.territory_names)
        self.land = [t for t in range(n) if not d.is_water[t] and not d.impassable[t]]
        land = set(self.land)
        self.neighbors = [tuple(x for x in d.neighbors[t] if x in land) if t in land else ()
                          for t in range(n)]
        self.dist = [self._bfs(t) for t in range(n)]

    def _bfs(self, start: int) -> List[int]:
        dist = [UNREACHABLE] * len(self.d.territory_names)
        if start not in self.land:
            return dist
        dist[start] = 0
        q = deque([start])
        while q:
            t = q.popleft()
            for x in self.neighbors[t]:
                if dist[x] == UNREACHABLE:
                    dist[x] = dist[t] + 1
                    q.append(x)
        return dist

    def first_step(self, start: int, goal: int) -> Optional[int]:
        """Next territory on a shortest land route from start to goal (lowest id on ties)."""
        d = self.dist[goal][start]
        if d == UNREACHABLE or d == 0:
            return None
        return min(x for x in self.neighbors[start] if self.dist[goal][x] == d - 1)

    def nearest(self, start: int, goals) -> Optional[int]:
        """Closest of goals other than start itself (Utils.findNearest needs a route with steps)."""
        best = None
        for g in goals:
            d = self.dist[g][start]
            if 0 < d < UNREACHABLE and (best is None or d < self.dist[best][start]):
                best = g
        return best

    def distance_to_enemy(self, s, player: int, t: int) -> int:
        """Land distance from t to the closest territory owned by a player at war with player."""
        enemy = [x for x in self.land if s.owner[x] != NULL_PLAYER and self.d.is_at_war(player, s.owner[x])]
        return min((self.dist[e][t] for e in enemy), default=UNREACHABLE)
