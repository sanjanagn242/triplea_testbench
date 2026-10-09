"""Map-agnostic graph view of a TripleA game, for adjacency, reachability and distance queries.

Three layers:
  MapGraph   static, built once from the `gameXml` sent in `game_start`: territories are networkx
             nodes, <connection> elements are edges, territory/unit-type attachments are attributes.
  GameState  dynamic, rebuilt from each request's `state` observation: owners, units, PUs.
  MapView    optional live matplotlib view of a GameState.

Movement domains: "land" excludes water and impassable territories, "sea" is water only, "air" is
everything passable. Pass `domain=None` for the raw map graph.

Never print to stdout here: stdout carries the agent protocol.
"""

import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import networkx as nx

HERE = Path(__file__).resolve().parent
MAP_SEARCH_ROOTS = (HERE.parent.parent / "games", Path.home() / "triplea" / "downloadedMaps")

DOMAINS = ("land", "sea", "air")


# ---- static map ------------------------------------------------------------------------------

class UnitType:
    def __init__(self, name, options):
        self.name = name
        self.movement = int(options.get("movement", 0))
        self.attack = int(options.get("attack", 0))
        self.defense = int(options.get("defense", 0))
        self.is_air = _flag(options, "isAir")
        self.is_sea = _flag(options, "isSea")
        self.is_infrastructure = _flag(options, "isFactory") or _flag(options, "isInfrastructure")
        self.can_produce = _flag(options, "isFactory") or _flag(options, "canProduceUnits")
        self.is_aa = _flag(options, "isAA") or _flag(options, "isAAforCombatOnly")
        self.is_infantry = _flag(options, "isInfantry")
        self.can_blitz = _flag(options, "canBlitz")
        self.is_artillery = _flag(options, "artillery")
        self.artillery_supportable = _flag(options, "artillerySupportable")
        self.options = options

    @property
    def domain(self):
        return "air" if self.is_air else "sea" if self.is_sea else "land"

    def __repr__(self):
        return f"UnitType({self.name}, move={self.movement}, {self.domain})"


class ProductionRule:
    def __init__(self, name, costs, results):
        self.name = name
        self.costs = costs  # {resource: quantity}
        self.results = results  # {unit type or resource: quantity}

    @property
    def cost(self):
        return self.costs.get("PUs", 0)

    def __repr__(self):
        return f"ProductionRule({self.name}, {self.costs} -> {self.results})"


class MapGraph:
    """Static map: territories, adjacency and unit types. Built once per game."""

    def __init__(self, name=""):
        self.name = name
        self.G = nx.Graph()
        self.players = []
        self.unit_types = {}
        self.production_rules = {}
        self.frontiers = {}  # {player: [rule names]}
        self._views = {}
        self._dist = {}

    @classmethod
    def from_xml(cls, xml_text):
        root = ET.fromstring(xml_text.encode("utf-8") if isinstance(xml_text, str) else xml_text)
        props = {p.get("name"): p.get("value") for p in root.iter("property")}
        info = root.find("info")
        m = cls(props.get("mapName") or (info.get("name") if info is not None else ""))

        for t in root.iter("territory"):
            m.G.add_node(t.get("name"), water=t.get("water") == "true", impassable=False,
                         production=0, capital=None, victory_city=0)
        for c in root.iter("connection"):
            m.G.add_edge(c.get("t1"), c.get("t2"))
        m.players = [p.get("name") for p in root.iter("player")]

        for a in root.iter("attachment"):
            options = {o.get("name"): o.get("value") for o in a.iter("option")}
            target = a.get("attachTo")
            if a.get("type") == "territory" and target in m.G:
                node = m.G.nodes[target]
                node["production"] = int(options.get("production", 0))
                node["capital"] = options.get("capital")
                node["victory_city"] = int(options.get("victoryCity", 0))
                node["impassable"] = _flag(options, "isImpassable")
            elif a.get("type") == "unitType":
                m.unit_types[target] = UnitType(target, options)

        for r in root.iter("productionRule"):
            m.production_rules[r.get("name")] = ProductionRule(
                r.get("name"),
                {c.get("resource"): int(c.get("quantity")) for c in r.iter("cost")},
                {x.get("resourceOrUnit"): int(x.get("quantity")) for x in r.iter("result")})
        frontier_rules = {f.get("name"): [fr.get("name") for fr in f.iter("frontierRules")]
                          for f in root.iter("productionFrontier")}
        for pp in root.iter("playerProduction"):
            m.frontiers[pp.get("player")] = frontier_rules.get(pp.get("frontier"), [])
        return m

    @classmethod
    def from_observation(cls, state):
        """Fallback when no XML is available: the observation carries neighbors and flags too."""
        m = cls(state.get("mapName", ""))
        for t in state["territories"]:
            m.G.add_node(t["name"], water=t.get("water", False),
                         impassable=t.get("impassable", False),
                         production=t.get("production", 0), capital=t.get("capitalFor"),
                         victory_city=t.get("victoryCityValue", 0))
        for t in state["territories"]:
            for n in t.get("neighbors", []):
                m.G.add_edge(t["name"], n)
        m.players = [p["name"] for p in state["players"]]
        return m

    # -- territory facts --

    def territories(self):
        return list(self.G.nodes)

    def is_water(self, t):
        return self.G.nodes[t]["water"]

    def capitals(self):
        """{player: territory}"""
        return {d["capital"]: t for t, d in self.G.nodes(data=True) if d["capital"]}

    def victory_cities(self):
        return [t for t, d in self.G.nodes(data=True) if d["victory_city"]]

    def domain_of(self, unit_type):
        ut = self.unit_types.get(unit_type)
        return ut.domain if ut else "land"

    def unit_cost(self, unit_type):
        """PUs per unit from the cheapest production rule that builds it (None if not buildable)."""
        costs = [r.cost / r.results[unit_type] for r in self.production_rules.values()
                 if unit_type in r.results and r.results[unit_type] > 0]
        return min(costs) if costs else None

    def purchasable(self, player):
        """[ProductionRule] in the player's frontier."""
        return [self.production_rules[r] for r in self.frontiers.get(player, [])
                if r in self.production_rules]

    # -- graph queries --

    def view(self, domain=None):
        """Subgraph a unit of `domain` may move through (read-only, cached)."""
        if domain is None:
            return self.G
        if domain not in self._views:
            nodes = self.G.nodes
            keep = {
                "land": lambda t: not nodes[t]["water"] and not nodes[t]["impassable"],
                "sea": lambda t: nodes[t]["water"],
                "air": lambda t: not nodes[t]["impassable"],
            }[domain]
            self._views[domain] = nx.subgraph_view(self.G, filter_node=keep)
        return self._views[domain]

    def _graph(self, domain, avoid):
        g = self.view(domain)
        if avoid:
            avoid = set(avoid)
            g = nx.subgraph_view(g, filter_node=lambda t: t not in avoid)
        return g

    def neighbors(self, t, domain=None):
        g = self.view(domain)
        return list(g.neighbors(t)) if t in g else []

    def distance(self, src, dst, domain=None, avoid=()):
        """Fewest moves from src to dst, or None if unreachable. `avoid` nodes are impassable
        (src and dst are always allowed)."""
        if not avoid:
            return self._all_pairs(domain).get(src, {}).get(dst)
        g = self._graph(domain, set(avoid) - {src, dst})
        try:
            return nx.shortest_path_length(g, src, dst)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def path(self, src, dst, domain=None, avoid=()):
        """Shortest route [src, ..., dst] (usable as a move's `route`), or None."""
        g = self._graph(domain, set(avoid) - {src, dst})
        try:
            return nx.shortest_path(g, src, dst)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def reachable(self, src, steps, domain=None, avoid=()):
        """{territory: distance} within `steps` moves of src, src included at 0."""
        g = self._graph(domain, set(avoid) - {src})
        if src not in g:
            return {}
        return nx.single_source_shortest_path_length(g, src, cutoff=steps)

    def distance_from_set(self, sources, domain=None):
        """{territory: distance to the nearest of `sources`} (multi-source BFS)."""
        g = self.view(domain)
        sources = [s for s in sources if s in g]
        if not sources:
            return {}
        return nx.multi_source_dijkstra_path_length(g, sources)

    def objectives(self):
        """Highest-production territories: the map's natural targets (e.g. the CTF flag)."""
        best = max((d["production"] for _, d in self.G.nodes(data=True)), default=0)
        return [t for t, d in self.G.nodes(data=True) if best and d["production"] == best]

    def _all_pairs(self, domain):
        # Precomputed once per domain; maps have at most a few hundred territories.
        if domain not in self._dist:
            self._dist[domain] = dict(nx.all_pairs_shortest_path_length(self.view(domain)))
        return self._dist[domain]

    def __repr__(self):
        return (f"MapGraph({self.name}, {self.G.number_of_nodes()} territories, "
                f"{self.G.number_of_edges()} connections)")


# ---- dynamic state ---------------------------------------------------------------------------

class Unit:
    def __init__(self, unit_id, unit_type, owner, movement=1, movement_left=1, attack=0,
                 defense=0, hits=0, properties=None, uid=None):
        self.id = unit_id
        self.uid = uid  # forward-model unit id (set when built from a SimulationState)
        self.unit_type = unit_type
        self.owner = owner
        self.range = movement
        self.movement_left = movement_left
        self.attack = attack
        self.defense = defense
        self.hits = hits
        self.moved = movement_left < movement  # Track if unit has moved this turn
        self.properties = properties if properties is not None else {}

    @classmethod
    def from_json(cls, u):
        return cls(u["id"], u["type"], u.get("owner") or "Neutral",
                   movement=int(u.get("movement", 1)),
                   movement_left=int(float(u.get("movementLeft", u.get("movement", 1)))),
                   attack=u.get("attack", 0), defense=u.get("defense", 0), hits=u.get("hits", 0),
                   properties=u)

    @property
    def domain(self):
        p = self.properties
        return "air" if p.get("air") else "sea" if p.get("sea") else "land"

    def __repr__(self):
        return f"Unit({self.unit_type}, {self.owner}, id={self.id[:8]})"


class Territory:
    def __init__(self, name, owner="Neutral"):
        self.name = name
        self.owner = owner
        self.units = []

    def unit_counts(self):
        """{(owner, unit_type): count}, for labels and quick strength checks."""
        return Counter((u.owner, u.unit_type) for u in self.units)

    def __repr__(self):
        return f"Territory({self.name}, owner={self.owner}, units={len(self.units)})"


class Player:
    def __init__(self, name, pu, capital=None):
        self.name = name
        self.PU = pu
        self.capital = capital
        self.unplaced = {}  # {unit_type: quantity}

    def __repr__(self):
        return f"Player({self.name}, PU={self.PU})"


class GameState:
    """One observation laid over a MapGraph."""

    def __init__(self, map_graph, state, whoAmI):
        self.map = map_graph
        self.whoAmI = whoAmI
        self.round = state.get("round", 0)
        self.step = state.get("step", "")
        self.active_player = state.get("activePlayer")
        # Turn history the observation does not carry (empty at the start of a turn).
        self.conquered = set()  # territories conquered this turn
        self.blitzed = set()  # ...of which conquered by blitzing through

        self.territories = {}
        for t in state["territories"]:
            territory = Territory(t["name"], t.get("owner") or "Neutral")
            territory.units = [Unit.from_json(u) for u in t.get("units", [])]
            self.territories[territory.name] = territory

        capitals = map_graph.capitals()
        self.players = {}
        for p in state["players"]:
            player = Player(p["name"], p.get("resources", {}).get("PUs", 0),
                            capitals.get(p["name"]))
            player.unplaced = dict(Counter(_unit_type(u) for u in p.get("unitsInReserve", [])))
            self.players[player.name] = player
        self.current_player = self.players.get(whoAmI)

    @classmethod
    def from_sim(cls, map_graph, definition, sim, whoAmI):
        """GameState view of a triplea_fm SimulationState (live bridge state or a rollout state).
        Units carry `uid`, the forward-model id that MoveAction/PlaceAction expect."""
        from triplea_fm import rules
        from triplea_fm.definition import NULL_PLAYER

        d = definition
        g = cls.__new__(cls)
        g.map = map_graph
        g.whoAmI = whoAmI
        g.round = sim.round
        g.step = d.steps[sim.step].name
        g.active_player = (d.player_names[sim.decision.actor]
                           if sim.decision is not None else None)

        def pname(o):
            return "Neutral" if o == NULL_PLAYER else d.player_names[o]

        g.territories = {name: Territory(name, pname(sim.owner[i]))
                         for i, name in enumerate(d.territory_names)}
        capitals = map_graph.capitals()
        g.players = {name: Player(name, sim.balances[i][d.treasury_resource], capitals.get(name))
                     for i, name in enumerate(d.player_names)}
        for u in sim.units.values():
            ut = d.unit_types[u.type]
            if u.loc < 0:
                if u.owner != NULL_PLAYER:
                    unplaced = g.players[d.player_names[u.owner]].unplaced
                    unplaced[ut.name] = unplaced.get(ut.name, 0) + 1
                continue
            g.territories[d.territory_names[u.loc]].units.append(
                Unit(str(u.id), ut.name, pname(u.owner), movement=ut.movement,
                     movement_left=rules.movement_left(d, u), attack=ut.attack,
                     defense=ut.defense, hits=u.hits, uid=u.id,
                     properties={"air": ut.is_air, "sea": ut.domain == 2,
                                 "infrastructure": ut.is_infrastructure}))
        g.conquered = {d.territory_names[t] for t in sim.conquered}
        g.blitzed = {d.territory_names[t] for t in sim.blitzed}
        g.current_player = g.players.get(whoAmI)
        return g

    def owner(self, t):
        return self.territories[t].owner

    def owned_by(self, owner=None):
        owner = owner or self.whoAmI
        return [t for t in self.territories.values() if t.owner == owner]

    def units_of(self, owner=None):
        """[(territory_name, unit)] for every unit `owner` (default: me) has on the map."""
        owner = owner or self.whoAmI
        return [(t.name, u) for t in self.territories.values() for u in t.units if u.owner == owner]

    def enemy_occupied(self, owner=None):
        """Territories holding units of anyone other than `owner` (default: me)."""
        owner = owner or self.whoAmI
        return {t.name for t in self.territories.values()
                if any(u.owner != owner for u in t.units)}

    def frontier(self, owner=None):
        """Territories owned by `owner` (default: me) that border someone else's land."""
        owner = owner or self.whoAmI
        return [t for t in self.owned_by(owner)
                if any(self.territories[n].owner != owner
                       for n in self.map.neighbors(t.name, "land"))]


# ---- visualization ---------------------------------------------------------------------------

def find_centers(map_name, roots=MAP_SEARCH_ROOTS):
    """Path to the map's centers.txt, searching testbench games and downloaded maps."""
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.glob(f"**/{map_name}/**/centers.txt")):
            return path
    return None


def load_centers(path):
    """Territory centers from a centers.txt, flipped so north is up in matplotlib."""
    pos = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if "(" not in line:
            continue
        name, coords = line.split("(", 1)
        x, y = coords.split(")", 1)[0].split(",")[:2]
        pos[name.strip()] = (float(x), -float(y))
    return pos


class MapView:
    """Live matplotlib window that redraws a GameState without blocking the agent."""

    NEUTRAL = "lightgray"
    WATER = "#cfe8ff"

    def __init__(self, map_graph, whoAmI, centers_path=None):
        self.map = map_graph
        self.whoAmI = whoAmI
        centers_path = centers_path or find_centers(map_graph.name)
        centers = load_centers(centers_path) if centers_path else {}
        if centers and all(t in centers for t in map_graph.G):
            self.pos = centers
        else:
            self.pos = nx.kamada_kawai_layout(map_graph.G)
        palette = ["#7f7f7f", "#d62728", "#2ca02c", "#ff7f0e", "#1f77b4", "#9467bd", "#8c564b",
                   "#e377c2", "#bcbd22", "#17becf"]
        self.owner_colors = {p: palette[i % len(palette)] for i, p in enumerate(map_graph.players)}
        self.fig = None

    def _init_figure(self):
        import matplotlib.pyplot as plt

        G = self.map.G
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(12, 12))
        nx.draw_networkx_edges(G, self.pos, ax=self.ax, edge_color="#999999")
        self.node_collection = nx.draw_networkx_nodes(G, self.pos, ax=self.ax, node_size=900)
        xs = [x for x, _ in self.pos.values()]
        ys = [y for _, y in self.pos.values()]
        self._dx = (max(xs) - min(xs)) * 0.03
        dy = (max(ys) - min(ys)) * 0.035
        name_pos = {n: (x, y - dy) for n, (x, y) in self.pos.items()}
        nx.draw_networkx_labels(G, name_pos, ax=self.ax, font_size=7)
        self.ax.set_axis_off()
        self.fig.subplots_adjust(right=0.75)
        self.label_texts = []
        self.resource_text_box = None

    def _fill(self, game, node):
        owner = game.territories[node].owner
        if owner in self.owner_colors:
            return self.owner_colors[owner]
        return self.WATER if self.map.is_water(node) else self.NEUTRAL

    def draw(self, game):
        import matplotlib.pyplot as plt

        if self.fig is None:
            self._init_figure()
        nodes = list(self.map.G.nodes)

        self.node_collection.set_facecolor([self._fill(game, n) for n in nodes])
        self.node_collection.set_edgecolor(
            ["gold" if game.territories[n].owner == self.whoAmI else "black" for n in nodes])
        self.node_collection.set_linewidth(2.0)

        for txt in self.label_texts:
            txt.remove()
        self.label_texts = []
        for n in nodes:
            counts = game.territories[n].unit_counts()
            if counts:
                x, y = self.pos[n]
                label = "\n".join(f"{qty} {utype} ({owner})"
                                  for (owner, utype), qty in counts.items())
                self.label_texts.append(
                    self.ax.text(x + self._dx, y, label, fontsize=7, ha="left", va="center"))

        if self.resource_text_box is not None:
            self.resource_text_box.remove()
        lines = []
        for player in game.players.values():
            line = f"{player.name}: {player.PU} PUs"
            if player.unplaced:
                units_str = ", ".join(f"{qty} {utype}" for utype, qty in player.unplaced.items())
                line += f" | Unplaced: {units_str}"
            lines.append(line)
        self.resource_text_box = self.ax.text(
            1.02, 0.5, "\n".join(lines), transform=self.ax.transAxes, fontsize=11,
            verticalalignment="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", edgecolor="black"))
        self.ax.set_title(f"{self.whoAmI} view - {self.map.name}, round {game.round}, {game.step}")

        self.fig.canvas.draw_idle()
        plt.pause(0.001)  # let the GUI event loop repaint without blocking the agent

    def save(self, game, path):
        self.draw(game)
        self.fig.savefig(path, bbox_inches="tight")


def _flag(options, key):
    return str(options.get(key, "false")).lower() == "true"


def _unit_type(u):
    return (u.get("type") or u.get("unitType")) if isinstance(u, dict) else str(u)


if __name__ == "__main__":
    # Replay an agent log to watch the game: python map_graph.py ../../logs/agent-Germans-game-1.txt
    import json
    import sys

    import matplotlib.pyplot as plt

    m = view = None
    for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
        if "request: " not in line:
            continue
        request = json.loads(line.split("request: ", 1)[1])
        if request.get("type") == "game_start" and m is None:
            m = MapGraph.from_xml(request["gameXml"])
            view = MapView(m, request["playerName"])
        elif request.get("type") == "turn_request" and view is not None:
            view.draw(GameState(m, request["state"], request["playerName"]))
            plt.pause(float(sys.argv[2]) if len(sys.argv) > 2 else 0.5)
    plt.ioff()
    plt.show()
