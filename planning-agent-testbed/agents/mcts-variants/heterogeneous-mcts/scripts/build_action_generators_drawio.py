"""Builds docs/heterogeneous-mcts-action-generators.drawio (UML: action generation per phase).

Usage: python3 scripts/build_action_generators_drawio.py docs/heterogeneous-mcts-action-generators.drawio
"""
import sys
from xml.sax.saxutils import quoteattr

OUT = sys.argv[1]

ACTION = ("rounded=1;whiteSpace=wrap;html=1;arcSize=18;fillColor=#dae8fc;strokeColor=#6c8ebf;"
          "fontSize=11;align=left;spacingLeft=8;spacingRight=6;")
RESULT = ACTION.replace("#dae8fc", "#d5e8d4").replace("#6c8ebf", "#82b366")
FORCED = ACTION.replace("#dae8fc", "#ffe6cc").replace("#6c8ebf", "#d79b00")
DECISION = "rhombus;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;fontSize=10;"
START = "ellipse;fillColor=#000000;strokeColor=#000000;html=1;"
END = "ellipse;shape=endState;fillColor=#000000;strokeColor=#000000;html=1;"
NOTE = ("shape=note;whiteSpace=wrap;html=1;size=14;fillColor=#f5f5f5;strokeColor=#999999;"
        "fontSize=10;align=left;verticalAlign=top;spacingLeft=6;spacingRight=6;spacingTop=4;")
LANE = ("swimlane;html=1;startSize=28;fillColor=#f8f8f8;strokeColor=#999999;fontStyle=1;"
        "fontSize=12;swimlaneFillColor=none;dashed=1;")
TITLE = "text;html=1;align=left;verticalAlign=top;fontSize=18;fontStyle=1;whiteSpace=wrap;"
SUB = "text;html=1;align=left;verticalAlign=top;fontSize=11;whiteSpace=wrap;fontColor=#555555;"
EDGE = "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=open;endSize=8;fontSize=10;"
DEP = EDGE + "dashed=1;"
CLS = ("swimlane;fontStyle=1;align=center;startSize=26;html=1;whiteSpace=wrap;childLayout=stackLayout;"
       "horizontal=1;horizontalStack=0;resizeParent=1;resizeParentMax=0;resizeLast=0;collapsible=1;"
       "marginBottom=0;fillColor=#dae8fc;strokeColor=#6c8ebf;swimlaneFillColor=#ffffff;")
MOD = ("rounded=1;arcSize=2;dashed=1;fillColor=none;strokeColor=#999999;verticalAlign=top;align=left;"
       "spacingLeft=10;fontStyle=1;fontSize=13;html=1;whiteSpace=wrap;")
ROW = ("text;strokeColor=none;fillColor=none;align=left;verticalAlign=top;spacingLeft=4;spacingRight=4;"
       "overflow=hidden;rotatable=0;points=[[0,0.5],[1,0.5]];portConstraint=eastwest;html=1;"
       "whiteSpace=wrap;fontSize=10;")
SEP = "line;strokeWidth=1;fillColor=none;align=left;verticalAlign=middle;spacingTop=-1;spacingLeft=3;spacingRight=3;rotatable=0;labelPosition=right;points=[];portConstraint=eastwest;"


class Page:
    def __init__(self, pid, name, w, h):
        self.pid, self.name, self.w, self.h = pid, name, w, h
        self.cells = []
        self.n = 0

    def _id(self):
        self.n += 1
        return f"{self.pid}-c{self.n}"

    def cell(self, value, style, x, y, w, h, cid=None, parent="1"):
        cid = cid or self._id()
        self.cells.append(
            f'<mxCell id="{cid}" value={quoteattr(value)} style="{style}" vertex="1" parent="{parent}">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        return cid

    def edge(self, src, dst, label="", style=EDGE, extra=""):
        cid = self._id()
        self.cells.append(
            f'<mxCell id="{cid}" value={quoteattr(label)} style="{style}{extra}" edge="1" parent="1" '
            f'source="{src}" target="{dst}"><mxGeometry relative="1" as="geometry"/></mxCell>')
        return cid

    def header(self, title, sub):
        self.cell(title, TITLE, 40, 10, self.w - 80, 30)
        self.cell(sub, SUB, 40, 42, self.w - 80, 36)

    def cls(self, title, attrs, ops, x, y, w, row=18):
        cid = self._id()
        height = 26 + row * (len(attrs) + len(ops)) + (8 if attrs and ops else 0)
        self.cell(title, CLS, x, y, w, height, cid=cid)
        yy = 26
        for a in attrs:
            self.cell(a, ROW, 0, yy, w, row, parent=cid)
            yy += row
        if attrs and ops:
            self.cell("", SEP, 0, yy, w, 8, parent=cid)
            yy += 8
        for o in ops:
            self.cell(o, ROW, 0, yy, w, row, parent=cid)
            yy += row
        return cid

    def xml(self):
        return (f'<diagram id="{self.pid}" name={quoteattr(self.name)}><mxGraphModel dx="1400" dy="900" '
                f'grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" '
                f'pageScale="1" pageWidth="{self.w}" pageHeight="{self.h}" math="0" shadow="0"><root>'
                f'<mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(self.cells) +
                '</root></mxGraphModel></diagram>')


def legend(p, x, y):
    p.cell("<b>Legend</b>", "text;html=1;fontSize=11;align=left;", x, y, 200, 20)
    p.cell("step", ACTION, x, y + 24, 120, 26)
    p.cell("actions returned (tree edges)", RESULT, x + 130, y + 24, 200, 26)
    p.cell("forced moves (not an edge)", FORCED, x, y + 58, 200, 26)
    p.cell("guard", DECISION, x + 210, y + 54, 90, 34)


# ---- 1: class diagram ----------------------------------------------------------------------------

def page_classes():
    p = Page("g-classes", "1 Action generators (classes)", 1700, 1120)
    p.header("Heterogeneous MCTS: action generators, class diagram",
             "HMCTSGameState.legal_moves() builds a node's edges from the decision its state is at. Purchase "
             "edges come from HeterogeneousMCTS.affordable_units; combat and non-combat move edges come from "
             "the generator modules, reached through generators.ag. The other search variants use copies of "
             "these generator modules.")

    p.cell("«module» generators.py", MOD, 20, 90, 380, 150)
    gen = p.cls("generators", [
        "+ ag: module  = action_generators_cy (Cython build)",
        "      or action_generators (pure Python fallback)",
        "+ IMPLEMENTATION: str",
        "TRIPLEMIND_PURE_PYTHON=1 forces the Python version",
    ], [], 35, 120, 350)

    p.cell("«module» hmcts.py", MOD, 20, 260, 380, 720)
    st = p.cls("HMCTSGameState", [
        "- sim: fm.SimulationState",
        "- current_player: str",
        "- phase: 'purchase' | 'combatMove' | 'nonCombatMove' | None",
        "- actions: [PurchaseUnit | CombatAction | NoncombatAction]",
        "- actionIndex: int  (next action expand() tries)",
        "- next_unit: int  (purchase: first candidate offered)",
        "- excluded: set  (combat / non-combat: territories decided)",
        "- moved: set  (non-combat: fm ids of units moved)",
    ], [
        "+ view(): map_graph.GameState",
        "+ legal_moves()  «dispatches on phase»",
        "+ apply_move(action)",
        "+ apply_moves([fm.MoveAction])  (moved += units)",
        "+ getNextAction(): action | None",
    ], 35, 290, 350)
    ms = p.cls("HeterogeneousMCTS  (excerpt)", [
        "- _unit_rules: {player: [PurchaseUnit]}  (cache)",
    ], [
        "+ unit_rules(player): [PurchaseUnit]",
        "+ affordable_units(sim, player, first=0): [PurchaseUnit]",
        "+ start_phase(state)  (non-combat: land_air_moves)",
        "+ finish_phase(state)  (non-combat: positioning moves)",
        "+ positioning_moves(state): [fm.MoveAction]",
        "+ expand(node) / pass_phase(state)",
        "+ _rollout_purchase / _rollout_combat",
        "+ _rollout_noncombat / _rollout_rest_of_noncombat",
        "+ play_phase(initial_state, chosen)",
    ], 35, 600, 350)
    pu = p.cls("PurchaseUnit", [
        "- rule_id: int",
        "- unit_type: str",
        "- costs: ((resource id, amount), ...)",
        "- strength: float  ((attack + defense) / cost)",
        "- index: int  (place in the candidate list)",
        "- end_phase: bool",
    ], [
        "+ action(): fm.PurchaseAction(((rule_id, 1),))",
    ], 440, 90, 330)

    p.cell("«module» action_generators.py  (and action_generators_cy.pyx)", MOD, 820, 90, 440, 560)
    ca = p.cls("CombatAction", [
        "- to_terr: str",
        "- moves: [fm.MoveAction]  ([] = skip to_terr)",
        "- strength: int",
        "- end_phase: bool",
    ], [], 440, 330, 330)
    ag_fn = p.cls("«functions» combat move", [], [
        "+ combat_actions(game, d, excluded, player)",
        "      : ([CombatAction], excluded)",
        "+ enemy_territories_by_priority(game, player)",
        "+ get_reachable_units(game, target, player)",
        "+ calculate_attack_strength(m, units)",
        "+ defender_strength(m, territory, player)",
        "+ combat_blocked(game) / _air_can_land(...)",
    ], 835, 120, 410)
    ag_sh = p.cls("«functions» shared helpers", [
        "THREAT_COVER, HOME_COVER, HOME_KEEP, FRONTIER_MIN",
    ], [
        "+ _reach_routes(game, player, start, range, domain)",
        "+ _air_routes(game, start, moves)",
        "+ _group_moves(d, [(unit, route)]): [fm.MoveAction]",
        "+ _landing_spots / _enemy_threat / _home_factory",
        "+ defense_value(u) / attack_value(u)",
        "+ noncombat_move(...)  (greedy whole phase, rollouts)",
        "+ purchase_actions(...)  (greedy, not used by hmcts)",
    ], 835, 330, 410)

    p.cell("«module» noncombat_generators.py", MOD, 820, 680, 860, 300)
    nca = p.cls("NoncombatAction", [
        "- to_terr: str",
        "- moves: [fm.MoveAction]  ([] = skip; END: positioning)",
        "- strength: float  (defence value sent)",
        "- end_phase: bool",
    ], [], 440, 520, 330)
    ncf = p.cls("«functions» non-combat move", [
        "COVER_LEVELS = (0.5, 1.0, 1.5)",
    ], [
        "+ land_air_moves(game, d, player, moved)",
        "+ noncombat_actions(game, d, excluded, moved, player)",
        "      : ([NoncombatAction], excluded)",
        "+ positioning_moves(game, d, player, moved)",
    ], 835, 710, 410)
    plan = p.cls("_Plan", [
        "- mine, frontier, landing, home_factory",
        "- strength: {territory: Σ defense_value}",
        "- want: {territory: cover wanted}",
        "- movable: {territory: [unit not in moved]}",
    ], [
        "+ destinations(u, src): {territory: route}",
        "+ shortfall(t) / priority(t)",
        "+ defense_gain(dst, v) / donor_loss(src, v)",
        "+ can_spare(src, v) / send(u, src, route)",
    ], 1270, 710, 395)

    view = p.cls("map_graph.GameState  (view)", [
        "built by GameState.from_sim; units carry fm ids",
        "- map: MapGraph, territories, players, round",
        "- conquered, blitzed",
    ], [
        "+ owned_by(owner) / owner(t)",
    ], 1300, 90, 370)

    p.edge(st, ms, "«call» purchase:<br>affordable_units", DEP)
    p.edge(ms, pu, "«create»", DEP)
    p.edge(st, ag_fn, "«call» combat move:<br>ag.combat_actions", DEP)
    p.edge(ag_fn, ca, "«create»", DEP)
    p.edge(st, ncf, "«call» non-combat move:<br>ncg.noncombat_actions", DEP)
    p.edge(ncf, nca, "«create»", DEP)
    p.edge(ncf, plan, "«use»", DEP)
    p.edge(plan, ag_sh, "«use» via ag", DEP)
    p.edge(ag_fn, ag_sh, "«use»", DEP)
    p.edge(gen, ag_fn, "ag", DEP)
    p.edge(ag_fn, view, "reads", DEP)
    p.edge(st, view, "view()", DEP, "exitX=1;exitY=0.05;")

    table = (
        '<table border="1" cellpadding="4" style="border-collapse:collapse;font-size:10px;">'
        '<tr style="background:#eeeeee;"><th>phase</th><th>generator (legal_moves)</th>'
        '<th>one tree edge</th><th>state carried between edges</th><th>nothing left</th>'
        '<th>forced moves</th><th>rollout policy</th></tr>'
        '<tr><td>purchase</td><td>HeterogeneousMCTS.affordable_units(sim, player, next_unit)</td>'
        '<td>PurchaseUnit: one unit of one production rule</td><td>next_unit = index of the last unit bought'
        '</td><td>[PurchaseUnit(end_phase)]</td><td>none</td><td>_rollout_purchase: ε-greedy over '
        'affordable_units, best (attack + defense) / PU</td></tr>'
        '<tr><td>combat move</td><td>ag.combat_actions(view, d, excluded, player)</td>'
        '<td>CombatAction: one attack set on the top target, or skip it</td><td>excluded = targets decided '
        'or unreachable</td><td>[CombatAction(end_phase)]</td><td>none (battles are fought later)</td>'
        '<td>_rollout_combat: combat_actions + ε-greedy / evaluate pick</td></tr>'
        '<tr><td>non-combat move</td><td>ncg.noncombat_actions(view, d, excluded, moved, player)</td>'
        '<td>NoncombatAction: one reinforcement set for the most needy territory, or skip it</td>'
        '<td>excluded = territories decided; moved = units moved</td><td>[NoncombatAction(end_phase, '
        'moves = positioning_moves)]</td><td>land_air_moves at start_phase; positioning moves at '
        'finish_phase</td><td>from the start: ag.noncombat_move (greedy); mid-phase: first (strongest) '
        'action per territory</td></tr></table>')
    p.cell(table, "text;html=1;align=left;verticalAlign=top;whiteSpace=wrap;overflow=fill;", 20, 1000,
           1660, 110)
    return p


# ---- 2: purchase ---------------------------------------------------------------------------------

def page_purchase():
    p = Page("g-purchase", "2 Purchase action generation", 1700, 1100)
    p.header("Purchase: activity diagram of legal_moves() at a purchase node",
             "One edge per affordable unit. Units are offered from the index of the last unit bought, so every "
             "multiset of units has exactly one path in the tree (no transpositions).")
    p.cell("HMCTSGameState  (hmcts.py)", LANE, 20, 90, 540, 990)
    p.cell("HeterogeneousMCTS  (hmcts.py)", LANE, 580, 90, 560, 990)

    s = p.cell("", START, 275, 130, 30, 30)
    a1 = p.cell("<b>legal_moves()</b>  phase == purchase<br>next_unit = index of the last unit bought "
                "(0 when the phase starts)", ACTION, 90, 185, 400, 60)
    a2 = p.cell("<b>affordable_units(sim, player, first = next_unit)</b>", ACTION, 640, 185, 440, 60)
    d1 = p.cell("player owns a factory<br>on a territory it owns?", DECISION, 750, 280, 220, 100)
    a3 = p.cell("<b>unit_rules(player)</b>  (cached per player)<br>for each rule in the player's production "
                "frontier keep it if: one result, quantity 1, paid in PUs only, cost &gt; 0, land unit, not "
                "infrastructure, not AA<br>→ PurchaseUnit(rule_id, unit_type, costs, strength = (attack + "
                "defense) / cost, index)", ACTION, 640, 420, 440, 130)
    a4 = p.cell("keep candidates[first:] whose every cost ≤ the player's balance (candidate order kept)",
                ACTION, 640, 585, 440, 55)
    d2 = p.cell("any unit<br>returned?", DECISION, 190, 680, 200, 90)
    r1 = p.cell("<b>actions</b> = the affordable units, in candidate order<br>(one tree edge each)",
                RESULT, 40, 820, 250, 70)
    r2 = p.cell("<b>actions</b> = [PurchaseUnit(end_phase=True)]<br>no_target → pass_phase", RESULT,
                310, 820, 230, 70)
    a5 = p.cell("<b>expand → apply_move(unit)</b><br>fm.PurchaseAction(((rule_id, 1),)) on the sim<br>"
                "next_unit = unit.index", ACTION, 40, 935, 250, 80)
    e = p.cell("", END, 410, 960, 30, 30)
    p.cell("END is never a tree edge: the phase ends, place and battles are played by the rollout "
           "policies, and the node becomes the next searched decision.", SUB, 310, 995, 240, 60)

    p.edge(s, a1)
    p.edge(a1, a2)
    p.edge(a2, d1)
    p.edge(d1, a3, "[yes]")
    p.edge(d1, d2, "[no] → []", extra="exitX=0;exitY=0.5;entryX=1;entryY=0.5;")
    p.edge(a3, a4)
    p.edge(a4, d2, extra="exitX=0.5;exitY=1;entryX=1;entryY=0.5;")
    p.edge(d2, r1, "[yes]")
    p.edge(d2, r2, "[no]")
    p.edge(r1, a5, "child node of the chosen edge")
    p.edge(a5, a1, "next legal_moves()", extra="exitX=0;exitY=0.5;entryX=0;entryY=0.5;")
    p.edge(r2, e)

    p.cell("<b>Why candidate order</b><br>Buying A then B and B then A would reach the same state. Offering "
           "only units at or after the last one bought keeps one path per set of units, so visits are not "
           "split over transpositions.", NOTE, 1170, 120, 500, 100)
    p.cell("<b>Outside the tree</b><br>• Rollouts: _rollout_purchase repeats affordable_units(first = 0) and "
           "picks with pick_rollout_move_epsilon_greedy (random with probability ε, else the best "
           "strength) until nothing is affordable.<br>• Answer: play_phase plays the chosen edges, then "
           "spends what is left greedily by strength.<br>• Place is never searched: _place_units puts every "
           "unit bought at the home factory (ag._home_factory).", NOTE, 1170, 240, 500, 170)
    p.cell("<b>action_generators.purchase_actions</b> is the greedy round-robin purchase of the live "
           "tripleMind agent. It uses the same candidate filter (land combat units paid in PUs) but the "
           "heterogeneous search does not call it.", NOTE, 1170, 430, 500, 90)
    legend(p, 1170, 560)
    return p


# ---- 3: combat move ------------------------------------------------------------------------------

def page_combat():
    p = Page("g-combat", "3 Combat move action generation", 1700, 1150)
    p.header("Combat move: activity diagram of legal_moves() at a combat-move node",
             "ag.combat_actions decides one target per call: the attack sets on the most important enemy "
             "territory not yet decided (strongest first), then a skip of it. The next node asks for the next "
             "target.")
    p.cell("HMCTSGameState  (hmcts.py)", LANE, 20, 90, 480, 1040)
    p.cell("action_generators.combat_actions  (via generators.ag)", LANE, 520, 90, 720, 1040)

    s = p.cell("", START, 245, 130, 30, 30)
    a1 = p.cell("<b>legal_moves()</b>  phase == combatMove<br>view() = GameState.from_sim (units carry fm ids)"
                "<br>excluded = targets already decided", ACTION, 60, 180, 400, 70)
    a2 = p.cell("<b>enemy_territories_by_priority(game, player)</b><br>enemy land territories next to mine, "
                "sorted by score: +1000 victory city, +500 per adjacent victory city, +100 × production, "
                "+50 per bordering territory of mine, −30 per neighbour with the same owner",
                ACTION, 560, 180, 560, 80)
    d1 = p.cell("next target<br>not in excluded?", DECISION, 740, 290, 200, 90)
    r0 = p.cell("return ([], excluded)", RESULT, 1000, 305, 220, 60)
    a3 = p.cell("<b>get_reachable_units(game, target, player)</b><br>my unmoved, mobile, non-sea units in "
                "my territories. _reach_routes (BFS) passes only through my territories; land units avoid "
                "territories conquered this turn without blitz. Air only if _air_can_land after the attack.",
                ACTION, 560, 410, 560, 85)
    d2 = p.cell("any attacker<br>reaches it?", DECISION, 740, 525, 200, 80)
    a4 = p.cell("excluded.add(target)<br>(pass over)", ACTION, 1000, 535, 220, 60)
    a5 = p.cell("sort attackers by donor_score: +100 if next to the target, +10 × attack<br>"
                "def = defender_strength: Σ defense of enemy non-infrastructure units", ACTION,
                560, 635, 560, 60)
    a6 = p.cell("<b>incremental buildup</b>: add attackers one at a time; strength = "
                "calculate_attack_strength (each supportable infantry paired with an artillery gets +1). "
                "Record the set when threshold × def &lt; strength &lt; 3.5 × def + 4 "
                "(threshold starts at 1.5, +0.1 after each set).", ACTION, 560, 720, 560, 80)
    a7 = p.cell("add the full force if its strength &gt; def and it differs from the last set<br>"
                "no set and def == 0: the cheapest single reachable unit (capture)", ACTION,
                560, 825, 560, 60)
    r1 = p.cell("<b>return</b> (actions, excluded ∪ {target})<br>actions = sets strongest first, as "
                "CombatAction(to_terr, moves = _group_moves: one MoveAction per route, strength), "
                "then skip(target)", RESULT, 560, 910, 560, 75)
    r2 = p.cell("<b>actions</b> = [CombatAction(end_phase=True)]<br>no_target → pass_phase", RESULT,
                60, 300, 400, 55)
    e = p.cell("", END, 245, 380, 30, 30)
    r3 = p.cell("<b>actions</b> = returned list (one tree edge each)", RESULT, 60, 920, 400, 50)
    a8 = p.cell("<b>expand → apply_move(attack)</b><br>apply each MoveAction with the forward model "
                "(no dice; the battles are fought in the battle phase by fast_battle). A skip applies "
                "nothing.", ACTION, 60, 1000, 400, 80)

    p.edge(s, a1)
    p.edge(a1, a2)
    p.edge(a2, d1)
    p.edge(d1, r0, "[none left]")
    p.edge(r0, r2, extra="exitX=0.5;exitY=0;entryX=1;entryY=0.25;")
    p.edge(r2, e)
    p.edge(d1, a3, "[target]")
    p.edge(a3, d2)
    p.edge(d2, a4, "[no]")
    p.edge(a4, d1, "loop", extra="exitX=0.5;exitY=0;entryX=0.5;entryY=1;")
    p.edge(d2, a5, "[yes]")
    p.edge(a5, a6)
    p.edge(a6, a7)
    p.edge(a7, r1)
    p.edge(r1, r3)
    p.edge(r3, a8, "child node of the chosen edge")
    p.edge(a8, a1, "next legal_moves() (next target)", extra="exitX=0;exitY=0.5;entryX=0;entryY=0.5;")

    p.cell("<b>Branching</b><br>Each combat node has the attack sets on one target plus a skip, so a path "
           "through the phase decides the targets one at a time in priority order. A skipped target goes "
           "into excluded and is not offered again in this phase.", NOTE, 1270, 120, 410, 110)
    p.cell("<b>Outside the tree</b><br>• Rollouts: _rollout_combat calls combat_actions in a loop from the "
           "node's excluded set and picks with pick_combat_rollout_move (ε-greedy on strength, or "
           "evaluate_state of each option, set by TRIPLEA_MCTS_COMBAT_ROLLOUT).<br>• Answer: play_phase plays the "
           "moves of the chosen non-skip edges.", NOTE, 1270, 250, 410, 140)
    p.cell("<b>Cython</b><br>generators.ag is action_generators_cy when it is built "
           "(scripts/build_cython.sh), else action_generators.py. Both return the same actions.",
           NOTE, 1270, 410, 410, 80)
    legend(p, 1270, 520)
    return p


# ---- 4: non-combat move --------------------------------------------------------------------------

def page_noncombat():
    p = Page("g-noncombat", "4 Non-combat move action generation", 1750, 1200)
    p.header("Non-combat move: activity diagram from start_phase to finish_phase",
             "The greedy noncombat_move split in three: (1) forced air landings when the phase starts, (2) a "
             "search over reinforcements, one territory short of cover per node, (3) positioning moves as "
             "the phase ends.")
    p.cell("HeterogeneousMCTS / HMCTSGameState  (hmcts.py)", LANE, 20, 90, 500, 1090)
    p.cell("noncombat_generators  (helpers from generators.ag)", LANE, 540, 90, 740, 1090)

    s = p.cell("", START, 255, 130, 30, 30)
    a1 = p.cell("<b>start_phase(state)</b><br>excluded = {}, moved = {}", ACTION, 70, 180, 400, 55)
    f1 = p.cell("<b>land_air_moves(game, d, player)</b>  step 1<br>air that cannot end the phase where it "
                "stands flies to the reachable landing spot (_landing_spots) most short of cover; air with "
                "no spot is left to be lost, as in the engine", FORCED, 580, 175, 660, 70)
    a2 = p.cell("apply_moves(forced moves): moved += their units", FORCED, 70, 265, 400, 45)
    a3 = p.cell("<b>legal_moves()</b>  phase == nonCombatMove", ACTION, 70, 345, 400, 45)
    a4 = p.cell("<b>_Plan(game, d, player, moved)</b><br>strength[t] = Σ defense_value(u) (1 + 0.5 × defense)"
                "<br>want[t] = THREAT_COVER × _enemy_threat[t]; home factory: max(HOME_COVER × threat, "
                "HOME_KEEP × army); frontier: at least FRONTIER_MIN<br>movable = combat units not in moved, "
                "movement left, land or air", ACTION, 580, 300, 660, 100)
    a5 = p.cell("targets = my territories not in excluded with shortfall = want − strength &gt; 0, sorted "
                "by priority × shortfall (priority: home factory ×1.5, more production higher)", ACTION,
                580, 425, 660, 55)
    d1 = p.cell("next target?", DECISION, 810, 505, 200, 80)
    r0 = p.cell("<b>return</b> [NoncombatAction(end_phase=True, moves = positioning_moves(...))]",
                RESULT, 1050, 510, 210, 70)
    a6 = p.cell("donors: movable units in other territories that can end on the target (land: "
                "_reach_routes through my territories; air: _air_routes to a landing spot of mine)<br>"
                "score = defense_gain(target) − donor_loss(source), best first", ACTION, 580, 615, 660, 70)
    a7 = p.cell("<b>incremental buildup</b>: skip a unit whose source cannot spare it (can_spare); record a "
                "set each time the value sent reaches the next COVER_LEVELS share (0.5, 1.0, 1.5) of the "
                "shortfall; add 'everything that can be spared' if it differs from the last set.<br>"
                "excluded.add(target)", ACTION, 580, 710, 660, 85)
    d2 = p.cell("any set?", DECISION, 810, 820, 200, 70)
    r1 = p.cell("<b>return</b> (actions, excluded)<br>actions = sets strongest first, as NoncombatAction("
                "to_terr, moves = _group_moves, strength = value sent), then skip(target)", RESULT,
                580, 920, 660, 60)
    r2 = p.cell("<b>actions</b> = returned list (one tree edge each)", RESULT, 70, 925, 400, 50)
    a8 = p.cell("<b>expand → apply_move(reinforcement)</b><br>apply_moves(action.moves): moved += units",
                ACTION, 70, 1010, 400, 60)
    r3 = p.cell("<b>actions</b> = [END]  → no_target → pass_phase", RESULT, 70, 510, 400, 50)
    a9 = p.cell("<b>finish_phase(state)</b><br>apply the END action's positioning moves (step 3), "
                "then EndPhase", FORCED, 70, 590, 400, 60)
    e = p.cell("", END, 255, 680, 30, 30)

    p.edge(s, a1)
    p.edge(a1, f1)
    p.edge(f1, a2, extra="exitX=0.5;exitY=1;entryX=1;entryY=0.5;")
    p.edge(a2, a3)
    p.edge(a3, a4, extra="exitX=1;exitY=0.5;entryX=0;entryY=0.5;")
    p.edge(a4, a5)
    p.edge(a5, d1)
    p.edge(d1, r0, "[none left]")
    p.edge(r0, r3, extra="exitX=0.5;exitY=0;entryX=1;entryY=0.5;")
    p.edge(r3, a9)
    p.edge(a9, e)
    p.edge(d1, a6, "[target]")
    p.edge(a6, a7)
    p.edge(a7, d2)
    p.edge(d2, d1, "[no] pass over", extra="exitX=1;exitY=0.5;entryX=1;entryY=0.5;")
    p.edge(d2, r1, "[yes]")
    p.edge(r1, r2)
    p.edge(r2, a8, "child node of the chosen edge")
    p.edge(a8, a3, "next legal_moves()", extra="exitX=0;exitY=0.5;entryX=0;entryY=0.5;")

    p.cell("<b>positioning_moves(game, d, player, moved)</b>  step 3<br>Sources farthest from the frontier "
           "first. Each unit its source can spare goes to the reachable territory of mine with the best "
           "score = defense_gain + advance toward the frontier + 0.2 × max(0, 5 − distance to an "
           "objective) − donor_loss, if the score is &gt; 0 and the move gets closer to the frontier.",
           NOTE, 1310, 120, 420, 150)
    p.cell("<b>Outside the tree</b><br>• Rollout from the start of a phase: _rollout_noncombat plays the "
           "greedy ag.noncombat_move (the same three steps planned at once).<br>• Rollout of the rest of a "
           "phase the tree started: _rollout_rest_of_noncombat takes the first (strongest) action at each "
           "territory, then finish_phase.<br>• Answer: play_phase plays the chosen edges' moves, then "
           "positioning_moves.", NOTE, 1310, 290, 420, 170)
    p.cell("<b>Same fields as CombatAction</b><br>NoncombatAction has to_terr, moves, strength and "
           "end_phase, so expand, the metrics and the game records treat both move phases alike.",
           NOTE, 1310, 480, 420, 80)
    legend(p, 1310, 590)
    return p


pages = [page_classes(), page_purchase(), page_combat(), page_noncombat()]
with open(OUT, "w") as f:
    f.write('<mxfile host="app.diagrams.net" type="device">' + "".join(pg.xml() for pg in pages)
            + "</mxfile>")
