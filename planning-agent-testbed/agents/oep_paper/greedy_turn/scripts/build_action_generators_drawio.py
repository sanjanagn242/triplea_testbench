"""Builds docs/greedy-turn-action-generators.drawio (UML: move-sequence generation per phase).

Usage: python3 scripts/build_action_generators_drawio.py docs/greedy-turn-action-generators.drawio
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
GEN = "endArrow=block;endSize=12;endFill=0;html=1;edgeStyle=orthogonalEdgeStyle;rounded=0;"
CLS = ("swimlane;fontStyle=1;align=center;startSize=26;html=1;whiteSpace=wrap;childLayout=stackLayout;"
       "horizontal=1;horizontalStack=0;resizeParent=1;resizeParentMax=0;resizeLast=0;collapsible=1;"
       "marginBottom=0;fillColor=#dae8fc;strokeColor=#6c8ebf;swimlaneFillColor=#ffffff;")
CLS_OWN = CLS.replace("#dae8fc", "#e1d5e7").replace("#6c8ebf", "#9673a6")
MOD = ("rounded=1;arcSize=2;dashed=1;fillColor=none;strokeColor=#999999;verticalAlign=top;align=left;"
       "spacingLeft=10;fontStyle=1;fontSize=13;html=1;whiteSpace=wrap;")
SHARED = ("rounded=1;arcSize=1;fillColor=#fafafa;strokeColor=#6c8ebf;strokeWidth=2;verticalAlign=top;"
          "align=left;spacingLeft=10;fontStyle=1;fontSize=13;html=1;whiteSpace=wrap;")
ROW = ("text;strokeColor=none;fillColor=none;align=left;verticalAlign=top;spacingLeft=4;spacingRight=4;"
       "overflow=hidden;rotatable=0;points=[[0,0.5],[1,0.5]];portConstraint=eastwest;html=1;"
       "whiteSpace=wrap;fontSize=10;")
SEP = ("line;strokeWidth=1;fillColor=none;align=left;verticalAlign=middle;spacingTop=-1;spacingLeft=3;"
       "spacingRight=3;rotatable=0;labelPosition=right;points=[];portConstraint=eastwest;")

SHARED_BY = "greedy_action, greedy_turn, vanilla_mcts, ne_mcts, bb_mcts, oep"


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

    def cls(self, title, attrs, ops, x, y, w, row=18, style=CLS):
        cid = self._id()
        height = 26 + row * (len(attrs) + len(ops)) + (8 if attrs and ops else 0)
        self.cell(title, style, x, y, w, height, cid=cid)
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
    p.cell("actions returned (DFS children)", RESULT, x + 130, y + 24, 210, 26)
    p.cell("forced moves (not an action)", FORCED, x, y + 58, 200, 26)
    p.cell("guard", DECISION, x + 210, y + 54, 90, 34)


# ---- 1: modules and classes ----------------------------------------------------------------------

def page_classes():
    p = Page("q-classes", "1 Modules & classes", 1760, 1260)
    p.header("Greedy Turn: move-sequence generation, modules and classes",
             "The actions are generated by phases.py and the generator modules it calls. These files are "
             f"byte-identical copies in every all-phase OEP agent ({SHARED_BY}); ../check-shared-files.sh "
             "fails when they drift. Greedy Turn (greedy_turn.py) and Greedy Action (greedy_action/greedy.py) "
             "differ only in how they select among the actions.")

    p.cell(f"Shared modules: identical in {SHARED_BY}", SHARED, 400, 90, 1340, 900)

    # phases.py
    p.cell("«module» phases.py", MOD, 420, 125, 640, 850)
    ps = p.cls("PhaseState", [
        "- sim: fm.SimulationState",
        "- depth: int  (actions taken in the phase)",
        "- info: immutable phase bookkeeping",
    ], ["+ clone(): PhaseState"], 435, 155, 300)
    ph = p.cls("«abstract» Phase", [
        "- search  (provides d, map, me, whoAmI, view, apply, move_rng)",
        "- name, step_kind, end",
    ], [
        "+ start(sim): (PhaseState, forced moves)",
        "+ actions(state): [action]  «highest-rated first»",
        "+ step(state, action, rng): bool",
        "+ key(state)  «transposition key»",
        "+ action_key(action)",
        "+ end_of(state): END action",
        "+ finish(state, end, rng): bool",
        "+ levels_left(state, actions): int",
        "+ response_actions(sequence): [fm action]",
        "+ label(action): str",
    ], 750, 155, 295)
    pp = p.cls("PurchasePhase", [
        "info: ((rule id, units bought), ...)",
        "- _units: [PurchaseUnit]  (cached)",
    ], ["+ units(): candidates, best strength first"], 435, 480, 300)
    cp = p.cls("CombatPhase", [
        "info: frozenset(territories attacked)",
    ], ["ROLLOUT_STRENGTH_MARGIN = 1.5"], 435, 600, 300)
    np_ = p.cls("NoncombatPhase", [
        "info: (frozenset(decided), frozenset(unit ids moved))",
    ], ["+ end_of(state): END + positioning_moves"], 435, 700, 300)
    pu = p.cls("PurchaseUnit", [
        "- rule_id, unit_type, costs",
        "- strength: (attack + defense) / cost",
        "- end_phase: bool",
    ], [], 750, 480, 295)
    fn = p.cls("«functions»", [], [
        "+ make_phases(search): {name: Phase}",
        "+ complete_greedily(phase, state, rng)",
        "      best-rated action until END",
    ], 750, 610, 295)
    p.cell("END_KEY = \"END\"", SUB, 750, 700, 295, 20)

    # action_generators
    p.cell("«module» action_generators.py  (+ action_generators_cy.pyx)", MOD, 1080, 125, 645, 430)
    ca = p.cls("CombatAction", [
        "- to_terr: str",
        "- moves: [fm.MoveAction]",
        "- strength: int",
        "- end_phase: bool",
    ], [], 1095, 155, 270)
    agf = p.cls("«functions» used by CombatPhase", [], [
        "+ enemy_territories_by_priority(game, player)",
        "+ get_reachable_units(game, target, player)",
        "+ calculate_attack_strength(m, units)",
        "+ defender_strength(m, territory, player)",
        "+ _group_moves(d, [(unit, route)])",
    ], 1380, 155, 330)
    ags = p.cls("«functions» used by noncombat_generators", [
        "THREAT_COVER, HOME_COVER, HOME_KEEP, FRONTIER_MIN",
    ], [
        "+ _reach_routes / _air_routes / _landing_spots",
        "+ _enemy_threat / _home_factory / defense_value",
    ], 1380, 330, 330)
    p.cell("combat_actions (the hmcts / TripleMind attack sets) and noncombat_move are in the module "
           "but are not called by phases.py.", NOTE, 1095, 330, 270, 80)

    # noncombat_generators
    p.cell("«module» noncombat_generators.py", MOD, 1080, 575, 645, 400)
    na = p.cls("NoncombatAction", [
        "- to_terr: str",
        "- moves: [fm.MoveAction]",
        "- strength: float",
        "- end_phase: bool",
        "- excluded: frozenset  (decided after it)",
    ], [], 1095, 605, 270)
    ncf = p.cls("«functions»", ["COVER_LEVELS = (0.5, 1.0, 1.5)"], [
        "+ land_air_moves(game, d, player, moved)",
        "+ noncombat_actions(game, d, excluded, moved, player)",
        "+ positioning_moves(game, d, player, moved)",
    ], 1380, 605, 330)
    plan = p.cls("_Plan", [
        "- strength, want, movable, frontier, landing",
    ], [
        "+ shortfall / priority / destinations",
        "+ defense_gain / donor_loss / can_spare",
    ], 1380, 770, 330)

    gm = p.cls("«module» generators.py", [
        "ag = action_generators_cy, else action_generators",
        "TRIPLEMIND_PURE_PYTHON=1 forces Python",
    ], [], 1095, 860, 270)

    # the two searches
    p.cell("«module» greedy_turn.py  (this agent)", MOD, 20, 90, 360, 420)
    gt = p.cls("GreedyTurnSearch", [
        "- d, map, whoAmI, me, time_budget",
        "- move_rng: fm.SeededRng",
        "- phases: {name: Phase}",
    ], [
        "+ apply(sim, action, rng) / view(sim)",
        "+ search(root_sim, phase_name)",
        "      DFS over plans, transposition table",
        "+ evaluate(phase, node, seed, game)",
        "+ fight_battles(sim, rng)",
        "+ siblings / log_level",
    ], 35, 120, 330, style=CLS_OWN)
    nd = p.cls("Node", [
        "- state: PhaseState, action, depth",
        "- actions, end, values, children",
    ], [], 35, 385, 330, style=CLS_OWN)
    p.cell("«module» greedy_action/greedy.py", MOD, 20, 530, 360, 280)
    ga = p.cls("GreedyActionSearch", [
        "- d, map, whoAmI, me, time_budget, move_rng",
        "- phases: {name: Phase}",
    ], [
        "+ search(root_sim, phase_name)",
        "      one-ply best action per level",
        "+ evaluate(phase, state, action, seed, ...)",
        "+ fight_battles(sim, rng)",
    ], 35, 560, 330, style=CLS_OWN)
    cd = p.cls("Candidate", ["- action, values"], [], 35, 740, 330, style=CLS_OWN)
    p.cell("vanilla_mcts, ne_mcts, bb_mcts and oep import the Phase classes "
           "(PurchasePhase, CombatPhase, NoncombatPhase) and complete_greedily directly; only their "
           "selection differs.", NOTE, 20, 830, 360, 60)

    for sub in (pp, cp, np_):
        p.edge(sub, ph, "", GEN, "exitX=1;exitY=0.5;entryX=0;entryY=0.75;")
    p.edge(pp, pu, "«create»", DEP)
    p.edge(cp, agf, "«call»", DEP, "exitX=1;exitY=0.3;")
    p.edge(cp, ca, "«create»", DEP)
    p.edge(np_, ncf, "«call»", DEP)
    p.edge(ncf, na, "«create»", DEP)
    p.edge(ncf, plan, "«use»", DEP)
    p.edge(plan, ags, "«use» via ag", DEP)
    p.edge(gm, agf, "ag", DEP, "exitX=1;exitY=0.5;")
    p.edge(gt, fn, "make_phases /<br>complete_greedily", DEP, "exitX=1;exitY=0.2;")
    p.edge(ga, fn, "make_phases /<br>complete_greedily", DEP, "exitX=1;exitY=0.3;")
    p.edge(gt, nd, "1 *", DEP)
    p.edge(ga, cd, "1 *", DEP)
    p.edge(nd, ps, "state", DEP, "exitX=1;exitY=0.5;")

    table = (
        '<table border="1" cellpadding="4" style="border-collapse:collapse;font-size:10px;">'
        '<tr style="background:#eeeeee;"><th>phase</th><th>start (forced)</th><th>actions(state), '
        'highest-rated first</th><th>step(state, action)</th><th>key(state)</th><th>action_key</th>'
        '<th>END</th><th>levels_left</th><th>response_actions</th></tr>'
        '<tr><td>purchase</td><td>none; info = ()</td><td>affordable candidate units, best (attack + '
        'defense) / PU first, then END; only END without a factory</td><td>buy 1 unit; info = units '
        'bought</td><td>units bought (a multiset)</td><td>rule_id</td><td>always offered, last</td>'
        '<td>most units of one offered rule the balance pays for</td><td>one PurchaseAction with the '
        'counts</td></tr>'
        '<tr><td>combat move</td><td>none; info = {}</td><td>one all-in attack per target not attacked: '
        'strength &gt; 1.5 × defence (priority order), END, the other attacks</td><td>apply its moves; '
        'info ∪= {target}</td><td>(unit id, territory) of my moved units</td><td>to_terr</td><td>always '
        'offered, after the strong attacks</td><td>attacks offered</td><td>every attack\'s moves</td>'
        '</tr><tr><td>non-combat move</td><td>land_air_moves; info = ({}, units moved)</td><td>reinforce'
        'ment sets for the neediest territory, strongest first, then its skip; END with the positioning '
        'moves once none is left</td><td>apply its moves; info = (action.excluded, moved ∪ units)</td>'
        '<td>(decided territories, (unit id, territory) of units moved)</td><td>(to_terr, unit ids)'
        '</td><td>only when nothing is left; end_of(state) builds one anywhere</td><td>territories still '
        'short of cover</td><td>forced moves, every action\'s moves, END\'s positioning moves</td></tr>'
        '</table>')
    p.cell(table, "text;html=1;align=left;verticalAlign=top;whiteSpace=wrap;overflow=fill;", 20, 1010,
           1720, 150)
    return p


# ---- 2: purchase ---------------------------------------------------------------------------------

def page_purchase():
    p = Page("q-purchase", "2 Purchase", 1700, 1040)
    p.header("Purchase: PurchasePhase.actions / step / key",
             "Every affordable unit is offered at every node, best (attack + defense) per PU first, with END "
             "last. Different orders of the same units share a key, so the search keeps one of them.")
    p.cell("search  (GreedyTurnSearch / GreedyActionSearch)", LANE, 20, 90, 480, 920)
    p.cell("PurchasePhase  (phases.py)", LANE, 520, 90, 640, 920)

    s = p.cell("", START, 245, 130, 30, 30)
    a0 = p.cell("<b>phase.start(sim)</b><br>PhaseState(sim, depth 0, info = ())  no forced moves",
                ACTION, 580, 175, 520, 55)
    a1 = p.cell("<b>phase.actions(state)</b> at a plan", ACTION, 60, 265, 400, 45)
    d1 = p.cell("I own a factory on<br>a territory I own?", DECISION, 740, 255, 200, 90)
    r0 = p.cell("<b>return</b> [END]", RESULT, 990, 270, 150, 55)
    a2 = p.cell("<b>units()</b>  (computed once)<br>production frontier rules with one result, quantity 1, "
                "paid in PUs only, cost &gt; 0, land, not infrastructure, not AA → PurchaseUnit(rule_id, "
                "unit_type, costs, strength = (attack + defense) / cost)<br>sorted by (−strength, rule_id)",
                ACTION, 580, 375, 520, 100)
    a3 = p.cell("keep those whose every cost ≤ my balance (order kept)", ACTION, 580, 505, 520, 45)
    r1 = p.cell("<b>return</b> affordable units + [END]  (END last)", RESULT, 580, 580, 520, 50)
    r2 = p.cell("<b>actions</b> = children to try, in this order", RESULT, 60, 580, 400, 50)
    a4 = p.cell("<b>phase.step(state, unit)</b><br>apply fm.PurchaseAction(((rule_id, 1),)); info = "
                "sorted (rule id, count) pairs; depth + 1", ACTION, 580, 670, 520, 70)
    a5 = p.cell("<b>phase.key(state)</b> = info  (the multiset of units bought)<br>already seen → "
                "transposition, child skipped (Greedy Turn)", ACTION, 60, 670, 400, 70)
    a6 = p.cell("<b>END</b>: phase.finish → EndPhaseAction<br>response_actions(sequence) → one "
                "PurchaseAction with the count of each rule", RESULT, 60, 790, 400, 70)
    e = p.cell("", END, 245, 900, 30, 30)

    p.edge(s, a0, extra="exitX=1;exitY=0.5;")
    p.edge(a0, a1, "root plan", extra="exitX=0.5;exitY=1;entryX=1;entryY=0.5;")
    p.edge(a1, d1)
    p.edge(d1, r0, "[no]")
    p.edge(d1, a2, "[yes]")
    p.edge(a2, a3)
    p.edge(a3, r1)
    p.edge(r1, r2)
    p.edge(r0, r2, extra="exitX=0.5;exitY=1;entryX=1;entryY=0.25;")
    p.edge(r2, a4, "each non-END unit", extra="exitX=0.75;exitY=1;entryX=0;entryY=0.5;")
    p.edge(a4, a5, extra="exitX=0.5;exitY=1;entryX=1;entryY=0.75;")
    p.edge(a5, a1, "new child: its actions next", extra="exitX=0;exitY=0.5;entryX=0;entryY=0.5;")
    p.edge(r2, a6, "END chosen", extra="exitX=0.25;exitY=1;entryX=0.25;entryY=0;")
    p.edge(a6, e)

    p.cell("<b>Compared with heterogeneous-mcts</b><br>hmcts offers units only from the index of the last "
           "one bought (frontier order) and never offers END while something is affordable, so it needs no "
           "transposition table. Here END is offered at every plan, so stopping early is a candidate, and "
           "orders of the same units are merged by key().", NOTE, 1190, 120, 490, 130)
    p.cell("<b>Rating</b><br>evaluate_state counts neither units in hand nor positions, so purchase leaves "
           "almost always tie. The order of actions() then decides: the highest (attack + defense) per PU "
           "first, and the first of equal values wins. complete_greedily buys the first affordable unit "
           "until only END is left.", NOTE, 1190, 270, 490, 120)
    legend(p, 1190, 420)
    return p


# ---- 3: combat move ------------------------------------------------------------------------------

def page_combat():
    p = Page("q-combat", "3 Combat move", 1700, 1080)
    p.header("Combat move: CombatPhase.actions / step / key",
             "One all-in attack per enemy territory not attacked yet: every unmoved unit of mine that can reach "
             "it. Attacks that beat the defence by 1.5× come first in priority order, then END, then the "
             "others.")
    p.cell("search", LANE, 20, 90, 460, 960)
    p.cell("CombatPhase  (phases.py, generators via ag)", LANE, 500, 90, 700, 960)

    s = p.cell("", START, 235, 130, 30, 30)
    a0 = p.cell("<b>phase.start(sim)</b>: PhaseState(sim, 0, info = frozenset())", ACTION,
                560, 170, 580, 45)
    a1 = p.cell("<b>phase.actions(state)</b> at a plan", ACTION, 50, 250, 400, 45)
    a2 = p.cell("game = search.view(sim); strong = [], weak = []<br>for target in "
                "<b>ag.enemy_territories_by_priority</b>(game, me)  (enemy land next to mine, TripleMind "
                "priority)", ACTION, 560, 245, 580, 65)
    d1 = p.cell("next target?", DECISION, 750, 335, 200, 70)
    d2 = p.cell("target in info<br>(attacked)?", DECISION, 750, 430, 200, 80)
    a3 = p.cell("<b>ag.get_reachable_units</b>(game, target, me): my unmoved, mobile, non-sea units that "
                "can reach it (blitz rules; air only if it can land after)", ACTION, 560, 535, 580, 55)
    d3 = p.cell("any?", DECISION, 800, 610, 100, 60)
    a4 = p.cell("CombatAction(target, moves = <b>ag._group_moves</b>(all reachable), strength = "
                "<b>ag.calculate_attack_strength</b>(all reachable))<br>defence = "
                "<b>ag.defender_strength</b>(target)", ACTION, 560, 690, 580, 65)
    d4 = p.cell("strength &gt; 1.5 ×<br>defence?", DECISION, 750, 775, 200, 80)
    a5 = p.cell("strong.append", ACTION, 600, 795, 110, 40)
    a6 = p.cell("weak.append", ACTION, 990, 795, 110, 40)
    r1 = p.cell("<b>return</b> strong + [END] + weak", RESULT, 1000, 345, 180, 50)
    r2 = p.cell("<b>actions</b> = children to try, in this order", RESULT, 50, 345, 400, 50)
    a7 = p.cell("<b>phase.step(state, attack)</b><br>apply its MoveActions (no dice); info ∪= {target}",
                ACTION, 50, 440, 400, 60)
    a8 = p.cell("<b>phase.key(state)</b> = sorted (unit id, territory) of my moved units<br>already seen "
                "→ transposition, skipped (Greedy Turn)", ACTION, 50, 530, 400, 65)
    a9 = p.cell("<b>END</b>: phase.finish → EndPhaseAction; then the search plays the battle phase "
                "(fight_battles: dice or fast_battle, default casualties, no retreat) and rates the "
                "state with evaluate_state", RESULT, 50, 640, 400, 85)
    e = p.cell("", END, 235, 760, 30, 30)

    p.edge(s, a0, extra="exitX=1;exitY=0.5;")
    p.edge(a0, a1, "root plan", extra="exitX=0.5;exitY=1;entryX=1;entryY=0.5;")
    p.edge(a1, a2, extra="exitX=1;exitY=0.5;entryX=0;entryY=0.5;")
    p.edge(a2, d1)
    p.edge(d1, r1, "[none left]")
    p.edge(r1, r2, extra="exitX=0.5;exitY=0;entryX=1;entryY=0.25;")
    p.edge(d1, d2, "[target]")
    p.edge(d2, d1, "[yes]", extra="exitX=0;exitY=0.5;entryX=0;entryY=0.5;")
    p.edge(d2, a3, "[no]")
    p.edge(a3, d3)
    p.edge(d3, d1, "[no]", extra="exitX=1;exitY=0.5;entryX=1;entryY=0.5;")
    p.edge(d3, a4, "[yes]")
    p.edge(a4, d4)
    p.edge(d4, a5, "[yes]")
    p.edge(d4, a6, "[no]")
    p.edge(a5, d1, "loop", extra="exitX=0;exitY=0.5;entryX=0;entryY=0.75;")
    p.edge(a6, d1, "loop", extra="exitX=1;exitY=0.5;entryX=1;entryY=0.75;")
    p.edge(r2, a7, "each non-END attack")
    p.edge(a7, a8)
    p.edge(a8, a1, "new child: its actions next", extra="exitX=0;exitY=0.5;entryX=0;entryY=0.5;")
    p.edge(a8, a9, "END chosen")
    p.edge(a9, e)

    p.cell("<b>Compared with heterogeneous-mcts</b><br>hmcts calls ag.combat_actions: several attack sets "
           "of growing strength for the single most important target, plus a skip, one target per node. "
           "Here phases.py calls the same helpers but offers one all-in attack for every target at once; "
           "END plays the part of skipping the rest. Units moved by an earlier attack in the plan are no "
           "longer reachable, so later attacks in a plan get smaller.", NOTE, 1220, 120, 460, 165)
    p.cell("<b>Rating</b><br>Only the combat move is rated after battles. With dice, Greedy Turn resamples "
           "every leaf after a finished DFS (the same seed per sweep); Greedy Action samples each level's "
           "candidates round-robin within its time share (remaining time / levels_left).", NOTE,
           1220, 305, 460, 110)
    legend(p, 1220, 445)
    return p


# ---- 4: non-combat move --------------------------------------------------------------------------

def page_noncombat():
    p = Page("q-noncombat", "4 Non-combat move", 1750, 1100)
    p.header("Non-combat move: NoncombatPhase.start / actions / step / key / end_of",
             "Forced air landings at start; then one territory short of cover per action (its reinforcement "
             "sets strongest first, then its skip), from noncombat_generators; END carries the positioning "
             "moves.")
    p.cell("search", LANE, 20, 90, 460, 980)
    p.cell("NoncombatPhase  (phases.py)  →  noncombat_generators", LANE, 500, 90, 760, 980)

    s = p.cell("", START, 235, 130, 30, 30)
    f0 = p.cell("<b>phase.start(sim)</b>: <b>ncg.land_air_moves</b>(view, d, me): air that cannot end the "
                "phase where it stands flies to the landing spot most short of cover. Applied now; "
                "info = (frozenset(), units moved). Returned as the forced moves, sent first.",
                FORCED, 560, 165, 660, 70)
    a1 = p.cell("<b>phase.actions(state)</b> at a plan", ACTION, 50, 265, 400, 45)
    a2 = p.cell("<b>ncg.noncombat_actions</b>(view, d, excluded, moved, me)<br><b>_Plan</b>: strength "
                "= Σ defense_value; want = THREAT_COVER × enemy threat (home factory: HOME_COVER, "
                "HOME_KEEP; frontier: FRONTIER_MIN); movable = units not in moved", ACTION,
                560, 260, 660, 75)
    a3 = p.cell("targets = my territories not excluded with shortfall &gt; 0, by priority × shortfall",
                ACTION, 560, 360, 660, 45)
    d1 = p.cell("next target?", DECISION, 790, 430, 200, 70)
    r0 = p.cell("<b>return</b> [NoncombatAction(end_phase, excluded, moves = positioning_moves)]",
                RESULT, 1030, 435, 210, 70)
    a4 = p.cell("donors that can end on the target, by defense_gain − donor_loss; build sets whenever "
                "the value sent reaches 0.5, 1.0, 1.5 × shortfall (sources keep their own cover), plus "
                "everything that can be spared; excluded.add(target)", ACTION, 560, 525, 660, 70)
    d2 = p.cell("any set?", DECISION, 790, 615, 200, 65)
    r1 = p.cell("<b>return</b> sets strongest first + skip(target), each NoncombatAction carrying "
                "excluded (the territories decided once it is played)", RESULT, 560, 700, 660, 55)
    r2 = p.cell("<b>actions</b> = children to try, in this order", RESULT, 50, 440, 400, 50)
    a5 = p.cell("<b>phase.step(state, action)</b><br>apply its MoveActions; info = (action.excluded, "
                "moved ∪ its units)", ACTION, 50, 790, 400, 60)
    a6 = p.cell("<b>phase.key(state)</b> = (decided territories, (unit id, territory) of units moved)"
                "<br>already seen → transposition, skipped (Greedy Turn)", ACTION, 50, 875, 400, 65)
    a7 = p.cell("<b>END</b>: phase.finish plays END's positioning moves (a rejected one is skipped), "
                "then EndPhaseAction", RESULT, 50, 530, 400, 60)
    e = p.cell("", END, 235, 620, 30, 30)
    a8 = p.cell("<b>phase.end_of(state)</b>: END with <b>ncg.positioning_moves</b> from this state, also "
                "while territories are undecided (Greedy Action's rating of a candidate, "
                "complete_greedily)", ACTION, 560, 790, 660, 60)

    p.edge(s, f0, extra="exitX=1;exitY=0.5;")
    p.edge(f0, a1, "root plan", extra="exitX=0.25;exitY=1;entryX=1;entryY=0.5;")
    p.edge(a1, a2, extra="exitX=1;exitY=0.5;entryX=0;entryY=0.5;")
    p.edge(a2, a3)
    p.edge(a3, d1)
    p.edge(d1, r0, "[none left]")
    p.edge(r0, r2, extra="exitX=0.5;exitY=0;entryX=1;entryY=0.25;")
    p.edge(d1, a4, "[target]")
    p.edge(a4, d2)
    p.edge(d2, d1, "[no] pass over", extra="exitX=1;exitY=0.5;entryX=1;entryY=0.75;")
    p.edge(d2, r1, "[yes]")
    p.edge(r1, r2, extra="exitX=0;exitY=0.5;entryX=1;entryY=0.75;")
    p.edge(r2, a5, "each non-END action", extra="exitX=0.75;exitY=1;entryX=0.75;entryY=0;")
    p.edge(a5, a6)
    p.edge(a6, a1, "new child: its actions next", extra="exitX=0;exitY=0.5;entryX=0;entryY=0.5;")
    p.edge(r2, a7, "END offered", extra="exitX=0.25;exitY=1;entryX=0.25;entryY=0;")
    p.edge(a7, e)

    p.cell("<b>positioning_moves</b><br>Sources farthest from the frontier first; each unit its source can "
           "spare moves to the reachable territory of mine with the best defense_gain + advance toward the "
           "frontier + 0.2 × max(0, 5 − distance to an objective) − donor_loss, if positive.",
           NOTE, 1280, 120, 450, 110)
    p.cell("<b>Compared with heterogeneous-mcts</b><br>The same three steps and the same generator. This "
           "copy of noncombat_generators adds `excluded` to every NoncombatAction, so step() moves to a "
           "child without generating the parent's options again; and END is offered only once nothing is "
           "left, as in hmcts, but end_of() can build one anywhere.", NOTE, 1280, 250, 450, 130)
    legend(p, 1280, 410)
    return p


# ---- 5: sequence generation in Greedy Turn --------------------------------------------------------

def page_dfs():
    p = Page("q-dfs", "5 Greedy Turn sequence search", 1700, 1260)
    p.header("Greedy Turn: how the action sequences are generated and chosen (GreedyTurnSearch.search)",
             "A depth-first search over plans in the order actions() returns them. A plan where END is "
             "offered is also a leaf (ending the phase there); the transposition table skips plans reached "
             "by another order.")
    p.cell("GreedyTurnSearch  (greedy_turn.py)", LANE, 20, 90, 760, 1150)
    p.cell("Phase  (phases.py)", LANE, 800, 90, 520, 1150)

    s = p.cell("", START, 385, 130, 30, 30)
    b0 = p.cell("phase = phases[phase_name]; deadline = now + TRIPLEA_GREEDY_SECONDS", ACTION,
                200, 175, 400, 45)
    b1 = p.cell("<b>start(root_sim.clone())</b> → (root state, forced moves)", ACTION, 860, 175, 420, 45)
    b2 = p.cell("root = Node(root state); seen = {key(root)}; stack = [root]", ACTION, 200, 245, 400, 45)
    d1 = p.cell("stack and<br>time left?", DECISION, 300, 315, 200, 80)
    b3 = p.cell("node = stack.pop()", ACTION, 200, 420, 400, 40)
    b4 = p.cell("<b>actions(node.state)</b>, highest-rated first", ACTION, 860, 415, 420, 45)
    d2 = p.cell("END offered?", DECISION, 300, 485, 200, 70)
    b5 = p.cell("<b>evaluate</b>: clone; <b>finish(state, END)</b>; combat move: fight_battles; "
                "evaluate_state → node.values, leaves.append(node)", ACTION, 60, 580, 400, 65)
    b6 = p.cell("for each non-END action: child state = clone", ACTION, 200, 670, 400, 40)
    b7 = p.cell("<b>step(child, action, move_rng)</b>", ACTION, 860, 670, 420, 40)
    d3 = p.cell("rejected, or<br><b>key(child)</b> in seen?", DECISION, 300, 735, 200, 80)
    b8 = p.cell("transpositions + 1  (skip)", ACTION, 560, 755, 200, 40)
    b9 = p.cell("seen.add(key); node.children.append(Node(child))", ACTION, 200, 840, 400, 40)
    b10 = p.cell("stack.extend(reversed(children))  (the first action is searched first)", ACTION,
                 200, 905, 400, 40)
    b11 = p.cell("combat move with dice and DFS finished: resample every leaf until the deadline (same "
                 "seed per sweep)", ACTION, 40, 320, 230, 90)
    b12 = p.cell("<b>backup</b>: node value = best mean leaf value below it", ACTION, 40, 440, 230, 60)
    b13 = p.cell("from the root: best sibling (END or a child, first of equal values) until END",
                 ACTION, 40, 980, 400, 50)
    b14 = p.cell("stopped short of END: <b>complete_greedily(phase, state)</b>", ACTION, 860, 980, 420, 50)
    r1 = p.cell("<b>return</b> forced + <b>response_actions(sequence)</b>", RESULT, 860, 1070, 420, 50)
    e = p.cell("", END, 1055, 1150, 30, 30)

    p.edge(s, b0)
    p.edge(b0, b1, extra="exitX=1;exitY=0.5;entryX=0;entryY=0.5;")
    p.edge(b1, b2, extra="exitX=0.5;exitY=1;entryX=1;entryY=0.5;")
    p.edge(b2, d1)
    p.edge(d1, b3, "[yes]")
    p.edge(b3, b4, extra="exitX=1;exitY=0.5;entryX=0;entryY=0.5;")
    p.edge(b4, d2, extra="exitX=0.5;exitY=1;entryX=1;entryY=0.5;")
    p.edge(d2, b5, "[yes] leaf", extra="exitX=0;exitY=0.5;entryX=0.5;entryY=0;")
    p.edge(d2, b6, "[no]")
    p.edge(b5, b6, extra="exitX=0.5;exitY=1;entryX=0;entryY=0.5;")
    p.edge(b6, b7)
    p.edge(b7, d3, extra="exitX=0.5;exitY=1;entryX=1;entryY=0.25;")
    p.edge(d3, b8, "[yes]")
    p.edge(d3, b9, "[no]")
    p.edge(b9, b10)
    p.edge(b10, d1, "next node", extra="exitX=1;exitY=0.5;entryX=1;entryY=0.5;")
    p.edge(d1, b11, "[no]")
    p.edge(b11, b12)
    p.edge(b12, b13, extra="exitX=0.5;exitY=1;entryX=0;entryY=0.5;")
    p.edge(b13, b14, "[no END reached]")
    p.edge(b13, r1, "[END]", extra="exitX=0.5;exitY=1;entryX=0;entryY=0.5;")
    p.edge(b14, r1)
    p.edge(r1, e)

    p.cell("<b>Greedy Action (greedy_action/greedy.py)</b> uses the same Phase calls one level at a time: "
           "actions(state); each candidate rated by step + finish(end_of) (+ battles) + evaluate_state, "
           "within the level's time share (remaining / levels_left); commit the best with step(); stop at "
           "END; out of time: complete_greedily.", NOTE, 1340, 120, 340, 160)
    p.cell("vanilla_mcts, ne_mcts, bb_mcts and oep build their trees, genomes or rollouts from the same "
           "Phase interface.", NOTE, 1340, 300, 340, 70)
    legend(p, 1340, 400)
    return p


pages = [page_classes(), page_purchase(), page_combat(), page_noncombat(), page_dfs()]
with open(OUT, "w") as f:
    f.write('<mxfile host="app.diagrams.net" type="device">' + "".join(pg.xml() for pg in pages)
            + "</mxfile>")
