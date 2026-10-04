"""Rule and contract tests for triplea_fm on Capture the Flag (run: python3 -m unittest discover tests)."""

import json
import pathlib
import random
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import triplea_fm as fm  # noqa: E402
from triplea_fm import rules, toolkit  # noqa: E402
from triplea_fm.state import HELD, Decision, Unit, check_invariants  # noqa: E402

TESTBED = ROOT.parents[1]
CTF = (TESTBED / "games/capture-the-flag/map/capture_the_flag/map/games/capture_the_flag.xml").read_text()
MINIMAP = (TESTBED / "games/minimap/map/minimap/map/games/minimap.xml").read_text()
D, REPORT = fm.load_definition(CTF)
T = D.territory_index
P = D.player_index
U = D.unit_type_index
R = D.production_rule_index
RUS, ITA, GER, CHI = P["Russians"], P["Italians"], P["Germans"], P["Chinese"]


def at_step(s, step_name):
    """Put a hypothetical state at the start of `step_name`'s decision (transients as given)."""
    st = next(x for x in D.steps if x.name == step_name)
    s.step = st.id
    s.entered = True
    s.phase_pc = rules.PC_AWAIT
    kind = {"PURCHASE": "PURCHASE", "PLACE": "PLACE"}.get(st.kind, "MOVE")
    if st.kind == "PLACE":
        s.placement_cap = rules.placement_snapshot(D, s, st.actor)
        s.placement_used = {}
    s.decision = Decision(s.next_decision_id, st.actor, kind, st.id)
    s.next_decision_id += 1
    s.invalidate()
    return s


def add(s, type_name, terr, owner, moved=0):
    uid = s.next_unit_id
    s.next_unit_id += 1
    s.units[uid] = Unit(uid, U[type_name], owner, owner, T[terr] if terr != HELD else HELD, moved)
    s.invalidate()
    return uid


def clear_units(s, terr):
    for u in list(s.units.values()):
        if u.loc == T[terr]:
            del s.units[u.id]
    s.invalidate()


def move(units, *route, intent="NORMAL"):
    return fm.MoveAction(tuple(units), tuple(T[t] for t in route), intent)


def ok(s, a, rng=None):
    r = fm.transition(D, s, a, rng or fm.SeededRng(1))
    assert r.status in ("DECISION", "TERMINAL"), (r.validation, r.error)
    return r


class NormalizationTests(unittest.TestCase):
    def test_ctf_baseline(self):
        self.assertTrue(REPORT.supported)
        self.assertEqual(len(D.territory_names), 29)
        self.assertEqual(sum(len(n) for n in D.neighbors) // 2, 32)
        self.assertEqual(sum(D.production), 103)
        self.assertEqual(sum(q for *_, q in D.initial_units), 19)
        self.assertEqual([b[0] for b in D.initial_balances], [12, 15, 18, 21])
        self.assertEqual(D.dice_sides, 6)
        self.assertEqual(set(D.victory_threshold.values()), {4})
        self.assertIn("Allied Air Dependents", REPORT.inert)

    def test_minimap_supported_with_inert_neutral_charge(self):
        md, report = fm.load_definition(MINIMAP)
        self.assertTrue(report.supported)
        self.assertEqual(len(md.territory_names), 20)
        self.assertEqual(md.player_names, ("Russians", "Italians"))
        self.assertTrue(md.always_on_aa and md.pu_cap)
        self.assertIn("neutralCharge", report.inert)  # both neutrals are impassable

    def test_neutral_charge_refused_when_neutral_enterable(self):
        xml = MINIMAP.replace('<option name="isImpassable" value="true"/>', "", 1)
        _, report = fm.load_definition(xml, strict=False)
        self.assertEqual([f.feature_id for f in report.features], ["property:neutralCharge"])
        self.assertIn("neutral territory 1", report.features[0].explanation)

    def test_active_naval_unit_refused(self):
        xml = CTF.replace('<frontierRules name="buyAAGun"/>',
                          '<frontierRules name="buyAAGun"/><frontierRules name="buyTransport"/>')
        xml = xml.replace("<!-- advanced industrial production -->",
                          '<productionRule name="buyTransport"><cost resource="PUs" quantity="7"/>'
                          '<result resourceOrUnit="transport" quantity="1"/></productionRule>')
        _, report = fm.load_definition(xml, strict=False)
        codes = {(f.code, f.feature_id) for f in report.features}
        self.assertIn(("UNSUPPORTED_BY_PROFILE_V1", "unit:transport"), codes)


class LifecycleTests(unittest.TestCase):
    def test_initial_decision_skips_init_and_bids(self):
        s = fm.create_initial_state(D)
        self.assertEqual(s.round, 1)
        self.assertEqual(D.steps[s.step].name, "russianPurchase")
        self.assertEqual(s.decision.kind, "PURCHASE")
        self.assertEqual(s.decision.actor, RUS)

    def test_invalid_action_changes_nothing(self):
        s = fm.create_initial_state(D)
        rng = fm.SeededRng(7)
        before = json.dumps(s.to_json(), sort_keys=True)
        r = fm.transition(D, s, fm.PurchaseAction(((R["buyBomber"], 1),)), rng)
        self.assertEqual(r.status, "REJECTED")
        self.assertEqual(r.validation.code, "INSUFFICIENT_RESOURCES")
        self.assertIs(r.next_rng, rng)
        self.assertEqual(rng.counter, 0)
        self.assertEqual(json.dumps(s.to_json(), sort_keys=True), before)

    def test_purchase_creates_held_units_and_charges(self):
        s = fm.create_initial_state(D)
        r = ok(s, fm.PurchaseAction(((R["buyInfantry"], 4),)))
        s2 = r.next_state
        self.assertEqual(s2.balances[RUS][0], 0)
        self.assertEqual(len(s2.held_units(RUS)), 4)
        self.assertEqual(s2.decision.kind, "PURCHASE")  # stays until EndPhase
        self.assertEqual(fm.validate(D, s2, fm.PurchaseAction(((R["buyInfantry"], 1),))).code,
                         "INSUFFICIENT_RESOURCES")

    def test_determinism_and_serialization_roundtrip(self):
        def play(seed):
            s, rng, rnd = fm.create_initial_state(D), fm.SeededRng(seed), random.Random(seed)
            for _ in range(400):
                if fm.is_terminal(s):
                    break
                r = fm.transition(D, s, rnd.choice(toolkit.candidates(D, s)), rng)
                s, rng = r.next_state, r.next_rng
            return s
        a, b = play(3), play(3)
        self.assertEqual(json.dumps(a.to_json(), sort_keys=True), json.dumps(b.to_json(), sort_keys=True))
        restored = fm.deserialize_state(D, json.loads(json.dumps(fm.serialize_state(a))))
        self.assertEqual(json.dumps(restored.to_json(), sort_keys=True), json.dumps(a.to_json(), sort_keys=True))


class MovementTests(unittest.TestCase):
    def setUp(self):
        self.s = at_step(fm.create_initial_state(D), "russianCombatMove")

    def test_land_unit_captures_empty_neutral_and_stops(self):
        s = self.s
        inf = [u.id for u in s.units_in(T["RussianStart"]) if u.type == U["infantry"]][0]
        s2 = ok(s, move([inf], "RussianStart", "RussianStepOne")).next_state
        self.assertEqual(s2.owner[T["RussianStepOne"]], RUS)
        self.assertIn(T["RussianStepOne"], s2.conquered)
        self.assertNotIn(T["RussianStepOne"], s2.blitzed)  # neutrals are not blitzable

    def test_land_cannot_pass_through_neutral(self):
        s = self.s
        arm = [u.id for u in s.units_in(T["RussianBase"]) if u.type == U["armour"]][0]
        s.units[arm].loc = T["RussianStart"]
        s.invalidate()
        v = fm.validate(D, s, move([arm], "RussianStart", "RussianStepOne", "RussianStepTwo"))
        self.assertEqual(v.code, "BLOCKED_ROUTE")

    def test_armour_blitzes_empty_enemy_territory(self):
        s = self.s
        s.owner[T["RussianStepOne"]] = ITA
        arm = add(s, "armour", "RussianStart", RUS)
        s2 = ok(s, move([arm], "RussianStart", "RussianStepOne", "RussianStepTwo")).next_state
        self.assertEqual(s2.owner[T["RussianStepOne"]], RUS)
        self.assertIn(T["RussianStepOne"], s2.blitzed)
        self.assertEqual(s2.owner[T["RussianStepTwo"]], RUS)  # empty neutral end captured too

    def test_enemy_factory_blocks_blitz(self):
        s = self.s
        s.owner[T["RussianStepOne"]] = ITA
        add(s, "factory", "RussianStepOne", ITA)
        arm = add(s, "armour", "RussianStart", RUS)
        v = fm.validate(D, s, move([arm], "RussianStart", "RussianStepOne", "RussianStepTwo"))
        self.assertEqual(v.code, "BLOCKED_ROUTE")

    def test_infantry_cannot_blitz(self):
        s = self.s
        s.owner[T["RussianStepOne"]] = ITA
        s.owner[T["RussianStepTwo"]] = ITA
        inf = add(s, "infantry", "RussianStart", RUS, moved=-1 + 1)
        s.units[inf].moved = 0
        v = fm.validate(D, s, move([inf], "RussianStart", "RussianStepOne", "RussianStepTwo"))
        self.assertFalse(v.ok)

    def test_air_cannot_overfly_neutral_and_must_land(self):
        s = self.s
        f = add(s, "fighter", "RussianStart", RUS)
        v = fm.validate(D, s, move([f], "RussianStart", "RussianStepOne", "RussianStepTwo"))
        self.assertEqual(v.code, "BLOCKED_ROUTE")
        # Attack an adjacent enemy territory: needs 1 move back to friendly land (ok with 4 moves).
        s.owner[T["RussianStepOne"]] = ITA
        add(s, "infantry", "RussianStepOne", ITA)
        self.assertTrue(fm.validate(D, s, move([f], "RussianStart", "RussianStepOne")).ok)
        s.units[f].moved = 3  # one move left: can reach, but cannot return
        self.assertEqual(fm.validate(D, s, move([f], "RussianStart", "RussianStepOne")).code, "ILLEGAL_LANDING")

    def test_flyover_aa_can_kill_air(self):
        s = self.s
        s.owner[T["RussianStepOne"]] = ITA
        add(s, "aaGun", "RussianStepOne", ITA)
        s.owner[T["RussianStepTwo"]] = ITA
        add(s, "infantry", "RussianStepTwo", ITA)
        b = add(s, "bomber", "RussianStart", RUS)
        tape = fm.TapeRng([("aa:flyover", 6, (0,))])
        r = ok(s, move([b], "RussianStart", "RussianStepOne", "RussianStepTwo"), tape)
        self.assertNotIn(b, r.next_state.units)
        self.assertEqual(r.chance_outcomes, [("aa:flyover", 6, (0,))])

    def test_noncombat_cannot_enter_enemy_or_neutral(self):
        s = at_step(fm.create_initial_state(D), "russianNonCombatMove")
        inf = [u.id for u in s.units_in(T["RussianStart"])][0]
        self.assertEqual(fm.validate(D, s, move([inf], "RussianStart", "RussianStepOne")).code, "WRONG_PHASE")

    def test_combat_move_end_exhausts_moved_land_units(self):
        s = self.s
        arm = [u.id for u in s.units_in(T["RussianBase"]) if u.type == U["armour"]][0]
        s2 = ok(s, move([arm], "RussianBase", "RussianStart")).next_state
        s3 = ok(s2, fm.EndPhaseAction()).next_state
        self.assertEqual(rules.movement_left(D, s3.units[arm]), 0)

    def test_air_in_conquered_territory_is_lost_after_noncombat(self):
        s = at_step(fm.create_initial_state(D), "russianNonCombatMove")
        s.owner[T["RussianStepOne"]] = RUS
        s.conquered.add(T["RussianStepOne"])
        f = add(s, "fighter", "RussianStepOne", RUS, moved=4)
        s2 = ok(s, fm.EndPhaseAction()).next_state
        self.assertNotIn(f, s2.units)


class BattleTests(unittest.TestCase):
    def _attack_flag(self, tape_entries):
        s = at_step(fm.create_initial_state(D), "russianCombatMove")
        s.owner[T["RussianStepTwo"]] = RUS
        arms = [add(s, "armour", "RussianStepTwo", RUS) for _ in range(3)]
        s2 = ok(s, move(arms, "RussianStepTwo", "Flag")).next_state
        self.assertEqual(len(s2.battles), 1)
        return s2, arms, fm.TapeRng(tape_entries)

    def test_battle_simultaneous_fire_and_capture(self):
        # Attackers (3 armour, att 3) roll 0,0,0 -> 3 hits; null infantry (def 2) all hit but fire
        # back: rolls 0,5,5 -> 1 hit on attackers. Then the territory is captured.
        s2, arms, tape = self._attack_flag([("combat:attacker", 6, (0, 0, 0)),
                                            ("combat:defender", 6, (0, 5, 5))])
        r = ok(s2, fm.EndPhaseAction(), tape)  # battle is the only one -> fought automatically
        s3 = r.next_state
        self.assertEqual(s3.owner[T["Flag"]], RUS)
        self.assertEqual(sum(1 for a in arms if a in s3.units), 2)
        self.assertEqual(s3.decision.kind, "MOVE")  # noncombat move

    def test_casualty_choice_and_retreat_decisions(self):
        # Mixed defenders force a casualty decision for the null player.
        s = at_step(fm.create_initial_state(D), "russianCombatMove")
        s.owner[T["RussianStepTwo"]] = RUS
        add(s, "artillery", "Flag", fm.NULL_PLAYER)
        arms = [add(s, "armour", "RussianStepTwo", RUS) for _ in range(2)]
        s2 = ok(s, move(arms, "RussianStepTwo", "Flag")).next_state
        tape = fm.TapeRng([("combat:attacker", 6, (0, 5)), ("combat:defender", 6, (5, 5, 5, 5)),
                           ("combat:attacker", 6, (5, 5)), ("combat:defender", 6, (5, 5, 5))])
        r = ok(s2, fm.EndPhaseAction(), tape)
        s3 = r.next_state
        self.assertEqual(s3.decision.kind, "CASUALTIES")
        self.assertEqual(s3.decision.actor, fm.NULL_PLAYER)
        self.assertEqual(s3.decision.hit_batch.hits, 1)
        bad = fm.SelectCasualtiesAction(s3.decision.hit_batch.id, tuple(arms[:1]))
        self.assertEqual(fm.validate(D, s3, bad).code, "BAD_CASUALTIES")
        r2 = fm.transition(D, s3, toolkit.default_casualties(D, s3), r.next_rng)
        s4 = r2.next_state
        self.assertEqual(s4.decision.kind, "RETREAT")
        self.assertEqual(rules.retreat_destinations(D, s4, s4.battles[s4.decision.battle_id]),
                         [T["RussianStepTwo"]])
        r3 = fm.transition(D, s4, fm.RetreatAction(fm.RETREAT, T["RussianStepTwo"]), r2.next_rng)
        s5 = r3.next_state
        self.assertTrue(all(s5.units[a].loc == T["RussianStepTwo"] for a in arms))
        self.assertEqual(s5.owner[T["Flag"]], fm.NULL_PLAYER)

    def test_capital_capture_transfers_treasury(self):
        s = at_step(fm.create_initial_state(D), "russianCombatMove")
        clear_units(s, "ItalianBase")
        add(s, "factory", "ItalianBase", ITA)
        s.balances[ITA][0] = 9
        s.owner[T["RussianStepTwo"]] = RUS
        s.owner[T["ItalianStepTwo"]] = RUS
        arm = add(s, "armour", "ItalianStart", RUS)
        s.owner[T["ItalianStart"]] = RUS
        s2 = ok(s, move([arm], "ItalianStart", "ItalianBase")).next_state
        self.assertEqual(s2.owner[T["ItalianBase"]], RUS)
        self.assertEqual(s2.balances[ITA][0], 0)
        self.assertEqual(s2.balances[RUS][0], 12 + 9)
        fac = [u for u in s2.units_in(T["ItalianBase"]) if u.type == U["factory"]][0]
        self.assertEqual((fac.owner, fac.orig_owner), (RUS, ITA))

    def test_raid_damage_capped_by_production(self):
        s = at_step(fm.create_initial_state(D), "russianCombatMove")
        s.owner[T["ItalianStepTwo"]] = RUS
        bombers = [add(s, "bomber", "RussianBase", RUS) for _ in range(2)]
        s.owner[T["Flag"]] = RUS
        s.owner[T["RussianStepTwo"]] = RUS
        s.owner[T["ItalianStart"]] = RUS
        clear_units(s, "Flag")
        clear_units(s, "ItalianStart")
        route = ["RussianBase", "RussianStart", "RussianStepOne"]
        s.owner[T["RussianStepOne"]] = RUS
        # Bombers fly RussianBase -> ... -> ItalianBase (6 steps) is too far to return; use a closer
        # setup: put bombers at ItalianStart.
        for b in bombers:
            s.units[b].loc = T["ItalianStart"]
        s.invalidate()
        s.balances[ITA][0] = 30
        s2 = ok(s, move(bombers, "ItalianStart", "ItalianBase", intent="RAID")).next_state
        tape = fm.TapeRng([("raid:damage", 6, (5, 5))])  # 6 + 6 = 12 raw damage
        r = ok(s2, fm.EndPhaseAction(), tape)
        self.assertEqual(r.next_state.balances[ITA][0], 30 - 12)  # per-bomber cap 10 not reached
        s3 = r.next_state
        self.assertTrue(all(b in s3.units for b in bombers))


class MinimapTests(unittest.TestCase):
    """Always on AA, Territory Turn Limit (PU_CAP) and inert neutralCharge, on Minimap."""

    def setUp(self):
        self.d, _ = fm.load_definition(MINIMAP)
        self.t, self.p, self.u = self.d.territory_index, self.d.player_index, self.d.unit_type_index

    def _state(self, step_name, d=None):
        d = d or self.d
        s = fm.create_initial_state(d)
        st = next(x for x in d.steps if x.name == step_name)
        s.step, s.entered, s.phase_pc = st.id, True, rules.PC_AWAIT
        s.decision = Decision(s.next_decision_id, st.actor, "MOVE", st.id)
        s.next_decision_id += 1
        s.invalidate()
        return s

    def _add(self, s, type_name, terr, owner):
        uid = s.next_unit_id
        s.next_unit_id += 1
        s.units[uid] = Unit(uid, self.u[type_name], owner, owner, self.t[terr], 0)
        s.invalidate()
        return uid

    def _take(self, s, terr, owner):
        for u in [u for u in s.units.values() if u.loc == self.t[terr]]:
            del s.units[u.id]
        s.owner[self.t[terr]] = owner
        s.invalidate()

    def _move(self, units, *route, intent="NORMAL"):
        return fm.MoveAction(tuple(units), tuple(self.t[x] for x in route), intent)

    def test_always_on_aa_fires_in_noncombat_move(self):
        rus, ita = self.p["Russians"], self.p["Italians"]
        s = self._state("russianNonCombatMove")
        self._add(s, "aaGun", "blue north 2", ita)
        self._take(s, "blue north 1", rus)
        f = self._add(s, "fighter", "red north 2", rus)
        a = self._move([f], "red north 2", "blue north 2", "blue north 1")
        r = fm.transition(self.d, s, a, fm.TapeRng([("aa:flyover", 6, (0,))]))
        self.assertEqual(r.status, "DECISION", (r.validation, r.error))
        self.assertNotIn(f, r.next_state.units)
        self.assertEqual(r.chance_outcomes, [("aa:flyover", 6, (0,))])
        # Without the property the same move is deterministic (CTF behaviour).
        d_off, _ = fm.load_definition(MINIMAP, overrides={"Always on AA": False})
        s = self._state("russianNonCombatMove", d_off)
        self._add(s, "aaGun", "blue north 2", ita)
        self._take(s, "blue north 1", rus)
        f = self._add(s, "fighter", "red north 2", rus)
        r = fm.transition(d_off, s, a, fm.TapeRng([]))
        self.assertEqual(r.status, "DECISION", (r.validation, r.error))
        self.assertIn(f, r.next_state.units)
        self.assertEqual(r.chance_outcomes, [])

    def _raid(self, d):
        rus, ita = self.p["Russians"], self.p["Italians"]
        s = self._state("russianCombatMove", d)
        self._take(s, "blue north 1", rus)
        bombers = [self._add(s, "bomber", "blue north 1", rus) for _ in range(2)]
        s.balances[ita][d.treasury_resource] = 30
        r = fm.transition(d, s, self._move(bombers, "blue north 1", "blue capitol", intent="RAID"),
                          fm.SeededRng(1))
        self.assertEqual(r.status, "DECISION", (r.validation, r.error))
        tape = fm.TapeRng([("aa:raid", 6, (5, 5)), ("raid:damage", 6, (5, 5))])  # AA misses, 6 + 6
        r = fm.transition(d, r.next_state, fm.EndPhaseAction(), tape)
        self.assertEqual(r.status, "DECISION", (r.validation, r.error))
        return r.next_state.balances[ita][d.treasury_resource]

    def test_territory_turn_limit_caps_total_raid_loss(self):
        self.assertEqual(self.d.production[self.t["blue capitol"]], 10)
        self.assertEqual(self._raid(self.d), 30 - 10)  # 12 raw damage, capped at production
        d_off, _ = fm.load_definition(MINIMAP, overrides={"Territory Turn Limit": False})
        self.assertEqual(self._raid(d_off), 30 - 12)  # only the per-bomber WW2V2 cap applies

    def test_neutral_charge_guard_rejects_injected_enterable_neutral(self):
        rus = self.p["Russians"]
        d = self.d
        s = self._state("russianCombatMove")
        s.owner[self.t["red north 2"]] = fm.NULL_PLAYER  # hypothetical: passable null-owned land
        self._take(s, "red north 2", fm.NULL_PLAYER)
        inf = [u.id for u in s.units.values() if u.loc == self.t["red north 1"] and u.type == self.u["infantry"]][0]
        v = fm.validate(d, s, self._move([inf], "red north 1", "red north 2"))
        self.assertEqual(v.code, "UNSUPPORTED_BY_PROFILE_V1")

    def test_random_games_complete(self):
        d = self.d
        for g in range(3):
            rnd, s, rng = random.Random(g), fm.create_initial_state(d), fm.SeededRng(g)
            for _ in range(20000):
                if fm.is_terminal(s):
                    break
                r = fm.transition(d, s, rnd.choice(toolkit.candidates(d, s)), rng)
                self.assertIn(r.status, ("DECISION", "TERMINAL"), (r.validation, r.error))
                s, rng = r.next_state, r.next_rng
            check_invariants(d, s)


class EconomyTests(unittest.TestCase):
    def test_original_factory_unlimited_captured_factory_capped(self):
        s = fm.create_initial_state(D)
        s.owner[T["ItalianBase"]] = RUS
        for u in s.units_in(T["ItalianBase"]):
            if u.type == U["factory"]:
                u.owner = RUS  # orig_owner stays Italians
        held = [add(s, "infantry", HELD, RUS) for _ in range(12)]
        at_step(s, "russianPlace")
        self.assertEqual(s.placement_cap[T["RussianBase"]], -1)
        self.assertEqual(s.placement_cap[T["ItalianBase"]], 10)
        self.assertEqual(fm.validate(D, s, fm.PlaceAction(tuple(held[:11]), T["ItalianBase"])).code,
                         "CAPACITY_EXCEEDED")
        self.assertTrue(fm.validate(D, s, fm.PlaceAction(tuple(held), T["RussianBase"])).ok)

    def test_unplaced_units_destroyed_and_income_counts_conquests(self):
        s = fm.create_initial_state(D)
        add(s, "infantry", HELD, RUS)
        s.owner[T["RussianStepOne"]] = RUS
        s.conquered.add(T["RussianStepOne"])
        s.balances[RUS][0] = 0
        at_step(s, "russianPlace")
        s2 = ok(s, fm.EndPhaseAction()).next_state
        self.assertEqual(s2.held_units(RUS), [])
        self.assertEqual(s2.balances[RUS][0], 10 + 2 + 2)  # base + start + conquered territory
        self.assertEqual(s2.conquered, set())

    def test_no_income_or_purchase_without_capital(self):
        s = fm.create_initial_state(D)
        s.owner[T["RussianBase"]] = ITA
        s.balances[RUS][0] = 20
        at_step(s, "russianPurchase")
        self.assertEqual(fm.validate(D, s, fm.PurchaseAction(((R["buyInfantry"], 1),))).code,
                         "PHASE_OBLIGATION")
        at_step(s, "russianPlace")
        s2 = ok(s, fm.EndPhaseAction()).next_state
        self.assertEqual(s2.balances[RUS][0], 20)

    def test_victory_only_at_end_round(self):
        s = fm.create_initial_state(D)
        for base in ("ItalianBase", "GermanBase", "ChineseBase"):
            s.owner[T[base]] = RUS
        at_step(s, "chinesePlace")
        r = ok(s, fm.EndPhaseAction())
        self.assertEqual(r.status, "TERMINAL")
        self.assertEqual(r.terminal_outcome.winners, (D.alliance_of[RUS],))
        s0 = fm.create_initial_state(D)
        for base in ("ItalianBase", "GermanBase", "ChineseBase"):
            s0.owner[T[base]] = RUS
        at_step(s0, "russianPurchase")
        self.assertIsNone(ok(s0, fm.EndPhaseAction()).terminal_outcome)


class ToolkitTests(unittest.TestCase):
    def test_generators_are_sound(self):
        rnd = random.Random(11)
        s, rng = fm.create_initial_state(D), fm.SeededRng(11)
        for _ in range(1500):
            if fm.is_terminal(s):
                break
            cands = toolkit.candidates(D, s)
            self.assertTrue(cands)
            for a in cands:
                self.assertTrue(fm.validate(D, s, a).ok, a)
            r = fm.transition(D, s, rnd.choice(cands), rng)
            s, rng = r.next_state, r.next_rng
            check_invariants(D, s)

    def test_validate_batch_matches_validate(self):
        s = at_step(fm.create_initial_state(D), "russianCombatMove")
        actions = toolkit.move_candidates(D, s) + [move([999], "RussianStart", "RussianStepOne")]
        self.assertEqual([v.ok for v in fm.validate_batch(D, s, actions)],
                         [fm.validate(D, s, a).ok for a in actions])


if __name__ == "__main__":
    unittest.main()
