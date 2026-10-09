"""Battle simulator: resolves hypothetical land/air battles under the kernel's battle rules.

The forward-model counterpart of TripleA's odds calculator (odds.calculator.BattleCalculator).
Agents use it to estimate an attack or a defence before committing to it, e.g. win rate, expected
TUV swing and the typical survivors, without building a full game state around the battle.

Each run places the given units in the battle territory of a scratch state and fights one battle
with the kernel's battle program (rules._run_battle): defensive AA, artillery support,
simultaneous fire, stalemates and capture are exactly those of transition(). The decisions a
battle asks for are answered like the engine's calculator player (DummyPlayer) answers them:
  * casualties: toolkit.default_casualties (the default order of loss);
  * retreat: never, unless retreat_when_only_air is set, in which case the attacker retreats as
    soon as only air units are left (DummyPlayer's retreatWhenOnlyAirLeft).

Units are given as (unit type id, owner) pairs, so hypothetical units (not yet purchased, or a
possible enemy counter-attack) can be simulated. Results refer back to them by index; survivors
are the combat units left (infrastructure such as AA guns is never counted, it does not fight).

This is an agent-side helper built on the rule programs, like toolkit; the kernel does not import
it, and it never changes the state it is given.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from . import rules, toolkit
from .actions import CONTINUE, RETREAT, RetreatAction
from .definition import GameDefinition
from .state import CASUALTIES_D, NORMAL_BATTLE, RETREAT_D, Battle, SimulationState, Unit

ATTACKER, DEFENDER, DRAW = "ATTACKER", "DEFENDER", "DRAW"

UnitSpec = Tuple[int, int]  # (unit type id, owner)


@dataclass(frozen=True)
class BattleRun:
    """One simulated battle. Survivors are indices into the attackers/defenders arguments."""

    winner: str  # ATTACKER | DEFENDER | DRAW (BattleResults: nobody left on either side is a draw)
    attackers_left: Tuple[int, ...]
    defenders_left: Tuple[int, ...]
    rounds: int


@dataclass
class BattleSimResult:
    """Aggregate of several runs of the same battle (AggregateResults)."""

    runs: List[BattleRun]

    def _rate(self, winner: str) -> float:
        return sum(1 for r in self.runs if r.winner == winner) / len(self.runs) if self.runs else 0.0

    @property
    def attacker_win_rate(self) -> float:
        return self._rate(ATTACKER)

    @property
    def defender_win_rate(self) -> float:
        return self._rate(DEFENDER)

    @property
    def draw_rate(self) -> float:
        return self._rate(DRAW)

    @property
    def average_rounds(self) -> float:
        return sum(r.rounds for r in self.runs) / len(self.runs) if self.runs else 0.0

    @property
    def average_attackers_left(self) -> float:
        return sum(len(r.attackers_left) for r in self.runs) / len(self.runs) if self.runs else 0.0

    @property
    def average_defenders_left(self) -> float:
        return sum(len(r.defenders_left) for r in self.runs) / len(self.runs) if self.runs else 0.0

    def closest_to_average(self) -> Optional[BattleRun]:
        """The run whose survivor counts are closest to the averages
        (AggregateResults.getBattleResultsClosestToAverage); first such run on ties."""
        if not self.runs:
            return None
        aa, ad = self.average_attackers_left, self.average_defenders_left
        return min(self.runs, key=lambda r: abs(len(r.attackers_left) - aa) + abs(len(r.defenders_left) - ad))


def simulate_battle(d: GameDefinition, s: SimulationState, site: int, attackers: Sequence[UnitSpec],
                    defenders: Sequence[UnitSpec], rng, runs: int = 1,
                    retreat_when_only_air: bool = False) -> BattleSimResult:
    """Fight `runs` independent battles at `site` between the given units.

    `s` supplies territory ownership (which decides capture and allied/enemy relations); the units
    in it are ignored. The site is treated as owned by the first defender's owner, as the engine's
    calculator does. Dice come from `rng`, which is advanced (pass a copy to keep it unchanged).
    Air attacking land fights normally; land units in water and naval units are not supported
    (V1 profile).
    """
    if not attackers:
        raise ValueError("a battle needs at least one attacker")
    template = _scratch_state(d, s, site, attackers, defenders)
    n_att = len(attackers)
    out = []
    for _ in range(runs):
        out.append(_fight_once(d, template.clone(), site, n_att, rng, retreat_when_only_air))
    return BattleSimResult(out)


def _scratch_state(d, s, site, attackers, defenders) -> SimulationState:
    """A state holding only the battle units at `site` (attackers first, ids 1..n)."""
    w = SimulationState()
    w.definition_digest = s.definition_digest
    w.game_number = s.game_number
    w.round = s.round
    w.step = s.step
    w.step_runs = list(s.step_runs)
    w.entered = True
    w.owner = list(s.owner)
    w.orig_owner = list(s.orig_owner)
    if defenders:
        w.owner[site] = defenders[0][1]
    w.balances = [[0] * len(d.resource_names) for _ in d.player_names]
    for type_id, owner in list(attackers) + list(defenders):
        uid = w.next_unit_id
        w.next_unit_id += 1
        w.units[uid] = Unit(uid, type_id, owner, owner, site)
    return w


def _fight_once(d, w: SimulationState, site: int, n_att: int, rng, retreat_when_only_air) -> BattleRun:
    b = Battle(w.next_battle_id, NORMAL_BATTLE, site, w.units[1].owner, w.owner[site])
    w.next_battle_id += 1
    b.attackers = [uid for uid in range(1, n_att + 1)]
    b.origins = {uid: site for uid in b.attackers}  # scripted: retreat only for all-air attacks
    b.status = "ACTIVE"
    w.battles[b.id] = b
    w.active_battle = b.id
    events: list = []
    while b.id in w.battles:
        if not rules._run_battle(d, w, b, rng, events):
            break
        dec = w.decision
        if dec.kind == CASUALTIES_D:
            action = toolkit.default_casualties(d, w)
        elif dec.kind == RETREAT_D:
            alive = [w.units[u] for u in b.attackers if u in w.units]
            only_air = alive and all(d.unit_types[u.type].is_air for u in alive)
            action = RetreatAction(RETREAT, site) if retreat_when_only_air and only_air \
                else RetreatAction(CONTINUE)
        else:  # the battle program only pauses for casualties and retreats
            raise RuntimeError(f"unexpected decision {dec.kind} in a simulated battle")
        rules.apply_action(d, w, action, rng, events)
    winner = next((e[3] for e in events if e[0] == "battle_end" and e[1] == b.id), DRAW)
    if winner not in (ATTACKER, DEFENDER):
        winner = DRAW
    # Survivors are combat units; infrastructure is captured or stays, it never fights.
    survivors = [uid for uid in range(1, w.next_unit_id)
                 if uid in w.units and not d.unit_types[w.units[uid].type].is_infrastructure]
    attackers_left = tuple(uid - 1 for uid in survivors if uid <= n_att)
    defenders_left = tuple(uid - 1 - n_att for uid in survivors if uid > n_att)
    if not attackers_left and not defenders_left:
        winner = DRAW
    return BattleRun(winner, attackers_left, defenders_left, b.round)
