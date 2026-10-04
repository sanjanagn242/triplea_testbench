# Forward-model rule model for a reusable TripleA research testbench

## 1. Scope and fidelity contract

**This is a source analysis and proposed support contract, not an implementation.** Capture the Flag (CTF) is the first validation scenario. The architectural boundary is a versioned subset of TripleA mechanics, not a map name, unit vocabulary, board layout, or fixed turn order.

The forward-model contract is centered on an **agent-generated, supplied Action**:

```text
validate(GameDefinition, SimulationState, Action)
    -> ValidationResult = Valid | Invalid(reason)
transition(GameDefinition, SimulationState, Action, RNG)
    -> SimulationState'

XML + explicit setup overrides -> Scenario Parser -> effective GameDefinition
Agent constructs Action -> validate -> transition on the supplied hypothetical state
CTF.xml / another XML admitted by the same profile -> the same transition rules
```

| Responsibility | Agent | Forward model |
|---|---|---|
| Actions/search | Constructs candidates using its own representation, heuristics or learned policy; enumerates if desired; chooses what to consider; prunes; runs MCTS/planning | Interprets and validates only the supplied action; applies transition semantics |
| Beliefs/policies | Maintains beliefs and policies; supplies hypothetical states and all explicit player decisions | Treats the supplied state as its complete world; never fills beliefs or selects a policy |
| Game effects | Supplies decision Actions, including casualty selection and retreat | Resolves rule-mandated automatic effects and stochastic effects using the supplied RNG; returns resulting state |

**Action generation belongs entirely to the agent.** The forward model must not provide `legal_actions(state)`, `generate_actions(state)`, candidate generation, action enumeration, action templates, search-space generation or action pruning. GameDefinition may expose rule/configuration facts that the agent reads when constructing actions; it must not construct actions for the agent. Validation predicates may inspect supplied action components and local game facts without materializing a candidate-action space.

An agent may generate an invalid action. `validate` must reject it against GameDefinition and the supplied hypothetical SimulationState, with a rule-specific reason and no live-state consultation. Validation is not action generation. `transition` applies a valid supplied action; if called with an invalid action it must reject it before committed mutation/chance consumption, using a defined error protocol rather than manufacture a substitute action. Final API/error types are not designed here.

When execution reaches a player decision, the resulting SimulationState records the decision actor, context and already-realized effects, then waits for the next **agent-supplied Action**. Casualty selection, retreat, battle selection and raid-target decisions must not be chosen by a model policy or auto-selected merely because an engine UI/helper normally proposes defaults. Pending context may contain rule facts such as unresolved hits and participating units; it must not contain generated candidate actions or action templates. This preserves the transition return shape without requiring one Action to finish an entire battle. Rule-forced effects with no decision and engine-defined stochastic AA targeting remain transition semantics; any explicit choice remains an Action.

During planning, **no validator, helper, policy, adapter, cache, random source, delegate bridge, or transition may query authoritative/live TripleA state**. The supplied SimulationState is the complete world. It can be an agent-created hypothetical state or one sampled from a belief distribution. Unknown facts cannot be filled from TripleA, even to “validate” the hypothesis. Belief sampling occurs outside the transition and supplies a complete sample. An inconsistent state gets a local consistency error, not an authoritative correction. Historical reachability from the XML starting position is not required, although state invariants and the selected rule profile remain required.

An input must include the historical/transient information needed for its decision boundary. A territory/unit snapshot cannot identify movement already spent, previously conquered territories, or battle origins. If a mid-turn state omits that information, reject it as incomplete; do not reconstruct it from the live game or silently use zero values. A normalized start-of-turn state may explicitly declare those transient collections empty.

Fidelity means agreement with the pinned engine on legal committed actions, decision actors, chance distributions, state changes, ordering, automatic lifecycle effects, and terminal-check timing for admitted features. It does not imply reproducing networking, sounds, UI undo, or AI policies. Invalid actions should have a defined local rejection contract without partially committing a move or consuming new chance draws. TripleA's RPC permissiveness and UI conventions must be distinguished from intended game actions; questionable edges are listed in §13.

### Evidence baseline

- Engine: `triplea-game/triplea`, revision **2a8b4401d9a6442636fb1d90bf5659b15fa46343**.
- Map: `triplea-maps/capture_the_flag`, revision **8f5dd75ca3014713a3743b4904b9d5af76a73d26**, game version **1.0.1**. [Pinned XML][CTF]
- XML SHA-256: `23365e3e2a6f63ec0b76102ea773bd460acb92b890f36758de318c5267a169b0`.
- Previous analysis inspected: `capture-the-flag-forward-model-specification.md` in the adjacent `2026-10-01/x20/outputs` workspace. Its same engine/map revisions were verified locally. This report independently inspects the relevant implementation and expands beyond that report's CTF scope.
- Findings are **static source traces**. No engine build, parser execution, or differential match was run. REQUIRED_V1 is a proposed implementation obligation, not a claim that a functioning loader/model already certifies these features.
- XML `minimumVersion="1.8"` is a compatibility floor. This document specifies the pinned engine, not all historical releases or current GitHub HEAD. Source links below are revision-pinned.

Five semantic classifications are used throughout: **G** = GENERIC ENGINE SEMANTICS; **X** = XML-CONFIGURED BEHAVIOR; **D** = ENGINE DEFAULT / DERIVED CONFIGURATION; **C** = CTF-SPECIFIC CONFIGURATION; **O** = CURRENTLY UNSUPPORTED / OPTIONAL TRIPLEA FEATURE. A mechanic normally has G meaning, X parameters and D effective defaults simultaneously. C describes a scenario value, never a special transition algorithm. Profile status is separate: REQUIRED_V1, PLANNED_V2, OPTIONAL_LATER, or UNSUPPORTED (§10). The project scope, not customary Axis & Allies coverage, determines these statuses.

## 2. TripleA rule architecture

TripleA distributes rules across parser-created data, attachments, property accessors, delegates, predicates, and mutation code. A single validator is not the complete rule specification.

| Layer | Actual implementation and role | Forward-model consequence |
|---|---|---|
| XML construction | `GameParser.parse/parseTerritories/parseConnections`, `parseAttachments`, `setOptions`, owner/unit/resource initialization; `XmlGameElementMapper` maps legacy class names | Preserve ordered attachment processing, class/property aliases, defaults and initialization; loading just unit statistics is insufficient. [Parser][Mapper] |
| Domain objects | `GameData`, `GameMap`, `GameSequence`, `GameStep`, `Unit`, player/resource/production/relationship collections | Definition and mutable state coexist in GameData. A detached adapter must populate both correctly. [Map][Sequence][Step][Unit] |
| Attachments | `UnitAttachment`, `TerritoryAttachment`, `PlayerAttachment`, `RelationshipTypeAttachment`, support/effect/canal/tech/rule/trigger attachments | Resolve semantic capabilities through accessors, including shorthand and derived values. [UA][TA][PA][RA][Support] |
| Effective properties | `Properties` getters; `LegacyPropertyMapper`; per-step properties via `GameStepPropertiesHelper` | A missing raw property is not a universal zero/false rule; explicit step properties can override name-derived defaults. [Props][Legacy][Phase] |
| Phase lifecycle | `ServerGame.runStep/startStep/endStep`; delegate `start/end/saveState/loadState`; `BaseTripleADelegate` before/after trigger hooks | Apply phase entry/exit once; retain counters and lifecycle state when resuming. [Server][Base] |
| Movement | `MoveDelegate.move` → `MoveValidator.validateMove` → `MovePerformer` and `AaInMoveUtil` | Validation, dice exposures, transport bookkeeping, capture, battle updates and movement charges are separate operations. [MD][MV][MP][FlyAA] |
| Battle orchestration | `BattleDelegate`, `BattleTracker`, `DependentBattle`; normal/nonfighting/raid/air battle subclasses | Preserve attack provenance, battle dependencies and ongoing battle state. [BD][BT][Dependent][MFB][SBR] |
| Battle steps | `BattleStep.Order`, ordered execution stack; power/support/dice/casualty/retreat helpers | Combat is a sequence of chance and choice substeps, not a single binomial exchange. [BS][Power][Roll][Casualty] |
| Mutations | `ChangeFactory`/`CompositeChange`, `bridge.addChange`, `GameData.performChange` | A local bridge can apply changes only to the simulation's detached world; network bridges are inappropriate. [Changes][Dummy] |

A declared delegate is not necessarily scheduled. A shared delegate instance can keep bookkeeping across steps. Helpers often reach `unit.getData()`, `player.getData()`, current sequence properties, `getMoveDelegate()`, or `getBattleDelegate()`. These are safe only if every reachable reference belongs to the detached simulation world.

## 3. Generic engine semantics

V1 implements the following algorithms generically, with parameters supplied by the effective definition:

1. Sequence traversal and once-only initialization/entry/exit effects.
2. Validation and accounting of supplied land/air routes, including terrain/domain/relationship restrictions.
3. Combat/noncombat differences, blitz/capture history, and land-based air-landing feasibility/cleanup.
4. Land/air battle construction, dependency validation, participant updates and round/substep progression.
5. Resolved attack/defense power, conventional artillery support, AA, dice and hit allocation.
6. Validation/application of supplied casualty and ordinary retreat Actions, damage and delayed death/return fire.
7. Strategic bombing with the admitted target/damage/cap rules.
8. Territory takeover, infrastructure capture, original ownership, ordinary allied liberation and capital consequences.
9. Purchase/placement validation, producer limits, resource economy and scheduled victory/terminal checks.

V2 adds naval movement/combat, transports, carriers, amphibious assaults, submarines and naval bombardment. Their source traces remain preparation for future implementation, **not V1 requirements**. The V1 topology and capability representation retains land/water and domain distinctions, rather than assuming every territory is land. Water can be represented and traversed by air under admitted V1 rules; activating a naval subsystem requires V2 and must be rejected by V1.

**Mechanism is generic; applicability and support are definition/state/profile dependent.** “No sea territories in CTF” is a scenario fact. “Naval movement is unsupported by profile V1” is an explicit implementation boundary. Neither permits a land-only map representation. Inactive technology or naval declarations can be retained after a sound inactivity check, but active features cannot be silently ignored. Shorthands can expose effective mechanics even if derived flags are absent from raw XML.

The agent alone generates and selects battle, casualty, retreat, purchase, placement and movement Actions and maintains contingent policies. The model validates the supplied decision and applies its consequences. Rule-mandated equivalent-identity normalization and stochastic AA targeting must be distinguished from actual player choices (§7); reusing an engine default/WeakAi policy to select a player decision is prohibited. V2 transport assignment and bombardment choices will likewise remain agent-supplied Actions.

## 4. XML-configured game definition

These are required information, not proposed final classes. This inventory preserves future extensibility: naval/cargo/carrier/canal fields are **PLANNED_V2** definition metadata for detection and later semantics, not V1 execution requirements. V1 must retain or diagnose such declarations without activating the deferred rules. Basic resource identities, balances and admitted cost/result/income vectors are generic V1 data; additional economic systems remain separately classified.

| Definition information (X, then normalized using D) | What it controls |
|---|---|
| Engine/profile version, XML hash and explicit setup overrides | Stable semantic identity and replay compatibility |
| Territory IDs, water/land, symmetric adjacency, impassability | Routes and movement domains |
| Territory production and unit-production values, capital associations, VC values, original-factory flags, explicit original-owner metadata | Income, producer limits, capital/victory rules; these quantities need not be equal |
| Unit-type IDs and numeric values | Movement, attack/defense, rolls, hit points, AA dice, bombardment, production potential and bombing damage |
| Effective unit capabilities | Air/sea/land, infrastructure, blitz, movement prohibitions, carrier and transport costs/capacities, evasion/first strike/target restrictions, producer/construction flags |
| Support rules | Provider/target types and players, sides, faction, number, bonus and bonus group; engine shorthand expands into this |
| Players, alliances, relationship types and initial relationship matrix | Different permission tests for movement, landing, takeover and canals; alliance membership also drives VC victory |
| Resources, production/repair rules and per-player frontiers | Cost/result vectors and permitted purchasing; a production rule can create several units or resources |
| Ordered sequence, delegate binding/class, actor, max-run counters, step properties | Phase order, startup vs recurring steps, resets/repair/landing boundaries |
| All effective rule properties | Relevant version switches, neutral rules, transport/sub/AA/bombing/placement variants and termination thresholds |
| Initial ownership, units, resources, held units and initialized damage | Starting-state template, not mandatory board contents for every hypothetical state |
| Admitted player/canal/effect metadata | Multi-capital thresholds, PU destruction on capture, canal guards and excluded types; other attachments rejected unless supported |

Do not derive unit rules in generic transition code from names such as `armour`, `fighter`, or `factory`. Preserve explicitly traced legacy name-based initialization separately (for example the engine's named battleship adjustment in §5). A unit named `scout` may blitz; a differently named carrier may have capacity. Do not assume six dice sides, one output per purchase rule, a shared frontier, four players, one capital, or integer movement bookkeeping. `Route` and movement state use `BigDecimal`; territory-effect costs can be fractional. [Route][Effects]

Properties editable at setup must be resolved before a simulation. Client-derived income bonuses and disabled-player settings must be explicit setup inputs. No transition may read current client preferences. The initial profile can admit only 100% income, zero bonus and all players enabled while retaining the generic information boundary. [Bonus][Init]

Trigger-based definition mutation cannot fit an immutable definition merely by overwriting it. A future extension would need dynamic effective-rule overrides and trigger-use/condition state in SimulationState. V1 rejects such active systems (§10–11).

## 5. Engine defaults and derived configuration

### Parser and attachment normalization

- `GameParser.parseDiceSides`: omitted dice sides → **6**. `GameMap.addConnection` stores both directions in sets; duplicate XML edges do not multiply adjacency. [Parser][Map]
- Omitted unit-placement owner → the engine null player, **not territory owner**. `hitsTaken` defaults 0 and is constrained to below hit points; `unitDamage` defaults 0 and must be nonnegative. Unowned territories have null ownership; parser initialization supplies self/null relationships and derives allied/war relations from alliances before explicit relationship initialization. Null ownership is not a neutral diplomatic pact. [Parser][RelTracker]
- Raw property names pass through `LegacyPropertyMapper`; value-child text precedes the value attribute, and property values are inferred/cast to types. Attachments are processed in declaration/expanded-foreach order, with setter behavior and legacy attachment spelling correction. [Parser][Legacy]
- `UnitAttachment` ordinary movement/attack/defense default 0, ordinary attack/defense rolls 1, hit points 1, transport capacity/cost and carrier capacity/cost use sentinel defaults rather than generic zero. Default AA strength is 1, maximum AA rounds 1, and omitted AA targets resolve to air types. Read the exact accessors, not field values alone. [UA]
- `isFactory` sets damageable, infrastructure, can-produce, construction and factory-type construction limits. `isAA` sets combat/bombing/flyover AA, AA movement restrictions, rocket and infrastructure flags. Rocket capability alone does not grant rocket technology. [UA]
- `GameParser.parse` creates default TechAdvance definitions when the technology element is absent. Definition existence alone does not mean a player has acquired technology. `LegacyPropertyMapper` also maps isInfantry/isMechanized to isLandTransportable; this is another reason to inspect normalized options rather than raw flags. [Parser][Legacy]
- `artillery`/`artillerySupportable` create/update legacy `UnitSupportAttachment` rules, including temporary target records when target declarations precede providers. Effective support is +1 offensive strength for one eligible target per ordinary artillery; the supported types come from declarations. [Support]
- `isSub` is shorthand with derived first-strike/evasion, movement-through-enemies and target permissions, some conditional on global properties. It is not equivalent to one hard-coded modern submarine rule. [UA][Props]

### Initialization and phase-derived values

`InitializationDelegate.init` invokes legacy artillery/destroyer frontier changes, shipyards, two-hit battleships, original ownership, technology, skip-unused-bids, income bonuses, disabled-player handling, transport assignment and transient-state reset. `initOriginalOwner` records missing territory originals and infrastructure ownership. Starting land units at sea are assigned to transports greedily in declared order; insufficient capacity produces an error. It then resets load/movement flags caused by that setup. `initTwoHitBattleship` specifically looks up the unit named by Constants.UNIT_TYPE_BATTLESHIP and may overwrite its XML HP with 2 or 1 to match the effective Two hit battleship property. This legacy engine behavior must be normalized exactly; a renamed unit with HP 2 is not automatically the same initialization target. [Init]

Do this once for the default starting template. Do **not** run original-owner initialization over an arbitrary hypothetical midgame board: that would turn current occupiers into original owners and change liberation/production rules. Likewise, do not erase supplied movement, cargo or battle flags by re-running setup.

`GameStepPropertiesHelper` honors explicit combat, noncombat, repair, reset, combined-turn and remove-unlandable-air properties; fallback behavior uses step semantics/names. CTF supplies ordinary suffixes. A reusable loader must normalize the selected step properties rather than always reset on a hard-coded faction turn. [Phase][Step]

| Accessor/default at pinned revision | Interpretation and scope |
|---|---|
| `getWW2V2`, missing → false | A composite rules switch; CTF explicitly sets true |
| `getPuMultiplier`, missing → 1 | Income/bombing resource scale |
| `getLandBattleRounds`, `getSeaBattleRounds`, missing → -1 | Unlimited rounds, subject to battle termination/stalemate |
| `getNeutralsBlitzable`, `getNeutralsImpassable`, neutral flyover, missing → false | Independent permissions, not consequences of zero neutral charge |
| `getUnplacedUnitsLive`, missing → false | Placement-end deletion of unplaced units |
| `getLowLuck`, AA-only and bombing-only Low Luck, missing → false | Separate chance algorithms/selection paths |
| `getTransportCasualtiesRestricted`, `getIgnoreTransportInMovement`, missing → false | Distinguish old combat transports from casualty-restricted transports |
| `getSubmersibleSubs`, ignore-subs-in-movement, air-attack-sub restriction, defending-subs sneak attack, missing → false | Independent submarine branches; WW2V2 alters some branches |
| `getNavalBombardCasualtiesReturnFire`, missing → false | Remove bombardment casualties before general return fire |
| `getDamageFromBombingDoneToUnitsInsteadOfTerritories`, missing → false | Default raid charges treasury; true stores unit production damage |
| `getAlliedAirIndependent`, missing → false | Controls offensive allied air and dependencies; read the actual canonical name |
| VC thresholds absent | Total 18, Honorable 15, Projection 13 via `EndRoundDelegate.getVcAmount`; CTF overrides Total to 4 |

Source: [Properties][Props], [phase helper][Phase], [victory accessors][ER]. This table is not an exhaustive property registry. Every admitted property needs an explicit recorded effective value and its source.

**CTF legacy-property caution:** its XML has `Allied Air Dependents=true`; the inspected canonical constant/accessor is `Allied Air Independent` and a core-source search found no use of the former spelling. Do not invert or reinterpret that raw property by intuition. `LegacyPropertyMapper.mapPropertyName` only maps repair-timing names and preserves this spelling. The pinned loader therefore retains the CTF property without mapping it to the canonical accessor: canonical Allied Air Independent stays false. Treat the former as a known inert legacy property at this revision, not an inverse boolean. [Legacy]

## 6. Dynamic state required by transitions

The inventory below spans V1 and preserved future analysis. V1 requires the land/air, economy, ownership, battle and sequencing information relevant to its admitted mechanics. Cargo links/load-unload history, carrier-loss landing, amphibious origins/dependencies, submerged flags and bombardment records are **PLANNED_V2** extension information; V1 rejects their active use rather than simulate them. V1 may retain inert metadata for detection, but need not implement the corresponding mutable subsystem. Base movement/capture histories remain V1 even where a row also mentions future naval use.

| Mutable SimulationState information | Why board counts alone do not suffice |
|---|---|
| Definition fingerprint; sequence index/round; step run counts; active faction; decision actor; entry/exit progress | Resume the correct phase and avoid duplicated income/reset/victory |
| Current territory owners | Routing, takeover, landing, production, income and victory |
| Per-unit stable identity, type, owner, location or held collection | Selection, cargo linkage, damage, movement and dependent casualties |
| Historical original territory/infrastructure ownership | Liberation, capital return and original-factory production; cannot infer from current ownership |
| Resource balances per player; held purchased units | Affordability, raids, capital transfer, placement |
| Movement spent, bonus movement, was-in-combat, submerged and transport flags | Multiple calls, phase boundaries, eligibility and carrier movement |
| Transported-by links; unloaded cargo records/destinations; loaded-this-turn/after-combat; combat/amphibious-unload flags | Capacity includes unloaded cargo history; loading and unloading permissions depend on prior actions |
| Current-turn conquered/blitzed/fought territories; forbidden bombardment sources | Fresh-capture restrictions for routes, landing, canals, factories and AA exposure |
| Pending battles, types, attacker/defender participants, dependency graph | Naval-before-landing, bombing-before-land and ongoing selection constraints |
| Attack origin → unit IDs; amphibious origins/units; transport dependencies; bombardment assignments; raid-target maps | Retreat sites, shore fire eligibility and cargo destruction |
| Current battle round/substep, alive/active/waiting-to-die/damaged/retreated sets; stored dice and unallocated hits | Return fire, support/target recomputation and resuming choices without rerolls |
| Combat hits separately from bombing/production damage | Multi-hit combat damage and reduced factory production are different quantities |
| Per-territory bombing losses this turn | Raid caps across separate attacks |
| Placement-start unit/producer snapshot; produced-by-producer records and placement allocation | New factory cannot produce immediately; aggregate limits and multiple adjacent producers |
| Defending air unable to land and landing decision progress | Carrier losses trigger special defender relocation/cleanup |
| Declared terminal flag, winner collection and reason | A winning board condition is not yet terminal before its scheduled check |
| Bid allowance/spent, only if nonzero bids admitted later | Different funds and placement rules from recurring purchase/place |
| Future extension: relationships, acquired technologies, triggers, objectives, rule overrides | These become mutable if those features are admitted; v1 freezes/rejects them |

**V1 Action information:** acting decision player/context; selected stable unit IDs and an agent-constructed ordered route; raid vs ordinary attack and supplied targets; supplied battle choice; casualty/damage allocation; retreat/continue choice and destination; purchase rule quantities; placement units/destination; phase completion. These describe the information the model must interpret and validate, **not action templates or an action-generation API**. Player turn and decision actor differ: defenders supply casualty Actions during an attacker's turn. Player choices for the null player's units must also be supplied through the agent/policy layer, not an internal WeakAi.

**PLANNED_V2 Action information:** transport assignment/load/unload, naval battle decisions, submerge/evade, bombardment allocation/decline and carrier-loss landing choice. Repair-mode and other optional Actions belong only to profiles that admit their corresponding semantics. The model must never synthesize these choices; rule data in GameDefinition can help the agent construct them.

**RNG/chance information:** explicit algorithm/seed/stream cursor or injected random-request transcript, bounds and counts, and stored outcomes for unfinished steps. Ordinary combat, AA, AA target randomization and raids consume chance; movement routes and ordinary casualty/retreat decisions do not become chance merely because an AI chooses them. Separate policy randomness from engine chance. Stable participant/target ordering matters for exact event replay. SimulationState may retain already-realized outcomes; RNG supplies future draws. No hidden global random source.

## 7. Rule-by-rule source traces

Notation in each trace: **GD** required immutable information; **SS** mutable state; **A** decision input; **R** chance. Reuse A = directly call existing logic on detached engine objects where feasible; B = extract/refactor; C = independently implement against verified traces. A does not mean a function is pure or ready to use unchanged. Profile statuses below refer to §10's project scope. Shared engine functions may contain V1, V2 and optional branches; retaining a trace does not activate all branches in V1. Validation checks a supplied Action's rule predicates, never enumerates alternatives. Explicit player queries in engine code must become a pending decision followed by an agent-supplied Action.

### 7.1 Turn and phase sequencing — G/X/D; REQUIRED_V1

`GameParser` constructs the sequence and delegates; `ServerGame.runStep` skips a step at max runs, starts the delegate, obtains input only if required, ends it and advances `GameSequence`. `BaseTripleADelegate.start/end` maintains before/after lifecycle flags. `GameStepPropertiesHelper` supplies move/reset/repair/landing interpretations. [Parser][Server][Sequence][Base][Phase]

A skipped max-run step does not execute ordinary start/end lifecycle; a step needing no player input still executes its automatic lifecycle. CTF zero bids are disabled by initialization setting their max runs to zero. Do not equate “no legal decision” with “no automatic effects.”

GD: ordered steps, bindings, actors, properties/max runs. SS: index, round/run counters, lifecycle progress, decision actor. A: phase completion and phase-specific decisions. R: only admitted automatic chance effects. Reuse B for a local sequencer, A for interpretation helpers, C for a small independent lifecycle implementation; `ServerGame` brings network/player-manager concerns.

### 7.2 Movement and movement points — G/X/D; REQUIRED_V1

`MoveValidator.validateMove` checks selected units/route, basic movement, combat/noncombat, path blockers, air landing, transports, canals and fuel. `validateFirstRoute` checks map adjacency/relationship permissions; `validateBasic` checks ownership/dependents, budgets, domains, impassability and unit exclusions. `Route.getMovementCost` sums entered-territory costs; `MovePerformer.markMovementChange` adds spent movement. [MV][Route][MP]

Movement is cumulative across calls. Sea units cannot traverse land. Land needs a legal transport operation to enter water. Air ignores ordinary hostile ground blockers but retains terrain/neutral/relationship restrictions. Combat entry can exhaust ground movement; `MoveDelegate.end` exhausts moved non-air units at the combat-to-noncombat boundary unless the configured reset-at-end branch applies. Air retains its remaining turn budget. [MD][MV]

GD: graph, domains, stats/capabilities, effective phase/properties. SS: locations, spent/bonus movement, cargo, capture/battle history. A: units, route, transport map. R: applicable movement AA only. Reuse A/B for validator/Route; B for performer, which needs local changes, tracker, decisions and AA.

### 7.3 Combat vs noncombat routes, neutrals and terrain — G/X/D; REQUIRED_V1 core; O extensions

`validateCombat` checks attack ownership, neutral middle steps, prior non-blitzed captures, blitz and prohibited combat movement. `validateNonCombat` rejects hostile/neutral ground entry and unresolved combat; naval exceptions are property-sensitive. `validateNonEnemyUnitsOnPath`, `onlyIgnoredUnitsOnPath`, and `enemyDestroyerOnPath` distinguish ordinary blocking forces from air, submerged units, bypass-capable subs and optionally ignored transports/subs. [MV]

Null-player neutral land, a diplomatically neutral player, and unowned sea are distinct. Zero neutral charge does not allow blitz/flyover. `canPayToCrossNeutralTerritory` checks PU affordability and takeover charges it. Terrain effects can exclude types, prohibit blitz, modify decimal movement costs and alter combat strength. [MV][BT][Effects][TEA]

GD: impassability and neutral/property/relationship permissions. SS: owners, blockers, captures, resources. A: route/groups. R: none apart from AA. Core impassability and neutral permissions are v1; arbitrary territory effects, fuel and movement/stacking restriction systems are OPTIONAL_LATER and rejected when active. Reuse A/B for predicates/validators; C only with equivalent cost arithmetic and restrictions.

### 7.4 Blitz — G/X/D; REQUIRED_V1

`MoveValidator.validateCombat` and `Matches.territoryIsBlitzable` determine intermediate hostile traversal; untransported ground members need blitz capability. `BattleTracker.addEmptyBattle/takeOver` captures qualifying intermediate empty hostile land and records conquered/blitzed sets. `MovePerformer.markMovementChange` handles neutral/non-blitzed stop conditions. [MV][Matches][BT][MP]

WW2V2 or the explicit restriction makes hostile factories/AA block blitz. Neutral blitzability is independent. Captured-but-not-blitzed territory is not interchangeable with long-held friendly land. Blitz spends normal movement; it is not a combat bonus or replenishment.

GD: can-blitz flags and property bundle. SS: current owner, occupying units, conquered/blitzed history, movement. A: selected group/route. R: none. Reuse A for predicates, B for movement/capture coupling.

### 7.5 Air movement and landing — G/X/D; REQUIRED_V1 land landing; PLANNED_V2 carriers

`AirMovementValidator.validateAirCanLand` first tests legal allied, nonconquered landing land; otherwise checks remaining-range land and carrier solutions. It accounts for allied air, moving carriers, capacity already occupied, aircraft movement classes and potential carrier movement. `Matches.airCanLandOnThisAlliedNonConqueredLandTerritory` enforces ownership/relationship and fresh-capture restrictions. `MoveDelegate.end` and `AbstractPlaceDelegate.end` remove air unable to land when step rules request it. [Air][Matches][MD][Place][Phase]

Feasible landing is validated when committing movement; actual landing is a subsequent move/placement-dependent event. Survivors do not automatically return to origin. Attacking air alone cannot take land. A plane's retreat from combat can remove it from battle while leaving it at the site for later noncombat landing. [Retreat][BT]

V1 validates supplied air moves against admissible land landing and local movement/capture state; carrier-based landing solutions are PLANNED_V2. V1 must not reuse a carrier-search branch to admit an otherwise invalid air move.

GD: air stats, landing/neutral permissions, carrier rules (V2). SS: remaining movement, current captures, other air/carriers/held carrier units if that production variant is admitted. A: route and later landing moves. R: AA/battle draws. Reuse A/B for existing validator on complete detached data; independently reproducing its carrier search is a high-risk C option. Mixed fractional-cost carrier search contains TODOs (§13); reject that combination until verified.

### 7.6 Sea movement and canals — G/X/D; PLANNED_V2

**Future V2 preparation:** the source behavior and GD/SS/A/R implications below are preserved for later implementation. V1 must not execute this subsystem; effective activation yields UNSUPPORTED_BY_PROFILE_V1. Any additional advanced variants identified as OPTIONAL_LATER remain beyond the planned baseline V2 bundle.

Sea routing uses domain/hostility tests in `MoveValidator` and `Matches.seaCanMoveOver`. Bypass-capable subs cannot pass enemy destroyers. Hostile sea co-location and ignored transport/sub rules are distinct from territory ownership. Unowned water ordinarily does not become captured land-like ownership; convoy/sea-control rules are separate PLANNED_V2 naval-economic systems. [MV][Matches][BT]

`validateCanal`, `canPassThroughCanal`, `checkCanalStepAndOwnership` inspect canal attachments on the route, excluded types, combat-phase prohibition, control/relationship of guarding land, and whether guarding land was just conquered. The relationship permission alone is insufficient for a freshly captured canal guard. [MV][Canal]

GD: water graph and ordinary canals/guard lists/type exemptions. SS: naval units, owners, capture/movement/submerged state. A: naval route/group. R: none. Reuse A/B for canal and path checks. Multiple overlapping canal modes need tests before acceptance; do not extrapolate from a single canal.

### 7.7 Transports, loading and unloading — G/X/D; PLANNED_V2

**Future V2 preparation:** the source behavior and GD/SS/A/R implications below are preserved for later implementation. V1 must not execute this subsystem; effective activation yields UNSUPPORTED_BY_PROFILE_V1. Any additional advanced variants identified as OPTIONAL_LATER remain beyond the planned baseline V2 bundle.

`MoveValidator.validateTransport` requires one-step loading/unloading and explicit assignment, disallows moving ground before loading, enforces transport/cargo co-movement, loading-hostile-sea restrictions and phase/unload history. Land→sea→land cannot be one ordinary transport bridge action. `TransportUtils.mapTransports` and `TransportTracker` implement assignment/capacity/history. [MV][TransportUtils][Transport]

`getAvailableCapacity` subtracts both currently carried cost **and unloaded cargo cost**. `loadTransportChange` sets transported-by/loaded-this-turn and loaded-after-combat. `unloadTransportChange` records destination, unloaded list, combat/amphibious flags and conditionally retains transported-by until a dependent battle resolves. Allied-transports loading cannot be followed by unloading in the same tracked turn interval. A transport unloaded in combat is restricted in later noncombat; split unload destinations and loading after combat have dedicated predicates. `MovePerformer.markTransportsMovement` exhausts affected non-air units and applies these changes. [Transport][MP]

Combat transports with attack/defense values and casualty-restricted noncombat transports are different configurations; CTF's unused transport has defense 1. Transported cargo does not fire as a sea combatant, and losses can destroy dependent cargo/landing units. [Participants][Casualty][MFB]

GD: cargo costs, capacity, ownership/phase/property restrictions. SS: links plus all load/unload/combat history and pending dependencies. A: cargo/transport assignment, route and unload destination. R: naval combat only. Reuse A for trackers/costs, B for validation and execution. Retaining just “capacity left” cannot capture the rules.

### 7.8 Carriers and defending air after carrier loss — G/X/D; PLANNED_V2

**Future V2 preparation:** the source behavior and GD/SS/A/R implications below are preserved for later implementation. V1 must not execute this subsystem; effective activation yields UNSUPPORTED_BY_PROFILE_V1. Any additional advanced variants identified as OPTIONAL_LATER remain beyond the planned baseline V2 bundle.

`MoveValidator.carrierMustMoveWith` accounts for allied dependent air moving with owned carriers; `AirMovementValidator.carrierCapacity/carrierCost` and its landing search reserve room for allied occupants. Allied independent-air property also changes offensive participant filtering. [MV][Air][Participants][Props]

`MustFightBattle.checkDefendingPlanesCanLand` records stranded surviving defending air. `BattleDelegate.checkDefendingPlanesCanLand` performs cleanup: with WW2V2 or surviving-air relocation enabled it considers adjacent legal land/allied carrier capacity and asks for landing choices when needed, then kills any air still unable to land. This defender salvage operation is not ordinary attacker noncombat movement. [MFB][BD]

Future PLANNED_V2 carrier-production branches additionally inspect held carriers/combined turns and can move existing fighters onto new carriers: `AbstractPlaceDelegate.moveAirOntoNewCarriers/canUnitsBePlaced/validateNewAirCanLandOnCarriers`. All these carrier paths, including ordinary deployment and capacity-aware landing, are deferred to V2; none is a V1 implementation requirement. [Place][Air]

GD: capacity/cost, allied dependency/relocation properties. SS: air identities/remaining movement, carrier occupants, stranded defenders. A: carrier move/dependent selection and permitted defender landing choice. R: combat draws. Reuse B; a local full bridge is necessary for cleanup changes and defender choices.

### 7.9 Amphibious assaults — G/X/D; PLANNED_V2

**Future V2 preparation:** the source behavior and GD/SS/A/R implications below are preserved for later implementation. V1 must not execute this subsystem; effective activation yields UNSUPPORTED_BY_PROFILE_V1. Any additional advanced variants identified as OPTIONAL_LATER remain beyond the planned baseline V2 bundle.

`BattleTracker.addMustFightBattleChange/getDependentAmphibiousAssault` makes a landing battle depend on its source sea battle; `addEmptyBattle` also preserves dependencies for undefended landing destinations. `DependentBattle`/`MustFightBattle.addAttackChange` record amphibious units and origins. Pending cargo remains linked so transport destruction can remove it from dependent battles. [BT][Dependent][MFB][Transport]

A sea victory must precede the dependent land attack; “unloaded units already stand on land” does not imply independence from the transport battle. Ordinary amphibious ground retreat is prohibited by `RetreatChecks`; WW2V2/attacker-plane/partial-amphibious switches determine air or partial retreat alternatives. Submarines-prevent-unescorted-landing is another explicit property branch in transport validation. [RetreatChecks][Retreat][MV]

GD: assault/retreat/escort properties. SS: origin maps, amphibious flags, cargo links, dependencies and casualties. A: unload, battle choice, legal retreat/continue. R: sea and land combat. Reuse B, with dependency transitions preserved. Marine bonuses and partial mixed amphibious retreat are future PLANNED_V2 amphibious variants; their individual configurations still need certification.

### 7.10 Combat creation and battle ordering — G/X/D; REQUIRED_V1

`MovePerformer` updates departures and arrivals, while `BattleTracker.addBattle/addMustFightBattleChange/addEmptyBattle` chooses normal/raid/nonfighting battle paths. Enemy infrastructure alone does not require ordinary fighting. Qualifying empty-land takeover can happen during movement. Aircraft moving away must be removed from their prior battle; membership cannot be “all units that visited.” [MP][BT][MFB]

`BattleDelegate.doInitialize` also creates battles for applicable hostile units already co-located, including states not created by the immediately preceding move. `fightBattle` rejects nonexistent battles, blocks dependent fights and requires finishing an already selected battle. Independent available battles are chosen by the attacker; naval-before-amphibious and raid-before-land dependencies constrain that choice. [BD][BT]

GD: supported battle types/phase/rules. SS: participants, origins, dependencies, current battle and co-location. A: agent-supplied battle choice and raid designation, validated for existence/order without enumerating alternatives. R: none at creation. Reuse B for tracker/delegate orchestration; C requires explicit graph-based ordering, not fixed territory order.

### 7.11 Combat rounds, attack/defense dice and multi-hit damage — G/X/D; REQUIRED_V1 land/air; PLANNED_V2 naval stages

`MustFightBattle.fight/pushFightLoopOnStack` builds ordered execution steps; `BattleStep.Order` orders AA, bombardment, removal of noncombatants, movement exhaustion, pre/post sub evasion, unprotected transports, first strike, general fire, casualty removal, ending and retreat/stalemate checks. Actual applicability and order are recomputed through properties and state. V1 executes only admitted land/air stages; sub/transport/bombardment stages are PLANNED_V2, and arbitrary nonnaval first-strike variants remain OPTIONAL_LATER. [MFB][BS]

For ordinary general fire, attackers roll/select casualties and defenders can return fire while waiting to die; removal follows the general firing stages. First-strike and bombardment removal have earlier conditional boundaries. Do not remove all selected defenders before their return-fire opportunity, or let AA deaths fire. [BS][FS][BombardClear][Casualty]

`CombatValueBuilder.mainCombatValue`, `MainOffenseCombatValue`, `MainDefenseCombatValue`, `PowerStrengthAndRolls` and support helpers resolve strength/roll count; `RollDiceFactory` selects RolledDice vs LowLuckDice. RolledDice uses zero-based draws `0..diceSides-1`, hit when draw < strength. Low Luck gives integer total-power/diceSides hits and a residual roll when nonzero. Multiple rolls and choose-best-roll are explicit alternative semantics, not automatically one hit per rolled die. [Power][Offense][Defense][PowerUnits][Roll][Rolled][LowLuck]

Preserved repair analysis (PLANNED_V2 for ordinary naval repair; OPTIONAL_LATER for active land/air automatic repair): `GameStepPropertiesHelper.isRepairUnits` selects combat-entry or end-turn repair timing. `MoveDelegate.repairMultipleHitPointUnits` uses the beginning-repair property to select only the active player's units; without that property the no-facility path considers damaged multi-hit units of all owners. `getLargestRepairRateForThisUnit` returns 1 when facilities are not required, so each invocation removes one combat hit, not arbitrarily all damage. Facility-dependent rates, repaired-unit conversion and damaged-carrier restrictions are deferred. Legacy "end of round" property names map to end-of-turn canonical keys. [MD][Phase][Legacy]

Hit points and accumulated combat hits govern damage/death. `CheckGeneralBattleEnd` checks exhausted sides; offense loss is tested before defense loss. `CheckStalemateBattleEnd` handles both sides unable to damage/target, and last-round limits. Mutual elimination cannot be implemented as an automatic attacking capture. [EndBattle][Stalemate]

GD: stats/rolls/HP/dice sides and admitted properties. SS: alive/active/waiting-to-die, hits, rounds/substeps, stored rolls and hits. A: casualties and retreat decisions. R: ordinary/first-strike/bombardment dice. Reuse A/B for power/dice, B for full battle execution; headless odds mode changes semantics (§12).

### 7.12 Artillery and support — G/X/D; REQUIRED_V1 conventional support; O arbitrary support variants

`UnitSupportAttachment.addRule/addTarget` normalizes shorthand. `SupportCalculator` filters providers by owner/type, eligible side/faction and supported types; the per-provider support count limits recipients. `AvailableSupports.getSortedSupport/giveSupportToUnit` groups/sorts and consumes support; separate strength/roll channels and bonus types affect stacking. [Support][SupportCalc][Available][Power]

CTF's infantry-target + offensive +1 strength + one-per-artillery rule is one configuration. Do not hard-code “artillery supports infantry by name” or apply support to defense automatically. Recalculate with surviving/active firing groups as the battle evolves, using engine ordering. Arbitrary enemy support, roll support, overlapping bonus groups and technology-modified support are later obligations.

GD: normalized provider/target/sides/faction/number/bonus/group/player rules. SS: current eligible participants and firing-stage allocation. A: no ordinary support-target action in this path; engine allocation is an algorithm. R: subsequent dice. Reuse B/A for support engine; C conventional subset only with matching allocation tests.

### 7.13 Casualty selection — G/X/D; REQUIRED_V1

`CasualtySelector.selectCasualties` gets target-specific hits/dependents, computes default losses, auto-selects when compelled/equivalent and otherwise queries the hit player's remote policy; checks hit count, damage/death and available targets. It also normalizes equivalent aircraft selection toward lower movement for kills and higher movement for damage. `SelectMainBattleCasualties`, firing-group splits and `MarkCasualties` apply this within the appropriate stage. [Casualty][MainCasualty][Participants][MarkCasualties]

Casualties are constrained by target groups and remaining HP, not arbitrary removal from the whole army. Cargo/dependents are additional losses, not additional independently selected firing casualties. Casualty-restricted transport removal has separate unprotected-unit steps. The null player gets a WeakAi in this engine path; that is preserved source behavior, not an authorization for the forward model to choose. Replace explicit engine policy queries with pending decisions and agent-supplied Actions for the null player as for other decision actors. Automatic outcomes with no genuine choice must be distinguished from default suggestions. [Casualty][Unprotected][MFB]

GD: HP/target restrictions/transport policy. SS: eligible targets, damage, dependents and pending hits. A: legal killed/damaged allocation when a choice exists. R: none for ordinary player casualty choices; AA may randomize (§7.15). Reuse B because synchronous remote-player calls need decision suspension followed by an agent-supplied casualty Action; policy execution belongs outside the forward model. Exact equivalent-identity normalization needs validation (§13).

### 7.14 Retreat — G/X/D; REQUIRED_V1 ordinary retreat; PLANNED_V2 naval/sub retreat

`MustFightBattle.getAttackerRetreatTerritories` derives sites from attack-origin maps, excluding hostile/blocking/fought/incompatible-domain sites; WW2V2/WW2V3 excludes air-only origins. All-air battles can retreat in place. `RetreatChecks.canAttackerRetreat` excludes ordinary amphibious retreat and defenseless-transport auto-win cases. `OffensiveGeneralRetreat` selects the applicable retreater. [MFB][RetreatChecks][Retreat]

`RetreaterGeneral.computeChanges` removes attacking air from participants while keeping it at battle site and physically moves the non-air retreating group/dependents to the chosen site. Air landing and movement are still needed later. A defeated defender does not have an ordinary general retreat action. Submarine evade/submerge choices have their own checks and timing. [GeneralRetreat][SubRetreat]

GD: variant permissions. SS: origins, survivors, enemy blockers, amphibious/submerged state, movement and participants. A: continue or permitted retreat/submerge and destination. R: none. Reuse B; never substitute arbitrary adjacent friendly territory as a retreat option.

### 7.15 AA — G/X/D; REQUIRED_V1 classic AA; O advanced batteries

`AaInMoveUtil.getTerritoriesWhereAaWillFire` processes middle-step eligible AA, normally excludes destination (battle handles it), may include origin to prevent split-route evasion unless already fought there, and has last-step/Always-on-AA/territory-restricted branches. It prevents undo after revealed AA dice. [FlyAA]

Battle AA uses `OffensiveAaFire/DefensiveAaFire`, `FiringGroupSplitterAa`, `AaPowerStrengthAndRolls`, `RollDiceFactory.rollAaDice` and `AaCasualtySelector.getAaCasualties`. AA categories/target sets/shot limits/round limits and overlap determine firing; do not multiply identical classic batteries into arbitrary extra shots. Ordinary target-linked rolls, random fallback and Low Luck casualty grouping are distinct algorithms. `Choose AA Casualties` changes the choice path. [BattleAA][AAGroups][AAPower][Roll][AACasualty]

GD: AA capability/strength/sides/targets/limits and exposure properties. SS: ordered moving targets, prior fought territories, surviving targets and stored draws. A: route; chosen AA casualties only where enabled. R: AA dice and applicable target draws. Reuse A/B for dice/selection; B for route exposure and battle integration. Offensive, multi-round, damage-only, overstacking and custom target AA are OPTIONAL_LATER; loader must recognize their nondefault configuration.

### 7.16 Submarines and destroyers — G/X/D; PLANNED_V2

**Future V2 preparation:** the source behavior and GD/SS/A/R implications below are preserved for later implementation. V1 must not execute this subsystem; effective activation yields UNSUPPORTED_BY_PROFILE_V1. Any additional advanced variants identified as OPTIONAL_LATER remain beyond the planned baseline V2 bundle.

`UnitAttachment` sub shorthand and property-sensitive accessors derive evasion, first strike, passage and targeting. `OffensiveFirstStrike.calculateState` keeps a WW2V2 sub firing stage even under destroyer presence; otherwise destroyers can move the firing into regular order. `ClearFirstStrikeCasualties` determines which side's waiting casualties are removed, with a special WW2V2 two-sided case. `TargetGroup.newTargetGroups/findTargets` applies cannot-target/cannot-be-targeted-by; friendly destroyers cancel the latter restriction, not every cannot-target restriction. [UA][FirstStrike][FS][Targets]

`OffensiveSubsRetreat`, `DefensiveSubsRetreat`, `EvaderRetreat` and `SubmergeSubsVsOnlyAirStep` handle timed evade choices, destroyer blocking and air-only matchups. Submerged units can remain in the territory while not participating/blocking; reset logic later resurfaces them. “Sub fires first and cannot hit planes” is insufficient to cover these transitions. [SubRetreat][DefSubRetreat][Evade][SubAir][MD]

GD: property bundle and normalized sub/destroyer capabilities. SS: active vs waiting/surviving units, destroyers, submerged flags, retreat sites. A: evade/continue. R: first-strike and ordinary dice. Reuse B for stage/choice orchestration and A for target helpers. Custom first-strike/evasion/target systems beyond the admitted sub template are OPTIONAL_LATER.

### 7.17 Bombardment — G/X/D; PLANNED_V2

**Future V2 preparation:** the source behavior and GD/SS/A/R implications below are preserved for later implementation. V1 must not execute this subsystem; effective activation yields UNSUPPORTED_BY_PROFILE_V1. Any additional advanced variants identified as OPTIONAL_LATER remain beyond the planned baseline V2 bundle.

`BattleDelegate.addBombardmentSources/getPossibleBombardingTerritories/selectBombardingBattle` finds eligible owned bombarding ships at qualifying amphibious origins, excludes disallowed/pending sea sources, offers opt-out/target choice, and applies optional per-amphibious-ground-unit cap. `NavalBombardment.valid` restricts firing to first-round land combat and exhausts ships' movement. Bombardment power uses `BombardmentCombatValue`. [BD][Bombard][BombardPower]

`ClearBombardmentCasualties.execute` removes defenders before general fire when return fire is false. New battle execution initially marks casualties with ReturnFire.ALL and then clears at this conditional boundary; the older embedded return-fire branch exists for old saves. Do not infer current behavior solely from `NavalBombardment.calculateReturnFire`. [Bombard][BombardClear]

GD: bombard flags/strength and cap/return-fire properties. SS: amphibious provenance, ship assignments, sea history, first-round casualties. A: bombard/decline and target. R: bombard dice. Reuse B orchestration, A/B power/dice. Defer bombardment support/technology beyond ordinary templates.

### 7.18 Strategic bombing — G/X/D; REQUIRED_V1 treasury damage; OPTIONAL_LATER unit damage/repair

Movement distinguishes raid from ordinary combat and selects legal damageable targets. `BattleTracker.addBombingBattle` creates raids/dependencies. `StrategicBombingRaidBattle.fight` runs AA, then rolls/damage and ends without ordinary attack-vs-defense rounds. Its `rollDice/findCost` draws map-dice values, adds 1, handles bomber roll/bonus rules, caps each bomber at territory production under WW2V2 or the damage-limit property, then separately applies aggregate per-turn caps if enabled. [MP][BT][SBR]

With damage-to-units false, the engine records territory losses and deducts capped damage × PU multiplier from defender cash, bounded by treasury. It does not reduce territory production simply because the property description says “territories.” With true, it assigns unit damage, respects remaining damage limits and production/operational effects; repair rules later reduce that separate damage. [SBR][Purchase][UnitUtils]

V1 includes CTF's ordinary treasury mode, one-roll bombers, standard AA and explicit per-bomber/per-turn cap branches. Damageable factory repair is architecturally relevant but deferred because max damage, operational thresholds, destruction and placement interactions require additional trace/test coverage; its presence must cause rejection, never conversion to treasury loss.

GD: raid capability, eligible targets, dice and cap/multiplier mode. SS: raid participants/targets, treasury, losses-so-far, capture dependencies; future unit damage. A: raid/target and battle selection. R: raid AA and damage dice. Reuse B for raids; A/B RNG helper; full engine raid code reaches move-delegate loss state and display/client branches.

### 7.19 Territory capture, ownership and liberation — G/X/D; REQUIRED_V1 ordinary rules

`BattleTracker.addEmptyBattle/takeOver`, `getNewOwnerForTakeOver`, `addChangeChangeOwnership`, `captureOrDestroyUnits` and allied-capital takeover code apply ownership and infrastructure changes. Qualifying land force can take empty enemy land during movement; after combat, qualifying surviving ground is needed. Air-only success does not capture land. Infrastructure is captured according to rules, not treated as ordinary combat casualties. [BT][MFB]

Original allied territories generally return to original owner if that player has a qualifying capital, the territory is its capital, or the special unowned-capital branch applies. Allied-capital liberation can return original territories held by allies. Own territory, allied territory and enemy original territory are not equivalent. Keep original ownership separately. Capture-to-third-party, unit-type conversion or destroy-instead-of-capture hooks exist and are OPTIONAL_LATER. [BT]

GD: relationship/capture permissions, original-owner template, capital metadata, infrastructure flags. SS: original/current ownership, surviving ground/infrastructure, capture history. A: qualifying movement/combat already chosen. R: only preceding combat. Reuse B for takeover; independent C implementations need liberation and infrastructure tests, not just `owner=attacker`.

### 7.20 Capitals — G/X/D; REQUIRED_V1 ordinary and multi-capital thresholds

`TerritoryAttachment.doWeHaveEnoughCapitalsToProduce` permits capital-less factions by default or requires an owned capital; PlayerAttachment can set retained-capital production thresholds. `AbstractEndTurnDelegate.canPlayerCollectIncome` and purchase lifecycle use capital gating. `BattleTracker.addChangesOnTakeOverCapitol` tests the capital association and current owner, considers retained-capital thresholds, removes the original capital owner's current PUs and transfers or destroys them. [TA][PA][Income][Purchase][BT]

Capital loss does not generally erase the player, remove its pieces or cancel remaining movement phases. Capturing a third-party occupier's original capital does not automatically steal that occupier's treasury. Multiple capitals and factions with none must be handled from metadata. Token loss/Pacific points are separate deferred features.

GD: capital associations, thresholds, destroys-PUs policy. SS: owners and treasuries. A: takeover indirectly. R: none. Reuse A for gate predicates, B for consequences. Validate the income gate as well as purchase; do not add a blanket capital requirement to placement where engine predicates do not have one.

### 7.21 Purchases and repairs — G/X/D; REQUIRED_V1 basic resource production rules; O repair variants

`PurchaseDelegate.purchase` aggregates costs/results, checks affordability and max-built limits (field + held units), creates units in the player's held collection or resource outputs, subtracts costs and applies a CompositeChange. `delegateCurrentlyRequiresUserInput` checks affordability/capital eligibility. Production frontiers specify available rules; costs cannot be assumed from unit names. [Purchase][Production]

The raw purchase RPC is not a comprehensive legal-action validator: the visible purchase body does not recheck the lifecycle capital gate and every submitted rule's frontier membership. A forward-model validation contract must check the supplied rule vector against frontier membership, nonnegative quantities and decision-phase permissions, and document that boundary rather than duplicate arbitrary crafted RPC exploits.

`purchaseRepair` computes resource cost and unit repairs and applies bombing damage reduction in unit-damage mode. Bid purchase differs from normal purchase, including remaining-bid funds; zero CTF bids can be skipped without bid support. [Purchase][Bid]

GD: resources/frontiers/cost-result vectors, build limits/capital gate. SS: treasury/held units/field counts; future damaged units. A: purchase quantity vector; future repair unit-rule allocation. R: none. Reuse A/B for aggregation; B for safe action adapter. V1 must represent resources generically and validate admitted cost/result vectors and balances without a hard-coded PU-only data model. CTF's PU configuration is the first conformance target. Basic additional-resource purchase/results and territory-income behavior require full trace/certification before admitting those configurations; this is a REQUIRED_V1 obligation, not permission to approximate. Objectives, upkeep, war bonds and automatic unit/resource generators remain OPTIONAL_LATER.

### 7.22 Production, factories and placement — G/X/D; REQUIRED_V1 ordinary producers/construction

`AbstractPlaceDelegate.isValidPlacement` checks in-hand units, production, capacity and type/destination restrictions. `canProduce/getAllProducers` require owned eligible producers, prohibit fresh-captured production by default, and use a placement-start snapshot so a new producer cannot immediately produce. For future V2, sea deployment uses adjacent eligible producers, with enemy-sea deployment permitted under WW2V2 or explicit property; this is not a V1 placement path. [Place]

`getMaxUnitsToBePlacedFrom` gives unlimited capacity for an original factory under its original owner unless a separate limit applies; otherwise `UnitUtils.getProductionPotentialOfTerritory/getBiggestProducer/getHowMuchCanUnitProduce` resolves production. It chooses the biggest eligible producer, **not the sum of all producer capacities**. Capture removes the original-owner unlimited branch. Factory construction has per-type/territory/turn limits; construction capacity is not identical to ordinary units. [Place][UnitUtils][UA]

V1 multiple land placement Actions share producer usage. For V2, adjacent sea placements may be reallocated between producers by engine logic; fixing a first greedy assignment can incorrectly reject later legal batches. `end` deletes unplaced units unless Unplaced Units Live is enabled and performs air checks. Placement has producer predicates rather than a separate universal capital gate. [Place]

GD: producer/construction capacities and territory values, original-factory status, placement variants. SS: snapshot/held units/ownership/history/producer allocation. A: units/destination, phase completion. R: none. Reuse B strongly preferred for future V2 multiple-producer sea allocation; V1 C must match validation outcomes for supplied land placement Actions. Factory upgrades, consuming/requiring units and unrestricted placement are OPTIONAL_LATER. Carrier-production variants belong to future V2 work, with each advanced variant separately certified.

### 7.23 Resources and income — G/X/D; REQUIRED_V1 basic resource economy; O extended economic systems

`AbstractEndTurnDelegate.start` gates income, sums collectible owned production, subtracts blockade loss, scales by PU multiplier, prevents negative treasury, then applies war bonds, other resources/objectives/bonuses/upkeep and repair/ownership hooks where configured. End exit clears turn battle state. `EndTurnDelegate` supplies resource-producing-unit, national-objective and other-resource effects. [Income][EndTurn][Bonus]

V1 ordinary collection counts newly conquered collectible land at the scheduled end turn even though same-turn factory placement may be prohibited. Capital loss blocks the ordinary path. Convoy collection and blockade zones alter collectibility/losses, so the ordinary “sum all territory production” formula is admitted only when those systems are absent. Resource identities/stocks and admitted basic cost/result/territory-income vectors are REQUIRED_V1 generic information. The additional-resource path in EndTurnDelegate must be fully traced and tested before its configurations are certified; the XML's resource names must not determine a new simulator implementation. Extended economic generators/objectives/upkeep remain OPTIONAL_LATER.

GD: production, multiplier, capital and collection rules. SS: ownership/treasury/capture state. A: phase completion. R: none in the ordinary path; future blockade/war-bonds effects may draw chance. Reuse A/B for basic collection, B for lifecycle; client bonus settings must be frozen locally.

### 7.24 Victory and terminal conditions — G/X/D; REQUIRED_V1 VC checks; O other victory modes

`EndRoundDelegate.start` chooses enabled checks; `checkVictoryCities/getVcAmount` sum VC values of alliance-owned territories and compare the configured/default threshold. `signalGameOver` records winners once and routes a display/stop notification through the selected client-network branch. CTF WW2V2 returns before the older named-power military-victory fallback. Economic, Pacific VP and trigger victory paths are distinct. [ER]

Terminal means the scheduled victory check has declared victory, not immediate board evaluation after a capture. Rollout horizon is a research truncation condition, not a TripleA draw. The research harness should stop on the locally recorded engine victory declaration; any client continuation convention belongs outside this profile.

GD: check timing, alliance membership/VC thresholds and enabled supported modes. SS: current owners, sequence and declared terminal/winners. A: completion through scheduled checks. R: none for VC checks. Reuse B/A for checks with a local stop callback. Simultaneous qualifying alliances/order need tests; no fabricated tie-break (§13).

### 7.25 Stochastic behavior and additional encountered mechanics — G/X/D/O

Combat RNG request ranges/counts originate in `RollDiceFactory`, `RolledDice/LowLuckDice`, AA casualty random selection and `StrategicBombingRaidBattle`. Seed equality alone does not imply replay equality; ordering and request counts must match. [Roll][Rolled][LowLuck][AACasualty][SBR]

Other material hooks encountered: before/after triggers (`BaseTripleADelegate`); initialized/acquired technology (`InitializationDelegate`, `TechTracker`, attachment stat getters); territory combat/movement effects; fuel charging (`Route.getMovementFuelCostCharge`); stacking/movement/placement restrictions; resource/automatic unit creation (`EndTurnDelegate`); disabled-player asset removal; air/naval bases and bonus movement (`MoveDelegate`/`MovePerformer`); politics/relationship changes and battle initialization after new war; scramble/interceptors/air battles and kamikaze in BattleDelegate; construction requirements/consumption/capture conversion; convoy ownership/blockades; rockets; bids. [Base][Init][UA][Effects][Route][MD][MP][BD][Place][EndTurn][BT]

These are real semantics, not harmless XML decoration. Their entry points were identified, but their full algorithms are **not established by this report**. They receive later/unsupported classifications below and trigger rejection if active. No unsupported behavior is assigned an invented approximation.

## 8. Capture the Flag as a concrete instantiation

All facts here are **C**, configured through X and interpreted with G/D. [XML][CTF]

| Fact | Effective CTF configuration |
|---|---|
| Graph | 29 land territories, 32 unique undirected edges; four repeated Flag edges collapse |
| Players/turn blocks | Russians → Italians → Germans → Chinese; four singleton alliances, ordinary opponents at war |
| Phases | Initialization once, each bid/bid-place once (zero bids skipped), then each player's Purchase/CombatMove/Battle/NonCombatMove/Place/EndTurn; EndRound after Chinese |
| Initial resources | 12/15/18/21 PUs respectively; different starting cash, same initial production 12 per faction |
| Initial pieces | Each Base: factory + armour; each Start: 2 infantry; Flag: 3 null-player infantry; 19 units total |
| Production/VC | Four Bases each production 10, originalFactory, own capital and 1 VC; Flag production 15, **0 VC**; other 24 territories production 2; total 103 |
| Purchases | Six rules shared by all: infantry 3, artillery 4, armour 5, fighter 12, bomber 15, aaGun 5; one unit per rule; no buy-factory/naval/repair frontier |
| Land/air stats | Infantry 1 move/1 attack/2 defense; artillery 1/2/2; armour 2/3/2 + blitz; fighter 4/3/4; bomber 6/4/1 + raid; AA 1 move; factory 0 |
| Additional defined types | Transport, destroyer, battleship, carrier, submarine; no sea graph/initial pieces/purchase producer makes them baseline-reachable; battleship explicitly HP 1 |
| Effective rules | WW2V2=true; d6 default; Low Luck/AA Low Luck=false; neutral charge 0; neutral blitz/flyover default false; ordinary AA; default treasury bombing mode |
| Victory | Total Victory=true; each singleton alliance threshold 4 VCs at EndRound; ownership of all four Bases qualifies; Flag control alone never terminates |
| Inactive systems | No scheduled tech research/active starting tech, triggers/politics, canals, territory effects, naval placement or amphibious origin in baseline |

The scenario loader should obtain all of this from XML plus engine normalization. It must never recognize the name “Capture The Flag” to select special rules. V1 may retain CTF's inactive naval type/capability declarations as definition metadata after proving they are not activated. CTF's ordinary land types also declare transportCost and its fighter declares carrierCost; these compatibility values alone do not activate cargo or carrier behavior. A hypothetical state that activates naval units, cargo, carrier landing, amphibious combat, submergence or bombardment must be rejected by V1 even if the type was retained. V2 may admit such hypothetical uses when its profile and state invariants permit them; the CTF graph still has no water location for ordinary sea units. Distinguish unsupported feature from invalid state and from a feature merely absent from initial reachability.

CTF baseline is a **candidate SUPPORTED** configuration for the proposed v1 once implementation/tests exist. Nonzero bids, active advanced AA variants, income overrides or team edits require fresh effective-configuration certification. Changing Total Victory to false removes its VC termination path; the model must not invent elimination victory. Map notes are not rule triggers.

## 9. Assumptions in the previous CTF specification that must not be generalized

The previous report correctly labels many items baseline-not-applicable. The problem would be promoting those restrictions into the reusable kernel. The following contrasts identify specific scoped statements from its §§2–9 and their replacements.

| Previous CTF-scoped fact/restriction | Generic consequence |
|---|---|
| All land; naval/transport/carrier/amphibious rules “NOT APPLICABLE” | V1 topology retains land/water without implementing naval rules; activating naval systems requires V2 and explicit V1 refusal. A water node alone need not activate them (e.g. air overflight). |
| Four named factions and fixed six-phase blocks | Load actors, step sequence, max runs and admitted step properties; never infer turn order from player-list order alone |
| Singleton alliances; allied liberation absent | Allied co-defense, dependent carrier air and original-owner restoration are required generic paths |
| No territory effects; cost one per entered territory | Base profile can explicitly reject active effects, but cost-one is a supported-profile restriction rather than universal TripleA semantics |
| One capital/base and loss blocks purchase/income | Allow no-capital/multiple-capital metadata; losing capital is not automatic elimination or cancellation of movement |
| Home original factory unlimited; captured Base limited by production 10 | Resolve original-factory/original-owner predicates and loaded producer values; neither 10 nor unlimited for all factories is generic |
| No factory/naval/repair purchase rules | Parse frontiers; ordinary land factory purchases can use generic V1 industry, naval purchase/placement requires V2, and production-damage repairs remain optional |
| Fixed six unit purchase costs and one-unit outputs | Load cost/result vectors, bundles, player frontiers and max-built limits |
| Every reachable combatant one HP | Generic HP data cannot assume one; basic land/air hit accounting belongs to V1, sea-unit damage/repair conformance belongs to V2 and advanced repair facilities remain optional |
| Submarine/carrier state fields absent in valid baseline | Preserve PLANNED_V2 state implications and extension points; active use is rejected by V1, not defaulted away |
| Treasury bombing, one roll, d6 and WW2V2 cap | Keep damage mode/roll/side/caps explicit; reject unsupported raid variants rather than reuse CTF arithmetic |
| Income is summed production with standard bonus settings | Collectibility, multiplier and capital gates are generic; blockade/objective/resource systems require detection |
| Total Victory at four Bases | Load VC values, alliance thresholds and scheduled check; never add flag possession/last-player-standing win logic |
| Neutral Flag defenders use ordinary units | Null-player identity and decision policy need explicit handling; null territory is not universally empty |
| `Allied Air Dependents` irrelevant without allied/naval state | Canonical Allied Air Independent defaults false; the CTF raw property is not mapped. Do not invert it or allow it to override the canonical value |
| “Impossible naval states” rejected under CTF-only scope | Validate definition + profile + supplied state: V1 rejects activated V2 features by profile; within supported mechanics, hypothetical states need not be initially reachable |

This is a scope correction, not a claim that the previous report implemented the wrong simulator. Its deferred naval rules and absent fields cannot serve as a complete map-independent specification.

## 10. Proposed Forward Model Rule Profiles V1 and V2

### Project scope and configuration certification

**V1:** Capture the Flag is the first complete conformance scenario, built on generic map topology, unit definitions and phase sequence; land movement, air movement, combat, blitz, artillery support, AA, bombing, purchase, placement, economy/resources, territory capture, ownership transitions and victory/terminal conditions. These are generic algorithms interpreted from XML, not a CTF-specific simulator. Other XMLs using the admitted V1 mechanics must run through the same model without implementation changes. Names, player count, stats, order, graph, production and victory values must remain configuration.

**V2:** naval movement/combat, transports, carriers, amphibious assaults, submarines and naval bombardment. V1 must not implement these. Preserve their definition metadata/source traces and architectural extension points, and reject any effective activation under V1. Supporting the traditional Axis & Allies feature set is not a reason to promote a V2 feature into V1.

CTF's effective WW2V2=true property/default bundle is the first configuration to certify. This is a conformance starting point, not a hard-coded turn sequence or map rule. Other property/version combinations need explicit certification, and every XML admitted by a profile must have exact supported semantics. Generic does not mean every TripleA feature or combination is supported. No uninspected map, including a named Minimap, is automatically certified.

V1 must expose only validation and transition semantics for agent-supplied Actions. No phase, casualty helper, reuse adapter or convenience API may enumerate/generate/prune actions. Explicit player decisions suspend execution into a pending state until the agent supplies the next Action.

| Mechanic | Classification | Profile status | Reason/boundary |
|---|---|---|---|
| Generic XML topology/entities/units/frontiers/sequence and normalization | X/D | REQUIRED_V1 | Load definitions; retain land/water and deferred capability metadata; never CTF-name dispatch |
| Detached hypothetical state, validation, transition and injected RNG | G | REQUIRED_V1 | Complete supplied-world contract, including rejection of invalid supplied Actions |
| Ordinary initialization/phase lifecycle and configurable actors/steps | G/X/D | REQUIRED_V1 | Generic sequencing; unsupported specialized delegates are detected |
| Land/air route legality, movement points and combat/noncombat limits | G/X/D | REQUIRED_V1 | V1 movement scope; no land-only topology assumption |
| Impassability, null-player neutrals and admitted charge/blitz/flyover properties | G/X/D | REQUIRED_V1 | Movement/ownership semantics needed beyond named CTF terrain |
| Blitz and conquered/blitzed/fought history | G/X/D | REQUIRED_V1 | Generic route validation/capture |
| Air land-landing feasibility and cleanup | G/X/D | REQUIRED_V1 | V1 air safety; no carrier fallback |
| Land/air normal/empty/raid battle creation, ordering and rounds | G/X/D | REQUIRED_V1 | Validate supplied battle choice and resolve automatic stages |
| Ordinary rolls/dice sides/HP/delayed deaths/stalemates | G/X/D | REQUIRED_V1 | Generic combat definitions; advanced damage abilities separately gated |
| Conventional artillery support | G/X/D | REQUIRED_V1 | Normalized provider/target types, not unit-name assumptions |
| Supplied casualty/damage and ordinary retreat decisions | G/X/D | REQUIRED_V1 | Agent Actions; model validates/applies, does not choose or enumerate |
| Classic AA/exposure and admitted ordinary/AA Low Luck variants | G/X/D | REQUIRED_V1 | CTF baseline and generic AA/chance paths; advanced batteries optional |
| Ordinary treasury bombing, standard AA and admitted cap/multiplier rules | G/X/D | REQUIRED_V1 | CTF bomber conformance; generic values/targets |
| Territory capture/infrastructure ownership/ordinary allied liberation | G/X/D | REQUIRED_V1 | Generic ownership transitions; advanced conversion/destruction optional |
| Capitals, PU consequences and no/single/multiple-capital metadata | G/X/D | REQUIRED_V1 | Generic economy/ownership predicates; configuration-specific thresholds |
| Basic resources/balances and purchase cost/result vectors/frontiers/build limits | G/X/D | REQUIRED_V1 | Resource identities and quantities are data; CTF PU case first; non-PU basic paths need full certification |
| Ordinary land producer capacities/construction and new/captured factory rules | G/X/D | REQUIRED_V1 | Generic V1 industry; no fixed Base production or factory names |
| Land placement and aggregate capacity/cleanup | G/X/D | REQUIRED_V1 | Validate supplied batches; unplaced retention is a configured rule |
| Basic territory resource income and multiplier/capital gates | G/X/D | REQUIRED_V1 | Generic resource economy; extended generators/objectives excluded |
| Configured VC/alliance checks at scheduled steps and terminal state | G/X/D | REQUIRED_V1 | Generic values/timing; no hard-coded CTF four-base or Flag win |
| Naval movement/combat and naval purchase/placement | G/X/D | PLANNED_V2 | Explicitly outside V1; preserved engine traces |
| Sea canals and naval passage/control rules | G/X/D | PLANNED_V2 | Naval topology semantics, not V1 route implementation |
| Own/allied transports, cargo, loading/unloading and dependent losses | G/X/D | PLANNED_V2 | Deferred transport subsystem; includes non-sea transport mechanics when activated |
| Carriers, capacity/landing, allied air dependencies and defender salvage | G/X/D | PLANNED_V2 | V1 cannot use carrier-based landing |
| Carrier/fighter production and landing variants | G/X/D | PLANNED_V2 | Future carrier scope; individual variants require source/test certification |
| Amphibious dependencies, marine/partial/air retreat interactions | G/X/D | PLANNED_V2 | Future assault subsystem; advanced variants separately certified |
| Submarines/destroyers, first strike, evasion, submergence and naval targeting | G/X/D | PLANNED_V2 | Naval special rules deferred; no V1 combat shortcut |
| Naval bombardment assignment, rounds and casualty timing | G/X/D | PLANNED_V2 | Distinct from V1 strategic bombing |
| Ordinary naval multi-hit repair timing | G/X/D | PLANNED_V2 | Retained naval damage analysis; not required just because classic ships use it |
| Land/air automatic combat-hit repair; repair facilities/damage-induced abilities | G/X/D/O | OPTIONAL_LATER | Base HP/damage accounting is V1; active extra repair/ability systems separately gated |
| Unit-production-damage bombing, repair/operational destruction | G/X/D/O | OPTIONAL_LATER | Beyond initial treasury bombing; cannot convert to treasury loss |
| Territory effects, decimal modified movement/combat bonuses/no-blitz/exclusions | G/X/D/O | OPTIONAL_LATER | Generic extension; no silent omission; keep decimal-capable representation |
| Air bases/bonus movement | G/X/D/O | OPTIONAL_LATER | Separate movement/landing rules |
| Naval bases | G/X/D | PLANNED_V2 | Naval subsystem extension; full variant behavior still requires certification |
| Custom support/roll/AA/non-navally-required first-strike/targeting/choose-best-roll | G/X/D/O | OPTIONAL_LATER | Advanced combat variants beyond conventional V1 scope |
| Nonzero bids | G/X/D/O | OPTIONAL_LATER | Different setup funds/placement; CTF zero steps can be disabled exactly |
| Granted technology and research/activation | G/X/D/O | OPTIONAL_LATER | Dynamic stats/abilities/frontiers; inactive definitions can be retained |
| Convoy routes/blockades and sea-zone economic control | G/X/D | PLANNED_V2 | Naval economy; V1 rejects active use |
| National objectives, upkeep, war bonds, automatic resource/unit creation | G/X/D/O | OPTIONAL_LATER | Extended economics, distinct from required basic resource vectors |
| Scrambling, interceptor/escort air battles | G/X/D/O | OPTIONAL_LATER | Extra decision/dependency systems not included by ordinary V1 air combat |
| Politics/dynamic relationships, triggers/conditions/scripted rule mutations | G/X/D/O | OPTIONAL_LATER | Requires dynamic rule/condition/use state |
| Air/land transport, paratroop transport interactions | G/X/D | PLANNED_V2 | Transport mechanics deferred; related tech/special-delegate systems not automatically certified |
| Rockets and nontransport special moves | G/X/D/O | OPTIONAL_LATER | Separate damage/movement mechanics |
| Fuel, stacking/player movement/attack/placement restrictions | G/X/D/O | OPTIONAL_LATER | Additional legality/economy constraints |
| Factory upgrades, consumes/requires units, capture conversion/destruction | G/X/D/O | OPTIONAL_LATER | Advanced construction/ownership transformations |
| Economic/Pacific/trigger/legacy military victory algorithms | G/X/D/O | OPTIONAL_LATER | Distinct victory mechanisms; required V1 configured VC semantics remain generic |
| Combined/interleaved turns with carrier or transport interactions | G/X/D | PLANNED_V2 | Generic sequence representation is V1; these activated interactions are V2 |
| Other unverified version bundles/unusual delegates/random setup | G/X/D/O | OPTIONAL_LATER | Certify exact semantics, never substitute baseline rules |
| Kamikaze/suicide-hit/specialized dynamic spawning | G/X/D/O | OPTIONAL_LATER | Additional ordering/chance rules |
| Action generation/enumeration/templates/search-space generation/pruning/planning/beliefs/policy selection by the model | Architectural boundary | UNSUPPORTED | Entirely agent-owned; definition facts and supplied-action validation remain allowed |
| Live edit/UI undo/network/PBEM/GUI as simulated actions | O | UNSUPPORTED | Committed local transitions; observation/export adapters separate |
| Arbitrary unregistered Java delegates/attachments or non-A&A loaders | O | UNSUPPORTED | Beyond supported semantics |
| Silent fallback, CTF-name dispatch or live-state completion | O | UNSUPPORTED | Violates fidelity/information contract |

REQUIRED_V1 is the current implementation obligation; unresolved required configurations cannot be certified until resolved. PLANNED_V2 is specifically scheduled future scope and is **unsupported by V1**. OPTIONAL_LATER is beyond these baseline commitments and unsupported until separately admitted. UNSUPPORTED is outside the project/model responsibility contract. Preserving source analysis does not imply implementation or certification, and planned V2 does not automatically include every historical variant.

## 11. Unsupported-feature detection requirements

The loader must produce a support result for the **effective setup and selected profile**, not just check XML syntax or familiar unit names. Detection inspects configured mechanics; it is not an action generator or search-space analyzer. V1 must reject activated V2 mechanics before execution, retaining a specific reason rather than silently ignoring them. A water node or an air unit's transport/carrier compatibility field alone is not necessarily activation; actual unit/cargo/landing/naval rule use is.

1. Parse safely and resolve variables/foreach, class/property aliases, attachments, typed values, initial relationships and setup overrides according to the pinned parser. Record XML locations and provenance for normalized effects.
2. Validate entity references, domains, capacities, damage values, costs/results, step interpretation and required defaults. Do not erase unknown options while parsing. For V1, initial cargo activates PLANNED_V2 and must be refused; cargo-assignment feasibility analysis is preserved for V2 certification.
3. Inspect loader class, scheduled delegate classes, delegate subclasses, attachment classes, every gameplay-affecting option/property/step property, initial technology/damage, and all ways of creating/reaching capabilities.
4. Expand shorthands and initialization effects before matching against a versioned capability registry. `isFactory`, `isAA`, `isSub`, artillery and legacy frontier modifications must expose their derived mechanics.
5. Match **values and combinations**, not just names: e.g. default classic AA may be supported while offensive/multi-round AA is not; both ordinary and pending-new-carrier landing require V2 under V1. Basic additional-resource vectors are V1 obligations, while automatic generators/upkeep are separately optional. Include version bundle, movement cost mode and admitted phase patterns.
6. Distinguish inactive declarations from effective producers. CTF declares an unscheduled technology delegate and false tech flags, so declarations alone do not reject it. But active starting tech, a tech step, trigger activation or mutable rule changes make it effective. If inactivity cannot be proved, report UNSUPPORTED with the reason.
7. Never prove inactivity using only the XML's initial board. Inspect initialization, purchase/placement frontiers, producers, applicable properties/triggers, scheduled steps and supplied-state feature use. V1 must reject active naval movement/combat, transports, carriers, amphibious assaults, submarines or naval bombardment. It may retain CTF's declared but inactive naval unit types and transportCost/carrierCost compatibility values: retention is **not execution support**. Record these as PLANNED_V2 metadata; the V1 state/action validation boundary must prohibit activating them through hypothetical injection, cargo links, raid/landing dependencies or supplied Actions. Other inactive declarations need a sound proof/explicit profile restriction too; otherwise refuse.
8. Classify unknown properties with an explicit registry: known display metadata may be ignored; known inert legacy spelling may be reported as inert after verification; unknown gameplay-capable names require refusal/manual review. Never treat every unknown property as UI metadata or every missing property as false.
9. Record all rejection reasons, the activating location/value, derived dependency and suggested later profile. Support checks must also run against supplied state's feature use: an XML can pass while the supplied state introduces active acquired tech, unsupported damage or omitted provenance.
10. Certification must cover nested helpers/client settings: freeze permitted settings and reject active effects that would otherwise read external configuration. Fingerprint normalized definition, selected profile and overrides.

Example eventual diagnostics (illustrative, not current executable output):

```text
SUPPORTED: profile forward-model-v1; engine=<pinned hash>; effectiveDefinition=<hash>
UNSUPPORTED_BY_PROFILE_V1:
  active transport capacity/cargo or transport purchase rule
    -> transport mechanics require Forward Model V2
  naval unit initialization/purchase/placement -> naval movement/combat require Forward Model V2
  carrier-based air landing -> carriers require Forward Model V2
  amphibious assault dependencies -> amphibious assaults require Forward Model V2
  active submarine capability -> submarines require Forward Model V2
  active shore bombardment -> naval bombardment requires Forward Model V2
UNSUPPORTED (OPTIONAL_LATER) because:
  unitAttachment("bomber").chooseBestRoll=true -> best-of-multiple raid/combat rolls
  property("Damage From Bombing Done To Units Instead Of Territories")=true
    -> persistent producer damage + repair/operational rules (OPTIONAL_LATER)
  scheduled delegate "PoliticsDelegate" -> dynamic relationship transitions
  triggerAttachment("reinforcement").placement -> scripted unit creation
INCOMPLETE_SIMULATION_STATE because:
  current phase Battle; attackingFromMap missing for battle <id>
```

At the state/action boundary, an activated V2 subsystem must receive the same profile-specific reason even when its declaration was inert in the loaded XML. For an ordinary supported mechanic, an illegal supplied move/purchase/casualty receives Invalid(reason), not a fabricated substitute or generated alternatives. Missing historical/transient fields receive an incomplete-state error. These distinguish unsupported rules, incomplete state and invalid Actions.

Names in diagnostics must use actual canonical parser keys and XML locations; the example is descriptive. A research option to use an approximation must create a separately labeled non-fidelity model/profile, not return SUPPORTED under this contract.

## 12. TripleA code reuse/refactoring opportunities

The reuse inventory retains V2 opportunities as preparation only. V1 reuse must restrict execution to admitted land/air/economy/ownership rules; passing through a shared TripleA helper must not accidentally enable its naval branches. No reused validator, UI helper, delegate or adapter may become an action generator. Engine remote-choice calls must suspend into pending decision state and consume a subsequent supplied Action, not call WeakAi/default policies or return enumerated action candidates.

### Detached GameData: established evidence and limits

`GameDataUtils.cloneGameData` serializes/restores a deep copy with configurable history behavior and requires the source lock. `translateIntoOtherGameData` translates references by serialization. `GameParser` can create GameData from XML; constructed hypothetical worlds can use engine data objects and local changes. **None of these mechanisms requires live authoritative GameData as the planning source.** Use an XML-derived detached template plus the supplied state's complete overrides, including removal of template units not in the hypothesis. Never clone a live game to fill hidden fields during a rollout. [Clone][Parser][Changes]

`BattleCalculator` provides concrete evidence of detached combat: clone/load data, translate supplied participant lists, replace the battle site's units, construct `MustFightBattle`, inject `DummyDelegateBridge`, fight repeatedly and invert selected changes. This establishes feasibility of detached battle execution, not correctness of arbitrary whole-game transitions. [Calc]

**Critical limits:** `battle.setHeadless(true)` changes several paths, including artificial retreat-site handling; `CasualtySelector` suppresses dependencies in headless mode. `DummyDelegateBridge.addChange` applies only `UnitDamageReceivedChange` and recursively selected composite children, ignoring other changes. It owns a hidden PlainRandomSource and fixed dummy policies. Therefore neither it nor aggregate odds calculation is a drop-in full forward-model transition. [Calc][Dummy][MFB][Casualty]

| Major mechanic | Direct reuse A | Extract/refactor B | Independent C judgment |
|---|---|---|---|
| XML/default normalization | GameParser, mapper, property getters, attachment setters on detached data | Export effective definition/provenance; restrict class registry and freeze overrides | High duplication risk; reuse parser semantics where practical |
| Graph/routes (V1), canals (V2 preparation) | GameMap/Route, MoveValidator helpers, Matches | Replace implicit delegate/sequence lookups with local context | Reasonable for bounded subset after differential tests |
| Move validation/air (V1), carriers/transports (V2 preparation) | MoveValidator/ AirMovementValidator / TransportTracker on complete detached objects | Decouple trackers, phase helpers, undo history and synchronous requests | Complex carrier/cargo legality makes direct independent rewrite risky |
| Move execution/capture | ChangeFactory with detached `performChange` | MovePerformer, BattleTracker ownership and move stages into local event runner | Needed for compact non-Java model, but source tests are mandatory |
| Battle creation/order | BattleTracker and dependency methods | BattleDelegate lifecycle and selected-battle decision interface | Small graph logic feasible; empty/raid/landing dependencies easy to miss |
| Power/support/dice | CombatValueBuilder, PowerStrengthAndRolls, RolledDice/LowLuckDice, injected RandomDiceGenerator | Remove GameSequence/UnitAttachment reach-through and extract parameter views | Good independent target only after conventional scope is explicit |
| Casualties/retreat | Target/HP helpers, casualty validation pieces | Replace remote Player calls with explicit pending decisions; preserve normalization | Avoid embedding Default/WeakAi policy as rules |
| Full normal combat | MustFightBattle operates with detached data/bridge | Local full-change bridge and resumable decisions; isolate UI/settings; avoid changed headless semantics | Independent state machine viable but expensive; detached engine is stronger oracle |
| AA/raid (V1), bombardment (V2 preparation) | Dice and target helper reuse | Route exposure, raid stages/loss ledger, bombardment allocation | Must preserve event and removal timing |
| Capital/liberation/income | TerritoryAttachment predicates, property accessors, basic resource calculations | Takeover/lifecycle checks, client-free BonusIncome behavior | Straightforward only after multiple capitals/allied restoration tests |
| Purchase/placement | Cost aggregation, UnitUtils potential, ChangeFactory | Validate supplied purchase rules against frontier data; consume supplied decisions; producer snapshots/allocation and cleanup | Ordinary purchase easy; multiple-producer sea placement is harder |
| Sequence/terminal | GameSequence helpers and VC computations | Client-free once-only sequencer and local victory callbacks | Reasonable independent implementation, preserving skipped vs automatic steps |

Recommended next engineering investigation: build an **offline authoritative oracle adapter**, independent of the live session, from parsed/constructed GameData + SimulationState + explicit decisions/RNG. Demonstrate isolation and full applied changes before choosing extraction versus a compact independent kernel. That is a future investigation, not code delivered here. Production can still use an independent model if conformance evidence is strong.

Audit requirements for reuse: no original-template mutable aliases across branches; all Unit/Player/Territory references translated consistently; local delegate save state/sequence populated; full changes applied; RNG injected; no automatic WeakAi/live bridges/settings; cloned history not shared if relevant; no UI/network callbacks; no headless shortcut that changes admitted rules. Cloning only GameData board fields without reconstructing required delegate/transient state is insufficient.

## 13. Unresolved questions

UNRESOLVED means no invented rule is supplied. These items must be inspected/tested before the corresponding configuration is certified. Allied cargo, carrier landing, canals, naval dependency cleanup and multiple-producer sea placement questions below are **PLANNED_V2**, not blockers for CTF/V1 conformance. Ordinary casualty Actions, purchase validation, basic resources, land placement, phase/terminal behavior and isolation remain V1 questions.

| Question | Established boundary; required next evidence |
|---|---|
| XML normalization integration | Static mapper resolves CTF Allied Air Dependents as an unmapped inert key. An offline XML load still needs to verify all effective defaults/shorthand output and reject unknown gameplay options. |
| Exact reset interval for allied cargo across interleaved/combined phases | TransportTracker uses loaded/unloaded flags and global reset changes. Trace GameStep reset defaults plus MoveDelegate lifecycle for each admitted sequence; test two allied players across combat/noncombat/turn changes. |
| Mixed carrier landing feasibility and stable identity choices | AirMovementValidator sorts aircraft and greedily considers capacity/origins; TODOs mention movement costs. Test heterogeneous costs/ranges, allied occupancy, carrier damage, multiple landing sites. Reject modified fractional movement initially. |
| Combat-hit repair interactions | Ordinary naval no-facility repair is PLANNED_V2; land/air automatic repair and facilities/conversion/damage-induced abilities are OPTIONAL_LATER. Traced rate/scope is preserved in §7.11; each active variant still needs tests. |
| Multiple overlapping canals | Trace full any/all canal property interactions and tests; test excluded unit types and just-conquered guard combinations. Ordinary single canal source behavior is established. |
| Full casualty normalization/decision representation | Engine can auto-pick equivalents and reassign aircraft identities. Inspect entire selector/order-of-losses validation plus tests to establish which identity choices are legitimate research actions. |
| Complete naval dependency cleanup | Trace `unitsLostInPrecedingBattle`/dependent removal and losing transport, retreat/reload, empty landing and multiple source zones. A dependency edge alone is not enough. |
| Multiple-producer placement order/reallocation | Inspect full `checkProduction/getMaxUnitsToBePlacedFrom` and placement tests, including same sea zone with two factories and sequential batches. No fixed greedy simplification certified. |
| Exact purchase protocol vs legal action set | RPC body differs from lifecycle/UI permissions. Verify nonnegative quantities, frontier membership/capital/phase gates with remote-interface and UI call paths; document intended legal committed actions. |
| Nonzero bid repeated purchase calls | Previous CTF report flags per-call `spent` handling. Inspect complete BidPurchaseDelegate and UI contract; offline repeated-call tests. OPTIONAL_LATER while unresolved. |
| Simultaneous victory and client stop integration | signalGameOver retains the first declaration while its caller iterates alliances; inspect alliance collection order and server/client stop paths, then test simultaneous qualification. Do not invent a player-order or draw rule. |
| Unit damage/repair/operational effects | Branch identified in raid/UnitUtils/purchase; inspect Unit damage-limit accessors, destruction, repair costs, factory exceptions and placement. OPTIONAL_LATER, reject active mode meanwhile. |
| Full unknown-feature registry | This is not a complete inventory of every attachment option/property. Generate/review registry from mapper, mutable properties, attachment validations and all reachable property usages before shipping a loader. |
| Named Minimap scenario compatibility | No concrete Minimap XML was supplied/inspected. Parse and certify an actual pinned map; do not claim compatibility by name. |
| Complete source isolation under reused delegates | Offline integration test needed: construct contradictory hypothetical state and inspect callbacks/reference graph to prove no live-state/settings access. No reuse prototype was built here. |

Optional mechanics' identified entry points are discovery evidence, not full algorithm specifications. Their implementation must be analyzed when promoted to a later profile. V1's unresolved required edges block that configuration's certification; V2 questions are retained future work and must not expand V1. Basic non-PU cost/result/territory-income and land/air HP cases need additional complete traces/tests before their XML configurations can be certified.

## 14. Recommended fidelity tests

**CTF is the first complete V1 conformance suite**, covering its full initialization, all recurring phases, land/air decisions, ordinary combat, blitz/support/AA/raids, purchases/placement, economy, ownership and scheduled terminal checks. A few passing battles or normalization checks do not constitute complete CTF conformance. No tests below were executed for this analysis.

All comparisons run against an **offline pinned TripleA oracle** initialized exclusively from the same definition and supplied hypothetical state. Neither validation nor rollout may consult the live session. The test harness/agent constructs and supplies Actions, including deliberately invalid candidates and every explicit player decision. Compare `validate` results/reasons and `transition` outcomes; do not require or add a legal_actions/generate_actions interface.

| V1 test family | Essential cases and comparison |
|---|---|
| Complete CTF baseline | XML hash; 29/32 topology; d6/null Flag defenders; shorthand/support; 19 pieces/103 production; initial funds vs income; zero bid skip/inactive tech and naval declarations; complete phase-to-phase action/dice replay through victory |
| Agent/model boundary | Invalid Actions constructed by agent rejected locally; no state mutation or RNG draw on rejection; pending casualty/retreat/battle/target decisions wait for supplied Actions; no internal policy/generation/enumeration/pruning; null-player choices supplied externally |
| Phase boundaries | No-input automatic effects vs max-run skips; resume after income without duplicate collection; moved armour exhausted for NCM; air remaining movement; delayed EndRound victory |
| Hypothetical isolation | Template/live board replaced by supplied hypothesis; hidden defenders added/removed; live callbacks fail; identical supplied inputs unaffected by authoritative board changes |
| Movement/blitz | Split calls, hostile occupied intermediate, AA/factory WW2V2 restrictions, neutral charge/flyover/blitz permissions, impassability, fresh nonblitz vs blitzed captures |
| Air land landing | Different ranges, hostile/neutral/freshly captured land invalid, feasible return at commitment, actual landing/cleanup; carrier-dependent solution rejected by V1 |
| Land/air battle ordering | Supplied independent battle choice, ongoing battle cannot switch, raid-before-land, initial hostile co-location, air enters then leaves, infrastructure-only target |
| Combat rounds | Ordinary return fire, AA early removal, mutual elimination, admitted land/air HP accounting, stalemate/round cap, resume stored outcomes without reroll |
| Conventional support | More/fewer providers than targets, renamed types, surviving support changes, strength caps and same normalized semantics under differing XML stats |
| Casualties/ordinary retreat | Defender Action actor, target/overdamage validation, equivalent-air normalization, origin-derived retreat, air-only origins, air remains for later landing; no policy-chosen retreat |
| AA/chance | Route origin/middle/destination handling, split-route exposure, admitted Low Luck/ordinary target distributions, classic battery limits, no undo/reroll after chance |
| Bombing | Supplied raid/target vs ordinary attack, AA kills bomber, per-bomber vs per-turn caps, cash floor/multiplier, landing after raid; no shore bombardment execution |
| Ownership/capitals | Air-only no capture, infrastructure originals, land-only allied liberation, multiple/no capitals, treasury transfer/destruction, third-party recapture, movement after capital loss |
| Land industry/resources | Different frontiers/cost/result vectors/bundles/build limits; held purchases; original/captured/new producers; multiple land batches; no naval placement; unplaced retention/loss; basic extra resource paths after certification |
| Income/terminal | New captures count income, capital gate/multiplier, VC values independent of names/production, generic alliances/thresholds/check timing, no implicit Flag/elimination/horizon victory |
| V2 refusal | Initialized or purchasable naval units, active cargo/transport, carrier landing, amphibious dependencies, submergence and bombardment: UNSUPPORTED_BY_PROFILE_V1 with V2 reason/location; CTF inert declarations retained |
| Other refusal | Active triggers/tech/scramble, unit-damage bombing, unadmitted effects, unknown gameplay delegate/property/attachment, politics, nonzero bid; no approximation |

### Additional small V1 XML/configuration fixtures

These are recommended future fixture definitions, not XMLs or simulator code implemented here. Each uses only admitted V1 mechanics and the same kernel:

| Fixture | Generic behavior exercised |
|---|---|
| Renamed CTF derivative | Rename players/territories/ordinary types, change statistics/prices/production/VC thresholds and permutation of IDs; honor genuine legacy initialization names separately; no CTF dispatch |
| Three-territory land corridor | Two players, neutral middle node, configurable blitz unit, hostile blockers/infrastructure and neutral charge; compare valid and invalid agent-supplied routes |
| Air-return triangle | Friendly launch/landing nodes and hostile raid/combat target; vary ranges and fresh captures; distinguish landing feasibility from actual cleanup |
| AA/support microbattle | Differently named artillery provider/recipient, AA, fighters/bomber; ordinary vs admitted Low Luck, target/casualty Actions and survivor support recomputation |
| Two land producers | Different original-owner/production capacities, new/captured factory and multi-batch placement without adjacent sea deployment |
| Land-only allied liberation | Three players with a fixed alliance, original ownership and no/single/multiple-capital configurations; capture and return of land/infrastructure |
| Basic resource/frontier variant | Different prices/bundled outputs and a second basic resource with territory income after its exact path is traced; no objectives, upkeep, tech or automatic generators |
| Reordered phase/VC variant | Different player count/names, admitted sequence/step-property arrangement, startup max runs, VC weights/thresholds and check timing |
| Air-over-water graph | Water node traversed by air between land territories, no naval units, transports/carriers/subs/amphibious/bombardment activation; proves topology is not land-only |

V2 naval/canal/cargo/carrier/amphibious/sub/bombardment conformance cases from the preserved source analysis are **future V2 tests**. During V1, activating those mechanics tests explicit refusal, not their simulation. Advanced optional variants likewise test refusal until admitted.

Use identical supplied Action transcripts and injected random-request transcripts to compare validation, state changes, pending decision actor/context, resources, unit identities/damage, sequence and terminal state. Chance distribution tests for small land/air combat, AA and raids complement exact replay. Compare outcomes for supplied candidates without asking the model to enumerate alternatives. Complete state comparison and metamorphic renaming/configuration/branch-isolation tests establish map independence; only changing XML should be needed for another certified V1 configuration.

### Revision-pinned source index

References used above point to the actual implementation, not general Axis & Allies descriptions. Method names in §7 identify the relevant transition/validation path. The inspected checkout is clean at the pinned engine revision. This index also provides the precise locations for follow-up unresolved inspection.

| Reference | Revision-pinned implementation |
|---|---|
| Parser | [engine/data/gameparser/GameParser.java][Parser] |
| Mapper | [engine/data/gameparser/XmlGameElementMapper.java][Mapper] |
| Legacy | [engine/data/gameparser/LegacyPropertyMapper.java][Legacy] |
| Map | [engine/data/GameMap.java][Map] |
| Sequence | [engine/data/GameSequence.java][Sequence] |
| Step | [engine/data/GameStep.java][Step] |
| Unit | [engine/data/Unit.java][Unit] |
| RelTracker | [engine/data/RelationshipTracker.java][RelTracker] |
| Route | [engine/data/Route.java][Route] |
| Server | [engine/framework/ServerGame.java][Server] |
| Clone | [engine/framework/GameDataUtils.java][Clone] |
| Changes | [engine/data/changefactory/ChangeFactory.java][Changes] |
| Production | [engine/data/ProductionRule.java][Production] |
| UA | [triplea/attachments/UnitAttachment.java][UA] |
| TA | [triplea/attachments/TerritoryAttachment.java][TA] |
| PA | [triplea/attachments/PlayerAttachment.java][PA] |
| RA | [triplea/attachments/RelationshipTypeAttachment.java][RA] |
| Support | [triplea/attachments/UnitSupportAttachment.java][Support] |
| TEA | [triplea/attachments/TerritoryEffectAttachment.java][TEA] |
| Canal | [triplea/attachments/CanalAttachment.java][Canal] |
| Props | [triplea/Properties.java][Props] |
| Base | [triplea/delegate/BaseTripleADelegate.java][Base] |
| Phase | [triplea/delegate/GameStepPropertiesHelper.java][Phase] |
| Init | [triplea/delegate/InitializationDelegate.java][Init] |
| MD | [triplea/delegate/MoveDelegate.java][MD] |
| MV | [triplea/delegate/move/validation/MoveValidator.java][MV] |
| MP | [triplea/delegate/MovePerformer.java][MP] |
| FlyAA | [triplea/delegate/AaInMoveUtil.java][FlyAA] |
| Air | [triplea/delegate/move/validation/AirMovementValidator.java][Air] |
| Matches | [triplea/delegate/Matches.java][Matches] |
| Effects | [triplea/delegate/TerritoryEffectHelper.java][Effects] |
| Transport | [triplea/delegate/TransportTracker.java][Transport] |
| TransportUtils | [triplea/util/TransportUtils.java][TransportUtils] |
| BD | [triplea/delegate/battle/BattleDelegate.java][BD] |
| BT | [triplea/delegate/battle/BattleTracker.java][BT] |
| Dependent | [triplea/delegate/battle/DependentBattle.java][Dependent] |
| MFB | [triplea/delegate/battle/MustFightBattle.java][MFB] |
| SBR | [triplea/delegate/battle/StrategicBombingRaidBattle.java][SBR] |
| BS | [triplea/delegate/battle/steps/BattleStep.java][BS] |
| Power | [triplea/delegate/power/calculator/CombatValueBuilder.java][Power] |
| Offense | [triplea/delegate/power/calculator/MainOffenseCombatValue.java][Offense] |
| Defense | [triplea/delegate/power/calculator/MainDefenseCombatValue.java][Defense] |
| PowerUnits | [triplea/delegate/power/calculator/PowerStrengthAndRolls.java][PowerUnits] |
| Roll | [triplea/delegate/dice/RollDiceFactory.java][Roll] |
| Rolled | [triplea/delegate/dice/calculator/RolledDice.java][Rolled] |
| LowLuck | [triplea/delegate/dice/calculator/LowLuckDice.java][LowLuck] |
| SupportCalc | [triplea/delegate/power/calculator/SupportCalculator.java][SupportCalc] |
| Available | [triplea/delegate/power/calculator/AvailableSupports.java][Available] |
| Casualty | [triplea/delegate/battle/casualty/CasualtySelector.java][Casualty] |
| AACasualty | [triplea/delegate/battle/casualty/AaCasualtySelector.java][AACasualty] |
| MainCasualty | [triplea/delegate/battle/steps/fire/SelectMainBattleCasualties.java][MainCasualty] |
| MarkCasualties | [triplea/delegate/battle/steps/fire/MarkCasualties.java][MarkCasualties] |
| Participants | [triplea/delegate/battle/steps/fire/general/FiringGroupSplitterGeneral.java][Participants] |
| Targets | [triplea/delegate/battle/steps/fire/general/TargetGroup.java][Targets] |
| FS | [triplea/delegate/battle/steps/fire/firststrike/ClearFirstStrikeCasualties.java][FS] |
| FirstStrike | [triplea/delegate/battle/steps/fire/firststrike/OffensiveFirstStrike.java][FirstStrike] |
| EndBattle | [triplea/delegate/battle/steps/change/CheckGeneralBattleEnd.java][EndBattle] |
| Stalemate | [triplea/delegate/battle/steps/change/CheckStalemateBattleEnd.java][Stalemate] |
| Unprotected | [triplea/delegate/battle/steps/change/RemoveUnprotectedUnits.java][Unprotected] |
| Retreat | [triplea/delegate/battle/steps/retreat/OffensiveGeneralRetreat.java][Retreat] |
| RetreatChecks | [triplea/delegate/battle/steps/RetreatChecks.java][RetreatChecks] |
| GeneralRetreat | [triplea/delegate/battle/steps/retreat/RetreaterGeneral.java][GeneralRetreat] |
| SubRetreat | [triplea/delegate/battle/steps/retreat/OffensiveSubsRetreat.java][SubRetreat] |
| DefSubRetreat | [triplea/delegate/battle/steps/retreat/DefensiveSubsRetreat.java][DefSubRetreat] |
| Evade | [triplea/delegate/battle/steps/retreat/EvaderRetreat.java][Evade] |
| SubAir | [triplea/delegate/battle/steps/retreat/sub/SubmergeSubsVsOnlyAirStep.java][SubAir] |
| Bombard | [triplea/delegate/battle/steps/fire/NavalBombardment.java][Bombard] |
| BombardClear | [triplea/delegate/battle/steps/change/ClearBombardmentCasualties.java][BombardClear] |
| BombardPower | [triplea/delegate/power/calculator/BombardmentCombatValue.java][BombardPower] |
| BattleAA | [triplea/delegate/battle/steps/fire/aa/DefensiveAaFire.java][BattleAA] |
| AAGroups | [triplea/delegate/battle/steps/fire/aa/FiringGroupSplitterAa.java][AAGroups] |
| AAPower | [triplea/delegate/power/calculator/AaPowerStrengthAndRolls.java][AAPower] |
| Purchase | [triplea/delegate/PurchaseDelegate.java][Purchase] |
| Place | [triplea/delegate/AbstractPlaceDelegate.java][Place] |
| Income | [triplea/delegate/AbstractEndTurnDelegate.java][Income] |
| EndTurn | [triplea/delegate/EndTurnDelegate.java][EndTurn] |
| ER | [triplea/delegate/EndRoundDelegate.java][ER] |
| Bid | [triplea/delegate/BidPurchaseDelegate.java][Bid] |
| UnitUtils | [triplea/UnitUtils.java][UnitUtils] |
| Bonus | [triplea/util/BonusIncomeUtils.java][Bonus] |
| Calc | [triplea/odds/calculator/BattleCalculator.java][Calc] |
| Dummy | [triplea/odds/calculator/DummyDelegateBridge.java][Dummy] |

[Parser]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/engine/data/gameparser/GameParser.java
[Mapper]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/engine/data/gameparser/XmlGameElementMapper.java
[Legacy]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/engine/data/gameparser/LegacyPropertyMapper.java
[Map]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/engine/data/GameMap.java
[Sequence]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/engine/data/GameSequence.java
[Step]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/engine/data/GameStep.java
[Unit]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/engine/data/Unit.java
[RelTracker]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/engine/data/RelationshipTracker.java
[Route]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/engine/data/Route.java
[Server]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/engine/framework/ServerGame.java
[Clone]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/engine/framework/GameDataUtils.java
[Changes]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/engine/data/changefactory/ChangeFactory.java
[Production]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/engine/data/ProductionRule.java
[UA]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/attachments/UnitAttachment.java
[TA]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/attachments/TerritoryAttachment.java
[PA]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/attachments/PlayerAttachment.java
[RA]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/attachments/RelationshipTypeAttachment.java
[Support]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/attachments/UnitSupportAttachment.java
[TEA]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/attachments/TerritoryEffectAttachment.java
[Canal]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/attachments/CanalAttachment.java
[Props]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/Properties.java
[Base]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/BaseTripleADelegate.java
[Phase]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/GameStepPropertiesHelper.java
[Init]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/InitializationDelegate.java
[MD]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/MoveDelegate.java
[MV]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/move/validation/MoveValidator.java
[MP]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/MovePerformer.java
[FlyAA]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/AaInMoveUtil.java
[Air]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/move/validation/AirMovementValidator.java
[Matches]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/Matches.java
[Effects]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/TerritoryEffectHelper.java
[Transport]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/TransportTracker.java
[TransportUtils]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/util/TransportUtils.java
[BD]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/battle/BattleDelegate.java
[BT]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/battle/BattleTracker.java
[Dependent]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/battle/DependentBattle.java
[MFB]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/battle/MustFightBattle.java
[SBR]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/battle/StrategicBombingRaidBattle.java
[BS]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/battle/steps/BattleStep.java
[Power]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/power/calculator/CombatValueBuilder.java
[Offense]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/power/calculator/MainOffenseCombatValue.java
[Defense]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/power/calculator/MainDefenseCombatValue.java
[PowerUnits]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/power/calculator/PowerStrengthAndRolls.java
[Roll]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/dice/RollDiceFactory.java
[Rolled]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/dice/calculator/RolledDice.java
[LowLuck]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/dice/calculator/LowLuckDice.java
[SupportCalc]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/power/calculator/SupportCalculator.java
[Available]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/power/calculator/AvailableSupports.java
[Casualty]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/battle/casualty/CasualtySelector.java
[AACasualty]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/battle/casualty/AaCasualtySelector.java
[MainCasualty]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/battle/steps/fire/SelectMainBattleCasualties.java
[MarkCasualties]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/battle/steps/fire/MarkCasualties.java
[Participants]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/battle/steps/fire/general/FiringGroupSplitterGeneral.java
[Targets]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/battle/steps/fire/general/TargetGroup.java
[FS]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/battle/steps/fire/firststrike/ClearFirstStrikeCasualties.java
[FirstStrike]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/battle/steps/fire/firststrike/OffensiveFirstStrike.java
[EndBattle]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/battle/steps/change/CheckGeneralBattleEnd.java
[Stalemate]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/battle/steps/change/CheckStalemateBattleEnd.java
[Unprotected]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/battle/steps/change/RemoveUnprotectedUnits.java
[Retreat]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/battle/steps/retreat/OffensiveGeneralRetreat.java
[RetreatChecks]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/battle/steps/RetreatChecks.java
[GeneralRetreat]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/battle/steps/retreat/RetreaterGeneral.java
[SubRetreat]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/battle/steps/retreat/OffensiveSubsRetreat.java
[DefSubRetreat]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/battle/steps/retreat/DefensiveSubsRetreat.java
[Evade]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/battle/steps/retreat/EvaderRetreat.java
[SubAir]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/battle/steps/retreat/sub/SubmergeSubsVsOnlyAirStep.java
[Bombard]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/battle/steps/fire/NavalBombardment.java
[BombardClear]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/battle/steps/change/ClearBombardmentCasualties.java
[BombardPower]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/power/calculator/BombardmentCombatValue.java
[BattleAA]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/battle/steps/fire/aa/DefensiveAaFire.java
[AAGroups]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/battle/steps/fire/aa/FiringGroupSplitterAa.java
[AAPower]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/power/calculator/AaPowerStrengthAndRolls.java
[Purchase]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/PurchaseDelegate.java
[Place]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/AbstractPlaceDelegate.java
[Income]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/AbstractEndTurnDelegate.java
[EndTurn]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/EndTurnDelegate.java
[ER]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/EndRoundDelegate.java
[Bid]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/delegate/BidPurchaseDelegate.java
[UnitUtils]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/UnitUtils.java
[Bonus]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/util/BonusIncomeUtils.java
[Calc]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/odds/calculator/BattleCalculator.java
[Dummy]: https://github.com/triplea-game/triplea/blob/2a8b4401d9a6442636fb1d90bf5659b15fa46343/game-app/game-core/src/main/java/games/strategy/triplea/odds/calculator/DummyDelegateBridge.java
[CTF]: https://github.com/triplea-maps/capture_the_flag/blob/8f5dd75ca3014713a3743b4904b9d5af76a73d26/map/games/capture_the_flag.xml


## Changes from Previous Rule Model

1. Action generation, candidate selection, enumeration, templates, pruning, MCTS/planning, beliefs and policies belong entirely to the agent.
2. The forward model is limited to interpreting/validating supplied Actions and applying transition semantics, including automatic/stochastic effects; explicit player decisions remain agent-supplied Actions.
3. V1 scope is reduced to the generic land/air/combat/industry/resource/ownership/terminal mechanics above, with Capture the Flag as its first complete conformance scenario and additional XML-only V1 fixtures.
4. Naval movement/combat, transports, carriers, amphibious assaults, submarines and naval bombardment move to PLANNED_V2. Their source analysis is retained as preparation; V1 rejects active use explicitly and keeps architecture/definition extensible.
