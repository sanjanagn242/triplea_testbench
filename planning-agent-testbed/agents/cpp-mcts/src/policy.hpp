// Fast heuristic policy, move priors and state evaluation for the C++ MCTS agent.
//
// Decisions are made in C++ from a flat state View; every proposed action is checked with the
// testbench forward-model library's validate() before it is returned, so no rules live here.
#pragma once

#include <algorithm>
#include <cmath>
#include <map>
#include <tuple>
#include <vector>

#include <triplea_fm/forward_model.hpp>

namespace agent {

using fm::Action;
using fm::Static;
using fm::Unit;
using fm::View;

class Policy {
 public:
  Policy(fm::ForwardModel& model, const Static& st) : fm_(model), st_(st) {
    // Agent heuristic: victory cities and capitals dominate territory value.
    for (int t = 0; t < st_.territories; ++t)
      value_.push_back(st_.production[t] + 12 * st_.victory[t] + (st_.capital_of[t] != fm::NULL_PLAYER ? 15 : 0));
  }

  // ---- shared helpers ----
  bool enemy_territory(const View& v, int p, int t) const {
    return v.owner[t] != p && !st_.allied(p, v.owner[t]);
  }
  bool neutral_land(const View& v, int t) const { return !st_.water[t] && v.owner[t] == fm::NULL_PLAYER; }
  bool can_land(const View& v, int p, int t) const {
    return !st_.water[t] && !v.conquered[t] && v.owner[t] != fm::NULL_PLAYER && st_.allied(p, v.owner[t]);
  }
  std::vector<const Unit*> units_at(const View& v, int t) const {
    std::vector<const Unit*> out;
    for (const Unit& u : v.units)
      if (u.loc == t) out.push_back(&u);
    return out;
  }
  std::vector<const Unit*> enemy_combat(const View& v, int p, int t) const {
    std::vector<const Unit*> out;
    for (const Unit& u : v.units)
      if (u.loc == t && !st_.allied(p, u.owner) && !st_.types[u.type].infrastructure) out.push_back(&u);
    return out;
  }
  int attack_power(const std::vector<const Unit*>& units) const {
    int support = 0, total = 0;
    for (const Unit* u : units) support += st_.types[u->type].artillery ? 1 : 0;
    for (const Unit* u : units) {
      const auto& t = st_.types[u->type];
      if (t.infrastructure) continue;
      int a = t.attack;
      if (t.supportable && support > 0) { ++a; --support; }
      total += a;
    }
    return total;
  }
  int defense_power(const std::vector<const Unit*>& units) const {
    int total = 0;
    for (const Unit* u : units)
      if (!st_.types[u->type].infrastructure) total += st_.types[u->type].defense;
    return total;
  }
  std::vector<int> target_distance(const View& v, int p) const {
    std::vector<int> targets;
    for (int t = 0; t < st_.territories; ++t)
      if (!st_.water[t] && v.owner[t] != p && !st_.allied(p, v.owner[t]) && value_[t] > 0)
        targets.push_back(t);
    std::vector<int> d(st_.territories, 0);
    if (targets.empty()) return d;
    for (int t = 0; t < st_.territories; ++t) {
      int best = 1 << 20;
      for (int x : targets) best = std::min(best, st_.dist[t][x]);
      d[t] = best;
    }
    return d;
  }

  // Value in [0, 1] for `player`: 1/0 at terminal states, else its share of a score combining
  // victory cities, production, capital ownership, unit value and treasury.
  double evaluate(const View& v, int player) const {
    if (v.terminal) return (v.winner >= 0 && v.winner == st_.alliance_of[player]) ? 1.0 : 0.0;
    std::vector<double> score(st_.players, 1.0);
    for (int t = 0; t < st_.territories; ++t) {
      int o = v.owner[t];
      if (o >= 0) score[o] += 3.0 * st_.production[t] + 30.0 * st_.victory[t];
      int c = st_.capital_of[t];
      if (c >= 0 && o == c) score[c] += 25.0;
    }
    for (const Unit& u : v.units)
      if (u.owner >= 0 && !st_.types[u.type].infrastructure) score[u.owner] += std::max(1, st_.types[u.type].cost);
    for (int p = 0; p < st_.players; ++p) score[p] += v.balance[p];
    double total = 0;
    for (double s : score) total += s;
    return score[player] / total;
  }

  // ---- policy ----
  Action act(int state, const View& v) {
    switch (v.decision) {
      case fm::DEC_PURCHASE: return purchase(state, v);
      case fm::DEC_MOVE: return v.step == fm::STEP_COMBAT ? combat_move(state, v) : noncombat_move(state, v);
      case fm::DEC_PLACE: return place(state, v);
      case fm::DEC_BATTLE_SELECTION: return select_battle(v);
      case fm::DEC_CASUALTIES: return casualties(v);
      case fm::DEC_RETREAT: return retreat(v);
      default: return {fm::END_PHASE};
    }
  }

  Action purchase(int state, const View& v) {
    int p = v.actor;
    for (const Unit& u : v.units)
      if (u.loc == fm::HELD && u.owner == p) return {fm::END_PHASE};
    struct Opt { int rule, cost; const fm::UnitType* t; };
    std::vector<Opt> options;
    for (int r : st_.frontier[p]) {
      const auto& rule = st_.rules[r];
      if (rule.result_type < 0 || rule.other_resources || rule.cost <= 0) continue;
      const auto& t = st_.types[rule.result_type];
      if (t.infrastructure || t.domain == 2) continue;
      options.push_back({r, rule.cost, &t});
    }
    if (options.empty()) return {fm::END_PHASE};
    auto defensive = *std::max_element(options.begin(), options.end(), [](const Opt& a, const Opt& b) {
      return std::make_tuple(double(a.t->defense) / a.cost, -a.cost) < std::make_tuple(double(b.t->defense) / b.cost, -b.cost);
    });
    auto offensive = *std::max_element(options.begin(), options.end(), [](const Opt& a, const Opt& b) {
      return std::make_tuple((a.t->attack + 0.5 * a.t->movement) / a.cost, -a.cost) <
             std::make_tuple((b.t->attack + 0.5 * b.t->movement) / b.cost, -b.cost);
    });
    int budget = v.balance[p];
    std::map<int, int> counts;
    bool toggle = true;
    while (true) {
      const Opt* pick = toggle ? &offensive : &defensive;
      if (budget < pick->cost) pick = budget >= defensive.cost ? &defensive : nullptr;
      if (!pick) break;
      counts[pick->rule]++;
      budget -= pick->cost;
      toggle = !toggle;
    }
    if (counts.empty()) return {fm::END_PHASE};
    Action a{fm::PURCHASE};
    for (auto& [r, c] : counts) { a.push_back(r); a.push_back(c); }
    return fm_.validate(state, a) ? a : Action{fm::END_PHASE};
  }

  Action place(int state, const View& v) {
    int p = v.actor;
    std::vector<int> held;
    for (const Unit& u : v.units)
      if (u.loc == fm::HELD && u.owner == p && !st_.types[u.type].factory) held.push_back(u.id);
    if (held.empty()) return {fm::END_PHASE};
    std::vector<fm::Cap> caps = v.caps;
    std::sort(caps.begin(), caps.end(), [&](const fm::Cap& a, const fm::Cap& b) {
      return std::make_tuple(a.cap >= 0, -value_[a.territory]) < std::make_tuple(b.cap >= 0, -value_[b.territory]);
    });
    for (const auto& c : caps) {
      int room = c.cap < 0 ? int(held.size()) : c.cap - c.used;
      if (room <= 0) continue;
      Action a{fm::PLACE, c.territory};
      for (int i = 0; i < room && i < int(held.size()); ++i) a.push_back(held[i]);
      if (fm_.validate(state, a)) return a;
    }
    return {fm::END_PHASE};
  }

  Action select_battle(const View& v) const {
    int best = -1;
    for (const auto& b : v.battles)
      if (b.attacker == v.actor && !b.active && (best < 0 || b.id < best)) best = b.id;
    return {fm::SELECT_BATTLE, best};
  }

  // Default order of loss: lowest power first (artillery counts its support), then cheapest,
  // then air with the least movement left.
  Action casualties(const View& v) const {
    bool attackers_hit = !v.defenders_hit;
    int supportable = 0;
    for (int id : v.eligible) supportable += st_.types[v.unit(id).type].supportable ? 1 : 0;
    std::vector<int> ids = v.eligible;
    auto key = [&](int id) {
      const Unit& u = v.unit(id);
      const auto& t = st_.types[u.type];
      int power = attackers_hit ? t.attack : t.defense;
      if (attackers_hit && t.artillery && supportable > 0) ++power;
      return std::make_tuple(power, t.cost, t.air() ? u.move_left : 0, id);
    };
    std::sort(ids.begin(), ids.end(), [&](int a, int b) { return key(a) < key(b); });
    Action a{fm::CASUALTIES, v.batch_id};
    int need = std::min<int>(v.hits, ids.size());
    for (int i = 0; i < need; ++i) a.push_back(ids[i]);
    return a;
  }

  Action retreat(const View& v) const {
    int site = -1;
    for (const auto& b : v.battles)
      if (b.id == v.battle_id) site = b.site;
    std::vector<const Unit*> mine;
    for (const Unit& u : v.units)
      if (u.loc == site && u.owner == v.actor) mine.push_back(&u);
    std::vector<const Unit*> enemies;
    for (const Unit& u : v.units)
      if (u.loc == site && !st_.allied(v.actor, u.owner)) enemies.push_back(&u);
    if (attack_power(mine) >= 0.6 * defense_power(enemies) || v.retreat_destinations.empty())
      return {fm::RETREAT, 0, -1};
    return {fm::RETREAT, 1, v.retreat_destinations.front()};
  }

  Action combat_move(int state, const View& v) {
    int p = v.actor;
    std::map<int, std::vector<const Unit*>> movable;
    for (const Unit& u : v.units) {
      const auto& t = st_.types[u.type];
      if (u.owner == p && u.loc >= 0 && u.move_left > 0 && t.land() && t.can_move_in_combat)
        movable[u.loc].push_back(&u);
    }
    if (movable.empty()) return {fm::END_PHASE};
    double best_score = -1e18;
    Action best;
    for (auto& [t, units] : movable) {
      for (int n : st_.neighbors[t]) {
        if (st_.water[n] || v.owner[n] == p || st_.allied(p, v.owner[n])) continue;
        auto enemies = enemy_combat(v, p, n);
        int defense = defense_power(enemies);
        std::vector<const Unit*> already;
        for (const Unit& u : v.units)
          if (u.loc == n && u.owner == p) already.push_back(&u);
        double score;
        Action a;
        if (enemies.empty()) {
          if (!already.empty()) continue;
          const Unit* pick = *std::min_element(units.begin(), units.end(), [&](const Unit* x, const Unit* y) {
            return std::make_pair(st_.types[x->type].cost, x->id) < std::make_pair(st_.types[y->type].cost, y->id);
          });
          score = value_[n] + 1;
          a = {fm::MOVE, 0, 1, pick->id, t, n};
        } else {
          std::vector<const Unit*> pool = already;
          for (int src : st_.neighbors[n]) {
            auto it = movable.find(src);
            if (it != movable.end()) pool.insert(pool.end(), it->second.begin(), it->second.end());
          }
          int power = attack_power(pool);
          if (power < aggression_ * defense + 1) continue;
          score = value_[n] + 2.0 * (power - defense);
          std::vector<int> ids;
          for (const Unit* u : units) ids.push_back(u->id);
          std::sort(ids.begin(), ids.end());
          a = {fm::MOVE, 0, int(ids.size())};
          a.insert(a.end(), ids.begin(), ids.end());
          a.push_back(t);
          a.push_back(n);
        }
        if (score > best_score && fm_.validate(state, a)) {
          best_score = score;
          best = a;
        }
      }
    }
    return best.empty() ? Action{fm::END_PHASE} : best;
  }

  Action noncombat_move(int state, const View& v) {
    int p = v.actor;
    std::vector<int> dist = target_distance(v, p);
    std::vector<const Unit*> order;
    for (const Unit& u : v.units) order.push_back(&u);
    std::sort(order.begin(), order.end(), [](const Unit* a, const Unit* b) { return a->id < b->id; });
    for (const Unit* u : order) {
      if (u->owner != p || u->loc < 0 || u->move_left <= 0) continue;
      const auto& t = st_.types[u->type];
      int at = u->loc;
      if (t.air()) {
        if (can_land(v, p, at)) continue;
        Action a = land_air(state, v, *u, dist);
        if (!a.empty()) return a;
        continue;
      }
      if (!t.land()) continue;
      if (st_.capital_of[at] == p) {
        int guards = 0;
        for (const Unit* x : units_at(v, at))
          if (x->owner == p && !st_.types[x->type].infrastructure) ++guards;
        if (guards <= 2) continue;
      }
      int best = -1;
      for (int n : st_.neighbors[at]) {
        bool enemy_present = false;
        for (const Unit* x : units_at(v, n))
          if (!st_.allied(p, x->owner)) enemy_present = true;
        if (v.owner[n] == p && dist[n] < dist[at] && !enemy_present && (best < 0 || dist[n] < dist[best])) best = n;
      }
      if (best >= 0) {
        std::vector<int> ids;
        for (const Unit* x : units_at(v, at))
          if (x->owner == p && x->type == u->type && x->move_left > 0) ids.push_back(x->id);
        std::sort(ids.begin(), ids.end());
        Action a{fm::MOVE, 0, int(ids.size())};
        a.insert(a.end(), ids.begin(), ids.end());
        a.push_back(at);
        a.push_back(best);
        if (fm_.validate(state, a)) return a;
      }
    }
    return {fm::END_PHASE};
  }

  // Shortest flight to a legal landing spot (air may not fly over neutral land).
  Action land_air(int state, const View& v, const Unit& u, const std::vector<int>& dist) {
    int p = v.actor;
    std::vector<int> parent(st_.territories, -2), depth(st_.territories, -1);
    std::vector<int> queue{u.loc};
    parent[u.loc] = -1;
    depth[u.loc] = 0;
    for (size_t q = 0; q < queue.size(); ++q) {
      int t = queue[q];
      if (depth[t] >= u.move_left) continue;
      for (int n : st_.neighbors[t]) {
        if (parent[n] != -2 || st_.impassable[n] || neutral_land(v, n)) continue;
        parent[n] = t;
        depth[n] = depth[t] + 1;
        queue.push_back(n);
      }
    }
    std::vector<int> spots;
    for (int t : queue)
      if (t != u.loc && can_land(v, p, t)) spots.push_back(t);
    std::sort(spots.begin(), spots.end(), [&](int a, int b) {
      return std::make_pair(depth[a], dist[a]) < std::make_pair(depth[b], dist[b]);
    });
    for (int dest : spots) {
      std::vector<int> route;
      for (int t = dest; t != -1; t = parent[t]) route.push_back(t);
      std::reverse(route.begin(), route.end());
      Action a{fm::MOVE, 0, 1, u.id};
      a.insert(a.end(), route.begin(), route.end());
      if (fm_.validate(state, a)) return a;
    }
    return {};
  }

  // Heuristic prior used to order and limit tree expansions.
  double prior(const View& v, const Action& a) const {
    int p = v.actor;
    switch (a[0]) {
      case fm::END_PHASE: return 0.5;
      case fm::PURCHASE: {
        int units = 0;
        for (size_t i = 2; i < a.size(); i += 2) units += a[i];
        return 0.5 + 0.01 * units;
      }
      case fm::MOVE: {
        int n = a[2];
        int start = a[3 + n], end = a.back();
        std::vector<const Unit*> units;
        bool land = false;
        for (int i = 0; i < n; ++i) {
          units.push_back(&v.unit(a[3 + i]));
          land |= st_.types[units.back()->type].land();
        }
        if (v.step == fm::STEP_COMBAT && enemy_territory(v, p, end)) {
          if (a[1] == 1) return 0.3;
          int defense = defense_power(enemy_combat(v, p, end));
          for (const Unit& u : v.units)
            if (u.loc == end && u.owner == p) units.push_back(&u);
          int power = attack_power(units);
          if (defense == 0) return land ? 1.0 + value_[end] / 10.0 : 0.05;
          return double(power) / (defense + 1) + value_[end] / 20.0 - 0.5;
        }
        std::vector<int> dist = target_distance(v, p);
        return 0.4 + 0.1 * (dist[start] - dist[end]);
      }
      default: return 0.5;
    }
  }

 private:
  fm::ForwardModel& fm_;
  const Static& st_;
  double aggression_ = 1.2;
  std::vector<int> value_;
};

}  // namespace agent
