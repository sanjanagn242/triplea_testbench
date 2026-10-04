// Open-loop MCTS over one phase of the agent's own decisions.
//
// Tree edges are the agent's actions within the current phase (forward-model ActionToolkit
// candidates, ordered by a heuristic prior, with progressive widening). Every iteration clones
// the root state inside the testbench forward-model library, replays the edge sequence with fresh dice
// (chance is sampled, not branched), rolls out all players with the heuristic policy for a fixed
// number of rounds, and backs up the evaluation.
#pragma once

#include <chrono>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include <triplea_fm/forward_model.hpp>

#include "policy.hpp"

namespace agent {

struct Node {
  Action action;
  std::vector<std::unique_ptr<Node>> children;
  std::vector<Action> untried;
  bool expanded = false;
  int n = 0;
  double w = 0.0;
};

struct SearchConfig {
  double time_budget = 2.0;
  int max_iterations = 100000;
  int rollout_rounds = 2;
  size_t max_children = 24;
  double exploration = 0.7;
  double widening = 2.0;
  int max_rollout_steps = 5000;
};

class PhaseSearch {
 public:
  PhaseSearch(fm::ForwardModel& model, const Static& st, uint64_t seed)
      : fm_(model), st_(st), policy_(model, st), rnd_(seed) {}

  SearchConfig config;
  int iterations = 0;

  std::vector<Action> search(int root_state) {
    View root_view = fm_.view(root_state);
    const int me = root_view.actor;
    Node root;
    expand(root_state, root_view, root);
    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::duration<double>(config.time_budget);
    iterations = 0;
    while (iterations < config.max_iterations && std::chrono::steady_clock::now() < deadline) {
      ++iterations;
      int s = fm_.clone(root_state);
      int rng = fm_.rng_new(static_cast<long long>(rnd_() >> 2));
      std::vector<Node*> path{&root};
      Node* node = &root;
      View v = fm_.view(s);
      while (same_phase(root_view, v)) {
        if (!node->expanded) expand(s, v, *node);
        size_t allowed = std::max<size_t>(1, size_t(config.widening * std::sqrt(node->n + 1.0)));
        if (!node->untried.empty() && node->children.size() < allowed) {
          auto child = std::make_unique<Node>();
          child->action = node->untried.front();
          node->untried.erase(node->untried.begin());
          if (fm_.apply(s, child->action, rng) >= fm::REJECTED) break;
          node->children.push_back(std::move(child));
          node = node->children.back().get();
          path.push_back(node);
          v = fm_.view(s);
          break;
        }
        Node* child = select(*node, s);
        if (!child) break;
        fm_.apply(s, child->action, rng);
        node = child;
        path.push_back(node);
        v = fm_.view(s);
      }
      double value = rollout(s, rng, me, root_view.round, v);
      for (Node* n : path) {
        n->n += 1;
        n->w += value;
      }
      fm_.free(s);
      fm_.rng_free(rng);
    }
    return best_sequence(root_state, root_view, root);
  }

 private:
  fm::ForwardModel& fm_;
  const Static& st_;
  Policy policy_;
  std::mt19937_64 rnd_;  // agent randomness only; game dice come from the forward model's RNG

  static bool same_phase(const View& root, const View& v) {
    return !v.terminal && v.decision != fm::DEC_NONE && v.actor == root.actor &&
           v.step_id == root.step_id && v.decision == root.decision && v.round == root.round;
  }

  void expand(int s, const View& v, Node& node) {
    node.expanded = true;
    node.untried = fm_.candidates(s);
    std::vector<std::pair<double, size_t>> scored;
    for (size_t i = 0; i < node.untried.size(); ++i) scored.push_back({-policy_.prior(v, node.untried[i]), i});
    std::stable_sort(scored.begin(), scored.end());
    std::vector<Action> ordered;
    for (size_t i = 0; i < scored.size() && i < config.max_children; ++i)
      ordered.push_back(node.untried[scored[i].second]);
    node.untried = std::move(ordered);
  }

  Node* select(Node& node, int s) {
    Node* best = nullptr;
    double best_score = -1e18;
    double log_n = std::log(node.n + 1.0);
    for (auto& ch : node.children) {
      if (!fm_.validate(s, ch->action)) continue;  // open loop: dice can invalidate an edge
      double score = ch->w / (ch->n + 1e-9) + config.exploration * std::sqrt(log_n / (ch->n + 1e-9));
      if (score > best_score) {
        best_score = score;
        best = ch.get();
      }
    }
    return best;
  }

  double rollout(int s, int rng, int me, int start_round, View v) {
    const int horizon = start_round + config.rollout_rounds;
    for (int steps = 0; !v.terminal && v.decision != fm::DEC_NONE && v.round <= horizon &&
                        steps < config.max_rollout_steps;
         ++steps) {
      Action a = policy_.act(s, v);
      fm::Status st = fm_.apply(s, a, rng);
      if (st == fm::REJECTED || st == fm::FAILED) {
        if (fm_.apply(s, {fm::END_PHASE}, rng) >= fm::REJECTED) break;
      }
      v = fm_.view(s);
    }
    return policy_.evaluate(v, me);
  }

  std::vector<Action> best_sequence(int root_state, const View& root_view, Node& root) {
    std::vector<Action> seq;
    int s = fm_.clone(root_state);
    int rng = fm_.rng_new(12345);
    Node* node = &root;
    View v = fm_.view(s);
    while (same_phase(root_view, v)) {
      Node* best = nullptr;
      if (node) {
        for (auto& ch : node->children) {
          if (ch->n == 0 || !fm_.validate(s, ch->action)) continue;
          if (!best || ch->n > best->n || (ch->n == best->n && ch->w > best->w)) best = ch.get();
        }
      }
      Action a = best ? best->action : policy_.act(s, v);  // beyond the tree: greedy completion
      node = best;
      if (fm_.apply(s, a, rng) >= fm::REJECTED) break;
      seq.push_back(a);
      if (a[0] == fm::END_PHASE) break;
      v = fm_.view(s);
    }
    fm_.free(s);
    fm_.rng_free(rng);
    return seq;
  }
};

}  // namespace agent
