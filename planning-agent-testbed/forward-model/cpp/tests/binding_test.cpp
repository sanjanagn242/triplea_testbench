// Tests for the C++ binding of the forward-model library (run: forward-model/cpp/tests/run-tests.sh).
#include <triplea_fm/forward_model.hpp>

#include <fstream>
#include <iostream>
#include <random>
#include <sstream>

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
  std::cout << (ok ? "ok   " : "FAIL ") << what << '\n';
  if (!ok) ++failures;
}

std::string play(fm::ForwardModel& model, int seed, int max_steps) {
  int s = model.initial_state();
  int rng = model.rng_new(seed);
  std::mt19937 pick(seed);
  for (int i = 0; i < max_steps && !model.view(s).terminal; ++i) {
    auto cands = model.candidates(s);
    if (cands.empty()) break;
    if (model.apply(s, cands[pick() % cands.size()], rng) != fm::DECISION &&
        !model.view(s).terminal)
      return "apply failed";
  }
  std::string snapshot = model.snapshot_json(s);
  model.free(s);
  model.rng_free(rng);
  return snapshot;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: binding_test <capture_the_flag.xml>\n";
    return 2;
  }
  std::ifstream in(argv[1]);
  std::stringstream xml;
  xml << in.rdbuf();

  fm::ForwardModel model;
  check(model.load_xml(xml.str()), "CTF definition is supported");
  fm::Static st = model.static_info();
  check(st.territories == 29 && st.players == 4, "static info: 29 territories, 4 players");

  int s = model.initial_state();
  fm::View v = model.view(s);
  check(v.decision == fm::DEC_PURCHASE && v.actor == 0 && v.round == 1,
        "initial decision is Russians' purchase in round 1");
  check(v.units.size() == 19, "19 starting units");

  int rng = model.rng_new(1);
  check(model.apply(s, {fm::PURCHASE, 0, 99}, rng) == fm::REJECTED, "unaffordable purchase is rejected");
  check(!model.validate(s, {fm::MOVE, 0, 1, 1, 0, 1}), "move during purchase is invalid");

  int c = model.clone(s);
  model.apply(c, {fm::PURCHASE, 0, 4}, rng);  // 4 infantry for 12 PUs
  check(model.view(c).balance[0] == 0 && model.view(s).balance[0] == 12, "clones are independent");

  auto cands = model.candidates(s);
  bool all_valid = !cands.empty();
  for (const auto& a : cands) all_valid = all_valid && model.validate(s, a);
  check(all_valid, "every toolkit candidate validates");

  check(play(model, 7, 600) == play(model, 7, 600), "same seed and choices reproduce the same state");
  check(play(model, 7, 600) != play(model, 8, 600), "different seeds diverge");

  std::cout << (failures ? "FAILED" : "PASSED") << " (" << failures << " failures)\n";
  return failures ? 1 : 0;
}
