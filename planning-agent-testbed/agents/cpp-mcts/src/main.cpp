// C++ MCTS agent for the planning-agent testbench.
//
// Speaks the testbench protocol (one JSON request line in, one JSON response line out on
// stdin/stdout). Search, rollout policy and evaluation are C++; game dynamics come from the
// testbench forward-model library through its C++ binding (<triplea_fm/forward_model.hpp>).
//
// Environment:
//   TRIPLEA_MCTS_SECONDS   base search time per decision (default 2.0)
//   TRIPLEA_MCTS_ROLLOUT   rollout horizon in rounds (default 2)
//   TRIPLEA_MCTS_SEED      search seed (default 0)
//   TRIPLEA_AGENT_LOG      1 enables logging, anything else disables it (set in run-agent.sh)
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>

#include <triplea_fm/forward_model.hpp>

#include "mcts.hpp"

namespace {

std::string env(const char* name, const std::string& fallback) {
  const char* v = std::getenv(name);
  return v ? std::string(v) : fallback;
}

const std::map<std::string, double> kPhaseShare = {
    {"purchase", 0.6}, {"combatMove", 1.5}, {"nonCombatMove", 0.8}, {"place", 0.3}};

class Agent {
 public:
  Agent(const std::filesystem::path& agent_dir)
      : agent_dir_(agent_dir),
        base_seconds_(std::stod(env("TRIPLEA_MCTS_SECONDS", "2.0"))),
        rollout_rounds_(std::stoi(env("TRIPLEA_MCTS_ROLLOUT", "2"))),
        seed_(std::stoull(env("TRIPLEA_MCTS_SEED", "0"))),
        logging_(env("TRIPLEA_AGENT_LOG", "0") == "1") {}

  std::string handle(const std::string& line) {
    auto req = model_.parse_request(line);  // type, phase, playerName, gameNumber
    const std::string& type = req[0];
    if (type == "game_start") return game_start(req[2], req[3]);
    if (type == "turn_request") return turn(req[1]);
    return R"({"error": "unknown request type"})";
  }

  void log(const std::string& message) {
    if (!logging_) return;
    std::time_t now = std::time(nullptr);
    char stamp[16];
    std::strftime(stamp, sizeof stamp, "%H:%M:%S", std::localtime(&now));
    std::cerr << stamp << ' ' << message << std::endl;
    if (!log_path_.empty()) std::ofstream(log_path_, std::ios::app) << stamp << ' ' << message << '\n';
  }

 private:
  std::filesystem::path agent_dir_;
  fm::ForwardModel model_;
  double base_seconds_;
  int rollout_rounds_;
  uint64_t seed_;
  bool logging_;
  bool supported_ = false;
  std::unique_ptr<fm::Static> static_;
  std::unique_ptr<agent::PhaseSearch> search_;
  std::filesystem::path log_path_;

  std::string game_start(const std::string& player, const std::string& game_number) {
    if (logging_) {
      std::filesystem::path logs = agent_dir_.parent_path().parent_path() / "logs";
      std::filesystem::create_directories(logs);
      log_path_ = logs / ("agent-" + player + "-game-" + game_number + ".txt");
      std::ofstream(log_path_) << game_number << '\n';
    }
    supported_ = model_.start_game() >= 0;
    for (const auto& m : model_.messages()) log((supported_ ? "" : "profile refusal: ") + m);
    if (supported_) {
      static_ = std::make_unique<fm::Static>(model_.static_info());
      search_ = std::make_unique<agent::PhaseSearch>(model_, *static_, seed_);
      search_->config.rollout_rounds = rollout_rounds_;
      log("ready as " + player);
    } else {
      log("forward model unavailable for this game; answering with empty actions");
    }
    return R"({"ready": true, "protocolVersion": 1})";
  }

  std::string turn(const std::string& phase) {
    if (phase == "battle") return R"({"fightAll": true})";
    auto share = kPhaseShare.find(phase);
    if (share == kPhaseShare.end() || !supported_) return "{}";
    int state = model_.state_from_request();
    if (state < 0) {
      for (const auto& m : model_.messages()) log(phase + ": state rejected by forward model: " + m);
      return "{}";
    }
    search_->config.time_budget = base_seconds_ * share->second;
    auto started = std::chrono::steady_clock::now();
    std::vector<fm::Action> sequence = search_->search(state);
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    int round = model_.view(state).round;
    model_.free(state);
    std::string response = model_.encode_response(phase, sequence);
    log("round " + std::to_string(round) + " " + phase + ": " + std::to_string(search_->iterations) +
        " iterations in " + std::to_string(seconds).substr(0, 4) + "s -> " +
        std::to_string(sequence.size()) + " actions");
    log("  action: " + response);
    return response;
  }
};

}  // namespace

int main(int, char** argv) {
  std::ios::sync_with_stdio(false);
  std::filesystem::path agent_dir = std::filesystem::canonical(argv[0]).parent_path().parent_path();
  try {
    Agent agent(agent_dir);
    std::string line;
    while (std::getline(std::cin, line)) {
      if (line.empty()) continue;
      std::string response;
      try {
        response = agent.handle(line);
      } catch (const std::exception& e) {
        agent.log(std::string("error: ") + e.what());
        response = R"({"error": "agent failure; see agent log"})";
      }
      std::cout << response << '\n' << std::flush;
    }
  } catch (const std::exception& e) {
    std::cerr << "fatal: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
