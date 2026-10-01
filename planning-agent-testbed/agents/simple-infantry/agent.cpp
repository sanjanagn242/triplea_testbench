#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <optional>
#include <queue>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <variant>
#include <vector>

namespace {

struct Json {
  using Array = std::vector<Json>;
  using Object = std::map<std::string, Json>;
  using Value = std::variant<std::nullptr_t, bool, double, std::string, Array, Object>;
  Value value = nullptr;

  Json() = default;
  Json(std::nullptr_t) : value(nullptr) {}
  Json(bool item) : value(item) {}
  Json(int item) : value(static_cast<double>(item)) {}
  Json(double item) : value(item) {}
  Json(const char* item) : value(std::string(item)) {}
  Json(std::string item) : value(std::move(item)) {}
  Json(Array item) : value(std::move(item)) {}
  Json(Object item) : value(std::move(item)) {}

  bool is_object() const { return std::holds_alternative<Object>(value); }
  bool is_array() const { return std::holds_alternative<Array>(value); }
  const Object& object() const { return std::get<Object>(value); }
  const Array& array() const { return std::get<Array>(value); }
  const std::string& string() const { return std::get<std::string>(value); }
  const Json* get(std::string_view key) const {
    if (!is_object()) return nullptr;
    const auto found = object().find(std::string(key));
    return found == object().end() ? nullptr : &found->second;
  }
  std::string as_string(std::string fallback = {}) const {
    if (const auto* item = std::get_if<std::string>(&value)) return *item;
    return fallback;
  }
  double as_number(double fallback = 0) const {
    if (const auto* item = std::get_if<double>(&value)) return *item;
    if (const auto* item = std::get_if<std::string>(&value)) {
      try {
        return std::stod(*item);
      } catch (...) {
      }
    }
    return fallback;
  }
  bool as_bool(bool fallback = false) const {
    if (const auto* item = std::get_if<bool>(&value)) return *item;
    return fallback;
  }
};

void append_utf8(std::string& out, std::uint32_t codepoint) {
  if (codepoint <= 0x7f) {
    out.push_back(static_cast<char>(codepoint));
  } else if (codepoint <= 0x7ff) {
    out.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
    out.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
  } else if (codepoint <= 0xffff) {
    out.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
    out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
    out.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
  } else {
    out.push_back(static_cast<char>(0xf0 | (codepoint >> 18)));
    out.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
    out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
    out.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
  }
}

class JsonParser {
 public:
  explicit JsonParser(std::string_view source) : source_(source) {}
  Json parse() {
    skip_space();
    Json result = parse_value();
    skip_space();
    if (position_ != source_.size()) fail("extra text after JSON value");
    return result;
  }

 private:
  std::string_view source_;
  std::size_t position_ = 0;

  [[noreturn]] void fail(const char* message) const {
    throw std::runtime_error(std::string("invalid JSON: ") + message);
  }
  void skip_space() {
    while (position_ < source_.size()
           && (source_[position_] == ' ' || source_[position_] == '\n'
               || source_[position_] == '\r' || source_[position_] == '\t')) {
      ++position_;
    }
  }
  bool consume(char expected) {
    if (position_ < source_.size() && source_[position_] == expected) {
      ++position_;
      return true;
    }
    return false;
  }
  Json parse_value() {
    skip_space();
    if (position_ >= source_.size()) fail("unexpected end of input");
    switch (source_[position_]) {
      case '{': return parse_object();
      case '[': return parse_array();
      case '"': return Json(parse_string());
      case 't': expect_word("true"); return Json(true);
      case 'f': expect_word("false"); return Json(false);
      case 'n': expect_word("null"); return Json(nullptr);
      default: return Json(parse_number());
    }
  }
  Json parse_object() {
    ++position_;
    Json::Object result;
    skip_space();
    if (consume('}')) return Json(std::move(result));
    do {
      skip_space();
      if (position_ >= source_.size() || source_[position_] != '"') fail("object key expected");
      std::string key = parse_string();
      skip_space();
      if (!consume(':')) fail("colon expected");
      result.insert_or_assign(std::move(key), parse_value());
      skip_space();
      if (consume('}')) return Json(std::move(result));
      if (!consume(',')) fail("comma expected");
    } while (true);
  }
  Json parse_array() {
    ++position_;
    Json::Array result;
    skip_space();
    if (consume(']')) return Json(std::move(result));
    do {
      result.push_back(parse_value());
      skip_space();
      if (consume(']')) return Json(std::move(result));
      if (!consume(',')) fail("comma expected");
    } while (true);
  }
  static int hex_digit(char value) {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    if (value >= 'A' && value <= 'F') return value - 'A' + 10;
    return -1;
  }
  std::uint32_t parse_hex4() {
    if (source_.size() - position_ < 4) fail("short unicode escape");
    std::uint32_t result = 0;
    for (int i = 0; i < 4; ++i) {
      const int digit = hex_digit(source_[position_++]);
      if (digit < 0) fail("invalid unicode escape");
      result = (result << 4) | static_cast<std::uint32_t>(digit);
    }
    return result;
  }
  std::string parse_string() {
    if (!consume('"')) fail("string expected");
    std::string result;
    while (position_ < source_.size()) {
      const unsigned char current = static_cast<unsigned char>(source_[position_++]);
      if (current == '"') return result;
      if (current < 0x20) fail("control character in string");
      if (current != '\\') {
        result.push_back(static_cast<char>(current));
        continue;
      }
      if (position_ >= source_.size()) fail("incomplete escape");
      switch (source_[position_++]) {
        case '"': result.push_back('"'); break;
        case '\\': result.push_back('\\'); break;
        case '/': result.push_back('/'); break;
        case 'b': result.push_back('\b'); break;
        case 'f': result.push_back('\f'); break;
        case 'n': result.push_back('\n'); break;
        case 'r': result.push_back('\r'); break;
        case 't': result.push_back('\t'); break;
        case 'u': {
          std::uint32_t codepoint = parse_hex4();
          if (codepoint >= 0xd800 && codepoint <= 0xdbff) {
            if (source_.size() - position_ < 6 || source_[position_] != '\\'
                || source_[position_ + 1] != 'u') fail("invalid unicode surrogate pair");
            position_ += 2;
            const std::uint32_t low = parse_hex4();
            if (low < 0xdc00 || low > 0xdfff) fail("invalid unicode surrogate pair");
            codepoint = 0x10000 + ((codepoint - 0xd800) << 10) + (low - 0xdc00);
          }
          append_utf8(result, codepoint);
          break;
        }
        default: fail("invalid escape");
      }
    }
    fail("unterminated string");
  }
  double parse_number() {
    const std::size_t start = position_;
    if (consume('-')) {}
    if (consume('0')) {
    } else {
      if (position_ >= source_.size() || source_[position_] < '1' || source_[position_] > '9')
        fail("invalid number");
      while (position_ < source_.size() && source_[position_] >= '0' && source_[position_] <= '9')
        ++position_;
    }
    if (consume('.')) {
      if (position_ >= source_.size() || source_[position_] < '0' || source_[position_] > '9')
        fail("invalid fraction");
      while (position_ < source_.size() && source_[position_] >= '0' && source_[position_] <= '9')
        ++position_;
    }
    if (consume('e') || consume('E')) {
      if (!consume('+')) consume('-');
      if (position_ >= source_.size() || source_[position_] < '0' || source_[position_] > '9')
        fail("invalid exponent");
      while (position_ < source_.size() && source_[position_] >= '0' && source_[position_] <= '9')
        ++position_;
    }
    try {
      return std::stod(std::string(source_.substr(start, position_ - start)));
    } catch (...) {
      fail("invalid number");
    }
  }
  void expect_word(std::string_view word) {
    if (source_.substr(position_, word.size()) != word) fail("invalid literal");
    position_ += word.size();
  }
};

std::string json_escape(std::string_view value) {
  std::ostringstream output;
  output << '"';
  for (const unsigned char character : value) {
    switch (character) {
      case '"': output << "\\\""; break;
      case '\\': output << "\\\\"; break;
      case '\b': output << "\\b"; break;
      case '\f': output << "\\f"; break;
      case '\n': output << "\\n"; break;
      case '\r': output << "\\r"; break;
      case '\t': output << "\\t"; break;
      default:
        if (character < 0x20) {
          output << "\\u00" << "0123456789abcdef"[(character >> 4) & 0xf]
                 << "0123456789abcdef"[character & 0xf];
        } else {
          output << static_cast<char>(character);
        }
    }
  }
  output << '"';
  return output.str();
}

std::string stringify(const Json& value) {
  if (std::holds_alternative<std::nullptr_t>(value.value)) return "null";
  if (const auto* item = std::get_if<bool>(&value.value)) return *item ? "true" : "false";
  if (const auto* item = std::get_if<double>(&value.value)) {
    std::ostringstream out;
    out.precision(15);
    out << *item;
    return out.str();
  }
  if (const auto* item = std::get_if<std::string>(&value.value)) return json_escape(*item);
  if (const auto* item = std::get_if<Json::Array>(&value.value)) {
    std::string out = "[";
    for (std::size_t i = 0; i < item->size(); ++i) {
      if (i) out += ',';
      out += stringify((*item)[i]);
    }
    return out + ']';
  }
  const auto& item = std::get<Json::Object>(value.value);
  std::string out = "{";
  bool first = true;
  for (const auto& [key, child] : item) {
    if (!first) out += ',';
    first = false;
    out += json_escape(key) + ':' + stringify(child);
  }
  return out + '}';
}

std::string xml_decode(std::string value) {
  const std::vector<std::pair<std::string, std::string>> entities = {
      {"&quot;", "\""}, {"&apos;", "'"}, {"&lt;", "<"}, {"&gt;", ">"}, {"&amp;", "&"}};
  for (const auto& [entity, replacement] : entities) {
    std::size_t at = 0;
    while ((at = value.find(entity, at)) != std::string::npos) {
      value.replace(at, entity.size(), replacement);
      at += replacement.size();
    }
  }
  return value;
}

using Attributes = std::map<std::string, std::string>;
Attributes xml_attributes(const std::string& tag) {
  static const std::regex attribute(
      R"ATTR(([A-Za-z_:][A-Za-z0-9_.:-]*)\s*=\s*(?:"([^"]*)"|'([^']*)'))ATTR");
  Attributes result;
  for (auto match = std::sregex_iterator(tag.begin(), tag.end(), attribute);
       match != std::sregex_iterator(); ++match) {
    const auto& m = *match;
    result[ m[1].str() ] = xml_decode(m[2].matched ? m[2].str() : m[3].str());
  }
  return result;
}

std::string attr(const Attributes& values, const std::string& name, std::string fallback = {}) {
  const auto found = values.find(name);
  return found == values.end() ? fallback : found->second;
}

std::vector<std::pair<Attributes, std::string>> xml_elements(
    const std::string& xml, const std::string& element) {
  const std::regex element_regex(
      "<" + element + R"(\b([^>]*?)(?:/\s*>|>([\s\S]*?)</)" + element + R"(\s*>))");
  std::vector<std::pair<Attributes, std::string>> result;
  for (auto match = std::sregex_iterator(xml.begin(), xml.end(), element_regex);
       match != std::sregex_iterator(); ++match) {
    result.emplace_back(xml_attributes((*match)[1].str()), (*match)[2].str());
  }
  return result;
}

std::map<std::string, std::string> options(const std::string& body) {
  std::map<std::string, std::string> result;
  for (const auto& [attributes, unused] : xml_elements(body, "option")) {
    (void) unused;
    result[attr(attributes, "name")] = attr(attributes, "value");
  }
  return result;
}

struct MapRules {
  std::string infantry_type;
  int infantry_cost = 0;
  std::map<std::string, std::string> rule_by_player;
  std::map<std::string, std::string> infantry_by_player;
  std::map<std::string, std::vector<std::string>> targets_by_player;
  std::set<std::string> impassable;
};

int parse_int(const std::string& value, int fallback = 0) {
  try {
    std::size_t consumed = 0;
    const int result = std::stoi(value, &consumed);
    return consumed == value.size() ? result : fallback;
  } catch (...) {
    return fallback;
  }
}

MapRules parse_map_rules(const std::string& xml) {
  std::set<std::string> infantry_types;
  for (const auto& [attributes, body] : xml_elements(xml, "attachment")) {
    if (attr(attributes, "name") == "unitAttachment"
        && options(body)["isInfantry"] == "true") {
      infantry_types.insert(attr(attributes, "attachTo"));
    }
  }

  struct Production { std::string type; std::string name; int cost; };
  std::map<std::string, Production> production;
  for (const auto& [attributes, body] : xml_elements(xml, "productionRule")) {
    const std::string rule_name = attr(attributes, "name");
    std::string result_type;
    for (const auto& [result_attributes, unused] : xml_elements(body, "result")) {
      (void) unused;
      const std::string candidate = attr(result_attributes, "resourceOrUnit");
      if (infantry_types.contains(candidate)) {
        result_type = candidate;
        break;
      }
    }
    if (result_type.empty()) continue;
    int cost = 0;
    for (const auto& [cost_attributes, unused] : xml_elements(body, "cost")) {
      (void) unused;
      if (attr(cost_attributes, "resource") == "PUs")
        cost += parse_int(attr(cost_attributes, "quantity"));
    }
    if (cost > 0) production[rule_name] = Production{result_type, rule_name, cost};
  }

  std::map<std::string, std::set<std::string>> rules_by_frontier;
  for (const auto& [attributes, body] : xml_elements(xml, "productionFrontier")) {
    for (const auto& [rule_attributes, unused] : xml_elements(body, "frontierRules")) {
      (void) unused;
      rules_by_frontier[attr(attributes, "name")].insert(attr(rule_attributes, "name"));
    }
  }
  std::vector<std::pair<std::string, std::string>> frontier_by_player;
  for (const auto& [attributes, unused] : xml_elements(xml, "playerProduction")) {
    (void) unused;
    frontier_by_player.emplace_back(attr(attributes, "player"), attr(attributes, "frontier"));
  }

  MapRules result;
  for (const auto& [player, frontier] : frontier_by_player) {
    std::optional<Production> best;
    for (const auto& rule_name : rules_by_frontier[frontier]) {
      const auto production_rule = production.find(rule_name);
      if (production_rule != production.end()
          && (!best || std::tie(production_rule->second.cost, production_rule->second.name)
                           < std::tie(best->cost, best->name))) {
        best = production_rule->second;
      }
    }
    if (best) {
      result.rule_by_player[player] = best->name;
      result.infantry_by_player[player] = best->type;
      if (result.infantry_type.empty()) result.infantry_type = best->type;
      if (result.infantry_cost == 0) result.infantry_cost = best->cost;
    }
  }
  for (const auto& [attributes, body] : xml_elements(xml, "attachment")) {
    if (attr(attributes, "name") != "territoryAttachment") continue;
    const std::string territory = attr(attributes, "attachTo");
    const auto territory_options = options(body);
    if (territory_options.contains("isImpassable") && territory_options.at("isImpassable") == "true")
      result.impassable.insert(territory);
    const auto capital = territory_options.find("capital");
    const int victory_value = territory_options.contains("victoryCity")
                                  ? parse_int(territory_options.at("victoryCity"))
                                  : 0;
    if (capital != territory_options.end() && !capital->second.empty())
      result.targets_by_player[capital->second].push_back(territory);
    else if (victory_value > 0)
      result.targets_by_player["*"].push_back(territory);
  }
  return result;
}

std::string safe_name(const std::string& name) {
  std::string result;
  for (const unsigned char character : name) {
    if ((character >= 'a' && character <= 'z') || (character >= 'A' && character <= 'Z')
        || (character >= '0' && character <= '9') || character == '_' || character == '.'
        || character == '-')
      result.push_back(static_cast<char>(character));
    else
      result.push_back('_');
  }
  return result.empty() ? "player" : result;
}

std::string timestamp() {
  const auto now = std::chrono::system_clock::now();
  const auto time = std::chrono::system_clock::to_time_t(now);
  std::tm utc{};
#if defined(_WIN32)
  gmtime_s(&utc, &time);
#else
  gmtime_r(&time, &utc);
#endif
  char buffer[32];
  std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%S+00:00", &utc);
  return buffer;
}

class SimpleInfantryAgent {
 public:
  Json process(const Json& request) {
    const Json* type = request.get("type");
    if (!type) throw std::runtime_error("message type is missing");
    if (type->as_string() == "game_start") return start_game(request);
    if (type->as_string() == "turn_request") return handle_turn(request);
    throw std::runtime_error("unknown message type: " + type->as_string());
  }

  void log(const std::string& event, const Json::Object& details = {}) const {
    std::ostringstream line;
    line << timestamp() << ' ' << event;
    if (game_number_ > 0) line << " game_number=" << game_number_;
    for (const auto& [key, value] : details) line << ' ' << key << '=' << stringify(value);
    line << '\n';
    if (log_path_.empty()) {
      std::cerr << line.str();
      return;
    }
    std::error_code error;
    std::filesystem::create_directories(log_path_.parent_path(), error);
    std::ofstream output(log_path_, std::ios::app);
    if (!output) {
      std::cerr << "Could not write agent log " << log_path_ << ": " << line.str();
      return;
    }
    output << line.str();
  }

 private:
  std::string player_name_;
  MapRules rules_;
  Json state_;
  int game_number_ = 0;
  std::filesystem::path log_path_;

  static const Json& required(const Json& object, const std::string& key) {
    const Json* value = object.get(key);
    if (!value) throw std::runtime_error("missing field " + key);
    return *value;
  }
  static std::string text(const Json& object, const std::string& key) {
    return required(object, key).as_string();
  }
  static Json::Array array(const Json& object, const std::string& key) {
    const Json& value = required(object, key);
    if (!value.is_array()) throw std::runtime_error("field " + key + " must be an array");
    return value.array();
  }
  static std::string timestamp_detail(const Json& object, const std::string& key) {
    const Json* value = object.get(key);
    return value ? stringify(*value) : "null";
  }
  Json start_game(const Json& request) {
    const Json* version = request.get("schemaVersion");
    if (!version || version->as_number(-1) != 1) throw std::runtime_error("unsupported protocol version");
    player_name_ = text(request, "playerName");
    game_number_ = static_cast<int>(required(request, "gameNumber").as_number(1));
    log_path_ = std::filesystem::current_path() / "planning-agent-testbed" / "logs"
                / ("agent-" + safe_name(player_name_) + "-game-" + std::to_string(game_number_) + ".txt");
    std::error_code error;
    std::filesystem::create_directories(log_path_.parent_path(), error);
    if (std::filesystem::exists(log_path_)) {
      log("agent_process_restarted", {{"player", Json(player_name_)}});
    } else {
      std::ofstream output(log_path_);
      output << "Game number: " << game_number_ << '\n';
    }
    rules_ = parse_map_rules(text(request, "gameXml"));
    state_ = required(request, "initialState");
    log("game_start", {{"game", state_.get("gameName") ? *state_.get("gameName") : Json(nullptr)},
                        {"map", state_.get("mapName") ? *state_.get("mapName") : Json(nullptr)},
                        {"player", Json(player_name_)},
                        {"round", state_.get("round") ? *state_.get("round") : Json(nullptr)},
                        {"rules_loaded", Json(true)}});
    return Json::Object{{"ready", Json(true)}, {"protocolVersion", Json(1)}};
  }

  Json handle_turn(const Json& request) {
    if (player_name_.empty()) throw std::runtime_error("game_start must be received before turn_request");
    const Json* version = request.get("schemaVersion");
    if (!version || version->as_number(-1) != 1) throw std::runtime_error("unsupported protocol version");
    state_ = required(request, "state");
    const std::string phase = text(request, "phase");
    const Json* request_id = request.get("requestId");
    log("turn_request", {{"game", state_.get("gameName") ? *state_.get("gameName") : Json(nullptr)},
                         {"player", Json(player_name_)},
                         {"round", state_.get("round") ? *state_.get("round") : Json(nullptr)},
                         {"phase", Json(phase)},
                         {"request_id", request_id ? *request_id : Json(nullptr)}});
    Json action;
    if (phase == "purchase") action = purchase();
    else if (phase == "combatMove") action = combat_moves();
    else if (phase == "battle") action = Json::Object{{"fightAll", Json(true)}};
    else if (phase == "place") action = place();
    else action = Json::Object{};
    log("action_sent", {{"game", state_.get("gameName") ? *state_.get("gameName") : Json(nullptr)},
                        {"player", Json(player_name_)},
                        {"round", state_.get("round") ? *state_.get("round") : Json(nullptr)},
                        {"phase", Json(phase)},
                        {"action", action}});
    return action;
  }

  Json purchase() const {
    const Json* players = state_.get("players");
    const Json* player_state = nullptr;
    if (players && players->is_array()) {
      for (const Json& player : players->array()) {
        if (player.get("name") && player.get("name")->as_string() == player_name_) {
          player_state = &player;
          break;
        }
      }
    }
    const auto rule = rules_.rule_by_player.find(player_name_);
    if (!player_state || rule == rules_.rule_by_player.end() || rules_.infantry_cost <= 0) {
      return Json::Object{{"purchaseCount", Json(0)},
                          {"rule", Json(nullptr)},
                          {"unitType", rules_.infantry_type.empty() ? Json(nullptr) : Json(rules_.infantry_type)}};
    }
    const Json* resources = player_state->get("resources");
    const Json* pus = resources ? resources->get("PUs") : nullptr;
    const int money = pus ? static_cast<int>(pus->as_number()) : 0;
    return Json::Object{{"purchaseCount", Json(std::max(0, money / rules_.infantry_cost))},
                        {"rule", Json(rule->second)},
                        {"unitType", Json(rules_.infantry_type)}};
  }

  Json place() const {
    const auto& territories = array(state_, "territories");
    std::set<std::string> factories;
    for (const Json& territory : territories) {
      if (text(territory, "owner") != player_name_ || required(territory, "water").as_bool()) continue;
      for (const Json& unit : array(territory, "units")) {
        if (text(unit, "owner") == player_name_ && required(unit, "infrastructure").as_bool()) {
          factories.insert(text(territory, "name"));
          break;
        }
      }
    }
    std::set<std::string> capital_targets;
    const auto capitals = rules_.targets_by_player.find(player_name_);
    if (capitals != rules_.targets_by_player.end()) {
      capital_targets.insert(capitals->second.begin(), capitals->second.end());
    }
    std::set<std::string> preferred;
    std::set_intersection(factories.begin(), factories.end(), capital_targets.begin(), capital_targets.end(),
                          std::inserter(preferred, preferred.begin()));
    const std::set<std::string>& candidates = preferred.empty() ? factories : preferred;
    return Json::Object{{"placeAt", candidates.empty() ? Json(nullptr) : Json(*candidates.begin())},
                        {"unitType", rules_.infantry_type.empty() ? Json(nullptr) : Json(rules_.infantry_type)}};
  }

  std::vector<std::string> enemy_targets() const {
    std::vector<std::string> targets;
    for (const auto& [owner, territories] : rules_.targets_by_player) {
      if (owner != player_name_ && owner != "*")
        targets.insert(targets.end(), territories.begin(), territories.end());
    }
    if (targets.empty()) {
      const auto all = rules_.targets_by_player.find("*");
      if (all != rules_.targets_by_player.end()) targets = all->second;
    }
    std::sort(targets.begin(), targets.end());
    return targets;
  }

  Json combat_moves() const {
    const auto& territories = array(state_, "territories");
    std::map<std::string, const Json*> by_name;
    for (const Json& territory : territories) by_name[text(territory, "name")] = &territory;
    const std::vector<std::string> targets = enemy_targets();
    if (targets.empty()) return Json::Object{{"moves", Json::Array{}}};

    std::map<std::string, int> distance;
    std::map<std::string, std::string> next_step;
    std::queue<std::string> pending;
    for (const auto& target : targets) {
      if (by_name.contains(target) && !rules_.impassable.contains(target)
          && !distance.contains(target)) {
        distance[target] = 0;
        pending.push(target);
      }
    }
    while (!pending.empty()) {
      const std::string current = pending.front();
      pending.pop();
      std::vector<std::string> neighbors;
      for (const Json& neighbor : array(*by_name.at(current), "neighbors"))
        neighbors.push_back(neighbor.as_string());
      std::sort(neighbors.begin(), neighbors.end());
      for (const auto& neighbor : neighbors) {
        if (!by_name.contains(neighbor) || rules_.impassable.contains(neighbor)
            || distance.contains(neighbor)) continue;
        distance[neighbor] = distance[current] + 1;
        next_step[neighbor] = current;
        pending.push(neighbor);
      }
    }

    Json::Array moves;
    for (const Json& territory : territories) {
      const std::string source = text(territory, "name");
      Json::Array unit_ids;
      for (const Json& unit : array(territory, "units")) {
        if (text(unit, "owner") == player_name_ && text(unit, "type") == rules_.infantry_type
            && required(unit, "movementLeft").as_number() > 0) {
          unit_ids.emplace_back(text(unit, "id"));
        }
      }
      if (unit_ids.empty()) continue;
      const auto step = next_step.find(source);
      if (step == next_step.end() || !by_name.contains(step->second)) continue;
      const Json& destination = *by_name.at(step->second);
      if (required(destination, "water").as_bool()
          || (destination.get("impassable") && destination.get("impassable")->as_bool())) continue;
      int enemy_defenders = 0;
      for (const Json& unit : array(destination, "units")) {
        if (text(unit, "owner") != player_name_ && text(unit, "type") == rules_.infantry_type)
          ++enemy_defenders;
      }
      if (enemy_defenders > 0 && unit_ids.size() < static_cast<std::size_t>(enemy_defenders)) continue;
      moves.emplace_back(Json::Object{{"from", Json(source)},
                                      {"to", Json(step->second)},
                                      {"unitIds", Json(std::move(unit_ids))}});
    }
    std::sort(moves.begin(), moves.end(), [](const Json& left, const Json& right) {
      return std::tie(left.get("from")->string(), left.get("to")->string())
             < std::tie(right.get("from")->string(), right.get("to")->string());
    });
    return Json::Object{{"moves", Json(std::move(moves))}};
  }
};

}  // namespace

int main() {
  SimpleInfantryAgent agent;
  std::string line;
  while (std::getline(std::cin, line)) {
    Json request;
    Json response;
    try {
      request = JsonParser(line).parse();
      response = agent.process(request);
    } catch (const std::exception& error) {
      const Json* type = request.get("type");
      agent.log("protocol_error", {{"message", Json(error.what())},
                                    {"request_type", type ? *type : Json(nullptr)}});
      response = Json::Object{{"error", Json(error.what())}};
    }
    std::cout << stringify(response) << '\n' << std::flush;
  }
  return 0;
}
