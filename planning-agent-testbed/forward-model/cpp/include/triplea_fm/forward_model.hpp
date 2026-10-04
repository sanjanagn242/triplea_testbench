// C++ binding of the testbench forward-model library (triplea_fm), header-only.
//
// There is one forward model: the triplea_fm kernel. This binding embeds the CPython interpreter
// and calls the library's plain-data binding API (triplea_fm/flat.py), so C++ agents get exactly
// the same rules, dice and validation as Python agents without implementing anything.
//
// Build flags: `forward-model/cpp/triplea-fm-config --cflags` and `--libs`. The Python package
// location is compiled in (TRIPLEA_FM_PYTHON_DIR) and can be overridden at run time with the
// TRIPLEA_FM_PYTHONPATH environment variable. One ForwardModel per process.
//
// Usage:
//   fm::ForwardModel model;                 // starts the embedded interpreter
//   model.load_xml(xml_text);               // or start_game() for a testbench game_start request
//   int s = model.initial_state();
//   int rng = model.rng_new(42);
//   while (!model.view(s).terminal) model.apply(s, model.candidates(s).front(), rng);
//
// Action encodings, status codes and the static/view layouts are documented in flat.py.
#pragma once

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fm {

using Action = std::vector<int>;  // see triplea_fm/flat.py for the encoding

enum ActionCode { END_PHASE = 0, PURCHASE = 1, MOVE = 2, SELECT_BATTLE = 3, CASUALTIES = 4,
                  RETREAT = 5, PLACE = 6 };
enum StepCode { STEP_OTHER = 0, STEP_PURCHASE = 1, STEP_COMBAT = 2, STEP_BATTLE = 3,
                STEP_NONCOMBAT = 4, STEP_PLACE = 5 };
enum DecisionCode { DEC_NONE = -1, DEC_PURCHASE = 0, DEC_MOVE = 1, DEC_BATTLE_SELECTION = 2,
                    DEC_CASUALTIES = 3, DEC_RETREAT = 4, DEC_PLACE = 5 };
enum Status { DECISION = 0, TERMINAL = 1, REJECTED = 2, FAILED = 3 };

constexpr int NULL_PLAYER = -1;
constexpr int HELD = -2;

struct UnitType {
  int domain, movement, attack, defense;
  bool can_blitz, infrastructure, can_move_in_combat, artillery, supportable, strategic_bomber,
      factory;
  int cost;
  bool air() const { return domain == 1; }
  bool land() const { return domain == 0; }
};

struct Rule {
  int cost, other_resources, result_type, result_qty;
};

struct Static {
  int territories = 0, players = 0, alliances = 0;
  std::vector<int> production, victory, capital_of, alliance_of;
  std::vector<bool> water, impassable;
  std::vector<std::vector<int>> neighbors, frontier, dist;
  std::vector<UnitType> types;
  std::vector<Rule> rules;

  void parse(const std::vector<int>& v) {
    size_t i = 0;
    auto next = [&]() { return v.at(i++); };
    territories = next();
    players = next();
    int nut = next(), nrules = next();
    alliances = next();
    next();  // treasury resource id
    for (int t = 0; t < territories; ++t) {
      production.push_back(next());
      victory.push_back(next());
      capital_of.push_back(next());
      water.push_back(next() != 0);
      impassable.push_back(next() != 0);
      std::vector<int> n(next());
      for (int& x : n) x = next();
      neighbors.push_back(n);
    }
    for (int p = 0; p < players; ++p) alliance_of.push_back(next());
    for (int u = 0; u < nut; ++u) {
      UnitType t{};
      t.domain = next(); t.movement = next(); t.attack = next(); t.defense = next();
      t.can_blitz = next(); t.infrastructure = next(); t.can_move_in_combat = next();
      t.artillery = next(); t.supportable = next(); t.strategic_bomber = next();
      t.factory = next(); t.cost = next();
      types.push_back(t);
    }
    for (int r = 0; r < nrules; ++r) {
      Rule rule{};
      rule.cost = next(); rule.other_resources = next(); rule.result_type = next();
      rule.result_qty = next();
      rules.push_back(rule);
    }
    for (int p = 0; p < players; ++p) {
      std::vector<int> f(next());
      for (int& x : f) x = next();
      frontier.push_back(f);
    }
    // All-pairs graph distances (ignoring ownership), a convenience for agents.
    dist.assign(territories, std::vector<int>(territories, 1 << 20));
    for (int a = 0; a < territories; ++a) {
      std::vector<int> queue{a};
      dist[a][a] = 0;
      for (size_t q = 0; q < queue.size(); ++q) {
        int t = queue[q];
        for (int n : neighbors[t]) {
          if (!impassable[n] && dist[a][n] > dist[a][t] + 1) {
            dist[a][n] = dist[a][t] + 1;
            queue.push_back(n);
          }
        }
      }
    }
  }

  bool allied(int a, int b) const {
    if (a == b) return true;
    if (a == NULL_PLAYER || b == NULL_PLAYER) return false;
    return alliance_of[a] == alliance_of[b];
  }
};

struct Unit {
  int id, type, owner, loc, move_left, in_combat;
};

struct Cap {
  int territory, cap, used;
};

struct BattleInfo {
  int id, site, attacker, raid, active;
};

struct View {
  int round = 0, step_id = 0, step = 0, decision = DEC_NONE, actor = 0, terminal = 0, winner = -1;
  std::vector<int> owner, conquered, balance;
  std::vector<Unit> units;
  std::vector<Cap> caps;
  int battle_id = -1, batch_id = -1, hits = 0, defenders_hit = 0;
  std::vector<int> eligible, retreat_destinations;
  std::vector<BattleInfo> battles;
  std::unordered_map<int, int> index;  // unit id -> position in units

  void parse(const std::vector<int>& v) {
    size_t i = 0;
    auto next = [&]() { return v.at(i++); };
    round = next(); step_id = next(); step = next(); decision = next(); actor = next();
    terminal = next(); winner = next();
    int nt = next();
    owner.resize(nt);
    for (int& x : owner) x = next();
    conquered.resize(nt);
    for (int& x : conquered) x = next();
    balance.resize(next());
    for (int& x : balance) x = next();
    units.resize(next());
    for (size_t k = 0; k < units.size(); ++k) {
      Unit& u = units[k];
      u.id = next(); u.type = next(); u.owner = next(); u.loc = next(); u.move_left = next();
      u.in_combat = next();
      index[u.id] = static_cast<int>(k);
    }
    caps.resize(next());
    for (Cap& c : caps) { c.territory = next(); c.cap = next(); c.used = next(); }
    battle_id = next(); batch_id = next(); hits = next(); defenders_hit = next();
    eligible.resize(next());
    for (int& x : eligible) x = next();
    retreat_destinations.resize(next());
    for (int& x : retreat_destinations) x = next();
    battles.resize(next());
    for (BattleInfo& b : battles) {
      b.id = next(); b.site = next(); b.attacker = next(); b.raid = next(); b.active = next();
    }
  }

  const Unit& unit(int id) const { return units.at(index.at(id)); }
};

class ForwardModel {
 public:
  // python_dir: directory containing the triplea_fm package; empty = default_python_dir().
  explicit ForwardModel(const std::string& python_dir = "") {
    Py_Initialize();
    const std::string dir = python_dir.empty() ? default_python_dir() : python_dir;
    PyObject* sys_path = PySys_GetObject("path");  // borrowed
    PyObject* entry = PyUnicode_FromString(dir.c_str());
    PyList_Insert(sys_path, 0, entry);
    Py_DECREF(entry);
    module_ = PyImport_ImportModule("triplea_fm.flat");
    if (!module_) fail("import triplea_fm.flat from " + dir);
  }

  static std::string default_python_dir() {
    if (const char* env = std::getenv("TRIPLEA_FM_PYTHONPATH")) return env;
#ifdef TRIPLEA_FM_PYTHON_DIR
    return TRIPLEA_FM_PYTHON_DIR;
#else
    return ".";
#endif
  }
  ~ForwardModel() {
    for (auto& kv : functions_) Py_XDECREF(kv.second);
    Py_XDECREF(module_);
    Py_Finalize();
  }
  ForwardModel(const ForwardModel&) = delete;
  ForwardModel& operator=(const ForwardModel&) = delete;

  // ---- protocol helpers ----
  std::vector<std::string> parse_request(const std::string& line) {
    PyObject* r = call("parse_request", {str(line)});
    std::vector<std::string> out;
    for (Py_ssize_t i = 0; i < PyList_Size(r); ++i) {
      PyObject* item = PyList_GetItem(r, i);
      out.push_back(PyUnicode_Check(item) ? PyUnicode_AsUTF8(item) : std::to_string(PyLong_AsLong(item)));
    }
    Py_DECREF(r);
    return out;
  }
  // Testbench game_start (after parse_request): returns the controlled player id or -1.
  int start_game() { return as_int(call("start_game", {})); }
  // Direct load of a game XML (no testbench request): true if supported by the profile.
  bool load_xml(const std::string& xml) { return as_int(call("load_xml", {str(xml)})) == 1; }
  std::vector<std::string> messages() {
    PyObject* r = call("messages", {});
    std::vector<std::string> out;
    for (Py_ssize_t i = 0; i < PyList_Size(r); ++i) out.push_back(PyUnicode_AsUTF8(PyList_GetItem(r, i)));
    Py_DECREF(r);
    return out;
  }
  int state_from_request() { return as_int(call("state_from_request", {})); }
  std::string encode_response(const std::string& phase, const std::vector<Action>& actions) {
    PyObject* list = PyList_New(static_cast<Py_ssize_t>(actions.size()));
    for (size_t i = 0; i < actions.size(); ++i) PyList_SetItem(list, i, ints(actions[i]));
    PyObject* r = call("encode_response", {str(phase), list});
    std::string out = PyUnicode_AsUTF8(r);
    Py_DECREF(r);
    return out;
  }

  // ---- simulation ----
  Static static_info() {
    Static s;
    s.parse(int_list(call("static_info", {})));
    return s;
  }
  int initial_state() { return as_int(call("initial_state", {})); }
  std::string snapshot_json(int state) {
    PyObject* r = call("snapshot_json", {num(state)});
    std::string out = PyUnicode_AsUTF8(r);
    Py_DECREF(r);
    return out;
  }
  int clone(int state) { return as_int(call("clone", {num(state)})); }
  void free(int state) { Py_DECREF(call("free", {num(state)})); }
  int rng_new(long long seed) { return as_int(call("rng_new", {PyLong_FromLongLong(seed)})); }
  void rng_free(int rng) { Py_DECREF(call("rng_free", {num(rng)})); }
  View view(int state) {
    View v;
    v.parse(int_list(call("view", {num(state)})));
    return v;
  }
  std::vector<Action> candidates(int state) {
    PyObject* r = call("candidates", {num(state)});
    std::vector<Action> out;
    for (Py_ssize_t i = 0; i < PyList_Size(r); ++i) out.push_back(to_ints(PyList_GetItem(r, i)));
    Py_DECREF(r);
    return out;
  }
  bool validate(int state, const Action& a) { return as_int(call("validate", {num(state), ints(a)})) == 1; }
  Status apply(int state, const Action& a, int rng) {
    return static_cast<Status>(as_int(call("apply", {num(state), ints(a), num(rng)})));
  }

 private:
  PyObject* module_ = nullptr;
  std::unordered_map<std::string, PyObject*> functions_;

  [[noreturn]] static void fail(const std::string& what) {
    PyErr_Print();
    throw std::runtime_error("forward model bridge failed: " + what);
  }
  static PyObject* num(long v) { return PyLong_FromLong(v); }
  static PyObject* str(const std::string& s) { return PyUnicode_FromStringAndSize(s.data(), s.size()); }
  static PyObject* ints(const Action& a) {
    PyObject* list = PyList_New(static_cast<Py_ssize_t>(a.size()));
    for (size_t i = 0; i < a.size(); ++i) PyList_SetItem(list, i, PyLong_FromLong(a[i]));
    return list;
  }
  static std::vector<int> to_ints(PyObject* list) {
    Py_ssize_t n = PyList_Size(list);
    std::vector<int> out(static_cast<size_t>(n));
    for (Py_ssize_t i = 0; i < n; ++i) out[i] = static_cast<int>(PyLong_AsLong(PyList_GetItem(list, i)));
    return out;
  }
  static std::vector<int> int_list(PyObject* r) {
    std::vector<int> out = to_ints(r);
    Py_DECREF(r);
    return out;
  }
  static int as_int(PyObject* r) {
    int v = static_cast<int>(PyLong_AsLong(r));
    Py_DECREF(r);
    return v;
  }
  // Calls triplea_fm.flat.<name>(*args); steals references to args; returns a new reference.
  PyObject* call(const char* name, std::initializer_list<PyObject*> args) {
    auto it = functions_.find(name);
    if (it == functions_.end()) {
      PyObject* f = PyObject_GetAttrString(module_, name);
      if (!f) fail(name);
      it = functions_.emplace(name, f).first;
    }
    PyObject* tuple = PyTuple_New(static_cast<Py_ssize_t>(args.size()));
    Py_ssize_t i = 0;
    for (PyObject* a : args) PyTuple_SetItem(tuple, i++, a);
    PyObject* result = PyObject_CallObject(it->second, tuple);
    Py_DECREF(tuple);
    if (!result) fail(name);
    return result;
  }
};

}  // namespace fm
