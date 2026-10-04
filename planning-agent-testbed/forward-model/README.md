# Forward-model library

The planning-agent testbench provides the forward model as a library. Agent developers use it to
simulate the game in tree search, planning or RL; they never implement game rules themselves. It
is independent of the TripleA engine: nothing here talks to a running game. It implements
[forward-model-implementation-spec.md](../forward-model-implementation-spec.md) (profile
`forward-model-v1`, land/air subset; see the Python README for status and deviations).

There is **one** forward model, the `triplea_fm` kernel. Every language reaches the same kernel
through a binding, so all agents simulate identical rules, dice and validation.

```
                        ┌────────────────────── forward-model/ (testbench library) ──────────────────────┐
Python agent ──import──▶│ python/triplea_fm   kernel: definition, state, rules, kernel, rng, actions      │
                        │                     agent-side: toolkit (candidates), baselines (opponent/eval), │
                        │                                 testbench (observation ↔ state)                  │
C++ agent ──#include──▶ │ cpp/include/triplea_fm/forward_model.hpp ──CPython──▶ triplea_fm.flat (plain-data│
                        │                                                         binding API)            │
future Java/RPC ──────▶ │ (bind to triplea_fm.flat the same way)                                          │
                        └─────────────────────────────────────────────────────────────────────────────────┘
TripleA engine (Java) ◀──testbench JSON protocol──▶ agents   (the library is not involved in live play)
```

| Path | Contents |
|---|---|
| `python/triplea_fm/` | The kernel and agent-side helpers ([python/README.md](python/README.md)) |
| `python/triplea_fm/flat.py` | Plain-data binding API (ints, strs, int lists, integer handles) that every non-Python binding calls |
| `python/tests/` | Rule, contract and toolkit tests |
| `cpp/include/triplea_fm/forward_model.hpp` | Header-only C++ binding (`fm::ForwardModel`, `fm::View`, `fm::Static`) |
| `cpp/triplea-fm-config` | Compiler and linker flags for C++ users (like pkg-config) |
| `cpp/tests/` | C++ binding tests (`cpp/tests/run-tests.sh`) |
| `../tools/arena.py` | Offline arena: the library stands in for the engine; agents run as real subprocesses |

## Using it

**Python.** Make `forward-model/python` importable (`pip install -e planning-agent-testbed/forward-model/python`,
or set `PYTHONPATH`, as `agents/python-mcts/run-agent.sh` does):

```python
import triplea_fm as fm
from triplea_fm import toolkit
d, report = fm.load_definition(xml_text)
s, rng = fm.create_initial_state(d), fm.SeededRng(1)
r = fm.transition(d, s, toolkit.candidates(d, s)[0], rng)
```

**C++.** Include one header and build with the config script. Requirements: C++20 and the Python
3.10+ development headers.

```cpp
#include <triplea_fm/forward_model.hpp>
fm::ForwardModel model;                       // embeds the interpreter, loads triplea_fm
model.load_xml(xml_text);                     // or parse_request(line) + start_game() in an agent
int s = model.initial_state(), rng = model.rng_new(1);
fm::View v = model.view(s);                   // decision, owners, units, battles, ...
model.apply(s, model.candidates(s).front(), rng);
```
```bash
g++ -std=c++20 $(forward-model/cpp/triplea-fm-config --cflags) agent.cpp -o agent \
    $(forward-model/cpp/triplea-fm-config --libs)
```

**Inside a testbench agent** (either language): build the root state from each `turn_request` with
`testbench.LiveBridge` (C++: `parse_request` + `state_from_request`), search with `clone` /
`validate` / `apply`, and encode the chosen actions with the bridge (C++: `encode_response`).
`agents/python-mcts` and `agents/cpp-mcts` are complete examples. Each contains only its search
and heuristics.

## Tests

```bash
(cd planning-agent-testbed/forward-model/python && python3 -m unittest discover tests)
planning-agent-testbed/forward-model/cpp/tests/run-tests.sh
```

## Extending

Rules change in one place: `python/triplea_fm`. A new language gets a binding over
`triplea_fm.flat`, not a new model. If the kernel is later moved to the native kernel with a C ABI
recommended by spec §14, `flat.py` documents the operations that ABI must provide. The C++ header
keeps the same class interface, so agents need no changes.
