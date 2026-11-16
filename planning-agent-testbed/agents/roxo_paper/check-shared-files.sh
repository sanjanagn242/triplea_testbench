#!/usr/bin/env bash
# The roxo_paper agents share their actions, playout and helpers as identical copies (phases.py,
# playout.py): they differ only in how they select. This fails, listing the differing copies, when
# they drift. Edit a shared file in one agent, copy it to the others, then run this.
set -euo pipefail

dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
agents=(ne_mcts bb_mcts oep)
shared=(phases.py playout.py weak_ai.py land_graph.py action_generators.py action_generators_cy.pyx
        generators.py noncombat_generators.py map_graph.py helper.py fast_battle.py)

status=0
for f in "${shared[@]}"; do
  sums="$(for a in "${agents[@]}"; do md5sum "$dir/$a/$f"; done)"
  if [ "$(cut -d' ' -f1 <<<"$sums" | sort -u | wc -l)" -ne 1 ]; then
    echo "differs: $f" >&2
    sed "s|$dir/||" <<<"$sums" >&2
    status=1
  fi
done
[ "$status" -eq 0 ] && echo "shared files identical in: ${agents[*]}"
exit "$status"
