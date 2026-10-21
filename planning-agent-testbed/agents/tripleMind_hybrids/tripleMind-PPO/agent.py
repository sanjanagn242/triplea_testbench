#!/usr/bin/env python3
"""TripleMind with a PPO policy choosing the attack formation (testbench entry point).

Inference only: the checkpoint is loaded once, the policy picks the highest-probability formation
(argmax) and nothing is trained. The class is ppo/agent.py's TripleMindPPO; training lives in
ppo/train.py (run_ppo.py train).

Environment:
  TRIPLEMIND_PPO_CHECKPOINT  policy checkpoint (default runs/ppo-selfplay/checkpoints/best.pt)
  TRIPLEMIND_PPO_MODE        greedy (default) or sample
  TRIPLEMIND_PPO_THREADS     torch threads (default 1)
  TRIPLEMIND_VIEW            1 shows TripleMind's live map window, 0 hides it (set in run-agent.sh)
  TRIPLEA_AGENT_SEED         seed of the agent's randomness (default 0)
  TRIPLEA_AGENT_LOG          1 enables logging, anything else disables it (set in run-agent.sh)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from ppo.agent import TripleMindPPO  # noqa: E402

if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("TRIPLEMIND_PPO_THREADS", "1")))
    agent = TripleMindPPO()
    agent.record_game_results = True
    agent.run()
