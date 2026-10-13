import json, sys, collections, glob, os

# Usage: python visualize.py <file.jsonl | directory> [more files/dirs...]
paths = []
for arg in sys.argv[1:] or ["records"]:
    if os.path.isdir(arg):
        paths.extend(sorted(glob.glob(os.path.join(arg, "*.jsonl"))))
    else:
        paths.append(arg)
if not paths:
    sys.exit("no .jsonl records found")

# Mirror everything printed into summary/<name>.txt next to this script.
summary_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "summary")
os.makedirs(summary_dir, exist_ok=True)
name = os.path.splitext(os.path.basename(paths[0]))[0] if len(paths) == 1 else "all-games"
summary_path = os.path.join(summary_dir, name + ".txt")
summary_file = open(summary_path, "w")


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


sys.stdout = Tee(sys.__stdout__, summary_file)

phase_counts = collections.Counter()
policy_counts = collections.Counter()
bytes_by_field = collections.Counter()
observers = collections.Counter()
wins = collections.Counter()
games_seen = 0
n_children = []

for path in paths:
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["type"] == "header":
                m = r["meta"]
                print(f"\n{os.path.basename(path)}: {m['agent']} as {m['player']}, "
                      f"game {m['game_number']} on {m['map_name']}")
                continue
            if r["type"] == "result":
                games_seen += 1
                wins["won" if r["won"] else "lost"] += 1
                print("  result:", r["winners"], "rounds", r["rounds_played"],
                      "steps", r["num_steps"], "-", r["status"])
                continue
            phase_counts[r["phase"]] += 1
            observers[r["state"]["observationPlayer"], r["state"]["activePlayer"]] += 1
            if r["policy_info"]:
                policy_counts[r["phase"]] += 1
                n_children.append(len(r["policy_info"]["children"]))
            for k, v in r.items():
                bytes_by_field[k] += len(json.dumps(v))
            for k, v in r["state"].items():
                bytes_by_field["state." + k] += len(json.dumps(v))

print("\nrecords per phase:", dict(phase_counts))
print("records WITH policy_info per phase:", dict(policy_counts))
print("(observationPlayer, activePlayer):", dict(observers))
print("games (result lines):", games_seen, dict(wins))
if n_children:
    print("MCTS options per decision: min", min(n_children),
          "max", max(n_children), "avg", sum(n_children)/len(n_children))
print("\nbytes by field (top):")
for k, v in bytes_by_field.most_common(10):
    print(f"  {k:30s} {v/1e6:8.2f} MB")

sys.stdout = sys.__stdout__
summary_file.close()
print(f"\nsummary saved to {summary_path}")
