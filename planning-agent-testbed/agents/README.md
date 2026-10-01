# Planning agents

Put each external agent in its own folder under `agents/`. Each folder must contain an
`agent.json` manifest describing its stable ID, display name, language, entrypoint, and launch
command. The testbench discovers these manifests when it loads agent choices; they all use the
same engine-side `ExternalAgentPlayer`. Adding an agent does not require a Java class or a new
engine player label.

For example, `simple-infantry/agent.json` starts the C++ example through its local shell launcher.
The command array supports `${repoRoot}`, `${agentDir}`, and `${entrypoint}` substitutions. Paths
in the manifest are relative to the agent folder when used by the launch command. The process
communicates with TripleA over newline-delimited JSON on standard input/output.

Simulation configuration files store external assignments by stable manifest ID:

```json
"agents": {
  "Russians": "simple-infantry"
}
```

`agents.json` is a human-readable inventory of agent IDs; the per-agent `agent.json` is the
authoritative launch definition. See [the agent developer guide](../docs/AGENT_DEVELOPER_GUIDE.md)
for the message protocol, action format, and runtime requirements.
