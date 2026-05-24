# Architecture

`agent-framework` thesis: tiny inspectable agent runtime with tools, memory, planning steps, and traceable execution.

```mermaid
flowchart LR
    A0["Task"]
    A1["Planner"]
    A2["Tool Registry"]
    A3["Observation"]
    A4["Memory"]
    A5["AgentResult Trace"]
    A0 --> A1
    A1 --> A2
    A2 --> A3
    A3 --> A4
    A4 --> A5
```

## Design Rules

- Keep the public API small enough to inspect in one sitting.
- Make demos run locally without network credentials.
- Put correctness checks in tests, conformance scripts, or benchmark scripts
  instead of relying on README claims.
- Prefer explicit failure modes over surprising implicit behavior.
