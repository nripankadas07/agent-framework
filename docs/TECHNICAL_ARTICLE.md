# agent-framework: agents from scratch, but actually usable

This is a launch-ready technical article draft for the repository. It is meant
to explain the idea, not inflate traction.

## The Problem

Many agent stacks hide the loop. This repo keeps the loop visible: plan, act, observe, remember, finish.

## The Core Idea

A planner returns an `AgentStep`; the runtime executes a named tool and records the observation. The list of steps is the trace.

## Why It Teaches

You can replace the rule-based planner with an LLM call without changing the runtime contract.

## Limitations

No sandbox is implied for arbitrary tools. Tool safety belongs to the tool author, and examples avoid unsafe `eval()`.

## Try It

Run the README demo from a clean checkout. If the demo needs credentials, it is
not a good flagship demo.
