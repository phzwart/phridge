# phridge refactor — staged prompts for a fast coding model

Nine plans, each split into stages. One stage = one prompt = one commit.
Run plans in order A → I; stages inside a plan in order. Every stage ends
with a verify block that must pass before the next stage starts.

| Plan | What it does | Stages | Risk |
|---|---|---|---|
| A | Walls, quarantine, dead shims | A1–A3 | low (moves + deletes) |
| B | Contracts package, manifests, dependency tiers, pyproject | B1–B4 | low |
| C | One cctbx adapter layer (`shims/cctbx/`) | C1–C2 | low–medium |
| D | Relocate plugins into `r/ q/ d/ transforms/ report/`; conformance suites | D1–D5 | medium (many moves) |
| E | Split the `intensity_ll` monolith into Q components + calibrate policy | E1–E3 | **high** (the one piece of real surgery) |
| F | Core: registry, move sets, graph, objective, partitions, config, receipts | F1–F5 | medium (new code, well specified) |
| G | Controller state machine, optimizers, policies, builders, CLI | G1–G5 | medium |
| H | Retire `legacy/` | H1–H2 | low |
| I | First post-refactor applications: multi-crystal pieces, EM observed kinds | I1–I2 | medium |

How to use with a fast model (Grok fast or similar):

1. Paste `PREAMBLE.md` first, then the stage prompt. Never paste two stages at once.
2. Do not let it skip the `Verify` block. If verify fails, paste the failure
   back with "fix only what is needed to make Verify pass; do not widen scope".
3. Keep the model on a branch per plan (`refactor/A`, `refactor/B`, …). Merge
   to main after the plan's last stage.
4. Environment: the full test suite needs cctbx-base and torch. Stages say
   which tests must run; the rest may be collected-only (`pytest --co`).
5. The design these prompts implement is the doc "Refinement sandbox — design
   specification" (tabs: spec, refactor plan). Where a prompt says "per spec",
   it means that doc. Do not paste the whole doc into the model; the prompts
   carry what each stage needs.

File inventory the prompts assume (commit 6180392, 2026-10-05). If the tree
has moved, run `git ls-files` first and adjust paths; do not guess.
