# MemRL Adaptation Schedules

This directory contains sidecar experiment schedules for testing whether MemRL
online updates adapt to user habits. These schedules reuse the existing
`uv run mw eval` pipeline and do not modify the default benchmark structure.

## Entrypoint

Run a schedule with:

```bash
python scripts/run_memrl_adaptation_experiment.py \
  --schedule scripts/memrl_adaptation_schedules/online_adaptation_developer.json \
  --output-root artifacts/memrl_adapt_online_developer \
  --condition online
```

The runner writes:

- `adaptation_rounds_summary.csv`
- `adaptation_rounds_summary.json`
- per-round `run.log`, `result.txt`, screenshots, trajectories, and MemRL plans
- shared memory state under `<output-root>/memory_state`

## Baselines

Use the same schedule with different conditions:

```bash
# Full online memory: retrieve and write between rounds.
--condition online

# Static memory baseline: retrieve memory but freeze online writes.
--condition static_memory

# No-memory baseline: disable MemRL retrieval and freeze writes.
--condition no_memory
```

The main evidence is the gap between `online` and the two baselines over later
probe rounds.

## Ablations

The experiment runners expose four non-default ablation switches. They are
implemented through environment variables at runtime, so the default agent and
memory logic stay unchanged.

```bash
# 1. No profile-task transfer gate: keep same-task or unscoped memories only.
--ablation no_transfer_gate

# 2. Similarity-only retrieval: rank memories by semantic similarity only.
--ablation similarity_only

# 3. No Q-value / online utility update: retrieve from fixed memory.
--ablation no_q_update

# 4. Single-phase prompting: one prompt decides abstain / ask / act.
--ablation single_phase
```

You can repeat or comma-separate ablations:

```bash
python scripts/run_memrl_adaptation_experiment.py \
  --schedule scripts/memrl_adaptation_schedules/online_adaptation_developer.json \
  --output-root artifacts/memrl_adapt_similarity_only \
  --condition online \
  --ablation similarity_only
```

The same `--ablation` flag is available in
`scripts/run_memrl_online_rounds.py` and
`scripts/run_memrl_same_user_drift_experiment.py`.

## Schedules

`online_adaptation_developer.json`

- Round 1: cold-start silence probes.
- Rounds 2-3: evidence rounds with developer settings/communication habits.
- Rounds 4-5: held-out and retention probes.

`habit_switch_birthday_existing_profiles.json`

- Rounds 1-2: no birthday habit using the existing `grandma` profile.
- Round 3: switch evidence using the existing `student` profile, which has
  `birthday_wish_routine`.
- Rounds 4-5: post-switch probes using the student profile.

This is a non-invasive concept-drift smoke test. It uses existing profile
variants instead of changing task definitions. A stricter same-user drift test
would need a temporary profile patch mechanism or new task variants.
