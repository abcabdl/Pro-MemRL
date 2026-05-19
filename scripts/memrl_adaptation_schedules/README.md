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
