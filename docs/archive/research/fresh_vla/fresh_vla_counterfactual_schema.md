# FRESH-VLA Counterfactual Pair Schema

Each record is one expert continuation from a shared pre-feedback state. Records
with the same `pair_id` differ only in the counterfactual physical outcome and
the expert continuation that becomes correct after that outcome is observable.

| Field | Meaning |
| --- | --- |
| `pair_id` | Stable identifier for continuations sharing the same pre-feedback conditioning. |
| `branch_id` | Identifier for one counterfactual rollout branch. |
| `branch_outcome` | Human-readable outcome such as `attached` or `slipped`. |
| `observation` | Policy-visible observation at the shared decision time. |
| `robot_state` | Policy-visible proprioception at the shared decision time. |
| `language_instruction` | Instruction included in the policy conditioning. |
| `action_chunk` | Expert action continuation shaped `[H, D]`. |
| `event_time` | Step at which the physical event occurs. |
| `feedback_reveal_time` | First step at which policy-visible sensing distinguishes the outcome. |
| `action_divergence_time` | First persistently different expert action step across outcomes. |
| `gripper_transition_horizon` | Event baseline retained only for comparison. |
| `oracle_feedback_horizon` | Common action-prefix length; equal to `action_divergence_time`. |
| `per_step_branch_divergence` | Normalized expert action distance for every future step. |
| `is_deterministic_control` | Marks deterministic negative-control examples. |

For repeated expert rollouts, the initial divergence threshold is the 95th
percentile of within-branch normalized action distance. The oracle horizon is
the first step whose cross-branch distance stays above that threshold for the
configured persistence window. Reports must include threshold sensitivity.

Only `observation`, `robot_state`, and `language_instruction` may enter the
policy. Oracle horizons and all branch/event metadata are loss-side fields.
`scripts/fresh_vla/counterfactual_data.py` validates both the schema and this
policy-input whitelist.
