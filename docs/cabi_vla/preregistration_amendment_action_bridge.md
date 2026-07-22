# CABI-VLA Action-Bridge Amendment

Status: preregistered before any v12/v13 training

## Failure being corrected

The seed-41 v10b gate is `BASELINE_INVALID`: plain BC solves 1/4 observed
state-0 edges and static CABI solves 2/4. Both fit selected offline anchors, but
same-image action sensitivity is almost entirely target-driven:

| Policy | Source sensitivity | Target sensitivity |
|---|---:|---:|
| BC | 0.000031 | 0.567442 |
| static CABI | 0.000061 | 0.558364 |

The v7 view balanced source and target marginals on an incomplete 3x2 graph.
That assigned about twice as much loss mass to `white-left` and
`yellow_white-right` as to each red edge, making target a shortcut for source.
Static CABI also wrote less than 1% relative norm into the deployed prefix, so
its identifiable role geometry remained disconnected from action prediction.

## Frozen correction

1. v12 assigns exactly equal effective action-loss mass to all four observed
   edges, including the 1/4 reduction at tetrad-anchor microbatches.
2. v13 changes only the action-free anchors: source selection supervises source
   transport and pre-transport selection supervises target transport.
3. All 5,611 original action windows and episode-start states are retained.
4. `white-right` and `yellow_white-left` actions remain absent.
5. The action bridge uses `residual_scale=1.0`, selected once from the measured
   v10b adapter/prefix norm ratio. It is not swept.
6. The closure model differs from the bridge control only by
   `decoder_closure_weight=0.25` on action-free fourth corners.

## Seed-41 gate

Train from the same Pi0.5 checkpoint, data order, optimizer and 33,000-update
budget:

1. plain BC;
2. action bridge without decoder closure;
3. action bridge with decision-point decoder closure.

Evaluate the same state 0 at fixed `K=3`. No full validation is allowed until
an observed-edge baseline reaches at least 3/4 success. Plain BC is the direct
baseline when valid; otherwise the bridge-only model is the architecture-matched
baseline for isolating the action-free closure effect.

Advance to five-state validation only when the closure model:

- retains at least 3/4 observed-edge success;
- exceeds bridge-only held-out success at state 0;
- succeeds on at least one of the two action-free edges;
- does not lose more than one observed edge relative to bridge-only.

The five-state migration gate requires both held-out edges to improve, mean
held-out success gain of at least 10 percentage points, and observed-edge
degradation no worse than 5 percentage points. Only then are seeds 42/43 and
the sealed test split evaluated.

## Interpretation

- If plain BC becomes valid, v7's marginal balancing was the baseline defect.
- If bridge-only becomes valid but plain BC does not, explicit action coupling
  is necessary for grounding; closure still needs a separate transfer gain.
- If closure does not beat bridge-only on held-out behavior, clean role geometry
  is insufficient and this CABI formulation stops.
- If no model reaches 3/4 observed success, the benchmark/training chain remains
  invalid and no migration claim is made.
