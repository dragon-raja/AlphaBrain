# Policy-response surrogate Gate -1 results

Date: 2026-07-19

Decision: **STOP_POLICY_RESPONSE_SURROGATE**

## Validity

- 716/716 expected fit-partition CCV states were collected.
- 534 states from 18 source IDs were used for fitting and grouped alpha
  selection; 182 states from five nested evaluation source IDs were opened once
  for the frozen decision.
- The preregistration SHA-256 matched the collected run configuration.
- Missing and extra record counts were both zero.
- CCV holdout, original test, confirmation, and sealed states opened: zero.
- Endpoint responses used two coupled flow-noise draws from a namespace disjoint
  from the six-repeat continuation labels.

## Result

All point estimates weight source initial states equally.

| Feature probe | Oracle gain recovered | Pair concordance | Oracle top-set hit | Stable-grasp harm |
| --- | ---: | ---: | ---: | ---: |
| candidate prefix | 44.7% | 0.466 | 63.9% | 0.7% |
| immediate physical signature | 0.0% | 0.018 | 67.7% | 0.0% |
| endpoint policy response | 1.7% | 0.517 | 70.1% | 2.1% |
| candidate + response | 63.5% | 0.487 | 68.0% | 2.1% |

The available Oracle utility gain was `0.01168`, source-bootstrap 95% CI
`[0.00508, 0.01717]`, so the evaluation subset contains real action leverage.
Candidate+response recovered 63.5% of that point estimate, but its recovered-gain
interval was `[31.3%, 77.6%]`. Its improvement over the better candidate/direct
baseline had paired 95% CI `[-18.5, +50.7]` percentage points and therefore did
not exclude zero.

Response-only concordance was 0.517, below the frozen 0.55 requirement. Its
source-level recovered-gain ratios were highly inconsistent: approximately
`-164.0%, 78.8%, 5.4%, -2.3%, -0.1%`. Candidate+response improved three sources
but remained negative on two. This is not a stable response-sufficiency result.

## Frozen checks

Passed:

- available Oracle gain lower confidence bound above zero;
- candidate+response point recovery above 35%;
- point improvement over candidate/direct above 10 percentage points;
- stable-grasp harm at most 5%.

Failed:

- paired improvement lower confidence bound above zero;
- response-only pairwise concordance above 0.55.

The preregistration requires every check. The favorable combined point estimate
cannot override the failed source-level checks.

## Interpretation

One next-policy call is not a source-stable sufficient statistic for the useful
continuation ordering. The exact Oracle appears to benefit from repeated
interaction: candidate execution changes contact, the frozen policy responds,
that response changes the next physical state, and later replans expose the
decisive difference. Compressing this into one endpoint response, a local
physical summary, or another scalar head is not supported by this task.

This result stops the deployable response-surrogate and the paused FOVEA
first-passage reranker. It does not negate the exact continuation Oracle's
closed-loop gain, nor the 43.36% action-leverage rate in CCV Gate 0A.

## Next mechanism

The remaining evidence-backed option is training-only continuation selection:

1. use repeated frozen-policy continuations to select an in-support Pi0.5
   candidate at on-policy recovery states;
2. abstain where repeat-level ordering is unstable or the winner does not beat
   sample 0;
3. distill only accepted winner actions into the base policy with the original
   successful-data anchor;
4. deploy the updated Pi0.5 as `N=1`, with no simulator, reranker, response model,
   or extra policy call;
5. compare against random-candidate, immediate-physical, ordinary continuation,
   and the already harmful expert-recovery SFT controls.

This is continuation-selected self-distillation, not another loss-weighting or
offline expert replay experiment. It is justified only as the next falsifiable
pilot; it is not yet an original-method claim.

Machine-readable result:
`/share/longjunyu/fresh-vla/policy-response-vla/gate-minus1-v1/gate_minus1_result.json`.

