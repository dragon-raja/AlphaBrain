# LIBERO-Bind v0 data quality report

## Frozen artifact

- Suite: `/share/longjunyu/cabi-vla/libero-bind-v0`
- Teacher collection: `/share/longjunyu/cabi-vla/libero-bind-v0-train-v1`
- Leakage-safe training view: `/share/longjunyu/cabi-vla/libero-bind-v0-train-view-v3`
- Canonical train states: `0..34`
- Action-supervised edges: `red-left`, `red-right`, `white-left`,
  `yellow_white-right`
- Action-withheld edges: `white-right`, `yellow_white-left`

The collection contains 140 attempted episodes. Teacher task success is:

| Edge | Successful / attempted | Rate |
|---|---:|---:|
| `red-left` | 33 / 35 | 94.3% |
| `red-right` | 35 / 35 | 100.0% |
| `white-left` | 35 / 35 | 100.0% |
| `yellow_white-right` | 35 / 35 | 100.0% |

The failed `red-left` episodes are canonical states 7 and 30. Both remain in the
collection manifest and teacher-success denominator, but neither contributes action
windows or a CABI tetrad.

## Training view checks

The v3 view contains:

- 5,611 stride-5 behavior-cloning windows;
- 138 successful transport anchors;
- 64 complete CABI tetrads;
- zero loaded actions from either withheld edge.

For every retained tetrad, changing only the target role satisfies both checks:

1. agent-view image, wrist-view image, and robot state are exactly equal elementwise;
2. the corresponding 10-step teacher action chunks differ.

All 64 retained tetrads pass both checks. Their target-intervention action MSE has
minimum `0.5714286` and mean `0.5714324`.

Two candidate tetrads from canonical state 4 were excluded. The red-left and red-right
teacher episodes used different grasp fallback offsets, so their transport anchors no
longer shared the same physical conditioning. They remain valid BC trajectories but
cannot support a causal target-role intervention.

## Visual audit artifacts

Verified full-episode videos are stored at:

`/share/longjunyu/cabi-vla/libero-bind-v0-train-v1-videos-v1`

States 0, 1, 2, 3, and 5 are rendered for all four supervised edges. The artifact has
20 H.264 MP4 files and 20 AV1 WebM files. Every file decodes successfully as `yuv420p`;
the shortest trajectory has 140 frames. Codec and frame-level verification metadata
are recorded in the artifact `manifest.json`.

## Gate result

`PASS_DATA_GATE`

The data may be used for the preregistered BC/CABI comparison. This report does not
claim policy transfer or behavioral improvement.
