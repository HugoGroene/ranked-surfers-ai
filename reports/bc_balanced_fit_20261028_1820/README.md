# Balanced actor fitting diagnostic

Three independently sampled training subsets, each with 16 examples of every
one of the five actions (80 total), selected using seeds 77, 78 and 79. Each
trial starts from the larger BC checkpoint. Whole-run holdout data is unused.
NONE examples are retained in the real dataset; this diagnostic does not change
its sampling, manifests, approved bounds or labels.

Frozen 192-dimensional cached features; unchanged actor architecture; full-batch
Adam learning rate 0.001, gradient clipping 1.0, at most 3,000 updates. This higher
learning rate is diagnostic only, not a change to BC or PPO defaults. Success
was defined in advance as >=95% training accuracy and >=80% recall for every
action. No trial passed. These are fitting scores, not generalization scores.

| Seed | Initial accuracy | Final accuracy | LEFT recall | RIGHT recall |
| --- | ---: | ---: | ---: | ---: |
| 77 | 20.0% | 70.0% | 43.8% | 87.5% |
| 78 | 20.0% | 72.5% | 43.8% | 87.5% |
| 79 | 13.8% | 78.8% | 56.2% | 68.8% |

Each subset had 80 distinct exact feature vectors; there were no exact duplicate
features with contradictory action labels. The full cached training features
have nonzero variation (median feature standard deviation 4.646) and centered
numerical rank 162/192. These checks do not establish sufficient visual/action
information or correct label timing.

## Interpretation

The existing actor can produce LEFT and RIGHT when fitted to balanced examples.
A completely disabled lateral output is inconsistent with this result. However,
failing to fit 80 examples to the target after this particular optimization run
does not prove the representation cannot learn them. Optimization, frozen
feature informativeness, decision ambiguity and label timing remain candidates.
Do not interpret memorization accuracy as safe gameplay or as support for
launching an overnight continuation.

Recommended next step is a small offline optimization-control comparison on
these same fixed subsets (multiple learning rates and longer fitting budget),
plus inspecting representative lateral observation/action alignments. If the
actor still cannot fit, investigate whether frozen features distinguish the
required decisions before approving any extractor fine-tuning. Once fitting is
established, compare per-action and natural-distribution whole-run holdout
metrics in a separate BC experiment, retaining useful NONE examples.

## Preservation and reproducibility

No model weights were saved. All three trials changed only the four actor
tensors in RAM; frozen policy tensors were checked unchanged. SHA-256 checks
verified the source PPO checkpoint, larger BC checkpoint and feature cache
unchanged. Exact sample indices/origins, metrics every 100 updates, input hashes
and configuration are in result.json. No live game, PPO continuation, manifest
change or Git operation was performed.

Reproduce from the repo root, using a new output directory:

```powershell
& $py -B -m tools.diagnose_demo_bc_fit `
  --experiment ai_data/demonstrations_v13/bc_runs/bc_sampled_batch_20261028_1740_v2 `
  --output reports/bc_balanced_fit_NEW
```
