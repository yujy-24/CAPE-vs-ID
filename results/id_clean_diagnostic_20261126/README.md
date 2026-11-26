# ID clean generalization diagnosis — 2026-11-26

**Question.** Why does the ID seed-43 checkpoint score below zero on held-out LIBERO tasks? This is a read-only diagnosis: no checkpoint was retrained and no training protocol or main-table number was changed. The command was `python diagnose_id.py --ckpt cache_v3/ckpt/id_seed43/final.pt --out cache_v3/diagnostics/id_clean_20261126/report.json`. The run began at 16:49:45; the final baseline-corrected report completed at 17:03:47 with exit code 0 on the rented RTX 4090. The final script is at repository commit `9e8b7c6`; the model is the previously recorded seed-43 checkpoint.

## Data separation

- **Same-task, new episodes:** 26 episodes from 9 of the 32 training tasks. None of these 26 episode IDs appears as a training window's `episode` in the manifest. Their images were decoded again and encoded with the same frozen DINOv3 backbone. All valid consecutive pairs were evaluated once (2,678 pairs).
- **Unseen tasks:** the original 995 validation windows from 8 held-out tasks, giving 9,950 pairs. This reproduces the existing `id_direct` score. Removing repeated `(episode, frame)` pairs leaves 9,132 unique pairs.
- The same-task set covers only 9 training tasks and uses all consecutive pairs, whereas the unseen-task set comes from sampled 10-step windows. The gap is strong evidence of a generalization difference for these sampled sets, but is not a population-wide estimate for all 32 training tasks.

## Main numbers

| Evaluation set | ID R² | ID MAE | Training-action mean baseline R² / MAE | Previous-action baseline R² / MAE |
|---|---:|---:|---:|---:|
| New episodes, seen tasks (2,678 pairs) | **0.7447** | **0.0966** | −0.0151 / 0.2275 | **0.9596 / 0.0243** |
| Existing held-out tasks (9,950 pairs) | **−0.0405** | **0.2033** | −0.0161 / 0.2165 | **0.9679 / 0.0190** |
| Held-out tasks, unique pairs (9,132 pairs) | **−0.0507** | **0.2037** | −0.0171 / 0.2158 | **0.9678 / 0.0190** |

The mean baseline uses the action labels in the existing training-cache windows only. The same-task / unseen-task gap is consistent with task distribution shift. It does not establish a unique cause: camera, scene, action policy, action observability, or feature geometry could shift together. The previous-action baseline uses extra input (`a_{t-1}`) unavailable to this image-pair ID head, so it is **not a matched-input competitor**. Its high score shows extreme action persistence in this dataset and sets a useful diagnostic reference.

## Does the head use visual change?

| Set | Correct `(z_t,z_{t+1})` R² | Static `(z_t,z_t)` R² | Reversed `(z_{t+1},z_t)` R² |
|---|---:|---:|---:|
| New episodes, seen tasks | **0.7447** | 0.6762 | 0.4312 |
| Held-out tasks | **−0.0405** | −0.0781 | −0.1620 |

The second frame and its temporal order matter, but the static pair's high same-task score shows that single-frame cues already explain much of this head's output. These substitutions are diagnostic distribution shifts, not newly trained ablations; they do not quantify a causal fraction of learned information.

## Localization and timing

- Grouped R² on same-task new episodes: translation 0.760, rotation 0.505, gripper 0.723. On unseen tasks: translation −0.032, rotation −0.086, gripper −0.053.
- Held-out task 2 is particularly poor (R² −0.911; 1,850 pairs): gripper target mean 0.232 versus prediction mean 0.843. Task 38 is positive (R² 0.543; 950 pairs). This is task-dependent failure, not a uniform loss across every held-out task.
- In the fixed central-frame lag test, same-task R² at label offsets −2/−1/0/+1/+2 is 0.730/0.750/**0.753**/0.735/0.701. Held-out-task R² is −0.044/−0.039/**−0.037**/−0.040/−0.046. Zero offset is best by R², but its margin over −1 is small; a global one-frame misalignment is not supported as the main explanation.
- The full JSON also groups results by six-dimensional action magnitude and DINO feature-change norm. The low-magnitude and small-change bins have lower within-bin R², particularly on held-out tasks. Since each bin has its own target variance, these R² values alone do not prove larger absolute errors; use the accompanying MAE. Feature change is a proxy and must not be interpreted as ground-truth physical motion.

## Interpretation and next check

This diagnosis supports **task-dependent generalization failure** for the frozen DINOv3 + IDNet setup. It does not prove that action numeric semantics change across tasks, that the head learned task semantics, or that head capacity or missing video pretraining caused the gap. The next useful audit is to inspect action conventions and the task-2 gripper mismatch, then repeat the same-task test on a broader episode-disjoint split before making a population-level claim. To test the original ID representation mechanism, a separate experiment must allow the encoder to receive gradients from ID; the current encoder is frozen.

`report.json` contains all per-task and per-dimension metrics, means, standard deviations, baselines, lag sweeps and the exact 26 episode IDs. `run.log`, `started_at.txt`, `status.txt` and SHA-256 files preserve execution evidence. Binary checkpoints and feature caches remain excluded from Git.
