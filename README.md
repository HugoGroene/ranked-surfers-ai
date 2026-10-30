# Ranked Surfers AI — PPO V6.1.2 + Human Demonstrations V1.3

**Last updated: 30 October 2026.** Development branch: [`headless-parallel`](https://github.com/HugoGroene/ranked-surfers-ai/tree/headless-parallel).

Screenshot-based reinforcement learning for a **locally hosted Practice copy** of the Unity WebGL endless runner Ranked Surfers. This project uses Stable-Baselines3 PPO, native game-canvas screen capture, asynchronous nine-class YOLO segmentation, and a fail-closed gameplay/menu guard. The separate **experimental human-demonstration V1.3** workflow records manual gameplay, permits reviewing and trimming good segments, and can fine-tune a **copy** of the existing PPO model using behavioral cloning (BC). Neither recorder nor trainer is intended for public ranked/live matches.

> **Repository vs. local assets:** Source, tests and selected experiment reports are versioned; working checkpoints, raw demonstrations, review videos and feature caches remain local. A fresh clone does not include the latest models or a Python environment. The older Git-LFS milestone is not the current working policy. The 27 October workflow below is historical; use the current status and pipeline guide first.

## Current status: recovered 28 October development

**Headless parallel PPO (experimental):** up to six independent local Unity browsers
feed one shared, settled-learning PPO optimizer. A six-worker whole-canvas check
completed five updates and all learning audits at 18.4 optimized transitions/s;
this does not establish improved policy quality or long-run stability. The virtual
display driver is disabled. The dashboard provides a six-tile grid, persistent
survival curve and training/system statistics. Start with the
[headless parallel guide](README_HEADLESS_PARALLEL.md),
[whole-canvas profile](README_PPO_CANVAS_V1.md) and
[diagnostic details](README_PPO_BACKGROUND.md).

**29 October learning-correctness correction:** pause-aware training now uses
an episode-aware collector with pre-GAE boundary settlement, actor-masked
passive resync transitions, verified-time survival rewards, elapsed-time
discounting and full-rollout deadline completion. Reward coefficients and
network/schema are unchanged, but timing/return semantics deliberately differ
from historical runs. See [implementation and live gates](reports/ppo_settled_validation_20261029/README.md)
and [current pipeline usage](README_PPO_PIPELINE.md). The findings and proposed
next work below describe the earlier snapshot. These corrections have passed
short live gates; the first corrected learning-rate comparison is complete below. Use `--legacy-learning`
only for explicit historical reproduction.

**29 October current results:** both corrected PPO arms completed 32,768
transitions from the small BC pilot, with rollout 512 and 100-episode matched
evaluations. Fresh pilot mean: 4.6586 s; learning rate 0.0001: 4.7756 s;
0.0003: 4.4793 s. Neither established an improvement. No learning-rate winner
was adopted; repeat evaluations were deferred in favor of failure diagnosis.
See [exact results](reports/ppo_lr_comparison_20261029/README.md).

**Capture issue remains open:** inspected PPO/detector source images include
browser chrome, whereas BC records a calibrated canvas before applying its
crop. Their 512x288 images therefore have different framing. This affects PPO
capture as well as BC compatibility; it is not merely a recorder issue and has
not been fixed. See [diagnosis and limitations](reports/ppo_failure_diagnosis_20261029/README.md).

The priority is an environment and PPO learning pipeline that supports effective
autonomous learning from survival/death rewards. BC remains an optional starting
point, not an action quota or a requirement to imitate human movement forever.
Better runtime throughput is measured separately from better gameplay.

| Foundation | Implemented and verified | Remaining qualification |
| --- | --- | --- |
| Longer episodes | Default cap 300 seconds; death/timeout distinction, reset and bootstrap handling; a training episode reached 17.09 seconds | Multi-minute model survival is not demonstrated; pass 15 explicitly for historical comparisons |
| Persistent Chromium | One owned window and worker across the final seven-stage batch; six reuses and clean closure | Keep the desktop visible/unobstructed; Windows may still refuse initial focus |
| Fresher action inputs | Optional exact-observation safety reading, single-use identity/timestamp checks, 20 ms deadline and fresh-check fallback | Opt-in; median image age improved, rare fallback tails remain; Unity response time is not measured |
| Controlled PPO settings | Explicit saved-setting overrides, effective-setting reports, rollout 128/512, target-KL and learning-rate comparisons | No winning learning configuration established; exploration remains unresolved |

See [pipeline usage](README_PPO_PIPELINE.md),
[implementation/tests/latency validation](reports/ppo_pipeline_foundation_20261028/README.md)
and [remote/parallel scaling assessment](reports/ppo_pipeline_foundation_20261028/SCALING.md).
The policy architecture, four-frame RGB/557-D observation, five-action mapping,
reward coefficients and normal confidence thresholds remain unchanged. No
recurrent model, integrated demonstration loss, detector removal or parallel
simulation backend was added.

### Completed learning experiments

All entries below are deterministic 100-episode evaluations. Separate procedural
sessions and one training seed limit conclusions; use the matched baseline for
each experiment rather than treating all historical scores as interchangeable.

| Experiment | Mean survival | LEFT / RIGHT selections | Interpretation |
| --- | ---: | ---: | --- |
| Pre-v1.7.9 overnight source | 3.9933 s | 0 / 0 | Matched before reference |
| Eight-hour v1.7.9 PPO | 4.2970 s | 0 / 0 | Improvement over source inconclusive |
| Small actor-only BC pilot | 4.8276 s | 0 / 0 | Best observed historical 100-episode mean; preserve |
| Larger BC | 4.4800 s | 0 / 0 | No demonstrated improvement over pilot |
| Macro-selected BC | 3.6653 s | 91 / 9 | Lateral choices emerged, survival worsened |
| Pilot + 4,096 PPO steps | 4.6238 s | 0 / 0 | Short continuation did not establish improvement |
| Macro BC + 4,096 PPO steps | 4.5058 s | 1 / 0 | Recovered survival, no clear advantage over pilot |

The second recording session saved 291 runs at approximately 20.5 FPS. Under
the explicitly authorized sampled-review policy, 142 were selected with trims
and combined with nine approved pilot runs: 151 approved runs in total. Raw
recordings remain intact. Approval and timestamp consistency do not establish
optimal actions; no diagnostic recording was silently treated as expert data.
BC froze the extractor/critic and changed four actor tensors. Frozen-feature
caching and per-action selection diagnostics are included in the new tooling.

Reports: [overnight](reports/v179_overnight_comparison_20261028/README.md),
[pilot BC](reports/bc_pilot_evaluation_20261028_153047/README.md),
[larger BC](reports/bc_sampled_batch_20261028/README.md),
[BC optimization controls](reports/bc_optimization_controls_20261028/README.md),
[macro gameplay](reports/bc_macro_gameplay_20261028/README.md),
[BC-to-PPO continuations](reports/ppo_bc_continuation_ab_20261028/README.md).

### Final multi-hour experiment: complete before the Windows restart

The controlled batch completed at **23:30 on 28 October**, before the reported
00:35 restart. All 98,304 training steps (three independent 32,768-step arms)
and 400 evaluation episodes completed. Each arm started from the same preserved
pilot, not from the preceding arm. All used the optional faster guard and a
300-second cap. Learning rate 0.0003, batch 32, epochs 5, gamma 0.995 and entropy
0.002 were held fixed.

| Model | Rollout | Target KL | Evaluation mean | Training steps/s |
| --- | ---: | ---: | ---: | ---: |
| Fresh pilot baseline | — | — | 4.6494 s | — |
| A | 128 | None | 3.9852 s | 16.06 |
| B | 512 | None | 4.3799 s | 18.64 |
| C | 512 | 0.03 | 4.1714 s | 19.81 |

All three arms passed stability and reward replay audits; no guard abort/hard
stop occurred. None established a survival improvement. A regressed in this
sample; other differences remain uncertain. C recorded 63 KL early stops;
target KL is not a hard bound on update size. A retains a documented warning
that its final MENU boundary had no subsequent step to record the last terminal
penalty. Reward replay checks arithmetic, not Unity internal time or unbiased
reward caps/discounting.

[Full results, uncertainty intervals and provenance](reports/ppo_controlled_learning_20261028/README.md).
The final batch state is COMPLETE; its older live_session.json is a stale
RUNNING launch note, not evidence of an interrupted experiment. Recovery checks
verified ZIP CRCs and metadata of the three saved outputs and matched all 83
recorded source hashes. No expensive experiment needs repeating for recovery.

Preserved local reference:
`ai_data/demonstrations_v13/bc_runs/bc_20261028_152607/model.zip`
(SHA-256 `ca99e521298e783eec6f5b11de11fa21eaa03631842801457e4ea3974643c5bc`).
Outputs: `ai_data/rl_v2_runs/ppoctrl_20261028_02_train/latest.zip`,
`ppoctrl_20261028_04_train/latest.zip`, and `ppoctrl_20261028_06_train/latest.zip`.
Do not promote the newest checkpoint merely because it is newest.

### Next learning work

The correctness audit, focused fixes and first corrected learning-rate comparison
are complete. Neither learning rate earned promotion. Diagnose capture alignment,
action coverage and failures before drawing conclusions from further training.
BC remains an optional initialization; the current PPO collector does not replay
demonstrations or apply an integrated imitation loss. Recurrent memory is deferred.

The user requested a three-hour continuation from the eight-hour PPO plus larger
BC checkpoint (`bc_sampled_batch_20261028_1740_v2/latest.zip`). Commands were
provided; launch/completion is not yet confirmed. That checkpoint stores rollout
128 and learning rate 0.0003. The initially supplied command overrode these to
512 and 0.0001; the latter was a new conservative choice, not an adopted test
winner. Actual run metadata must establish which settings were used. Historical
and corrected training returns are not directly comparable; evaluate source and
output checkpoints under matching gameplay evaluation settings.

No cloud resources were provisioned. The persistent batch runner is sequential
automation, not parallel environment collection. Raw models/data stay local;
the published reports describe the exact assets needed to reproduce a run.

## 28 October update: v1.7.9 runtime and overnight evaluation

Current development branch: `claude-development`. The 27 October branch and
latest-checkpoint descriptions below are historical. See
[the v1.7.9 comparison](reports/v179_overnight_comparison_20261028/README.md)
for the current measured results and local checkpoint paths.

The pause-aware trainer completed an eight-hour continuation. It excludes
DOM-visible or visually stale Pause panels from post-optimizer observations,
requires four consecutive confident gameplay captures, and retains guarded
terminal rewards. PPO architecture, reward coefficients, classifier labels and
confidence thresholds are unchanged. Checkpoints and recordings remain local.

Matched 100-episode deterministic evaluations scored 3.99 s before and 4.30 s
after the continuation; the bootstrap 95% interval for the mean difference
was -0.16 to +0.75 s. Both selected zero LEFT/RIGHT actions and had zero guard
aborts. Improvement remains inconclusive; investigate lateral-action collapse
before another long continuation.

Offline runtime regressions (no installer payloads needed):

```powershell
& $py -B -m unittest discover -s tests -p "test_ppo_pause_runtime_v179.py" -v
& $py -B -m unittest discover -s tests -p "test_ppo_current_runtime.py" -v
```

The local startup focus helper now attempts to activate only its owned Chromium
window and requires both Windows foreground ownership and browser focus before
the first episode. If Windows denies activation, it allows eight seconds for
manual focus and then stops without gameplay input. Automatic activation was
denied in early live attempts; later owned-window activation and persistent
batches succeeded. This is not a guarantee of unattended startup on every run.
See [next-step research and focus limitations](reports/v179_overnight_comparison_20261028/NEXT_STEPS_RESEARCH.md).
Focus regressions: `& $py -B -m unittest discover -s tests -p "test_ppo_browser_focus.py" -v`.

## 28 October: autonomous PPO pipeline foundation

The current focus is reliable and efficient autonomous PPO learning; BC remains
optional. The new [pipeline guide](README_PPO_PIPELINE.md) covers 300-second
episode limits, a persistent single-window batch runner, explicit saved-setting
overrides and an opt-in lower-latency observation guard. Historical commands
passing a 15-second limit remain reproducible. The faster path is experimental;
short latency checks do not establish better gameplay or learning efficiency.
See [validation and scaling](reports/ppo_pipeline_foundation_20261028/README.md).

## Historical results — 27 October 2026

### Controlled PPO evaluations

| Policy / reference | Deterministic episodes | Mean survival | Median survival | At least 5 s | At least 10 s | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Earlier PPO checkpoint | 100 | 3.11 s | 3.35 s | 0 | — | Published older comparison |
| No-action reference | 100 | 3.48 s | 3.37 s | 12 | — | Published older comparison; not a learned policy |
| 120-minute PPO continuation | 100 | 3.84 s | 4.16 s | 15 | — | Published in `reports/v612_learning_proof_20261026_153745/` |
| **Latest 7-hour PPO continuation** | **100** | **4.61 s** | **4.21 s** | **44** | **2** | **Local result; report not yet committed** |

The latest evaluation's longest survival was **10.25 s**. The latest seven-hour training continuation logged **374,703 additional PPO steps**, **5,417 episodes**, a training-episode mean of **4.33 s**, a training best of **15.03 s**, **14.87 steps/s including resets**, and **zero guard aborts**. Training and evaluation episode averages measure different distributions and should not be conflated.

**Latest local checkpoint (not automatically available in a fresh Git clone):**

```text
ai_data/rl_v2_runs/v612_overnight_next_420min_press5_20261027_094116/latest.zip
```

**Action-collapse diagnosis:** In the latest deterministic 100-episode evaluation, the policy selected **zero LEFT and zero RIGHT** actions. Recorded counts were `NONE=1978`, `JUMP=2779`, `ROLL=3169`, `LEFT=0`, `RIGHT=0`. Mean predicted action probabilities were `NONE=21.66%`, `LEFT=0.56%`, `RIGHT=0.15%`, `JUMP=36.59%`, `ROLL=41.04%`. The key experimental hypothesis is that carefully approved human lane-change examples can introduce LEFT/RIGHT behavior while preserving the model's existing timing. **BC has not yet been shown to improve survival.**

Small changes in 100-episode means can reflect procedural/evaluation variance. Compare checkpoint evaluations using identical game, guard, detector and action settings, and repeat close comparisons if needed.

## How the system works

| Component | Current implementation |
| --- | --- |
| RL agent | Stable-Baselines3 `PPO('MultiInputPolicy')`, five discrete actions: `NONE=0`, `LEFT=1`, `RIGHT=2`, `JUMP=3`, `ROLL=4` |
| Images | Four successive 512×288 RGB game-crop frames, stacked oldest-first as **12×288×512** uint8 |
| Numerical observation | **557 float32 features**: nine-class segmentation/hazard summaries, image-column features, action history, capture/detection timing and action context |
| Feature extractor | `RGBHazardExtractor`: RGB CNN → 128 features; auxiliary MLP **557→96→64**, then concatenated (192 features total) |
| Detector | Nine-class YOLO segmentation, typically `--imgsz 960`, asynchronous **one-worker** inference approximately every 3 decisions; stale results are gated |
| PPO input | Actions executed in the local Practice browser by the PPO environment; safety checked before actions |
| Timing | PPO target `--fps 30`, uncapped, typical proven action tap **5 ms** (`--press-delay-ms 5` must be passed explicitly; script default is 10 ms) |
| Rewards | Survival time + periodic milestones and death penalty; **no** hard-coded hazard/action reward or inferred track-lane safety rules |
| Safety | `rs_v612_gameplay_guard.py` prevents keyboard actions if gameplay cannot be verified; no system can eliminate the capture-to-keypress race entirely |
| Human demonstrations | **V1.3** manual-only recorder with calibrated MSS capture, per-run folders, candidate/approved segments, action timestamps, and actor-only BC branch |

The YOLO worker is **already asynchronous**; switching to an LSTM, compressing the 557-D auxiliary vector, changing rewards, or changing keypress durations should not be mixed into the initial demonstration experiment. Image-column locations are not ground-truth track lanes. Local practice only.

## Repository map

| Location | Purpose |
| --- | --- |
| `train_rs_v612_ppo.py` | PPO offline check, input probe, training and deterministic/stochastic evaluations |
| `rs_v612_gameplay_guard.py` | Fail-closed visual gate against clicking/acting in menus or uncertain states |
| `rs_v61_features.py`, `rs_v6_ppo_features.py`, `rs_v61_rewards.py` | V6.1 observation schema and reward computation |
| `rs_v5_segmentation_live_benchmark.py`, `rs_v5_shadow_segmentation.py` | Native game crop, detector and historical benchmark dependencies |
| `train_rs_v3_ppo.py`, `rs_practice_controller.py`, `rs_v4_screen_states.py` | **Required** inherited capture/controller/extractor/classifier code; not safe to delete merely because names are older |
| `rs_v612_demo.py` | **V1.3** manually played demonstrations: capture, save, review, segment approval |
| `rs_v612_demo_bc.py` | Clone approved demonstrations into **new** PPO model checkpoint; original PPO untouched |
| `test_rs_v612_demo.py`, `test_rs_v612_guard.py`, `test_rs_v61.py` | Offline demonstration, guard and observation tests |
| `README_DEMONSTRATIONS_V612.md` | Full V1.3 manual-play, save, preview, trim, troubleshooting and BC instructions |
| `server.py` | Local Unity server with required WebGL content-encoding and cross-origin headers |
| `checkpoints/` | Selected older backed-up weights via Git LFS; **not** necessarily latest local checkpoint |
| `reports/` | Selected committed benchmark reports; newest 4.61 s report is **local only** until explicitly added |
| `ai_data/rl_v2_runs/` | Local PPO checkpoints, logs and evaluation reports (Git-ignored) |
| `ai_data/demonstrations_v13/` | **Recommended current** local replay root; review/BC data **must not be blindly staged** |
| `rankedsurfers-downloaded/` | Unity WebGL files served locally; respect redistribution rights |
| `development/archive/`, `development/local/` | Historical/reference material and ignored local diagnostics |

`ppo_direct_observation_probe.py` and a locally modified `train_rs_v612_ppo.py` were reported present in the **original PC project**, but were **not pushed** with the recorder debug branch. Do not assume these are in a fresh clone. The tracked PPO trainer remains suitable for evaluating compatible V6.1.2 models.

## Windows PowerShell — setup and launch

Use the project directory that contains `server.py`. For example, a fresh clone may be in `Documents\ranked-surfers-ai-new` rather than the original `Documents\ranked-surfers-ai`; choose the appropriate path.

```powershell
cd "$HOME\Documents\ranked-surfers-ai"

# Restore tracked selected model assets (requires Git LFS installed).
git lfs install
git lfs pull

# Verify this computer has the required Python environment.
.\.venv\Scripts\python.exe -c "import torch, stable_baselines3, ultralytics, playwright, mss; print('CUDA:',torch.cuda.is_available())"
```

**Fresh-clone limitation:** Git does not include `.venv`, most `ai_data` assets, or the latest 4.61 s checkpoint. Install/restore the compatible Python dependencies and transfer required local files from the original PC project before attempting training or demonstrations. At minimum, the commands below refer to these **local assets**:

```text
ai_data/v4_screen_states_rebuild_01/screen_state_candidate_v2.pt
ai_data/v5_hazards/runs/segment_round04/weights/best.pt
ai_data/rl_v2_runs/v612_overnight_next_420min_press5_20261027_094116/latest.zip
```

The first two are the current classifier/detector paths used by the recorder; the third is necessary to reproduce the latest PPO model or fine-tune it. These paths may need to be copied from your original installation.

### Terminal 1 — serve the local Practice game

```powershell
cd "$HOME\Documents\ranked-surfers-ai"
.\.venv\Scripts\python.exe server.py
```

Leave the terminal open. Server URL: `http://localhost:8000`. Use `server.py`, **not** a generic web server, because WebGL assets require special headers.

### Offline tests (no game input)

```powershell
cd "$HOME\Documents\ranked-surfers-ai"

.\.venv\Scripts\python.exe -m unittest test_rs_v612_guard test_rs_v61 -v
.\.venv\Scripts\python.exe -m unittest test_rs_v612_demo -v
.\.venv\Scripts\python.exe train_rs_v612_ppo.py --mode check --resolution 160x96
```

The user confirmed **12/12 demonstration tests pass** after installing V1.3 on Windows (27 October). These are *offline unit tests*, not proof of real-time capture speed or improved PPO performance.

## Human demonstrations V1.3 — recommended next experiment

**Important:** Stop PPO training/evaluation before running the human recorder. The recorder launches its own Chromium instance and **never injects movement or restart keys**.

### Terminal 2 — record new runs separately inside the same repo

```powershell
cd "$HOME\Documents\ranked-surfers-ai"

.\.venv\Scripts\python.exe -u rs_v612_demo.py record `
  --root "ai_data\demonstrations_v13" `
  --fps 20 `
  --detect-every 3 `
  --imgsz 960
```

Let Chromium load, select fullscreen **before playing**, keep the canvas unobstructed, and wait for `[CALIBRATION]` to complete. Click **Practice** manually and use arrows or WASD/Space. Restart the game manually with **R**. After a good run or a confirmed death, **Alt+Tab** to the recorder's PowerShell window. Type one command followed by Enter:

| Terminal key | Effect |
| --- | --- |
| **S** | Save latest buffered run as **candidate** (not used by BC); **recommended for the first tests** |
| **P** | Save as candidate and open the full-canvas preview |
| **G** | Save + approve, excluding the last 0.7 s by default (**not** a guarantee mistakes are excluded) |
| **T** | Save + approve manually selected intervals, e.g. `0.2-4.8,5.2-8.6` |
| **R** | Discard the buffered unsaved run (**terminal R** differs from manually restarting the game) |
| **Q** | Quit the recorder |

Each saved replay goes to `ai_data/demonstrations_v13/runs/demo_YYYYmmdd_HHMMSS_micros/` with preview, `manifest.json`, raw RGB/aux/actions/timestamps, validity mask and raw key events. `frames.npz` stores unique 512×288 model images; the **preview video** preserves full-canvas framing. **Reviewing a saved replay changes only approval intervals in its manifest**—not the raw frames or actions.

**First validate V1.3 on the actual machine:**

```powershell
# View the unmodified capture region used by the recorder.
explorer "ai_data\demonstrations_v13\capture_preview_full_canvas.png"

# Find the newest saved replay and inspect its event-label health.
$run = Get-ChildItem "ai_data\demonstrations_v13\runs" -Directory |
  Sort-Object LastWriteTime -Descending | Select-Object -First 1
Get-Content "$($run.FullName)\manifest.json" -Raw | ConvertFrom-Json |
  Select-Object status,duration_seconds,recorded_keydown_count,valid_action_samples,action_counts_valid | Format-List
Invoke-Item "$($run.FullName)\preview.mp4"
```

Check the terminal's measured `[CAPTURE] ... FPS`, framing, actual keypress counts and valid action labels. If the preview is cropped wrongly, gameplay lags, or action validity is zero, keep the replay a **candidate** and fix capture before training. V1.2's last uploaded laggy trial recorded **51 frames in 14.07 s (~3.55 FPS; median interval 272 ms)**, producing **zero valid training samples**; V1.3 was developed to address that and has not yet received a confirmed real-world FPS measurement.

### Review and trim **after** saving with G/S/P

```powershell
# $run is the folder located above.
.\.venv\Scripts\python.exe rs_v612_demo.py review `
  "$($run.FullName)" --status approved --segments "0.2-5.5"

# Or keep it excluded from training until corrected:
.\.venv\Scripts\python.exe rs_v612_demo.py review `
  "$($run.FullName)" --status candidate
```

You may keep multiple nonoverlapping intervals, such as `"0.2-3.5,4.2-7.0"`. **Never assume surviving parts were all good**: remove the actual wrong move(s), potentially well before death. Candidate/rejected runs are excluded from BC.

For the detailed workflow and troubleshooting, see **[README_DEMONSTRATIONS_V612.md](README_DEMONSTRATIONS_V612.md)**.

## Train a separate behavioral-cloning copy — only after verified demos

Collect **at least three varied, genuinely good approved runs** and substantially more for a meaningful lane experiment, including real correct LEFT and RIGHT actions. The BC loader splits by **whole runs** for holdout; keeps non-NONE actions and subsamples NONE; freezes the visual/aux extractor and critic for the initial actor-focused test; and creates a **new** checkpoint, preserving the 4.61 s source.

```powershell
cd "$HOME\Documents\ranked-surfers-ai"
$best = "ai_data\rl_v2_runs\v612_overnight_next_420min_press5_20261027_094116\latest.zip"
if (-not (Test-Path $best)) { throw "Latest local PPO checkpoint is missing: $best" }

.\.venv\Scripts\python.exe -u rs_v612_demo_bc.py `
  --root "ai_data\demonstrations_v13" `
  --checkpoint $best `
  --epochs 3 `
  --batch-size 12 `
  --lr 0.00002
```

New BC outputs appear under `ai_data/demonstrations_v13/bc_runs/bc_*/model.zip`. **The BC training script does not evaluate the Unity game.** It reports holdout imitation loss; you must separately evaluate gameplay. Never overwrite the original PPO checkpoint.

## Benchmark or evaluate a checkpoint: 100 deterministic episodes

Use the **same** 5 ms input and YOLO/classifier settings for baseline and BC comparison. Keep the server running; no recorder or other PPO process running at the same time.

```powershell
cd "$HOME\Documents\ranked-surfers-ai"
$state = "ai_data\v4_screen_states_rebuild_01\screen_state_candidate_v2.pt"
$detector = "ai_data\v5_hazards\runs\segment_round04\weights\best.pt"

# Set this to the latest PPO source, or to a new BC model.zip to compare.
$checkpoint = "ai_data\rl_v2_runs\v612_overnight_next_420min_press5_20261027_094116\latest.zip"
foreach ($p in @($state,$detector,$checkpoint)) { if (-not (Test-Path $p)) { throw "Missing: $p" } }

$evalName = "v612_eval_100_" + (Get-Date -Format "yyyyMMdd_HHmmss")
.\.venv\Scripts\python.exe -u train_rs_v612_ppo.py `
  --mode eval --run-name $evalName --resume $checkpoint `
  --eval-policy deterministic --eval-episodes 100 --max-minutes 45 `
  --resolution 512x288 --fps 30 --rate-mode uncapped `
  --press-delay-ms 5 --capture-mode fast --resize-method linear `
  --state-model $state --state-confidence 0.90 `
  --detector $detector --imgsz 960 --conf 0.20 --detect-every 3
```

The evaluator writes a report to `ai_data/rl_v2_runs/<evalName>/v61_eval_report.json`. Read and verify **100 completed episodes** before comparing; a time limit, interrupted run or safety abort may yield fewer.

```powershell
$r = Get-Content "ai_data\rl_v2_runs\$evalName\v61_eval_report.json" -Raw | ConvertFrom-Json
$d = @($r.durations_seconds)
[PSCustomObject]@{
  Episodes = $r.eval_episodes
  MeanSeconds = $r.mean_survival_seconds
  MedianSeconds = $r.median_survival_seconds
  BestSeconds = $r.best_survival_seconds
  Survived5s = @($d | Where-Object { $_ -ge 5 }).Count
  Survived10s = @($d | Where-Object { $_ -ge 10 }).Count
  ActionCounts = ($r.action_counts | ConvertTo-Json -Compress)
  GuardAborts = $r.detector.v612_live_input_guard.uncertain_state_abort
} | Format-List
```

The **current known reference** is **4.61 s mean**, **4.21 s median**, **44/100 at least 5 s**, **2/100 at least 10 s**, and **no LEFT/RIGHT selected**. Successful BC should make lateral decisions available **without ruining survival**; thereafter consider short PPO continuation. No 100-episode evaluation of a BC model has been reported yet.

## Continue PPO only after a successful BC evaluation

A separate output name preserves the BC source. Example **60-minute experimental** continuation; do not run concurrently with recording or evaluation:

```powershell
$bc = "ai_data\demonstrations_v13\bc_runs\REPLACE_WITH_ACTUAL_BC_RUN\model.zip"
if (-not (Test-Path $bc)) { throw "BC checkpoint not found: $bc" }
$runName = "v612_post_demo_ppo_" + (Get-Date -Format "yyyyMMdd_HHmmss")

.\.venv\Scripts\python.exe -u train_rs_v612_ppo.py `
  --mode train --run-name $runName --resume $bc `
  --steps 1048576 --max-minutes 60 `
  --resolution 512x288 --fps 30 --rate-mode uncapped `
  --n-steps 128 --batch-size 32 --n-epochs 5 `
  --press-delay-ms 5 --capture-mode fast --resize-method linear `
  --state-model $state --state-confidence 0.90 `
  --detector $detector --imgsz 960 --conf 0.20 --detect-every 3 `
  --save-every-minutes 4
```

If a PPO checkpoint is resumed, its stored rollout/batch hyperparameters can override CLI suggestions; inspect the emitted `[RESUME]` message. Guard aborts or uncertain state must be investigated, not bypassed. The new checkpoint lands in `ai_data/rl_v2_runs/<runName>/latest.zip`.

## Other diagnostic/benchmark tools

- `train_rs_v612_ppo.py --mode check` — offline network/observation/serialization verification (run before changing model code).
- `train_rs_v612_ppo.py --mode probe` — **live local-Practice input pulse audit**; may send safe local game inputs. Check the flag help before running: `python train_rs_v612_ppo.py --help`.
- `rs_v5_segmentation_live_benchmark.py` — historic segmentation throughput/quality and preview benchmark. A fixed-JUMP test intentionally **does not evaluate a PPO policy**; do not report those trials as learned policy performance.
- `rs_v2_timing_probe_ab.py` — historical capture/input timing comparison; inspect its help for arguments.
- `ppo_direct_observation_probe.py` — separate **local-only** diagnostic script reported on the original machine, **not** part of this GitHub branch.
- `reports/v612_learning_proof_20261026_153745/` — older published 100-episode experimental evidence; do not present it as the latest seven-hour evaluation.

For a controlled future optimization: benchmark YOLO `--imgsz` 960 vs 768 vs 640 with matching saved frame quality and detector freshness, or test PPO `target_kl`/learning rate **one independent change at a time**. None of those changes is included in V1.3.

## Git, safe backups and second clone

**Known Git issue:** Existing `.gitignore` protects many older `ai_data` paths, but new `ai_data/demonstrations_v13/` folders may be **untracked and not ignored**. Never run `git add .`, `git add -A` or force-add entire `ai_data/`. Raw captured screen data can be large and may contain private pixels. Stage source, docs, and selected small evidence files explicitly:

```powershell
# Run inside the clone in which you want to commit documentation.
git status
git add README.md README_DEMONSTRATIONS_V612.md
git diff --cached --stat
git commit -m "Document latest PPO 4.61s result and V1.3 demonstrations"
git push origin demo-recorder-debug-20261027
```

If using the **new** `ranked-surfers-ai-new` clone, update only that folder:

```powershell
cd "$HOME\Documents\ranked-surfers-ai-new"
git switch demo-recorder-debug-20261027
git pull --ff-only
```

Preserve the newest `ai_data/rl_v2_runs/.../latest.zip` separately (ideally with a hash/backup). If you decide to publish a chosen new milestone, copy **only** its checkpoint to `checkpoints/` (Git LFS) and corresponding **actual**, unmodified JSON report to `reports/`. Never fabricate a benchmark JSON or assert a live test passed when it has not.

## Safety, limitations and next steps

1. Verify the **V1.3 live Windows FPS**, stable fullscreen/windowed capture, input lag, preview crop, and keypress-to-observation alignment with **candidate** recordings. V1.3's unit tests do not establish any of these empirically.
2. Curate genuinely good lateral examples; keep raw replays, exclude death and pre-death mistakes via explicit segments; train only approved runs.
3. Actor-only BC on a **copy** of the 4.61-second PPO checkpoint; evaluate both with matched **100 deterministic local-Practice episodes**. Keep the better model; don't overwrite either checkpoint.
4. Only after that result, consider further PPO or more complex imitation (DAgger-style corrections, decaying demonstration loss). Keep reward changes, LSTM, YOLO resize, and action timing separate from this experiment.

The code samples assume you have the required local model assets, working `.venv`, Unity game files and a compatible Windows display. A Python test passing or a ZIP being pushed **does not** prove real-time capture quality or better survival.

Whole-canvas migration: see [README_PPO_CANVAS_V1.md](README_PPO_CANVAS_V1.md). The explicit --framing-profile canvas_v1 option is shared by PPO training/evaluation, shared headless workers, new BC recording and BC training selection. It uses the calibrated whole canvas and retains separate raw safety inputs; detector weights and existing checkpoints/demos are preserved. Legacy framing remains the default for reproduction.

The shared-worker preview (--preview-port 8001) now provides a compact six-tile grid and a persistent rolling-mean episode-duration chart with a fixed requested-runtime x-axis. See README_PPO_BACKGROUND.md for metric definitions and reopening saved dashboards using tools.ppo_worker_preview --report PATH --port 8001.

Thirty-minute headless diagnostics can retain five-minute snapshots via --save-every-minutes 5. The dashboard now shows connected red graph points, PPO update statistics and sampled system performance using the existing telemetry stream. Preserve the run directory (including config.json, progress.csv, worker ledgers, survival.json, system_usage.jsonl and checkpoint snapshots) for post-run analysis.

Reaction-speed status: the dashboard now supports Hide previews for metrics-only viewing, and workers skip preview JPEG encoding while hidden. New timing diagnostics separate policy-image age from safety-frame age. Six-worker screenshot-request-to-command p95 remained about 190ms without preview images; optimize capture/decision alignment before assuming higher throughput produces fast late-game reactions. See reports/REACTION_LATENCY_20261030.md.
