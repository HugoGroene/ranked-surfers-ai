# Ranked Surfers AI — Project Status, Development History, and Research Roadmap


> **Status snapshot: 27 October 2026** | Active development branch: `demo-recorder-debug-20261027` | Last confirmed pre-documentation commit: `e0b1cf9` | RL: V6.1.2 PPO | Human demonstration recorder: V1.3
>
> This document is a developer/AI-agent handoff. It describes observed results and open questions; it is not a claim that the latest local checkpoint or recordings have been pushed to GitHub. See [README.md](README.md) and [README_DEMONSTRATIONS_V612.md](README_DEMONSTRATIONS_V612.md) for the canonical day-to-day commands.

**Repository:** https://github.com/HugoGroene/ranked-surfers-ai  
**Game:** https://rankedsurfers.com/  
**Currently supported environment:** locally served Unity WebGL Practice copy at `http://localhost:8000`.

### Long-term ambition and important boundary

The ultimate stated ambition is to build an **AI gameplay bot (an “AI cheat”) for Ranked Surfers**, able to observe the game, understand approaching hazards and execute all five possible movements autonomously. The current implementation does **not** establish functionality or permission for automating the public ranked service: all documented training and testing is on the **local Practice copy**. Site rules may prohibit automation or cheating; review the current site terms and obtain authorization for any online experiments. Do not confuse a local PPO survival benchmark with competitive online capability. This document does not provide evasion or anti-cheat bypass instructions.

## Executive summary

The repository is now at a fairly clear transition point: **the main problem is no longer “does PPO learn anything?”** The committed evidence shows that PPO did learn useful survival behavior, and the newer local continuation appears to have improved it further. The dominant technical failure is now **action collapse**: the policy has learned a strategy dominated by `JUMP`, `ROLL`, and `NONE`, while effectively abandoning `LEFT` and `RIGHT`. The newest documented 100-episode evaluation reports zero deterministic lateral actions despite nonzero lateral action probabilities. 

The repository’s latest pushed branch head is now **`55865ae`**, committed on 27 October 2026 with the message `Update latest PPO benchmarks and V1.3 documentation`; it follows the V1.3 recorder commit `943909d` and the initial demonstration-recorder commit `f39f2b9`. The branch history also documents the earlier V4 controller, V5.4 segmentation pipeline, V6.1.2 guard, PPO-learning proof, and workflow organization. 

The important strategic conclusion is therefore:

> **Do not immediately spend another night simply continuing the same PPO checkpoint. First establish that V1.3 can record high-quality human LEFT/RIGHT decisions at real gameplay speed, then test actor-only behavioral cloning against the untouched PPO baseline.**

That recommendation follows directly from the current failure mode. The latest documented model is not failing because it never acts; it is acting extensively, but its action distribution has collapsed around non-lateral behavior. The existing BC code is already structured as a conservative experiment: approved runs only, run-level validation holdout, preservation of non-NONE examples, required LEFT and RIGHT samples, actor-only fine-tuning, and a new output checkpoint rather than overwriting the PPO source. 

There is also an important scope issue. The **technical end goal you described is indeed an AI gameplay bot—an “AI cheat” in normal competitive-gaming terminology—for Ranked Surfers**. However, Public game rules and terms must be checked before any online use. Accordingly, the research README I created states the true technical ambition while drawing a clear line: development and testing should remain on the local Practice copy or another explicitly authorized environment unless the operator gives permission for online testing. 

## Where the project actually stands

The current active RL system is **PPO V6.1.2**, while the human-data experiment is **Demonstrations V1.3**. The current README describes a Stable-Baselines3 `PPO("MultiInputPolicy")` policy with five actions—`NONE`, `LEFT`, `RIGHT`, `JUMP`, and `ROLL`—using a four-frame RGB history together with a 557-dimensional auxiliary observation. 

The PPO implementation confirms that the observation is a `Dict` containing a `12×H×W` stacked image and the V6.1 auxiliary vector; the custom extractor combines visual and auxiliary features before separate policy/value heads. It also confirms that the YOLO detector is scheduled asynchronously rather than synchronously blocking every policy decision.  Stable-Baselines3’s current PPO implementation is the clipped PPO algorithm derived from the original PPO method; PPO remains a reasonable baseline here because it offers a relatively simple on-policy training loop and is widely used for pixel/control environments. 

The system can be summarized as:

```text
local Unity WebGL Practice game
             │
             ▼
      capture game pixels
             │
       ┌─────┴─────────────┐
       │                   │
       ▼                   ▼
screen/gameplay        native game crop
safety guard                │
                            ├──── async YOLO segmentation
                            │
                            ▼
                   4 RGB frames + 557 aux
                            │
                            ▼
                   PPO MultiInputPolicy
                            │
                            ▼
             NONE / LEFT / RIGHT /
                  JUMP / ROLL
                            │
                            ▼
                 guarded browser input
                            │
                            ▼
                  next observation
```

The reward design is intentionally not a hand-coded obstacle solver. The current environment rewards real elapsed survival, milestone progress, and penalizes confirmed death; the code explicitly avoids rewards based on coin detection, detector confidence, inferred lanes, or particular actions.  That is a good scientific property because it means an improvement cannot simply be attributed to encoding the desired obstacle response directly into the reward.

The perception layer uses nine segmentation categories: `coin`, `train_front`, `train_side`, `train_roof`, `ramp`, `barrier_jump`, `barrier_roll`, `barrier_either`, and `tunnel_entrance`. The committed evaluation schema records four RGB frames, 12 image channels, 467 V6 base features plus screen-column/action-context additions for **557 auxiliary dimensions**. 

### Benchmark state

The most useful distinction is between **committed evidence** and **new local evidence**.

| Policy/reference | Episodes | Mean | Median | ≥5 s | ≥10 s | Evidence status |
|---|---:|---:|---:|---:|---:|---|
| Earlier PPO | 100 | 3.11 s | 3.35 s | 0 | — | Earlier controlled comparison.  |
| No-action control | 100 | 3.48 s | 3.37 s | 12 | — | Earlier control.  |
| 120-minute PPO | 100 | **3.84 s** | **4.16 s** | 15 | — | Raw report committed.   |
| Latest seven-hour PPO | 100 | **4.61 s** | **4.21 s** | **44** | **2** | Documented from local result; newest raw report/checkpoint not committed.  |

The **3.84-second result is the strongest benchmark currently backed by a committed raw report**. That report records 6,678 steps in 424.05 seconds, about 15.75 steps/s including resets; gameplay-only intervals had a median of 49.04 ms and p95 of 64.94 ms. The detector completed 2,263 requests, its reported fresh fraction was 0.971, raw model-prediction time had a median of about 13.48 ms, and detection age at the decision point was much larger—about 99.3 ms median and 175.4 ms p95. 

That last comparison is particularly important:

```text
YOLO model inference median:        ~13.5 ms
detection age when PPO uses it:     ~99.3 ms
detection-age p95:                 ~175.4 ms
```

So **raw GPU inference speed is not the same thing as information freshness**. The next optimization work should profile the complete pipeline—capture, scheduling, inference, polling, observation construction, PPO inference, guard checks, and key dispatch—rather than simply trying to reduce YOLO’s `predict()` time. The current asynchronous implementation is already a sensible architecture for avoiding a blocking detector call every action. 

The committed 3.84-second policy was already displaying lateral collapse: deterministic action counts contained only `NONE`, `JUMP`, and `ROLL`, while its mean policy probabilities were approximately 33.18% NONE, 0.16% LEFT, 0.54% RIGHT, 40.43% JUMP, and 25.69% ROLL.  

The newer local 4.61-second model reportedly improved survival considerably—44% of its evaluation episodes reached five seconds—but still selected **zero LEFT and zero RIGHT deterministic actions**. Its recorded counts were `NONE=1978`, `JUMP=2779`, `ROLL=3169`, `LEFT=0`, `RIGHT=0`; LEFT and RIGHT together represented only 0.71% of mean predicted action probability. 

That is strong evidence that **lane behavior is the next bottleneck**.

## How the project evolved and what was learned

The Git history shows a coherent progression rather than a collection of unrelated scripts. The project began with the V4 migration on 20 October, restored required V3 controller dependencies, removed Linux-specific components for Windows compatibility, then checkpointed the V4 screen-state controller and approximately 31 Hz pipeline. V5.4 followed on 21 October with segmentation and another 31 Hz pipeline checkpoint. On 26 October, the project strengthened the V6.1.2 guard, stored a trained PPO checkpoint, recorded the learning-improvement evidence, and reorganized/documented the active pipeline. On 27 October the human demonstration workflow was added and then upgraded to V1.3 fast capture. 

That history explains why some older-named modules remain important. V6.1.2 deliberately inherits proven capture/controller/extractor components rather than duplicating them. The current README explicitly warns that older V3/V4/V5 files are still runtime dependencies and should not be deleted merely because their names predate V6.1.2. 

The demonstration work also produced useful negative results.

Early recording exposed a severe labeling problem: video could exist while physical-action labels were absent. V1.2 improved capture/input handling, but one real replay measured only **51 frames over 14.07 seconds—about 3.55 FPS—with a 272 ms median frame interval and zero valid training samples**, despite physical keydowns being present. 

V1.3 directly addresses that failure by changing the capture architecture. Its code takes a browser screenshot for calibration, locates that canvas in the physical desktop image, and then performs repeated capture through MSS rather than repeatedly asking Chromium to compose a screenshot. The calibration fails closed when its pixel match is too weak. 

The physical-input logic is also now much better designed. On Windows the recorder polls physical key state using `GetAsyncKeyState`, records arrows plus WASD/Space aliases, prefers Win32 timestamps over delayed browser callback timestamps when both exist, and deduplicates equivalent events. 

Most importantly, labels are causal. The recorder assigns a keypress to the observation interval **after** that observation and before the next frame; samples with long intervals or ambiguous multiple actions are invalidated rather than guessed. The code also updates action history only after constructing the current observation, avoiding leakage of the answer into the features. 

The V1.3 recorder passed your **12 offline tests**, but the project documentation correctly distinguishes that from real-time validation: real Windows FPS, latency, and valid-label yield still need a new live candidate replay. 

## Research-backed improvements

The literature supports the direction you are now taking, but it also suggests several refinements.

### Make evaluation more rigorous before chasing small gains

Stable-Baselines3’s own guidance emphasizes validating custom environments and carefully evaluating RL algorithms rather than assuming a single training outcome is representative.  More importantly, the RLiable work shows why point estimates can be misleading in deep RL and recommends interval estimates and score distributions rather than relying solely on means or medians from limited runs. 

For this project, every serious evaluation should therefore start storing:

```text
all episode survival times
mean
median
p25 / p75
best
>= 5 s fraction
>= 10 s fraction
action counts
mean action probabilities
bootstrap 95% CI for mean survival
bootstrap 95% CI for threshold rates
guard aborts
detector freshness
frame timing
exact checkpoint SHA-256
Git commit
exact CLI command
package versions
GPU / driver
display resolution and Windows scaling
```

The current 100-episode tests are much better than judging one spectacular episode, but **100 episodes from one trained checkpoint are still not independent training seeds**. The 4.61-second model should therefore be described as the current best observed checkpoint rather than as a statistically settled improvement until repeated evaluation or independent training runs support it. This interpretation follows the evaluation concerns identified by RLiable. 

### Human demonstrations are a good targeted intervention

The BC implementation is appropriately conservative. It splits validation by entire run, which prevents near-identical neighboring frames from the same episode leaking between train and validation data; it preserves real action examples, reduces some NONE dominance, requires both LEFT and RIGHT, and only changes the actor on the first experiment. 

The important experiment is therefore not “train BC until imitation accuracy looks good.” It is:

```text
PPO baseline
      │
      ├────────────── 100-episode matched evaluation
      │
      ▼
approved human demonstrations
      │
      ▼
actor-only BC
      │
      ▼
new independent checkpoint
      │
      ├────────────── same 100-episode evaluation
      │
      ▼
optional PPO continuation
```

Holdout imitation loss is only an intermediate diagnostic; the true objective is improved game survival and better lane-choice behavior. Your BC code itself already acknowledges that distinction by telling the user to evaluate the produced model with the normal deterministic 100-episode environment afterward. 

### DAgger is probably the next imitation-learning idea after BC

Plain behavioral cloning has a fundamental sequential-learning problem: the training set contains states visited by the human, but once the learned policy makes a mistake, it can visit a different state distribution. Ross, Gordon, and Bagnell’s DAgger paper explicitly identifies this issue: in sequential imitation learning, future observations depend on earlier actions, breaking the ordinary independent-and-identically-distributed assumption. Their approach iteratively gathers labels on states induced by the learner. 

That maps unusually well onto your game. A future **local Practice** workflow could be:

```text
agent plays
   ↓
record the states where it becomes uncertain or fails
   ↓
human supplies corrective action
   ↓
aggregate those states with approved demonstrations
   ↓
retrain a new policy copy
   ↓
repeat
```

HG-DAgger is another relevant research direction because it studies interactive imitation learning with human experts, making it a useful conceptual source for a future intervention-style recorder rather than continuous manual annotation. 

This should come **after** clean static BC. Doing DAgger before V1.3’s timing is trustworthy would simply aggregate more mislabeled data.

### Validate the detector scientifically

The YOLO component currently produces useful structured features, but the project appears much stronger on detector latency diagnostics than on committed held-out detector-accuracy metrics. Ultralytics exposes segmentation validation metrics including mask mAP, box mAP, and per-image precision/recall/F1 information. 

A proper detector benchmark should therefore record per-class values for the nine gameplay classes. This is especially important for classes whose misidentification changes the correct response, such as `barrier_jump`, `barrier_roll`, and trains.

Then optimize:

```text
imgsz 960
vs.
smaller validated resolutions

detect-every 3
vs.
nearby frequencies
```

based on both accuracy **and decision-age latency**. A configuration that saves 4 ms of inference but increases false negatives or makes detections older at the policy decision can make the agent worse.

### Do not jump to LSTM yet

A recurrent policy is a legitimate future idea because obstacle approach and motion are temporally dependent. SB3 Contrib supports recurrent PPO and explicitly carries LSTM state plus episode-start signals through predictions. 

But the order matters. The current project already has four-frame stacking, explicit action history, and timing features. Changing to recurrent PPO while simultaneously introducing human demonstrations would make it nearly impossible to determine what caused an improvement or regression.

The clean sequence is:

```text
current PPO + four-frame stack
           ↓
verify V1.3 capture
           ↓
actor-only BC
           ↓
matched gameplay benchmark
           ↓
DAgger/corrective data if useful
           ↓
detector/timing ablations
           ↓
then consider RecurrentPPO/LSTM
```

That is also consistent with the general RL engineering principle of establishing a working, measurable baseline before adding architectural complexity. 



## Operational file map

| File/folder | Role | Version-control state |
|---|---|---|
| `train_rs_v612_ppo.py` | Modes `check`, `probe`, `train`, `eval`; PPO policy and real-time controller | Tracked; local machine also had **uncommitted modifications** at last `git status` |
| `rs_v612_gameplay_guard.py` | Fail-closed, screen-state-gated browser input | Tracked |
| `rs_v61_features.py`, `rs_v6_ppo_features.py`, `rs_v61_rewards.py` | V6 feature encoding and survival rewards | Tracked |
| `rs_v5_segmentation_live_benchmark.py`, `rs_v5_shadow_segmentation.py` | Detector/capture dependencies, historical benchmarks | Tracked |
| `rs_practice_controller.py`, `train_rs_v3_ppo.py`, `rs_v4_screen_states.py` | Inherited controller, feature extractor, and classifier code | Tracked; **not safe to remove** |
| `rs_v612_demo.py` | V1.3 Windows manual game recorder, keyboard events and previews | Tracked on debug branch |
| `rs_v612_demo_bc.py` | Actor-only BC fine-tuning on approved demonstrations | Tracked on debug branch |
| `test_rs_v612_demo.py` | Recorder offline regression tests | Tracked on debug branch |
| `server.py` | Local Unity asset server and required WebGL headers | Tracked |
| `reports/v612_learning_proof_20261026_153745/` | Raw earlier 3.84-second benchmark evidence | Tracked |
| `checkpoints/` | Selected milestone models | Git LFS where configured |
| `ai_data/rl_v2_runs/` | Working training/evaluation runs, including 4.61-second checkpoint | **Local-only / ignored** |
| `ai_data/demonstrations_v13/` | Human replay data and BC outputs, with unique per-run folders | **Local-only; may not be ignored** |
| `ppo_direct_observation_probe.py` | Standalone local diagnostic probe | **Not pushed** at last local status |
| `rankedsurfers-downloaded/` | Local game assets for Practice server | Availability and redistribution rights must be verified |

### Fresh Windows clone setup

The GitHub clone does **not** automatically include `.venv`, private/local game assets, the newest local checkpoint or large recordings. Check these before running anything. Use PowerShell and adapt the folder path:

```powershell
cd "$HOME\Documents\ranked-surfers-ai"
git switch demo-recorder-debug-20261027
git lfs install
git lfs pull
.\.venv\Scripts\python.exe -c "import torch, stable_baselines3, ultralytics, playwright, mss; print('CUDA:',torch.cuda.is_available())"
```

In terminal 1, run the local server:

```powershell
cd "$HOME\Documents\ranked-surfers-ai"
.\.venv\Scripts\python.exe server.py
```

In terminal 2, verify offline wiring (requires installed dependencies and model files):

```powershell
cd "$HOME\Documents\ranked-surfers-ai"
.\.venv\Scripts\python.exe -m unittest test_rs_v612_demo -v
.\.venv\Scripts\python.exe -m unittest test_rs_v612_guard test_rs_v61 -q
.\.venv\Scripts\python.exe train_rs_v612_ppo.py --mode check --resolution 160x96
```

### Collect a human demonstration (V1.3)

This captures **your** manual input; it does not play or restart the game for you. Close concurrent PPO processes. Keep Chromium visible and unobstructed, use Practice, restart manually, and save the first run as **candidate** (`S`) before inspecting. Review the full-canvas video and physical LEFT/RIGHT labels. The requested 20 FPS is a target, **not a live measurement**.

```powershell
cd "$HOME\Documents\ranked-surfers-ai"
.\.venv\Scripts\python.exe -u rs_v612_demo.py record `
  --root "ai_data\demonstrations_v13" `
  --fps 20 --detect-every 3 --imgsz 960
```

Terminal review keys: `S` save candidate; `G` approve if valid; `T` approve exact good intervals; `P` candidate plus preview; `R` discard unsaved replay; `Q` quit. Broken timing/framing, zero valid actions or misaligned labels mean **no approval**.

For a later manual review, replace the directory and intervals:

```powershell
.\.venv\Scripts\python.exe rs_v612_demo.py review `
  "ai_data\demonstrations_v13\runs\YOUR_DEMO_FOLDER" `
  --status approved --segments "0.2-4.2,4.8-8.3"
```

### Actor-only behavioral cloning

Only after several varied **approved** runs contain actual valid LEFT and RIGHT decisions, train a *separate* model. This does not overwrite source PPO weights:

```powershell
$best = "ai_data\rl_v2_runs\v612_overnight_next_420min_press5_20261027_094116\latest.zip"
if (-not (Test-Path $best)) { throw "Local PPO checkpoint missing" }
.\.venv\Scripts\python.exe -u rs_v612_demo_bc.py `
  --root "ai_data\demonstrations_v13" `
  --checkpoint $best --epochs 3 --batch-size 12 --lr 0.00002
```

Output: `ai_data/demonstrations_v13/bc_runs/bc_*/model.zip`. BC validation loss is **not** a game benchmark. Run controlled Unity evaluation before declaring success.

### Deterministic 100-episode evaluation

Pass the **actual** compatible `latest.zip` or BC `model.zip`; preserve all arguments between comparisons:

```powershell
$checkpoint = "PATH_TO_REAL_CHECKPOINT.zip"
$evalName = "v612_comparison_" + (Get-Date -Format "yyyyMMdd_HHmmss")
$candidate = "ai_data\v4_screen_states_rebuild_01\screen_state_candidate_v2.pt"
$detector = "ai_data\v5_hazards\runs\segment_round04\weights\best.pt"
foreach ($file in @($checkpoint,$candidate,$detector)) {
  if (-not (Test-Path $file)) { throw "Missing: $file" }
}
.\.venv\Scripts\python.exe -u train_rs_v612_ppo.py `
  --mode eval --run-name $evalName --resume $checkpoint `
  --eval-policy deterministic --eval-episodes 100 --max-minutes 35 `
  --resolution 512x288 --fps 30 --rate-mode uncapped `
  --press-delay-ms 5 --capture-mode fast --resize-method linear `
  --state-model $candidate --state-confidence 0.90 `
  --detector $detector --imgsz 960 --conf 0.20 --detect-every 3
```

The evaluation report is `ai_data/rl_v2_runs/<evalName>/v61_eval_report.json`. Compare means, medians, five-/ten-second threshold counts, action histograms, guard stops and detector age; do **not** compare a stochastic policy run against a deterministic baseline without labeling the difference.

## Highest-priority unanswered questions

1. **Capture correctness:** Can V1.3 maintain smooth Windows gameplay and produce large numbers of *valid* labels at the requested rate? Do labels agree with the pre-action frames, the keys pressed and the physical movement on screen?
2. **Action collapse:** Are lateral actions disfavored because PPO exploration is inadequate, because action dispatch is unreliable, or because the policy's current observations do not predict when lane changes are safe? Distinguish hypotheses using controlled probes.
3. **BC improvement:** Does the separate BC checkpoint use LEFT/RIGHT in live deterministic evaluation, and does survival beat or at least match the untouched 4.61-second reference across repeated comparisons?
4. **Perception quality:** What are per-class held-out precision/recall and mask mAP for the nine classes? How do these metrics vary with `imgsz` 960/768/640?
5. **Timing:** How much of detector staleness arises from capture, queuing, inference, postprocessing, polling, observation building and keyboard dispatch?
6. **Generalization:** How stable are gains across independent PPO seeds, procedural scenarios, different display settings and longer evaluation windows?

## Reproducibility requirements

Record the original unmodified reports and checkpoint digests rather than rewriting benchmark history. For each new evaluation, save the exact Git SHA, CLI args, system/package versions, OS/monitor scaling, detector checkpoint, seed, episode durations, action counts and mean probabilities, step-rate/timing statistics, YOLO freshness/accuracy and guard stops. Bootstrap uncertainty intervals for 100-episode means and threshold rates are recommended; 100 episodes of a *single* training seed do not substitute for multiple seeds. Keep train and validation demonstration runs separate. Avoid changing YOLO resolution, reward, actor training, and PPO architecture all at once.

## Research and documentation references

References below are **sources to consider**, not claims of implemented project features:

- Schulman et al. (2017), **Proximal Policy Optimization Algorithms** — https://arxiv.org/abs/1707.06347
- Stable-Baselines3 **PPO** documentation — https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
- Stable-Baselines3 **RL Tips and Tricks** — https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
- Agarwal et al. (2021), **Deep RL at the Edge of the Statistical Precipice (RLiable)** — https://arxiv.org/abs/2108.13264
- Ross, Gordon & Bagnell (2011), **DAgger: A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning** — https://arxiv.org/abs/1011.0686
- Kelly et al., **HG-DAgger: Interactive Imitation Learning with Human Experts** — https://arxiv.org/abs/1810.02890
- **imitation** library (BC, DAgger, GAIL examples) — https://imitation.readthedocs.io/
- Ultralytics **Instance Segmentation** — https://docs.ultralytics.com/tasks/segment/
- Playwright Python documentation — https://playwright.dev/python/docs/intro
- Playwright screenshot API and calibration — https://playwright.dev/python/docs/screenshots
- **MSS** screen capture documentation — https://python-mss.readthedocs.io/
- **RecurrentPPO**, SB3 Contrib — https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html
- Unity WebGL documentation — https://docs.unity3d.com/
- Ranked Surfers official website — https://rankedsurfers.com/
- Ranked Surfers published rules/terms (verify current availability and applicability) — https://rankedsurfers.com/terms/

## Handoff rules for future developers and coding agents

**Inspect code and raw reports before changing assumptions.** Confirm actual CLI flags, feature shapes, and checkpoint compatibility. Reproduce offline tests. Validate V1.3 live with a *candidate* replay before approving data. Do not claim BC improves gameplay until matched deterministic evaluations prove it. Do not overwrite the 4.61-second checkpoint or quietly change the observation schema. Keep older V3/V4/V5 imported modules intact. Do not `git add .`: local replay/backup folders may be untracked but not excluded in `.gitignore`. Stage only intended documents, code and selected, reviewed report files. No activity against live or ranked services without appropriate authorization.
