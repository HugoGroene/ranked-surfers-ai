# Faster and remote PPO training

## What limits the current setup

PPO already optimizes minibatches. The single local game runs in real time;
capturing/acting more often improves reaction opportunities but does not make
the map advance faster. A larger GPU alone does not supply more independent game
experience. YOLO is asynchronous: its feature age is not a blocking wait of that
length on every action. Increasing its frequency previously made features newer
but slowed the action loop. The new latency path is measured separately in the
[foundation report](README.md).

Larger rollouts are also not parallel simulation. A 512-step rollout collects
four times as many transitions per update as a 128-step rollout, then does more
optimization per update. This reduces fixed update/resynchronization overhead
and changes training statistics, while retaining the same number of optimization
epochs per sample. It does not eliminate optimizer compute or guarantee faster
learning. The raw image buffer grows from 216 to 864 MiB in the installed SB3.

## Options

| Approach | What it can offer | Work still needed |
| --- | --- | --- |
| Dedicated remote GPU workstation/VM, one game | Frees the personal PC; possibly better runtime hardware | Validate an active graphical session, GPU WebGL, desktop capture, focus, package/assets and long-lived rendering |
| Multiple independent game workers | More aggregate real-time experience; shared/batched policy inference | Isolated displays/capture or a new capture backend, separate episode state, synchronized PPO collection and safe optimizer pausing |
| Engine-controlled stepping/acceleration | Potentially the largest experience-collection gain | Suitable Unity source/build integration and physics/observation parity validation |

The remote single-worker option is the nearest offloading path, not a promise of
large speedup. The current backend is not yet a validated cloud deployment, and
the updated local code/checkpoints are not automatically present in GitHub.
An interactive Windows GPU VM is one possible compatibility prototype. A Linux
virtual-display or headless renderer is another engineering option, but its
pixels, timing, graphics acceleration and guard behavior must be verified first.
Headless does not remove the rendering requirement of an image-based policy.

No provider, machine size or price is recommended before measuring one worker.
No cloud account was accessed, resources provisioned, files uploaded, packages
installed or billable job launched during this task.

## Parallel sampling is the main scaling project

SB3 supports collecting PPO rollouts from multiple independent environments via
[vectorized environments](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html).
However, this application cannot safely get that behavior by opening many tabs
on one foreground desktop: MSS sees the desktop and the controller requires
the correct visible/focused game. Each worker needs its own reliable observation
and input source. More workers also multiply rendering, detector, classifier,
buffer and inference costs. Speedup will be below the ideal worker count when
those resources or the slowest worker become limiting.

The current same-window batch runner is sequential automation, not this parallel
backend. A practical implementation path is:

1. Validate one remote worker against the local pixels, source-frame age,
   input behavior, terminal detection, reward timing and throughput.
2. Isolate two workers and verify that actions, screenshots, detections and
   episode/reset state cannot cross between them.
3. Collect synchronized PPO rollouts using current policy weights, with
   independent per-worker optimizer-pause clocks and terminal observations.
4. Benchmark shared/batched GPU inference for the actor and perception models.
5. Scale worker count only while measured transitions per wall-clock second
   improve without unacceptable observation age or reduced detector accuracy.

Replaying a fixed collection of recorded clips repeatedly is not equivalent to
collecting new PPO rollouts. Offline/imitation approaches can use recordings,
but they are separate learning algorithms/experiments. Current work deliberately
keeps BC optional and does not turn old demonstrations into on-policy PPO data.

## Faster-than-real-time game simulation

Unity ML-Agents documents engine time-scale settings and concurrent instances in
[its training guide](https://unity-technologies.github.io/ml-agents/Training-ML-Agents/).
Those features apply to environments integrated with that toolkit. This
repository contains packaged WebGL assets; no C# Unity project or ML-Agents
environment was found. Installing a package or renting GPUs cannot by itself
add synchronous stepping to this game build.

If an appropriate engine integration becomes available, prefer controlled
physics steps with equivalent observations/actions and game-time rewards.
Simply increasing game speed while leaving capture/input rates unchanged makes
the control problem harder and can change the learned dynamics. Any accelerated
environment needs transfer checks against the real-time local Practice game.

## Recommended order

Use the measured local pipeline foundation and persistent batches first. Keep
the old reference checkpoints. Next, prototype one remote worker if freeing the
PC is the immediate priority; build isolated parallel sampling if increasing
experience throughput is the priority. Engine stepping is the larger longer-term
opportunity when the required game integration is available. Do not pay for a
large GPU farm before the environment backend scales.
