"""Summarize the fixed controlled trial after all stages complete."""
import hashlib
import json
import re
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    batch = read(HERE / "batch/batch.json")
    if batch["status"] != "COMPLETE":
        raise RuntimeError("All stages must finish before analysis")
    plan = read(HERE / "plan.json")
    names = ["baseline", "A", "B", "C"]
    evaluations = {}
    durations = {}
    for label, stage in zip(names, [1, 3, 5, 7]):
        run = f"ppoctrl_20261028_{stage:02d}_eval"
        report = read(ROOT / "ai_data/rl_v2_runs" / run / "v61_eval_report.json")
        ds = np.asarray(report["durations_seconds"], dtype=float)
        assert len(ds) == report["eval_episodes"] == 100
        durations[label] = ds
        evaluations[label] = {
            "run": run, "episodes": len(ds), "mean_seconds": float(ds.mean()),
            "median_seconds": float(np.median(ds)), "best_seconds": float(ds.max()),
            "reached_5s": int((ds >= 5).sum()), "reached_10s": int((ds >= 10).sum()),
            "action_counts": report["action_counts"],
            "mean_action_probabilities": report["mean_action_probabilities"],
            "terminations": report.get("eval_termination_counts"),
            "guard": report["detector"]["v612_live_input_guard"],
            "gameplay_intervals_ms": report["gameplay_only_intervals_ms"],
        }
    rng = np.random.default_rng(123)
    comparisons = {}
    for newer, older in [("A", "baseline"), ("B", "baseline"), ("C", "baseline"), ("B", "A"), ("C", "B")]:
        a, b = durations[newer], durations[older]
        changes = rng.choice(a, (20000, len(a)), replace=True).mean(axis=1)
        changes -= rng.choice(b, (20000, len(b)), replace=True).mean(axis=1)
        comparisons[f"{newer}_minus_{older}"] = {
            "mean_difference_seconds": float(a.mean() - b.mean()),
            "bootstrap_95_percent_ci": np.quantile(changes, [.025, .975]).tolist(),
            "note": "Independent episode bootstrap, conditional on these trained models/sessions; excludes training-seed and map-session uncertainty.",
        }
    training = {}
    for label, stage in zip(["A", "B", "C"], [2, 4, 6]):
        run = f"ppoctrl_20261028_{stage:02d}_train"
        folder = ROOT / "ai_data/rl_v2_runs" / run
        report = read(folder / "v61_report.json")
        settings = read(folder / "v61_effective_ppo_settings.json")
        audits = read(HERE / "batch" / f"{run}_audits.json")
        log = (HERE / "batch" / f"{run}.log").read_text(encoding="utf-8")
        optimizer = {}
        for key in ["approx_kl", "clip_fraction", "entropy_loss", "explained_variance", "value_loss"]:
            vals = [float(v) for v in re.findall(r"\|\s*" + key + r"\s*\|\s*([-+0-9.eE]+)\s*\|", log)]
            optimizer[key] = {"count": len(vals), "median": float(np.median(vals)), "min": min(vals), "max": max(vals)} if vals else None
        with zipfile.ZipFile(folder / "latest.zip") as archive:
            data = json.loads(archive.read("data"))
        assert report["steps"] == 32768 and report["completed_requested_steps"]
        assert data["num_timesteps"] == plan["source_timesteps"] + 32768
        training[label] = {
            "run": run, "checkpoint": str(folder / "latest.zip"),
            "checkpoint_sha256": digest(folder / "latest.zip"),
            "steps": report["steps"], "wall_seconds": report["wall_seconds"],
            "steps_per_second": report["avg_steps_per_sec_including_resets"],
            "mean_training_survival": report["mean_survival_seconds"],
            "best_training_survival": report["max_survival_seconds"],
            "settings": settings, "audits": audits, "optimizer": optimizer,
            "early_stopping_events": log.count("Early stopping at step"),
            "saved_num_timesteps": data["num_timesteps"],
            "saved_n_updates_epochs": data["_n_updates"],
            "guard": report["detector"]["v612_live_input_guard"],
        }
    original_hashes = read(HERE / "code_sha256.json")
    changed_code = [p for p, h in original_hashes.items() if not (ROOT / p).exists() or digest(ROOT / p) != h]
    result = {
        "status": batch["status"], "evaluations": evaluations,
        "comparisons": comparisons, "training": training,
        "source_checkpoint_unchanged": digest(ROOT / plan["source_checkpoint"]) == plan["source_sha256"],
        "runtime_code_changed_during_trial": changed_code,
        "batch_browser_session": batch["browser_session"],
        "batch_source_checkpoints_unchanged": batch["source_checkpoints_unchanged"],
    }
    (HERE / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"evaluations": {k: {key: v[key] for key in ["mean_seconds", "median_seconds", "best_seconds", "reached_5s", "reached_10s"]} for k, v in evaluations.items()}, "comparisons": comparisons, "training": {k: {key: v[key] for key in ["steps_per_second", "early_stopping_events", "saved_n_updates_epochs"]} for k, v in training.items()}, "source_checkpoint_unchanged": result["source_checkpoint_unchanged"], "code_changed": changed_code}, indent=2))


if __name__ == "__main__":
    main()
