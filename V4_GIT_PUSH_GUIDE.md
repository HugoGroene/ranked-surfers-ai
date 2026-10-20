# Commit the finished V4 controller snapshot (Windows PowerShell)

Run these from `C:\Users\Hugo\Documents\ranked-surfers-ai`, with the new `README.md` placed at repository root. The commands **do not automatically push**; you review staged changes first. The `.gitignore` additions below preserve already-tracked files (Git continues to track previously committed items even if a new ignore rule matches them).

## 1. Add protective ignore rules (once)

Keep your existing `.gitignore` intact, append:

```powershell
@'

# V4 generated experiment data; keep reference templates separately
/ai_data/v4_*/
/ai_data/rl_v2_runs/
/ai_data/*.log
/ai_data/start_templates_ubuntu_backup/
/rs_v4_jump_speed_package.zip
'@ | Add-Content -Path .gitignore
```

Do not append this block repeatedly if you already did it.

## 2. Stage only source and documentation

```powershell
git add -- .gitignore README.md V4_GIT_PUSH_GUIDE.md README_RS_V4_JUMP_SPEED.md README_RS_V4_SCREEN_STATES.md
git add -- 'rs_v4_*.py'
```

This adds the new V4 benchmark and classifier scripts without the ZIP or raw data. Git pathspec wildcard `'rs_v4_*.py'` is intentional.

## 3. Stage required classifier checkpoint only

```powershell
Get-Item .\ai_data\v4_screen_states\screen_state_best.pt | Select-Object FullName,Length
Test-Path .\ai_data\v4_screen_states\screen_state_best.json
# Only stage the model if you have the right to share it and it is not too large.
git add -f -- ai_data/v4_screen_states/screen_state_best.pt
# Optional small metadata report (contains training/validation session names):
git add -f -- ai_data/v4_screen_states/screen_state_best.json
```

GitHub rejects ordinary individual files over 100 MiB; if the `.pt` file is that large, do **not** stage it normally. Use Git LFS or another permitted model distribution method instead. Verify its size and content before adding it. If the YOLO detector checkpoint isn't already tracked, decide separately whether to commit it; check `git ls-files checkpoints/yolo_hazards_best.pt`.

## 4. Decide about the changed Windows menu template

Your `git status` shows `ai_data/start_templates/ref_00.png` modified and `ref_01.png` through `ref_04.png` deleted. **If those changes were the deliberate Windows recalibration and you want this snapshot to reproduce that controller**, stage them:

```powershell
git add -u -- ai_data/start_templates/
```

If the template deletions were accidental, **do not** run that command; inspect the tracked versions first. Your untracked `ai_data/start_templates_ubuntu_backup/` should remain uncommitted unless you explicitly decide to preserve cross-platform templates in the repository.

## 5. Review, commit, push

```powershell
git status --short
git diff --cached --stat
git diff --cached --check
git diff --cached --name-status
```

Expected: source `.py`, README files, `.gitignore`, the two classifier checkpoint/report files, and optionally the intended template modifications/deletions. No raw dataset folders, reports from previous benchmarks, ZIPs, `.venv`, or private browser captures.

```powershell
git commit -m "Checkpoint working V4 screen-state controller and 31 Hz benchmark"
git push origin main
git status
```

If the working tree still shows untracked ignored/scratch files, that's okay; your **commit** is the snapshot, not every byte of the local project. `git push` requires your existing remote Git credentials. Confirm with `git log -1 --oneline` and `git status`.

### Quick verification in a fresh clone

Confirm Python source, the YOLO detector checkpoint, classifier checkpoint, and expected game assets are available. Python environment folders and generated screenshots are intentionally not included. If your downloaded Unity copy is tracked, check you have appropriate redistribution rights before making the repository public.
