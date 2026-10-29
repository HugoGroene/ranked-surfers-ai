> Subsequent update: under the user-authorized ten-run sampled-review policy,
> 142 new runs were approved and combined with the previous nine for a BC run.
> See [the sampled-batch experiment](../bc_sampled_batch_20261028/README.md).
> The original review page/queue is a pre-approval snapshot; exports from it
> require revalidation against current manifests before applying changes.

# Review the 291-run recording session

Open [review.html](review.html) in Microsoft Edge or Google Chrome. This is a
local offline page, with no server or external dependencies. Keep the report
folder in this location: its contact-sheet links refer to the original replay
folders. Browser-compatible H.264 video copies are in `video/`; original videos
are unchanged. All 291 copies have verified matching frame counts and FPS.

## How to review efficiently

1. Keep the default useful-examples sort. It prioritizes suggested LEFT/RIGHT
   coverage, while filters include long, medium and short runs.
2. Click **Play selected section**, optionally at 1.5x speed. Inspect the retained
   decisions, then use **Review last 7 seconds** to check where failure begins.
   Action buttons jump to one second before a recorded button press. The full
   contact sheet helps locate an unexpected restart or a missed earlier mistake.
3. Adjust start/end fields or use **Set start/end at current frame**. Use recorded
   timestamps displayed by this page, not the approximate video-player clock.
   If there are separated clean sections, add intervals. Flag uncertain runs or
   skip them rather than training from guesses.
4. Check the reviewed-section box, then **Keep reviewed segments**. It advances
   to the next run. **Skip** and **Flag** also advance. None of these buttons
   writes to a replay manifest or starts training.
5. Click **Download reviewed selection** when ready. Send the exported
   `demo_review_selection.json` to the assistant. It records your explicit keeps
   and exact frame-aligned cuts; only kept runs are selected for BC.

Progress and draft edits are stored in this browser. Use **Export progress
backup** before closing or moving computers, and **Import progress backup** to
resume. Start with a manageable batch; the first export does not require reviewing
all 291. Do not clear browser data until progress is exported. The test browser
was isolated and has not created any user decisions in your browser.

## What was audited

Session log: `recorder_20261028_160735_200276.log`. The saved-run list contains
exactly 291 unique recordings. There are 83 runs of at least 20 seconds, 75 runs
of 10-20 seconds, and 133 runs below 10 seconds. Total recorded time is about
72.07 minutes. All saved labels reproduce from the raw input events. This is
technical consistency, not proof that every action was good.

The source data contains 802 LEFT and 790 RIGHT valid labels. Automatic prefix
suggestions retain 570 LEFT and 549 RIGHT labels across 241 runs before visual
review. Suggestions use a leading trim of 2.15 seconds, trailing trim of 2.5
seconds, and stop before the first interior capture gap. These are starting
points, not the individually verified cuts used in the nine-run pilot. A start
animation can reappear later; short survival is not by itself a reason to reject
an otherwise good segment, and long survival is not an approval criterion.

208 runs have at least one >130 ms interval, often near the beginning. The page
checks retained four-frame histories and rejects windows crossing an invalid
gap. It does not automatically detect every collision, wrong decision or mixed
attempt. Inspect those visually. Runs with no useful automatic prefix remain
available for manual review/editing; none has been deleted.

All 291 manifests remain unchanged candidates. No checkpoint or raw recording
was modified. The prior nine approved pilot runs are not in this queue. No new
BC training has started.

## Files and verification

- `audit.json`: detailed recomputed diagnostics for every run.
- `queue.json`: review metadata, exact timestamps and unapproved suggestions.
- `suggested_cuts_NOT_APPROVED.csv`: spreadsheet view of suggestions, not approvals.
- `video_conversion.json`: frame-count/FPS verification for every review copy.
- `browser_qa.json`: successful playback, filters, export, draft persistence,
  approval checkbox and exact-cut parity checks in an isolated headless browser.
- `review_preview.png`: screenshot of the reviewer.

Seven offline planner tests passed. Browser checks verified that all 241
suggested windows match the Python validator, that edits invalidate a previous
keep until re-reviewed, and that the exported selection preserves exact bounds.

After receiving your reviewed selection, the next step is to validate and apply
only those explicit approvals, then train a separate actor-only BC checkpoint
from the untouched eight-hour PPO source with whole-run holdout and per-action
metrics. Current survival reference: the small BC pilot's 4.83-second mean over
100 deterministic episodes. More extensive optimization should be selected by
validation performance rather than maximizing epochs blindly.
