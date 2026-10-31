# Optional shared-memory Electron transfer

The existing Playwright backend and Electron socket transport remain available.
Use `--headless --headless-capture electron --electron-transport shared_memory`
to select Windows shared-memory pixels. Socket remains the Electron transport
default; no silent fallback occurs if shared-memory startup/validation fails.

The isolated Electron runtime also needs pinned Koffi 3.3.6:

```powershell
npm.cmd install --prefix development/local/electron_worker --no-audit --no-fund --save-exact koffi@3.3.6
```

Koffi supplies prebuilt native Windows FFI bindings; no Chromium fork or compiler
installation is required. The bridge calls CreateFileMappingW/MapViewOfFile and
memcpy. It still uses CPU bitmap output and two CPU copies, not GPU texture
sharing or zero-copy PyTorch tensors.

Each renderer has a randomly named Local-session mapping, separate from its
socket authentication token. Exactly one demanded frame occupies the mapping.
The producer completes pixel/header writes before sending metadata over the
existing authenticated loopback connection. Python validates magic/version,
dimensions, serial and byte count, copies the pixels to owned memory, and checks
the header again. Only after capture returns can another request overwrite the
slot. Old returned images stay valid. Both processes close their mapping handles
on shutdown. Existing viewport, timestamp, serial, pause and guard checks remain.

Optional `--reuse-frame-state` avoids re-running the frozen screen-state model
on an identical capture. Cache identity includes the image object, capture time,
capture serial, classifier object and confidence. Each new capture invalidates
the result. This does not create an independent confirmation vote, refresh a
timestamp, skip capturing, change confidence thresholds or retrain the classifier.
It targets redundant calls in refresh/boundary handling, not all preparation cost.
The screen classifier must remain frozen while this option is active.

For a short four-worker check, use the existing parallel command with the two
options above, `--reuse-frame-state --profile-reaction`, canvas_v1, a preserved
checkpoint, `--displays 1 2 3 4 --steps 1024 --max-minutes 2`, and a unique report
directory. Run the existing local server separately. All outputs stay separate
from the source model. Do not infer multi-hour stability from this short budget.

Fallbacks are `--electron-transport socket` or `--headless-capture playwright`.
The default paths are retained. See the [dated results](reports/SHARED_MEMORY_RESULTS_20261031.md).
Short four-worker shared-memory checks measured 38-39ms pooled command medians,
81-83ms p95 and 26.7-27.4 optimized transitions/s, with all audits passing.
These are not promises of faster learning or long-run stability. Exact-frame
reuse removed duplicate calls but showed no clear ordinary-latency advantage.
