"""Idle-only FP32 classifier parity on both maps, plus capture scheduling checks."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import unittest
from validation import ROOT, RESULT, digest, fingerprints, require_idle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--per-group', type=int, default=32, help='Images per map/class/session; 1..64')
    a = parser.parse_args()
    if not 1 <= a.per_group <= 64: parser.error('--per-group must be 1..64')
    require_idle()  # Before Torch/OpenCV imports or dataset reads.
    import cv2
    import numpy as np
    import rs_v4_screen_states as states
    from development.experiments.multimap_classifier.classifier_config import classifier_path
    from development.experiments.temporal_history.live_adapter import CapturePump
    from runtime import channels_last, paced_run
    import test_runtime
    import test_pump
    import test_resync_spacing
    cv2.setNumThreads(1)
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(m) for m in (test_runtime,test_pump,test_resync_spacing))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful(): raise SystemExit('Scheduling tests failed; classifier checks not started')
    paced_run(CapturePump.run)  # Reject upstream source drift without starting a pump.
    checkpoint = classifier_path()
    original = states.ScreenStateClassifier(checkpoint, threads=1)
    candidate = copy.copy(original)
    candidate.net = copy.deepcopy(original.net)
    hook = channels_last(candidate.net, candidate.torch)
    rows = []; errors = []; counts = {}; max_error = 0.
    # These thresholds cover typical controller gates. Preserve the full vector
    # too, rather than relying on accuracy or the top label alone.
    thresholds = (0.5, .55, .6, .65, .68, .7, .75, .8, .85, .9, .95, .98, .99, .995, .999)
    try:
        # Match the actual canonical preprocess for BGR and BGRA inputs.
        for label in states.CLASSES:
            for session in sorted((ROOT/'ai_data/multimap_screen_states/dataset'/label).iterdir()):
                map_id = next((m for m in ('halloween', 'tokyo') if m in session.name.lower()), None)
                if not session.is_dir() or map_id is None: continue
                files = sorted(p for p in session.iterdir() if p.suffix.lower() in ('.png','.jpg','.jpeg'))
                if not files: continue
                selected = sorted(set(np.linspace(0, len(files)-1, min(a.per_group, len(files)), dtype=int)))
                for index in selected:
                    file = files[index]; image = cv2.imread(str(file))
                    if image is None: raise ValueError('Unreadable classifier image: '+str(file))
                    if len(rows)%2: image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
                    outputs = {}; timings = {}
                    methods = [('original', original), ('candidate', candidate)]
                    for name, model in methods[::1 if len(rows)%2 == 0 else -1]:
                        started = time.perf_counter(); outputs[name] = model.predict(image)
                        timings[name] = (time.perf_counter()-started)*1000
                    x = np.array(list(outputs['original'].values()))
                    y = np.array(list(outputs['candidate'].values()))
                    err = float(np.max(np.abs(x-y))); max_error = max(max_error, err)
                    equal = (np.isfinite(x).all() and np.isfinite(y).all()
                             and np.allclose(x,y,atol=1e-6,rtol=1e-5) and x.argmax()==y.argmax()
                             and all(np.array_equal(x>=t,y>=t) for t in thresholds))
                    if not equal: errors.append(dict(file=str(file),original=x.tolist(),candidate=y.tolist()))
                    counts[map_id+'/'+label] = counts.get(map_id+'/'+label,0)+1
                    rows.append(dict(file=str(file.relative_to(ROOT)), **timings))
        if not all(counts.get(m+'/'+c,0) for m in ('halloween','tokyo') for c in ('GAMEPLAY','MENU')):
            errors.append(dict(error='Missing MENU/GAMEPLAY images for one or both maps'))
        report = dict(passed=result.wasSuccessful() and not errors, tests=result.testsRun,
                      sources=fingerprints(), classifier_sha256=digest(checkpoint), counts=counts,
                      maximum_probability_error=max_error, errors=errors, samples=rows,
                      note='Offline layout parity only; live freshness, tails, state-machine traces and cleanup remain unverified.')
        RESULT.parent.mkdir(parents=True,exist_ok=True)
        RESULT.write_text(json.dumps(report,indent=2))
        print('CHECK REPORT:',RESULT,flush=True)
        print(json.dumps({k:v for k,v in report.items() if k not in ('samples','sources','errors')},indent=2))
        if not report['passed']: raise SystemExit(1)
    finally: hook.remove()


if __name__ == '__main__': main()
