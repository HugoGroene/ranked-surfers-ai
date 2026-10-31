"""Startup provenance and failure-isolated finalization; no gameplay operations."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile


def worker_seed(base_seed, index):
    if base_seed is None:
        raise ValueError('A reproducible base seed is required')
    if not 0 <= index < 10000:
        raise ValueError('Invalid worker seed index')
    return (int(base_seed) + 1000003 * (index + 1)) % (2**32)


def atomic_json(path, value):
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix='.'+path.name+'.', dir=path.parent)
    candidate = Path(name)
    try:
        with os.fdopen(fd, 'w', encoding='utf8') as handle:
            json.dump(value, handle, indent=2, allow_nan=False, default=str)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(candidate, path)
    finally:
        candidate.unlink(missing_ok=True)


def finalize_run(report, checkpoint, cleanup, persist):
    """Try checkpoint/status before cleanup; retain every error and finish status.

    Caller preserves an already-active exception. Otherwise it raises if the
    returned list is nonempty. Even a second KeyboardInterrupt during one stage
    cannot skip all remaining stages. A process kill cannot be handled here.
    """
    errors = report.setdefault('finalization_errors', [])
    def attempt(name, fn):
        try:
            fn()
        except BaseException as exc:
            errors.append(dict(stage=name, error=repr(exc)))
            report['status'] = 'STOPPED'
    attempt('checkpoint', checkpoint)
    attempt('status_before_cleanup', persist)
    for name, fn in cleanup:
        attempt(name, fn)
    attempt('status_after_cleanup', persist)
    return errors


def source_manifest(root, *, assets=(), contract=None, seeds=None):
    root = Path(root)
    # Small source files only; never scan checkpoints, experiment images or logs.
    files = list(root.glob('*.py'))
    for folder in ('tools',):
        for extension in ('*.py', '*.cjs', '*.html', '*.cpp', '*.cc', '*.h'):
            files.extend((root/folder).rglob(extension))
    hashes = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(set(files)) if 'node_modules' not in p.parts}
    versions = {}
    for package in ('torch', 'stable-baselines3', 'gymnasium', 'numpy', 'opencv-python', 'playwright'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    asset_hashes = {}
    for item in assets:
        path = Path(item)
        if path.is_file():
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                for chunk in iter(lambda: handle.read(1024*1024), b''):
                    digest.update(chunk)
            asset_hashes[str(path)] = digest.hexdigest()
    return dict(version=1, source_sha256=hashes, assets_sha256=asset_hashes,
                dependencies=versions, observation_contract=contract, sampling_seeds=seeds,
                seed_note='Python/Torch sampling seeds do not attest Unity track seeds.')
