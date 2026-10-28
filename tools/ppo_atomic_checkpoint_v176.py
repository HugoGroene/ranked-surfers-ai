#!/usr/bin/env python3
"""Crash-resistant SB3 PPO latest.zip update, preserving latest.prev.zip.

Only the checkpoint inside the current unique run directory is modified.
A broken/incomplete new save never replaces the last successful latest.zip.
"""
from pathlib import Path
import os
import shutil
import tempfile
import zipfile


def _durable_zip(path):
    # Zip CRC protects against an incomplete or corrupted model save; the
    # full SB3 checkpoint shape is checked in the normal model.load smoke test.
    with zipfile.ZipFile(path, 'r') as archive:
        if not archive.namelist() or archive.testzip() is not None:
            raise ValueError('Incomplete PPO checkpoint ZIP')
    with open(path, 'r+b') as handle:
        os.fsync(handle.fileno())


def save_atomic_checkpoint(model, target):
    """Write a new .zip, validate it, backup previous, atomically replace.

    Must run on the training host/local filesystem. os.replace is atomic
    for renames within the same filesystem. Does not touch any --resume path.
    """
    target = Path(target)
    if target.name != 'latest.zip':
        raise ValueError('Atomic PPO checkpoint only permits local latest.zip')
    if not target.parent.is_dir():
        raise ValueError('Checkpoint run folder does not exist')
    temp_handle, temp_name = tempfile.mkstemp(
        prefix='.latest.v176.new.', suffix='.zip', dir=str(target.parent))
    os.close(temp_handle)
    candidate = Path(temp_name)
    backup_temp = None
    try:
        model.save(candidate)
        _durable_zip(candidate)
        if target.exists():
            # Never move/delete the only valid latest before the new save is
            # ready. Copy previous to a separate tempfile, then atomically
            # replace latest.prev.zip, leaving latest in place until commit.
            old_fd, old_temp = tempfile.mkstemp(
                prefix='.latest.v176.previous.', suffix='.zip', dir=str(target.parent))
            os.close(old_fd)
            backup_temp = Path(old_temp)
            shutil.copyfile(target, backup_temp)
            _durable_zip(backup_temp)
            os.replace(backup_temp, target.with_name('latest.prev.zip'))
            backup_temp = None
        os.replace(candidate, target)
    finally:
        candidate.unlink(missing_ok=True)
        if backup_temp is not None:
            backup_temp.unlink(missing_ok=True)
    return target
