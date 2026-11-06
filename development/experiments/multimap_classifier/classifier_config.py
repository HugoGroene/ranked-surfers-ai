"""Shared classifier selection for current multimap workflows."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
REGISTRY=Path(__file__).with_name('maps.json')


def classifier_path(override=None):
    value=override or json.loads(REGISTRY.read_text(encoding='utf8'))['active_classifier']
    path=Path(value)
    if not path.is_absolute():path=ROOT/path
    path=path.resolve()
    if not path.is_file():raise FileNotFoundError('Classifier not found: '+str(path))
    return path
