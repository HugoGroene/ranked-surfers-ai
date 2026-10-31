"""Validate checkpoint input meaning before starting a browser or loading weights."""
import json
import zipfile


def requested_contract(args):
    from rs_v61_features import AUX_DIM, FEATURE_VERSION, ACTION_NAMES
    from rs_v5_segmentation_live_benchmark import DEFAULT_CROP
    width, height = getattr(args, 'resolution', (512, 288))
    profile = getattr(args, 'framing_profile', 'canvas_v1')
    return dict(version=1, framing_profile=profile,
                detector_mode=getattr(args, 'detector_mode', 'off'),
                image_shape=[12, height, width], image_order='RGB_oldest_to_newest',
                resize_method=getattr(args, 'resize_method', 'linear'),
                crop=None if profile == 'canvas_v1' else list(getattr(args, 'crop', DEFAULT_CROP)),
                aux_dim=AUX_DIM, feature_version=FEATURE_VERSION,
                actions=list(ACTION_NAMES),
                assume_center_lane=bool(getattr(args, 'assume_center_lane', False)))


def validate_checkpoint(path, requested, *, allow_migration=False):
    # Inspect JSON only; this preflight does not unpickle/load the model.
    with zipfile.ZipFile(path) as archive:
        saved = json.loads(archive.read('data'))
    contract = saved.get('observation_contract')
    legacy = contract is None
    if legacy:
        contract = dict(framing_profile=saved.get('framing_profile', 'legacy'),
                        detector_mode=saved.get('detector_mode', 'on'))
    elif not isinstance(contract, dict) or contract.get('version') != 1:
        raise ValueError('Unsupported checkpoint observation contract')
    if not legacy and set(contract) != set(requested):
        raise ValueError('Incomplete or unknown observation contract fields')
    mismatches = {key: dict(saved=value, requested=requested.get(key))
                  for key, value in contract.items() if requested.get(key) != value}
    if mismatches and not allow_migration:
        raise ValueError('Checkpoint input contract mismatch: '+json.dumps(mismatches, sort_keys=True)+
                         '. Training migration requires --allow-input-migration; evaluation must match.')
    return dict(requested=requested, saved=contract, mismatches=mismatches,
                migration=bool(mismatches), legacy_metadata=legacy,
                note='Older checkpoints validate known framing/detector fields; remaining semantics are not attested.'
                     if legacy else 'Complete versioned contract validated.')


def attach_contract(model, contract):
    model.observation_contract = dict(contract)
    model.framing_profile = contract['framing_profile']
    model.detector_mode = contract['detector_mode']
