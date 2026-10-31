"""Copy a recorded aux vector into the live detector-unavailable representation."""
import numpy as np
from rs_v6_ppo_features import HAZARD_DIM
from rs_v61_features import AUX_DIM, V6_AUX_DIM, SCREEN_COLUMN_DIM


def detector_off_aux(aux):
    result = np.array(aux, dtype=np.float32, copy=True)
    if result.ndim < 1 or result.shape[-1] != AUX_DIM or not np.isfinite(result).all():
        raise ValueError('Invalid demonstration auxiliary features')
    result[..., :HAZARD_DIM] = 0.
    result[..., V6_AUX_DIM-1] = 1.  # unavailable detection age
    result[..., V6_AUX_DIM:V6_AUX_DIM+SCREEN_COLUMN_DIM] = 0.
    return result
