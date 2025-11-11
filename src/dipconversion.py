import numpy as np
import numbers

# all code from LoopStructal 1.6.6 duplicating here to minimise dependencies
# LoopStructural/utils/maths.py

def strikedip2vector(strike, dip) -> np.ndarray:
    """Convert strike and dip to a vector

    Parameters
    ----------
    strike : _type_
        _description_
    dip : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    vec = np.zeros((len(strike), 3))
    s_r = np.deg2rad(strike)
    d_r = np.deg2rad((dip))
    vec[:, 0] = np.sin(d_r) * np.cos(s_r)
    vec[:, 1] = -np.sin(d_r) * np.sin(s_r)
    vec[:, 2] = np.cos(d_r)
    vec /= np.linalg.norm(vec, axis=1)[:, None]
    return vec


def azimuthplunge2vector(
    plunge,
    plunge_dir,
    degrees: bool = True,
) -> np.ndarray:
    """Convert plunge and plunge direction to a vector

    Parameters
    ----------
    plunge : Union[np.ndarray, list]
        array or array like of plunge values
    plunge_dir : Union[np.ndarray, list]
        array or array like of plunge direction values

    Returns
    -------
    np.array
        nx3 vector
    """
    if isinstance(plunge, numbers.Number):
        plunge = np.array([plunge], dtype=float)
    else:
        plunge = np.array(plunge, dtype=float)
    if isinstance(plunge_dir, numbers.Number):
        plunge_dir = np.array([plunge_dir], dtype=float)
    else:
        plunge_dir = np.array(plunge_dir, dtype=float)
    if degrees:
        plunge = np.deg2rad(plunge)
        plunge_dir = np.deg2rad(plunge_dir)
    vec = np.zeros((len(plunge), 3))
    vec[:, 0] = np.sin(plunge_dir) * np.cos(plunge)
    vec[:, 1] = np.cos(plunge_dir) * np.cos(plunge)
    vec[:, 2] = -np.sin(plunge)
    return vec
