from scipy.spatial.transform import Rotation as R
import numpy as np


def quaternion_to_psi(quat):
    # Check if quaternion has zero norm
    if np.isclose(np.linalg.norm([quat.x, quat.y, quat.z, quat.w]), 0):
        # Handle zero norm quaternion
        # Here we return a default value, but you can handle it differently if needed
        return 0.0

    r = R.from_quat([quat.x, quat.y, quat.z, quat.w])
    return r.as_euler('zyx')[0]
