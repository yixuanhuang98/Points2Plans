import torch
import numpy as np
from pyquaternion import Quaternion


def rotation_to_quat(R):
    T = np.eye(4)
    T[:3,:3] = R
    return homogeneous_to_quat(T)


def homogeneous_to_position(T):
    return T[:3, 3]


def homogeneous_to_rotation(T):
    return T[:3, :3]


def homogeneous_to_quat(T):
    """
    Converts rotation matrix from homogeneous TF matrix to quaternion.
    """
    q = Quaternion(matrix=T) # w, x, y, z
    return np.array([q.x, q.y, q.z, q.w]) # Need to switch to x, y, z, w


def homogeneous_to_pose(T):
    """
    Converts homogeneous TF matrix to a 3D position and quaternion.
    """
    return homogeneous_to_position(T), homogeneous_to_quat(T)


def quat_to_homogeneous(q):
    """
    Converts quaternion to homogeneous TF matrix. Assumes (x, y, z, w) quaternion input.
    """
    return Quaternion(q[3], q[0], q[1], q[2]).transformation_matrix  # Quaternion is (w, x, y, z)


def quat_to_rotation(q):
    """
    Converts quaternion to rotation matrix. Assumes (x, y, z, w) quaternion input.
    """
    return quat_to_homogeneous(q)[:3,:3]


def pose_to_homogeneous(p, q):
    """
    Converts position and quaternion to a homogeneous TF matrix.
    """
    T = quat_to_homogeneous(q)
    T[:3, 3] = p
    return T


def homogeneous_transpose(T):
    """
    Converts TF to TF inverse, also use np.matmul instead of np.dot
    """
    new_T = np.eye(4)
    new_T[:3,:3] = T[:3,:3].T
    new_T[:3,3] = -np.matmul(new_T[:3,:3], T[:3,3])
    return new_T
    

def random_quat():
    q = Quaternion.random()
    return np.array([q.x, q.y, q.z, q.w])


def random_rotation():
    return quat_to_rotation(random_quat())
    

def get_x_axis_rotation(theta):
    return np.array([[1,             0,              0],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta),  np.cos(theta)]])


def get_y_axis_rotation(theta):
    return np.array([[ np.cos(theta), 0, np.sin(theta)],
                     [             0, 1,             0],
                     [-np.sin(theta), 0, np.cos(theta)]])


def get_z_axis_rotation(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta),  np.cos(theta), 0],
                     [            0,              0, 1]])


def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([x, y, z, w], dim=-1).view(shape)

    return quat


def quat_conjugate(a):
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).view(shape)


def quaternion_error(desired, current, square=False, numpy=False, flatten=False):
    q_c = quat_conjugate(current)
    q_r = quat_mul(desired, q_c)
    error = q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)
    if square:
        error = error**2
    if numpy:
        error = error.cpu().numpy()
    if flatten:
        error = error.flatten()
    return error


def position_error(desired, current, square=False, numpy=False, flatten=False):
    error = desired - current
    if square:
        error = error**2
    if numpy:
        error = error.cpu().numpy()
    if flatten:
        error = error.flatten()
    return error


def geodesic_error(R1, R2):
    return np.arccos(np.clip((np.trace(np.dot(R1, R2.T)) - 1.) / 2., -1, 1))
    


if __name__ == '__main__':
    R1 = random_rotation()
    R2 = random_rotation()

    print(geodesic_error(R1, R2))
