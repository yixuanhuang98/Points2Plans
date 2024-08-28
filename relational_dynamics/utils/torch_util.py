from math import pi

import torch
import torch.nn.functional as F

def convert_to_float_tensors(tensor_dict, keys=[]):
    keys = keys if keys else tensor_dict.keys()
    for k in keys:
        if torch.is_tensor(tensor_dict[k]):
            tensor_dict[k] = tensor_dict[k].float()
        else:
            tensor_dict[k] = torch.FloatTensor(tensor_dict[k])


def convert_to_long_tensors(tensor_dict, keys=[]):
    keys = keys if keys else tensor_dict.keys()
    for k in keys:
        if torch.is_tensor(tensor_dict[k]):
            tensor_dict[k] = tensor_dict[k].long()
        else:
            tensor_dict[k] = torch.LongTensor(tensor_dict[k])


def make_batch(x, n_batch=1):
    """
    Batchifies a tensor by adding a batch dim and repeating it over that
    dimension n_batch times.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    ndim = x.dim()
    x = x.unsqueeze(0)
    if n_batch > 1:
        x = x.repeat(n_batch, *[1]*ndim)  # unrolls list of 1's of length ndim
    return x

            
def move_batch_to_device(batch_dict, device):
    """
    Recursive function that moves a (nested) dictionary of tensors to the specified device.
    """
    for k, v in batch_dict.items():
        if isinstance(v, torch.Tensor):
            batch_dict[k] = v.to(device)
        elif isinstance(v, dict):
            move_batch_to_device(v, device)

        
def move_models_to_device(models_dict, device):
    """
    Assuming flat dictionary where values are all type torch.nn.Module.
    """
    for k, v in models_dict.items():
        models_dict[k] = v.to(device)


def set_models_to_train(models_dict):
    for v in models_dict.values():
        v.train()


def set_models_to_eval(models_dict):
    for v in models_dict.values():
        v.eval()


def load_state_dicts(models_dict, state_dicts):
    for k, v in models_dict.items():
        if k not in state_dicts:
            print(f"Model {k} does not have state to load")
            #ui_util.print_warning(f"Model {k} does not have state to load")
            continue
        v.load_state_dict(state_dicts[k])


def accumulate_parameters(models_dict):
    params = []
    for v in models_dict.values():
        params += list(v.parameters())
    return params




# torch quaternion functions from NVIDIA:

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

def random_quat(batch_size=1, device='cuda'):
    """
    Computes a random quaternion sampled uniformly from the unit sphere.

    Note this is primarily implemented so that it uses torch random generator
    instead of numpy to avoid issues with how torch/np random generators
    interact when training with randomization:
        https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/

    See: https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L261
    """
    r1 = torch.rand(batch_size).to(device).view(-1, 1)
    r2 = torch.rand(batch_size).to(device).view(-1, 1)
    r3 = torch.rand(batch_size).to(device).view(-1, 1)

    w = torch.sqrt(1.0 - r1) * (torch.sin(2.0 * pi * r2))
    x = torch.sqrt(1.0 - r1) * (torch.cos(2.0 * pi * r2))
    y = torch.sqrt(r1)       * (torch.sin(2.0 * pi * r3))
    z = torch.sqrt(r1)       * (torch.cos(2.0 * pi * r3))
    # Normalize just to be sure since there can be slight numerical differences from pure unit
    return F.normalize(torch.cat([x, y, z, w], dim=-1))


def random_rotation(batch_size=1, device='cuda'):
    return quat_to_rotation(random_quat(batch_size, device))


def quat_to_rotation(q):
    batch = q.size(0)    
    qx = q[:,0].view(batch, 1)
    qy = q[:,1].view(batch, 1)
    qz = q[:,2].view(batch, 1)
    qw = q[:,3].view(batch, 1)
    
    # Unit quaternion rotation matrices computatation  
    xx = qx*qx
    yy = qy*qy
    zz = qz*qz
    xy = qx*qy
    xz = qx*qz
    yz = qy*qz
    xw = qx*qw
    yw = qy*qw
    zw = qz*qw
    
    row0 = torch.cat((1-2*yy-2*zz, 2*xy - 2*zw, 2*xz + 2*yw), 1)
    row1 = torch.cat((2*xy+ 2*zw,  1-2*xx-2*zz, 2*yz-2*xw  ), 1)
    row2 = torch.cat((2*xz-2*yw,   2*yz+2*xw,   1-2*xx-2*yy), 1)
    
    matrix = torch.cat((row0.view(batch,1,3), row1.view(batch,1,3), row2.view(batch,1,3)),1)
    return matrix


if __name__ == '__main__':
    # print(random_rotation(1))

    a = torch.tensor([[1,2,3],
                      [4,5,6],
                      [7,8,9]])
    print(make_batch(a, 11).shape)
