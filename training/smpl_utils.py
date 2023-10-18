import os
import numpy as np
import torch
from torch_utils import persistence, misc
from kaolin.ops.mesh import check_sign
from kaolin.metrics.trianglemesh import point_to_mesh_distance
import torch.autograd as autograd


def get_eikonal_term(pts, sdf):
    eikonal_term = autograd.grad(outputs=sdf, inputs=pts,
                                 grad_outputs=torch.ones_like(sdf, requires_grad=False, device=pts.device),
                                 create_graph=True,
                                 retain_graph=True,
                                 only_inputs=True)[0]

    return eikonal_term

@misc.profiled_function
def batch_transform_normal(P, v, pad_ones=True):
    if pad_ones:
        homo = torch.ones((*v.shape[:-1], 1), dtype=v.dtype, device=v.device)
    else:
        homo = torch.zeros((*v.shape[:-1], 1), dtype=v.dtype, device=v.device)
    v_homo = torch.cat((v, homo), dim=-1)
    v_homo = torch.matmul(P, v_homo.unsqueeze(-1))
    v_ = v_homo[..., :3, 0]
    skinning_rotate_transform = torch.inverse(P)[..., :3, :3]
    return v_, skinning_rotate_transform

@misc.profiled_function
def batch_transform(P, v, pad_ones=True):
    if pad_ones:
        homo = torch.ones((*v.shape[:-1], 1), dtype=v.dtype, device=v.device)
    else:
        homo = torch.zeros((*v.shape[:-1], 1), dtype=v.dtype, device=v.device)
    v_homo = torch.cat((v, homo), dim=-1)
    v_homo = torch.matmul(P, v_homo.unsqueeze(-1))
    v_ = v_homo[..., :3, 0]
    return v_

@misc.profiled_function
def batch_index_select(data, inds, device):
    bs, nv = data.shape[:2]
    inds = inds + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    data = data.reshape(bs*nv, *data.shape[2:])
    return data[inds.long()]

@misc.profiled_function
def cal_sdf_batch(verts, faces, triangles, points, device):
    Bsize = points.shape[0]
    residues, _, _ = point_to_mesh_distance(points.contiguous(), triangles)
    pts_dist = torch.sqrt(residues.contiguous())
    # pts_signs = -2.0 * (check_sign(verts.cuda(), faces[0].cuda(), points.cuda()).float() - 0.5).to(device) # negative outside
    pts_signs = -2.0 * (check_sign(verts.cuda(), faces.cuda(), points.cuda()).float() - 0.5).to(device) # negative outside
    pts_sdf = (pts_dist * pts_signs).unsqueeze(-1)

    return pts_sdf.view(Bsize, -1, 1)

@misc.profiled_function
def face_vertices(vertices, faces, device):
    """ 
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """

    bs, nv = vertices.shape[:2]
    _, nf = faces.shape[:2]
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs*nv, vertices.shape[-1]))
    return vertices[faces.long()]

def get_canonical_pose():
    # Define a-pose
    # pose_canonical = [ 
    #     0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
    #     0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
    #     0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
    #     0.,  0.,  0.,  0.,  0.,-np.pi/4.,  0.,  0., np.pi/4.,  0.,-np.pi/6,  0., 0., np.pi/6,
    #     0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]
        
    # Define Da-pose
    pose_canonical = [ 
        0.0000,  0.0000,  0.5000,  0.0000,  0.0000, -0.5000,  0.0000,  0.0000,
        0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
        0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
        0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
        0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
        0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
        0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
        0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
        0.0000,  0.0000,  0.0000,  0.0000,  0.0000]
    
    # Define T-pose
    # pose_canonical = torch.from_numpy(np.array([ 
    #   0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
    #   0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
    #   0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
    #   0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
    #   0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
    #   0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
    #   0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
    #   0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
    #   0.0000,  0.0000,  0.0000,  0.0000,  0.0000])).float()


    # Define A-pose
    # pose_canonical = torch.from_numpy(np.array([ 
    #   0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
    #   0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
    #   0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
    #   0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,
    #   0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])).float()

    return pose_canonical

def grid_sample(image, optical):
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]

    ix = ((ix + 1) / 2) * (IW-1);
    iy = ((iy + 1) / 2) * (IH-1);
    with torch.no_grad():
        ix_nw = torch.floor(ix);
        iy_nw = torch.floor(iy);
        ix_ne = ix_nw + 1;
        iy_ne = iy_nw;
        ix_sw = ix_nw;
        iy_sw = iy_nw + 1;
        ix_se = ix_nw + 1;
        iy_se = iy_nw + 1;

    nw = (ix_se - ix)    * (iy_se - iy)
    ne = (ix    - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix)    * (iy    - iy_ne)
    se = (ix    - ix_nw) * (iy    - iy_nw)
    
    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW-1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH-1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW-1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH-1, out=iy_ne)
 
        torch.clamp(ix_sw, 0, IW-1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH-1, out=iy_sw)
 
        torch.clamp(ix_se, 0, IW-1, out=ix_se)
        torch.clamp(iy_se, 0, IH-1, out=iy_se)

    image = image.view(N, C, IH * IW)


    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) + 
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    return out_val

#----------------------------------------------------------------------------
def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index / N) % N
    samples[:, 0] = ((overall_index / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size
#----------------------------------------------------------------------------
