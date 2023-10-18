import torch
import random
import trimesh
import numpy as np
from math import atan2, asin, cos
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from scipy.spatial import Delaunay
from skimage.measure import marching_cubes
# import mcubes

import pytorch3d.io
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
)
from pytorch3d.ops.interp_face_attrs import interpolate_face_attributes
from pytorch3d.renderer.blending import (
    BlendParams,
    softmax_rgb_blend
)

def create_cameras(
    R=None, T=None,
    azim=0, elev=0., dist=1.,
    fov=12., znear=0.01,
    device="cuda") -> FoVPerspectiveCameras:
    """
    all the camera parameters can be a single number, a list, or a torch tensor.
    """
    if R is None or T is None:
        R, T = look_at_view_transform(dist=dist, azim=azim, elev=elev, device=device)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=znear, fov=fov)
    return cameras

def create_mesh_normal_renderer(
        cameras: FoVPerspectiveCameras,
        image_size: int = 256,
        blur_radius: float = 1e-6,
        light_location=((-0.5, 1., 5.0),),
        device="cuda",
        **light_kwargs,
):
    """
    If don't want to show direct texture color without shading, set the light_kwargs as
    ambient_color=((1, 1, 1), ), diffuse_color=((0, 0, 0), ), specular_color=((0, 0, 0), )
    """
    # We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=blur_radius,
        faces_per_pixel=5,
    )
    # We can add a point light in front of the object.
    lights = PointLights(
        device=device, location=light_location, **light_kwargs
    )
    phong_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=NormalShader(device=device)
    )

    return phong_renderer

def NormalCalcuate(meshes, fragments):
    verts = meshes.verts_packed()  # (V, 3)
    faces = meshes.faces_packed()  # (F, 3)
    vertex_normals = meshes.verts_normals_packed()  # (V, 3)
    faces_verts = verts[faces]
    faces_normals = vertex_normals[faces]
    pixel_normals = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_normals #  torch.ones_like()
    )
    return pixel_normals
    
class NormalShader(nn.Module):
    def __init__(self, device="cuda", blend_params=None):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()
    def forward(self, fragments, meshes, **kwargs):
        blend_params = kwargs.get("blend_params", self.blend_params)
        normals = NormalCalcuate(meshes, fragments)
        images = softmax_rgb_blend(normals, fragments, blend_params)[:,:,:,:3]

        images = F.normalize(images, dim=3)
        return images