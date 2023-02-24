import cv2
import ipdb
import numpy as np
from skimage import measure
import trimesh
import os
from subprocess import call
from scipy import ndimage
from ObjIO import load_obj_data
from VoxelizerUtil import voxelize, voxelize_2, save_volume, resize_volume
import copy
from Constants import consts
import torch

def save_obj_mesh_with_color(mesh_path, verts, faces, colors):
    """
    input
        mesh_path: XXX.obj
        verts    : (N, 3) in the mesh-coords.
        faces    : (N, 3), order not switched yet
        colors   : (N, 3), RGB, float 0 ~ 1
    """

    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        c = colors[idx]
        file.write('v %.4f %.4f %.4f %.4f %.4f %.4f\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1])) # switch the order, so that the computed normals later can face outwards from the mesh
    file.close()

def verts_canonization(verts, dim_w, dim_h):
    """
    translate & rescale the verts from [[0,2W),[0,2H),[0,2D)] ---> [(-0.33,0.33),(-0.5,0.5),(-0.33,0.33)]
    """

    # translate
    verts[:,0] -= dim_w # X, from [0,2W) to [-W,W)
    verts[:,1] -= dim_h # Y, from [0,2H) to [-H,H)
    verts[:,2] -= dim_w # Z, from [0,2D) to [-D,D)

    # rescale
    verts[:,0] /= (2.*dim_h) # X, from [-W,W) to (-0.33,0.33)
    verts[:,1] /= (2.*dim_h) # Y, from [-H,H) to (-0.5,0.5)
    verts[:,2] /= (2.*dim_h) # Z, from [-D,D) to (-0.33,0.33)

    return verts

def voxel_2_obj(meshVoxel, image_path, save_path):

    # MC
    verts, faces, normal, _ = measure.marching_cubes_lewiner(meshVoxel, level=0.5)

    # ipdb.set_trace()

    # normalize
    verts *= 2.0
    verts = verts_canonization(verts=verts, dim_h=128, dim_w=128)

    # 1, save the mesh
    # recon_obj = trimesh.Trimesh(verts, faces, process=True, maintains_order=True)
    # recon_obj.export(save_path)

    # ipdb.set_trace()
    # 2, save the mesh with color
    xyz_tensor = verts.T     # (3, N)
    # ipdb.set_trace()
    uv = xyz_tensor[:2, :][None].copy()     # (1, 2, N)
    image = cv2.imread(image_path).transpose(2, 0, 1)[None]

    # ipdb.set_trace()
    uv_tensor = torch.tensor(uv, dtype=torch.float32)
    image_tensor = torch.tensor(image[:1], dtype=torch.float32)
    color = index(image_tensor, uv_tensor).detach().cpu().numpy()[0].T   # (N, 3) RGB (-1, 1)
    color = color * 0.8 + 0.2

    # ipdb.set_trace()
    save_obj_mesh_with_color(save_path, verts, faces, color)



def binary_fill_from_corner_3D(input, structure=None, output=None, origin=0):
    # now True means outside, False means inside
    mask = np.logical_not(input)

    # mark 8 corners as True
    tmp = np.zeros(mask.shape, bool)
    for xi in [0, tmp.shape[0] - 1]:
        for yi in [0, tmp.shape[1] - 1]:
            for zi in [0, tmp.shape[2] - 1]:
                tmp[xi, yi, zi] = True

    # find connected regions from the 8 corners, to remove empty holes inside the voxels
    inplace = isinstance(output, np.ndarray)
    if inplace:
        ndimage.binary_dilation(tmp, structure=structure, iterations=-1,
                                mask=mask, output=output, border_value=0,
                                origin=origin)
        np.logical_not(output, output)
    else:
        output = ndimage.binary_dilation(tmp,
                                         structure=structure,
                                         iterations=-1,
                                         mask=mask,
                                         border_value=0,
                                         origin=origin)
        np.logical_not(output, output)  # now 1 means inside, 0 means outside

        return output

def voxelization_normalization(verts, useMean=True, useScaling=True):
    """
    normalize the mesh into H [-0.5,0.5]*(1-margin), W/D [-0.333,0.333]*(1-margin)
    """

    # ipdb.set_trace()

    vertsVoxelNorm = copy.deepcopy(verts)
    vertsMean, scaleMin = None, None

    if useMean:
        vertsMean = np.mean(vertsVoxelNorm, axis=0, keepdims=True)  # (1, 3)
        # ipdb.set_trace()
        vertsVoxelNorm -= vertsMean

    xyzMin = np.min(vertsVoxelNorm, axis=0); assert(np.all(xyzMin < 0))
    xyzMax = np.max(vertsVoxelNorm, axis=0); assert(np.all(xyzMax > 0))


    if useScaling:
        scaleArr = np.array([consts.threshWD/abs(xyzMin[0]), consts.threshH/abs(xyzMin[1]), consts.threshWD/abs(xyzMin[2]),
                             consts.threshWD/xyzMax[0], consts.threshH/xyzMax[1], consts.threshWD/xyzMax[2]])
        scaleMin = np.min(scaleArr)
        vertsVoxelNorm *= scaleMin

    return vertsVoxelNorm, vertsMean, scaleMin

def index(feat, uv):
    '''

    :param feat: [B, C, H, W] image features
    :param uv: [B, 2, N] uv coordinates in the image plane, range [-1, 1]
    :return: [B, C, N] image features at the uv coordinates
    '''
    # ipdb.set_trace()
    uv = uv.transpose(1, 2)     # [B, N, 2]
    uv = uv.unsqueeze(2)        # [B, N, 1, 2]
    # NOTE: for newer PyTorch, it seems that training results are degraded due to implementation diff in F.grid_sample
    # for old versions, simply remove the align_corners argument.

    # ipdb.set_trace()
    samples = torch.nn.functional.grid_sample(feat, uv)  # [B, C, N, 1]
    return samples[:, :, :, 0]  # [B, C, N]

def _voxelize(meshPath, voxel_h=128, voxel_w=128):

    VOXELIZER_PATH = "../voxelizer/build/bin"

    assert (os.path.exists(meshPath))
    # XYZ (128, 128, 128) voxels (not DHW, but WHD), 1 inside, 0 outside
    # init. args.
    dim_x, dim_y, dim_z = voxel_w, voxel_h, voxel_w
    new_volume = np.zeros((dim_x, dim_y, dim_z), dtype=np.uint8)  # 1 means inside the object

    # call CUDA code to voxelize the mesh
    call([os.path.join(VOXELIZER_PATH, 'voxelizer'), meshPath, meshPath + '.occvox'])

    # ipdb.set_trace()
    # convert .occvox to np.array
    with open(meshPath + '.occvox', 'r') as fp:
        for line in fp.readlines():
            indices = line.split(' ')
            vx, vy, vz = int(indices[0]), int(indices[1]), int(indices[2])
            new_volume[vx, vy, vz] = 1

    # remove the temp .occvox
    call(['rm', meshPath + '.occvox'])
    #
    # # binary
    voxels = new_volume
    voxels = binary_fill_from_corner_3D(voxels)

    return voxels


if __name__ == '__main__':
    VOXEL_H, VOXEL_W = 128, 128
    VOXEL_SIZE = 1 / VOXEL_H
    mesh_path = "./0000.obj"
    out_path = "./0000_volume.obj"
    image_path = "./000.png"

    voxels = _voxelize(mesh_path, VOXEL_H, VOXEL_W)

    # volume
    save_volume(voxels, out_path, 128, 128, VOXEL_SIZE)

    # voxelization_normalization
    # obj mesh
    out_path = out_path.replace(".obj", "_OBJ.obj")
    voxel_2_obj(voxels, image_path, out_path)
