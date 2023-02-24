import argparse
import os, sys
import cv2
import ipdb
import trimesh
import numpy as np
import random
import math
import random
from tqdm import tqdm
from lib.renderer.mesh import load_scan
from lib.renderer.gl.prt_render import PRTRender

def render_subject(subject, dataset, save_folder, size):
    scale = 100.0
    up_axis = 1
    smpl_type = "smplx"

    mesh_file = os.path.join(f'./data/{dataset}/scans/{subject}', f'{subject}.obj')
    smplx_file = f'./data/{dataset}/{smpl_type}/{subject}.obj'
    tex_file = f'./data/{dataset}/scans/{subject}/material0.jpeg'
    fit_file = f'./data/{dataset}/{smpl_type}/{subject}.pkl'

    vertices, faces, normals, faces_normals, textures, face_textures = load_scan(
        mesh_file, with_normal=True, with_texture=True
    )

    ipdb.set_trace()

    # center
    scan_scale = 1.8 / (vertices.max(0)[up_axis] - vertices.min(0)[up_axis])
    rescale_fitted_body, joints = load_fit_body(
        fit_file, scale, smpl_type=smpl_type, smpl_gender='male'
    )

    os.makedirs(os.path.dirname(smplx_file), exist_ok=True)
    trimesh.Trimesh(rescale_fitted_body.vertices / scale,
                    rescale_fitted_body.faces).export(smplx_file)

    vertices *= scale
    vmin = vertices.min(0)
    vmax = vertices.max(0)
    vmed = joints[0]
    vmed[up_axis] = 0.5 * (vmax[up_axis] + vmin[up_axis])

    rndr_smpl = ColorRender(width=size, height=size, egl=egl)
    rndr_smpl.set_mesh(
        rescale_fitted_body.vertices, rescale_fitted_body.faces, rescale_fitted_body.vertices,
        rescale_fitted_body.vertex_normals
    )
    rndr_smpl.set_norm_mat(scan_scale, vmed)

    # camera
    cam = Camera(width=size, height=size)
    cam.ortho_ratio = 0.4 * (512 / size)

    prt, face_prt = prt_util.computePRT(mesh_file, scale, 10, 2)
    rndr = PRTRender(width=size, height=size, ms_rate=16, egl=True)

    # texture
    texture_image = cv2.cvtColor(cv2.imread(tex_file), cv2.COLOR_BGR2RGB)

    tan, bitan = compute_tangent(normals)
    rndr.set_norm_mat(scan_scale, vmed)
    rndr.set_mesh(
        vertices,
        faces,
        normals,
        faces_normals,
        textures,
        face_textures,
        prt,
        face_prt,
        tan,
        bitan,
        np.zeros((vertices.shape[0], 3)),
    )
    rndr.set_albedo(texture_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', '--dataset', type=str, default="thuman2", help='dataset name')
    parser.add_argument('-out_dir', '--out_dir', type=str, default="./result", help='output dir')
    parser.add_argument('-size', '--size', type=int, default=512, help='render size')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    print(
        f"Start Rendering {args.dataset}, {args.size}x{args.size} size."
    )

    current_out_dir = args.out_dir
    os.makedirs(current_out_dir, exist_ok=True)
    print(f"Output dir: {current_out_dir}")

    subjects = np.loadtxt(f"./data/{args.dataset}/all.txt", dtype=str)
    for subject in subjects:
        render_subject(subject=subject,
                       dataset=args.dataset,
                       save_folder=current_out_dir,
                       size=args.size)

    print("Finish Rendering.")
