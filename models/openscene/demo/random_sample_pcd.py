import open3d
import numpy as np
import os
import sys
import torch
sys.path.append('/home/admin/Projects/LL3DA/')

source_pcd = torch.load('/mnt/nfs/share/Adaptive/openscene_dense_fts_distill/scene0204_00_xyz.pt').cpu().numpy()
source_color = torch.load('/mnt/nfs/share/Adaptive/openscene_dense_fts_distill/scene0204_00_color.pt').cpu().numpy()
source_fts = torch.load('/mnt/nfs/share/Adaptive/openscene_dense_fts_distill/scene0022_00_dense_fts.pt').cpu().numpy()

pcd = open3d.geometry.PointCloud()
pcd.points = open3d.utility.Vector3dVector(source_pcd)
pcd.colors = open3d.utility.Vector3dVector(source_color)

# open3d.io.write_triangle_mesh("/home/admin/Projects/LL3DA/src/openscene/demo/test_pointclouds/scene0204_00.ply", pcd)
np.save("/home/admin/Projects/LL3DA/src/openscene/demo/test_features/scene0022_00_fts.npy", source_fts)