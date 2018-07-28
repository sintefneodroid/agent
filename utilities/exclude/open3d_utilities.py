import numpy as np
import open3d

if __name__ == "__main__":

  print("Load a ply point cloud, print it, and render it")
  pcd = open3d.read_point_cloud('/home/heider/Datasets/pointclouds/office.ply')
  print(pcd)
  print(np.asarray(pcd.points))
  # open3d.draw_geometries([pcd])

  print("Downsample the point cloud with a voxel of 0.05")
  downsampled = open3d.voxel_down_sample(pcd, voxel_size=0.1)
  # open3d.draw_geometries([downpcd])

  print("Recompute the normal of the downsampled point cloud")
  open3d.estimate_normals(downsampled, search_param=open3d.KDTreeSearchParamHybrid(
      radius=0.1, max_nn=30))
  open3d.draw_geometries([downsampled])
