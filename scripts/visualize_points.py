import sys

import trimesh
import open3d as o3d
import numpy as np
import torch

def visualize_points(points, size=0.04):

    point_visual = []
    for i in range(len(points)):
        point = points[i]
        if point.shape[0] == 0:
            continue
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(point[:,0:3])
        pcd_o3d.paint_uniform_color(get_color(i))
        if point.shape[1]==6:
            pcd_o3d.normals = o3d.utility.Vector3dVector(point[:,3:6])

        # if i==0:
        o3d.io.write_point_cloud('./0808_box_{}.ply'.format(i), pcd_o3d)
        point_visual.append(pcd_o3d)
    # o3d.visualization.draw_geometries(point_visual)
    a = 1 

def get_color(i):
    if i == 0:
        color = [1, 0, 0]
    elif i == 1:
        color = [0, 1, 0]
    elif i == 2:
        color = [0, 0, 1]
    elif i == 3:
        color = [1, 1, 0]
    elif i == 4:
        color = [0, 1, 1]
    elif i == 5:
        color = [1, 0, 1]
    elif i == 6:
        color = [0.5, 0.5, 0.5]
    elif i == 7:
        color = [0, 0, 0]
    else:
        print("the color num is out of 8.")
    return color