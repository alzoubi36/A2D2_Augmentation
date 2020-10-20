
from pyglet.gl import *
import pywavefront
import open3d as o3
from pywavefront import visualization

obj = pywavefront.Wavefront('./3D-Models/Porsche Panamera Turbo.obj')
pc = obj.vertices
pcd = o3.geometry.PointCloud()
pcd.points = o3.utility.Vector3dVector(pc)
o3.visualization.draw_geometries([pcd])