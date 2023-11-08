import open3d as o3d
from PIL import Image
import numpy as np
import os 

# Load a triangle mesh from a file (replace 'your_mesh.ply' with your mesh file)
mesh = o3d.io.read_triangle_mesh("C:/Users/User/Desktop/kinect_3d_dev-master/saved_meshes/type1.ply")

# Customize the visual properties
visualizer = o3d.visualization.Visualizer()
visualizer.create_window()

image = "C:/Users/User/Downloads/package1_1.jpeg"
pil_image = Image.open(image)
o3d_image = o3d.geometry.Image(np.array(pil_image))

# texture = o3d.geometry.ImageTexture()
# texture.set_image(o3d_image)
visualizer.add_geometry(o3d_image)
line_set_mesh = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
visualizer.add_geometry(line_set_mesh)

visualizer.update_geometry(line_set_mesh)
visualizer.poll_events()
visualizer.update_renderer()

# Keep the window open for interaction
visualizer.run()
visualizer.destroy_window()