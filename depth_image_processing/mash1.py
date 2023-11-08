import open3d as o3d

# Load the first PLY mesh
mesh1 = o3d.io.read_triangle_mesh("C:/Users/User/Desktop/kinect_3d_dev-master/saved_meshes/turnobj2.ply")

# Load the second PLY mesh
mesh2 = o3d.io.read_triangle_mesh("C:/Users/User/Desktop/kinect_3d_dev-master/saved_meshes/turnobj3.ply")

# Create a list of the meshes you want to visualize together

translation_vector = [0.2, 0.2, 0.1]  # Adjust the values to specify the desired translation
mesh2.translate(translation_vector)

meshes = [mesh1, mesh2]

# Visualize the meshes in a single window
o3d.visualization.draw_geometries(meshes)