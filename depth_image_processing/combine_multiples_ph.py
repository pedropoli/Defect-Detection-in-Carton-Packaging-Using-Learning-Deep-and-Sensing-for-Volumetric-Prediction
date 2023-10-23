# This code has changes from its original version: kinect_3d_dev @janbijster

import transforms3d
import numpy as np
import utils
import time
import open3d
import functools
import colorsys
import os, sys
from scipy.spatial import Delaunay


# params
session_name = 'turnobj3'
snapshot_number = 18

num_sensors = 1
depth_images_folder = '../data/depth_scans'
calibration_matrices_folder = '../data/calibration'

filter = [
    [-0.4, 0.4],
    [-0.65, 2],
    [-0.7, 1.5]
]
# filter=[None, None, None]

# script
calibration_matrices_filename = '{}/{}.npz'.format(calibration_matrices_folder, session_name)
transformations = np.load(calibration_matrices_filename)

snapshots_processed = set([])
voxels_combined = None
all_snapshots = []

# def demo_manual_registration():
#     return 

def tetrahedron_volume(a, b, c, d):
    return np.abs(np.einsum('ij,ij->i', a-d, np.cross(b-d, c-d))) / 6

def get_volum(ch):
    simplices = np.column_stack((np.repeat(ch.vertices[0], ch.nsimplex),ch.simplices))
    tets = ch.points[simplices]
    volum = np.sum(tetrahedron_volume(tets[:, 0], tets[:, 1],tets[:, 2], tets[:, 3]))
    return volum

def merge_point_clouds(list_of_pcd):
    merged_pcd = open3d.geometry.PointCloud()
    for i in range(len(list_of_pcd)):
        merged_pcd+=list_of_pcd[i]
    return merged_pcd

def convex_hull(pcd):
    hull, _ = pcd.compute_convex_hull()
    hull_ls = open3d.geometry.LineSet.create_from_triangle_mesh(hull)
    hull_ls.paint_uniform_color((1, 0, 0))
    return hull_ls, hull

def downsample_point_clouds(list_of_pcd, voxel_size=0.02):
    point_cloud_ds = []
    for point in list_of_pcd:
        pcd_down = point.voxel_down_sample(voxel_size=voxel_size)
        point_cloud_ds.append(pcd_down)
    return point_cloud_ds

def get_voxel_size(point_cloud):
    voxel_grid = open3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size=0.05)
    voxel_size = voxel_grid.voxel_size
    return voxel_size

def optimizing_pose_graph(max_correspondence_distance_fine, pose_graph):
    option = open3d.pipelines.registration.GlobalOptimizationOption(
    max_correspondence_distance=max_correspondence_distance_fine,
    edge_prune_threshold=0.25,
    reference_node=0)
    with open3d.utility.VerbosityContextManager(open3d.utility.VerbosityLevel.Debug) as cm:
        open3d.pipelines.registration.global_optimization(
                                            pose_graph,
                                            open3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                                            open3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                                            option)

def pairwise_registration(source, target, max_correspondence_distance_coarse, max_correspondence_distance_fine):
    print("Apply point-to-plane ICP")
    icp_coarse = open3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        open3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = open3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        open3d.pipelines.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = open3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp

def full_registration(pcds, max_correspondence_distance_coarse,max_correspondence_distance_fine):
    pose_graph = open3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(open3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(pcds[source_id], pcds[target_id], max_correspondence_distance_coarse, max_correspondence_distance_fine)
            print("Build o3d.pipelines.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(open3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
                pose_graph.edges.append(open3d.pipelines.registration.PoseGraphEdge(source_id,target_id,transformation_icp,information_icp,
                                                             uncertain=False))
                
            else:  # loop closure case
                pose_graph.edges.append(
                    open3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=True))
    return pose_graph

# def surface_recontruction(list_of_pc):
#     combined_pc = open3d.geometry.PointCloud.concat()
#     return final_point_clouds

def demo_crop_geometry(pointcloud):
    vis = open3d.visualization.VisualizerWithEditing()
    vis.create_window()
    print("Demo for manual geometry cropping")
    print(
        "1) Press 'Y' twice to align geometry with negative direction of y-axis"
    )
    print("2) Press 'K' to lock screen and to switch to selection mode")
    print("3) Drag for rectangle selection,")
    print("   or use ctrl + left click for polygon selection")
    print("4) Press 'C' to get a selected geometry and to save it")
    print("5) Press 'F' to switch to freeview mode")
    vis.add_geometry(pointcloud)
    vis.run()
    cropped_point_cloud = vis.get_cropped_geometry()
    print(type(cropped_point_cloud))
    vis.destroy_window()
    return cropped_point_cloud

def select_polygon(list_of_pc):
    vis = open3d.visualization.VisualizerWithEditing()
    for pcd in list_of_pc:
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()  # user picks points
        vis.destroy_window()
        if cmd == 'e':
            vis.destroy_window()
            break
    return None

def crop_point_clouds(list_of_point_clouds):
    # especificar coordenadas para a regiao de interesse:
    min_bound = []
    max_bound = []
    cropped_point_clouds = []
    for pc in list_of_point_clouds:
        cropped_pc = pc.crop([min_bound, max_bound])
        cropped_point_clouds.append(cropped_pc)
    return cropped_point_clouds

def align_point_clouds(all_pointclouds):
    # especifica um point cloud de referencia:
    reference_pcd = all_pointclouds[0]
    aligned_point_clouds = []
    for i in range(1, len(all_pointclouds)):
        result = open3d.pipelines.registration.registration_icp(
            source=all_pointclouds[i], target=all_pointclouds[i - 1],
            max_correspondence_distance=0.05,  # ajustar ao valor de menor erro
            estimation_method=open3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=open3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=500)
        )
        aligned_point_cloud = all_pointclouds[i].transform(result.transformation)
        aligned_point_clouds.append(aligned_point_cloud)
        print(len(aligned_point_clouds))

    return aligned_point_clouds

def get_array_of_point_cloud(point_cloud):
    points = np.asarray(point_cloud.points)
    print(points)
    return points

def volume_select(visual, point_cloud):
    print("Select a volume by holding 'Shift' key and clicking and dragging the mouse.")
    visual.update_geometry(point_cloud)
    visual.poll_events()
    visual.update_renderer()

def mesh_point_cloud(list_of_aligned_pc):
    merged_mesh = open3d.geometry.TriangleMesh()
    for aligned_point_cloud in list_of_aligned_pc:
        mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_poisson(aligned_point_cloud, depth=9)
        merged_mesh += mesh[0]
    return merged_mesh

# Main loop:
while True:

    # Waiting for new snapshots loop:
    while True:
        # check for new unprocessed snapshots:
        available_run_images = os.listdir('{}/{}'.format(depth_images_folder, session_name))
        snapshots = {}
        process_snapshot = -1
        for image in available_run_images:
            snapshot_index = int(image[:4])
            sensor_index = int(image[5:7])
            if snapshot_index not in snapshots:
                snapshots[snapshot_index] = 1
            else:
                snapshots[snapshot_index] += 1
            if snapshots[snapshot_index] == num_sensors:
                # snapshot is ready to process
                if snapshot_index not in snapshots_processed:
                    # snapshot is not yet processed
                    process_snapshot = snapshot_index
                    break

        if process_snapshot == -1:
            print('no new snapshots ready, waiting...')
            time.sleep(1)
            cmd = input("Press q to quit")
            if cmd == 'q':
                sys.exit()
            if cmd == 'p':
                list_cropped_pc = []
                voxel_size = 0.02
                max_correspondence_distance_coarse = voxel_size * 15
                max_correspondence_distance_fine = voxel_size * 1.5
                for cloud in all_snapshots:
                    cropped_pc = demo_crop_geometry(cloud)
                    list_cropped_pc.append(cropped_pc)

                cropped_pc_ds = downsample_point_clouds(list_cropped_pc)
                alignes_ds_pcd = align_point_clouds(cropped_pc_ds)
                merged_pcd = merge_point_clouds(alignes_ds_pcd)    
                pcd_hull_ls, pcd_hull = convex_hull(merged_pcd)
                # print(type(pcd_hull))
                # print(type(pcd_hull_ls))
                # volume_pcd = get_volum(pcd_hull) 
                # print("Volume do objeto:", volume_pcd)
                open3d.visualization.draw_geometries([merged_pcd, pcd_hull_ls])

                # alignes_crop_pc = align_point_clouds(list_cropped_pc)
                # vis = open3d.visualization.Visualizer()
                # vis.create_window()
                # for cloud in alignes_crop_pc:
                #     vis.add_geometry(cloud)
                # vis.run()
                # vis.destroy_window()
                # vis.add_geometry(all_snapshots)



        else:
            print('new snapshot ready: {}'.format(process_snapshot))
            break

    # new snapshot to process.
    # load depth scans and combine to snapshot model:
    snapshot_pointcloud = None
    for sensor_index in range(num_sensors):
        depth_image_filename = '{}/{}/{:04d}_{:02d}.npy'.format(depth_images_folder, session_name, process_snapshot,
                                                                sensor_index)
        print('Opening image {}...'.format(depth_image_filename))
        depth_image = np.load(depth_image_filename)
        raw_pointcloud = depth_image     # salva pointcloud sem alterações
        pointcloud = utils.depth_image_to_pointcloud(depth_image, filter=filter) 
        if pointcloud is None:
            continue
        pointcloud.paint_uniform_color(utils.item_in_range_color(process_snapshot, 10, True))
        transformation = transformations[transformations.files[sensor_index]]
        pointcloud.transform(transformation)
        snapshot_pointcloud = utils.merge_pointclouds(snapshot_pointcloud, pointcloud)

    snapshots_processed.add(process_snapshot)
    if snapshot_pointcloud is None:
        continue

    # Got a new snapshot pointcloud.
    all_snapshots.append(snapshot_pointcloud)
    list_cropped_pc = []


print(len(all_snapshots))
    # for cloud in all_snapshots:
    #     cropped_pc = demo_crop_geometry(cloud)
    #     list_cropped_pc.append(cropped_pc)
    #     print(len(all_snapshots))

# for pointclouds in all_snapshots:
#     rearranged_pc = voxelization(pointclouds, 19)
#     open3d.draw_geometries(rearranged_pc)
