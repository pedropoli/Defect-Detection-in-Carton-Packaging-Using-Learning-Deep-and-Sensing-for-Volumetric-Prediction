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
import pathlib

# params
session_name = 'type2_scans_20'
snapshot_number = 20

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

# def tetrahedron_volume(a, b, c, d):
#     return np.abs(np.einsum('ij,ij->i', a-d, np.cross(b-d, c-d))) / 6

def save_triangle_mesh(mesh):
    file_path = "C:/Users/User/Desktop/kinect_3d_dev-master/saved_meshes/" + session_name + ".ply"
    open3d.io.write_triangle_mesh(file_path, mesh)
    print("File saved")
    return None

def get_volum(closed_line_set):
    # surface_mesh = open3d.geometry.LineSet.create_from_triangle_mesh(closed_line_set)
    volume = closed_line_set.get_volume()
    # print(closed_line_set.is_watertight())
    return abs(volume * 1000000)

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

def align_point_clouds(all_pointclouds, k):
    # especifica um point cloud de referencia:
    reference_pcd = all_pointclouds[0]
    registered_point_clouds = []
    inlier_rms = []
    for i in range(1, len(all_pointclouds)):
        source_cloud = all_pointclouds[i]
        result = open3d.pipelines.registration.registration_icp(
            source=source_cloud, target=all_pointclouds[0],
            max_correspondence_distance=0.05,  # ajustar ao valor de menor erro
            estimation_method=open3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=open3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=k)
        )
        aligned_point_cloud = all_pointclouds[i].transform(result.transformation)
        registered_point_clouds.append(aligned_point_cloud)
        inlier_rms.append(result.inlier_rmse)
    print(sum(inlier_rms)/(len(all_pointclouds)-1))
    return registered_point_clouds

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
                # alignes_ds_pcd_0002 = align_point_clouds(cropped_pc_ds,3)
                # alignes_ds_pcd_0001 = align_point_clouds(cropped_pc_ds,4)
                # alignes_ds_pcd_0000 = align_point_clouds(cropped_pc_ds,5)
                
                # alignes_ds_pcd = align_point_clouds(cropped_pc_ds)
                alignes_ds_pcd_0 = align_point_clouds(cropped_pc_ds,2)
                alignes_ds_pcd_1 = align_point_clouds(cropped_pc_ds,3)
                alignes_ds_pcd_2 = align_point_clouds(cropped_pc_ds,4)
                alignes_ds_pcd_3 = align_point_clouds(cropped_pc_ds,5)
                alignes_ds_pcd_4 = align_point_clouds(cropped_pc_ds,6)
                alignes_ds_pcd_5 = align_point_clouds(cropped_pc_ds,7)
                alignes_ds_pcd_6 = align_point_clouds(cropped_pc_ds,8)
                alignes_ds_pcd_7 = align_point_clouds(cropped_pc_ds,9)
                alignes_ds_pcd_8 = align_point_clouds(cropped_pc_ds,10)
                # alignes_ds_pcd_9 = align_point_clouds(cropped_pc_ds,50)
                # alignes_ds_pcd_10 = align_point_clouds(cropped_pc_ds,100)
                # alignes_ds_pcd_20 = align_point_clouds(cropped_pc_ds,200)
                # alignes_ds_pcd_30 = align_point_clouds(cropped_pc_ds,300)
                # alignes_ds_pcd_40 = align_point_clouds(cropped_pc_ds,400)
                # alignes_ds_pcd_50 = align_point_clouds(cropped_pc_ds,500)


                # alignes_ds_pcd_60 = align_point_clouds(cropped_pc_ds,1500)
                # alignes_ds_pcd_70 = align_point_clouds(cropped_pc_ds,1600)
                # alignes_ds_pcd_80 = align_point_clouds(cropped_pc_ds,5000)
                # alignes_ds_pcd_90 = align_point_clouds(cropped_pc_ds,10000)
                # alignes_ds_pcd_100 = align_point_clouds(cropped_pc_ds,15000)

                
                # alignes_ds_pcd_70 = align_point_clouds(cropped_pc_ds,0.7)
                # alignes_ds_pcd_80 = align_point_clouds(cropped_pc_ds,0.8)
                # alignes_ds_pcd_90 = align_point_clouds(cropped_pc_ds,0.9)
                # alignes_ds_pcd_100 = align_point_clouds(cropped_pc_ds,1.0)





                # merged_pcd = merge_point_clouds(alignes_ds_pcd)
                # mesh_pcd = mesh_point_cloud(alignes_ds_pcd)
                merged_pcd_0 = merge_point_clouds(alignes_ds_pcd_0)
                merged_pcd_1 = merge_point_clouds(alignes_ds_pcd_1)
                merged_pcd_2 = merge_point_clouds(alignes_ds_pcd_2)
                merged_pcd_3 = merge_point_clouds(alignes_ds_pcd_3)
                merged_pcd_4 = merge_point_clouds(alignes_ds_pcd_4)
                merged_pcd_5 = merge_point_clouds(alignes_ds_pcd_5)
                merged_pcd_6 = merge_point_clouds(alignes_ds_pcd_6)
                merged_pcd_7 = merge_point_clouds(alignes_ds_pcd_7)
                merged_pcd_8 = merge_point_clouds(alignes_ds_pcd_8)
                # merged_pcd_9 = merge_point_clouds(alignes_ds_pcd_9)
                # merged_pcd_10 = merge_point_clouds(alignes_ds_pcd_10)
                # merged_pcd_20 = merge_point_clouds(alignes_ds_pcd_20)
                # merged_pcd_30 = merge_point_clouds(alignes_ds_pcd_30)
                # merged_pcd_40 = merge_point_clouds(alignes_ds_pcd_40)
                # merged_pcd_50 = merge_point_clouds(alignes_ds_pcd_50)




                # merged_list = [merged_pcd_1, merged_pcd_2, merged_pcd_3, merged_pcd_4, merged_pcd_5, merged_pcd_6, merged_pcd_7, merged_pcd_8, merged_pcd_9]
                pcd_hull_ls0, pcd_hull0 = convex_hull(merged_pcd_0)
                pcd_hull_ls1, pcd_hull1 = convex_hull(merged_pcd_1)
                pcd_hull_ls2, pcd_hull2 = convex_hull(merged_pcd_2)
                pcd_hull_ls3, pcd_hull3 = convex_hull(merged_pcd_3)
                pcd_hull_ls4, pcd_hull4 = convex_hull(merged_pcd_4)
                pcd_hull_ls5, pcd_hull5 = convex_hull(merged_pcd_5)
                pcd_hull_ls6, pcd_hull6 = convex_hull(merged_pcd_6)
                pcd_hull_ls7, pcd_hull7 = convex_hull(merged_pcd_7)
                pcd_hull_ls8, pcd_hull8 = convex_hull(merged_pcd_8)
                # pcd_hull_ls9, pcd_hull9 = convex_hull(merged_pcd_9)    
                # pcd_hull_ls10, pcd_hull10 = convex_hull(merged_pcd_10)
                # pcd_hull_ls20, pcd_hull20 = convex_hull(merged_pcd_20)
                # pcd_hull_ls30, pcd_hull30 = convex_hull(merged_pcd_30)
                # pcd_hull_ls40, pcd_hull40 = convex_hull(merged_pcd_40)
                # pcd_hull_ls50, pcd_hull50 = convex_hull(merged_pcd_50)   




                # pcd_hull_list = [pcd_hull_ls1, pcd_hull_ls2, pcd_hull_ls3, pcd_hull_ls4, pcd_hull_ls5, pcd_hull_ls6, pcd_hull_ls7, pcd_hull_ls8, pcd_hull_ls9]
                print("----------------------")
                # print(merged_pcd_1)
                # print(merged_pcd_2)
                print("----------------------")
                volume0 = get_volum(pcd_hull0)
                volume1 = get_volum(pcd_hull1)
                volume2 = get_volum(pcd_hull2)
                volume3 = get_volum(pcd_hull3)
                volume4 = get_volum(pcd_hull4)
                volume5 = get_volum(pcd_hull5)
                volume6 = get_volum(pcd_hull6)
                volume7 = get_volum(pcd_hull7)
                volume8 = get_volum(pcd_hull8)
                # volume9 = get_volum(pcd_hull9)
                # volume10 = get_volum(pcd_hull10)
                # volume20 = get_volum(pcd_hull20)
                # volume30 = get_volum(pcd_hull30)
                # volume40 = get_volum(pcd_hull40)
                # volume50 = get_volum(pcd_hull50)

                print("Volume do objeto:", volume0)
                print("Volume do objeto:", volume1)
                print("Volume do objeto2:", volume2)
                print("Volume do objeto3:", volume3)
                print("Volume do objeto4:", volume4)
                print("Volume do objeto5:", volume5)
                print("Volume do objeto6:", volume6)
                print("Volume do objeto7:", volume7)
                print("Volume do objeto8:", volume8)
                # print("Volume do objeto9:", volume9)
                # print("Volume do objeto6:", volume10)
                # print("Volume do objeto7:", volume20)
                # print("Volume do objeto8:", volume30)
                # print("Volume do objeto9:", volume40)
                # print("Volume do objeto6:", volume50)

                # for merged_list, pcd in zip(merged_list, pcd_hull_list):
                #     vis = open3d.visualization.Visualizer()
                #     vis.create_window()    
                #     vis.add_geometry(merged_list)
                #     vis.add_geometry(pcd)
                #     vis.run()
                #     vis.destroy_window()

                # merged_pcd = merge_point_clouds(alignes_ds_pcd)    
                # pcd_hull_ls, pcd_hull = convex_hull(merged_pcd)
                # print(type(pcd_hull))
                # print(type(pcd_hull))
                # print(type(pcd_hull_ls))
                # print("Volume do objeto:", volume_pcd)
                # open3d.visualization.draw_geometries([merged_pcd, pcd_hull_ls])
                # save_triangle_mesh(pcd_hull)
                
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
