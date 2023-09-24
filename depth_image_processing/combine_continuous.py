# This code has changes from its original version: kinect_3d_dev @janbijster

import transforms3d
import numpy as np
import utils
import time
import open3d
import functools
import colorsys
import os, sys


# params
session_name = 'testvoxel2'
snapshot_number = 59
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


def voxelization(point_cloud):

    # voxel_grid=open3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size=0.0009)
    voxel_size = 0.0009
    # point_cloud=open3d.geometry.point_cloud_crop(point_cloud, min_bound, max_bound)
    # voxel_grid = open3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(point_cloud, voxel_size, min_bound, max_bound)
    voxel_grid=open3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size=0.0009)
    # vis = open3d.visualization.Visualizer()
    vis = open3d.visualization.VisualizerWithEditing()
    # vis = open3d.visualization.draw_geometries_with_editing(voxel_grid,window_name='Box Visualize', width=800, height=600)
    vis.create_window(window_name='Box Visualize', width=800, height=600)
    # point_cloud_cropped = crop_pc(point_cloud)
    coord_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    # vis.add_geometry(point_cloud)
    # vis.add_geometry(coord_frame)
    open3d.visualization.draw_geometries([point_cloud, coord_frame])
    # vis.add_geometry(voxel_grid)
    vis.run()
    vis.destroy_window()
    print(vis.get_picked_points())
    return voxel_grid

def pick_points(pcd):
    vis = open3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()

def crop_pc(point_cloud):
    min_bound = [-0.079, 0.51, 2.3]
    max_bound = [0.04, 0.66, 2.1]
    print("HERE")
    bbox = open3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    cropped_point_cloud = point_cloud.crop(bbox)
    # cropped_point_cloud = open3d.geometry.crop_point_cloud(point_cloud, min_bound, max_bound)
    return cropped_point_cloud

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
        # print(process_snapshot, snapshot_number)
        # if process_snapshot == snapshot_number:
        #     done = True
        if process_snapshot == -1:
            print('no new snapshots ready, waiting...')
            time.sleep(1)
            cmd = input("Press q to quit")
            if cmd == 'q':
                sys.exit()
            if cmd == 'p':
                rearranged_voxels = voxelization(pointcloud)
                    # get_volumetric(rearranged_voxels)

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
        pointcloud = utils.depth_image_to_pointcloud(depth_image, filter=filter)    #
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
    # open3d.visualization.draw_geometries(all_snapshots)


    # Voxelization Function: align poin clouds in one 3D image



# for pointclouds in all_snapshots:
#     rearranged_pc = voxelization(pointclouds, 19)
#     open3d.draw_geometries(rearranged_pc)
