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
session_name = 'meshtest'
snapshot_number = 31
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
    # icp = open3d.pipelines.registration.registration_icp()
    # icp.set_input_target(reference_pcd)
    aligned_point_clouds = []
    for point_cloud in all_pointclouds:
        # icp.set_input_source(point_cloud)
        # Aplica o algoritmo ICP para alinhar o point cloud fonte com a referencia
        result = open3d.pipelines.registration.registration_icp(
            source=point_cloud, target=reference_pcd,
            max_correspondence_distance=0.01,  # ajustar ao valor de menor erro
            estimation_method=open3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=open3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=8000)
        )
        aligned_point_cloud = point_cloud.transform(result.transformation)
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
                alignes_points = align_point_clouds(all_snapshots)
                meshed_pcs = mesh_point_cloud(alignes_points) 
                open3d.visualization.draw_geometries([meshed_pcs])

                # rearranged_voxels = voxelization(pointcloud)
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
