import bpy
import os
import numpy as np
import json

v_views = 2
h_views = 25
RESOLUTION = 1024
RESULTS_PATH = os.path.expanduser('D:/Users/taiko/Ryugu-GS/data_input/blender')
if not os.path.exists(RESULTS_PATH):
    os.makedirs(os.path.join(RESULTS_PATH, 'train'))

# Render Setting
scene = bpy.context.scene
scene.render.resolution_x = RESOLUTION
scene.render.resolution_y = RESOLUTION
scene.render.image_settings.file_format = str('PNG')
scene.render.film_transparent = True
scene.render.use_persistent_data = True

# Camera Setting
cam = scene.objects['Camera']
cam.location = (0, 40.0, 0.0)

# Camera Parent
camera_parent = bpy.data.objects.new("CameraParent", None)
camera_parent.location = (0, 0, 0)
scene.collection.objects.link(camera_parent)
cam.parent = camera_parent

# Camera Constraint
cam_constraint = cam.constraints.new(type='TRACK_TO')
cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
cam_constraint.up_axis = 'UP_Y'
cam_constraint.target = camera_parent

# Function to calculate camera parameters
def calculate_camera_params(camera, resolution_x, resolution_y):
    focal_length = camera.lens  # 焦点距離（mm）
    sensor_width = camera.sensor_width  # センサー幅（mm）
    sensor_height = camera.sensor_height  # センサー高さ（mm）
    
    # Fit sensor calculation
    sensor_fit = camera.sensor_fit
    if sensor_fit == 'VERTICAL':
        fx = (focal_length / sensor_height) * resolution_y
        fy = fx
    else:
        fx = (focal_length / sensor_width) * resolution_x
        fy = fx
    
    # Principal point
    cx = resolution_x / 2
    cy = resolution_y / 2

    return fx, fy, cx, cy

# Calculate camera parameters
fx, fy, cx, cy = calculate_camera_params(cam.data, RESOLUTION, RESOLUTION)

# Data to store in JSON file
out_data = {
    'camera_angle_x': cam.data.angle_x,
    'camera_params': {
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy
    },
    'frames': []
}

for i in range(v_views):
    for j in range(h_views):

        # Camera Rotation
        camera_parent.rotation_euler = np.array([np.pi / 2 / v_views * i, 0, 2 * np.pi / h_views * j])
        # Rendering
        filename = 'r_{0:03d}'.format(h_views * i + j)
        scene.render.filepath = os.path.join(RESULTS_PATH, 'train', filename)
        bpy.ops.render.render(write_still=True)

        # add frame data to JSON file
        def listify_matrix(matrix):
            matrix_list = []
            for row in matrix:
                matrix_list.append(list(row))
            return matrix_list
        frame_data = {
            'file_path': './train/' + filename,
            'transform_matrix': listify_matrix(cam.matrix_world)
        }
        out_data['frames'].append(frame_data)

with open(os.path.join(RESULTS_PATH, 'transforms_train.json'), 'w') as out_file:
    json.dump(out_data, out_file, indent=4)
