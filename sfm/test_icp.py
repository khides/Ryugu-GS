import cv2
import numpy as np

def calculate_rotation_matrix_opencv_correct(camera_direction):
    z_axis = np.array([0, 0, 1])

    if not np.allclose(camera_direction, z_axis):
        rotation_axis = np.cross(z_axis, camera_direction)
        if np.linalg.norm(rotation_axis) > 0:  # 回転軸の正規化
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        rotation_angle = np.arccos(np.dot(z_axis, camera_direction) /
                                   (np.linalg.norm(z_axis) * np.linalg.norm(camera_direction)))
        R_matrix, _ = cv2.Rodrigues(rotation_axis * rotation_angle)
    else:
        R_matrix = np.eye(3)

    return R_matrix

camera_direction = np.array([0.22361765, -0.8467474, 0.48271522])
R = calculate_rotation_matrix_opencv_correct(camera_direction)

# カメラ方向を復元
restored_camera_direction = R @ np.array([0, 0, 1])
print("Camera Direction:", camera_direction)
print("Restored Camera Direction:", restored_camera_direction)
