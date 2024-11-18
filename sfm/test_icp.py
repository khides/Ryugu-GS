import cv2
import numpy as np

def calculate_rotation_matrix_opencv_correct(camera_direction):
    # カメラの前方方向のベクトル（z軸）を定義
    z_axis = np.array([0, 0, 1])

    # カメラ方向が z 軸と異なる場合、回転行列 R を計算
    if not np.allclose(camera_direction, z_axis):
        rotation_axis = np.cross(z_axis, camera_direction)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)  # 正規化
        rotation_angle = np.arccos(np.dot(z_axis, camera_direction) / (np.linalg.norm(z_axis) * np.linalg.norm(camera_direction)))
        R_matrix, _ = cv2.Rodrigues(rotation_axis * rotation_angle)
    else:
        R_matrix = np.eye(3)  # z軸と一致する場合は単位行列

    return R_matrix

# 使用例
camera_direction = np.array([0.22361765, -0.8467474, 0.48271522])  # 任意のカメラ方向
R = calculate_rotation_matrix_opencv_correct(camera_direction)

# カメラ方向を復元
restored_camera_direction = R.T @ np.array([0, 0, 1])
print("Camera Direction:", camera_direction)
print("Restored Camera Direction:", restored_camera_direction)
