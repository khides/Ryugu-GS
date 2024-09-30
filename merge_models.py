
import numpy as np
import sqlite3
import struct
import cv2
import quaternion
from scipy.spatial import cKDTree
from sklearn.cluster import MiniBatchKMeans
import logging
from matplotlib import pyplot as plt
import os
import shutil
from typing import List, Tuple, Any

class Log():
    def __init__(self, filename) -> None:
        logging.basicConfig(filename=filename, level=logging.DEBUG, 
                            format='%(asctime)s %(levelname)s:%(message)s', filemode='w')
    
    def info(self, message: str) -> None:
        print(message)
        logging.info(message)
    
    def warning(self, message: str) -> None:
        print(message)
        logging.warning(message)
    
    def error(self, message: str) -> None:
        print(message)
        logging.error(message)
        
logger = Log("camera_pose_estimation.log")


def extract_descriptors_list_from_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT image_id, data FROM descriptors")
    descriptors = cursor.fetchall()
    
    all_descriptors = []
    image_feature_start_indices = {}
    current_index = 0
    for image_id, desc in descriptors:
        desc_array = np.frombuffer(desc, dtype=np.uint8).reshape(-1, 128).astype(np.float32)
        all_descriptors.append(desc_array)
        image_feature_start_indices[image_id] = current_index
        current_index += desc_array.shape[0]

    conn.close()
    all_descriptors = np.vstack(all_descriptors)
    # logger.info(f"Extracted feature start indices: {image_feature_start_indices}")
    return all_descriptors, image_feature_start_indices

def extract_descriptors_dict_from_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT image_id, data FROM descriptors")
    descriptors = cursor.fetchall()
    
    all_descriptors = {}
    for image_id, desc in descriptors:
        desc_array = np.frombuffer(desc, dtype=np.uint8).reshape(-1, 128).astype(np.float32)
        descriptors = all_descriptors.setdefault(image_id, [])
        descriptors.append(desc_array)
        all_descriptors[image_id] = descriptors

    conn.close()
    return all_descriptors

def extract_keypoints_dict_from_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT image_id, data FROM keypoints")
    keypoints = cursor.fetchall()
    
    all_keypoints = {}
    image_feature_start_indices = {}
    current_index = 0
    for image_id, keypoint in keypoints:
        keypoint_array = np.frombuffer(keypoint, dtype=np.float32).reshape(-1, 6)
        keypoints = all_keypoints.setdefault(image_id, [])
        keypoints.append(keypoint_array)
        all_keypoints[image_id] = keypoints

    conn.close()
    return all_keypoints


# 回転行列をクォータニオンに変換する関数
def quaternion_from_matrix(matrix) -> List[Any]:
    q = quaternion.from_rotation_matrix(matrix)
    return [q.x, q.y, q.z, q.w]

# Quaternionを回転行列に変換する関数
def quaternion_to_rotation_matrix(q) -> np.ndarray:
    q = np.quaternion(q[0], q[1], q[2], q[3])
    return quaternion.as_rotation_matrix(q)


# バイナリファイルを読み込む関数 (images.bin)
def read_images_bin(file_path) -> dict:
    images = {}
    with open(file_path, "rb") as f:
        num_reg_images = struct.unpack("Q", f.read(8))[0]
        for _ in range(num_reg_images):
            image_id = struct.unpack("I", f.read(4))[0]
            qvec = struct.unpack("4d", f.read(32))
            tvec = struct.unpack("3d", f.read(24))
            camera_id = struct.unpack("I", f.read(4))[0]
            name = ""
            while True:
                char = struct.unpack("c", f.read(1))[0]
                if char == b'\x00':
                    break
                name += char.decode("utf-8")
            num_points2D = struct.unpack("Q", f.read(8))[0]
            xys = []
            point3D_ids = []
            for _ in range(num_points2D):
                xys.append(struct.unpack("2d", f.read(16)))
                point3D_ids.append(struct.unpack("Q", f.read(8))[0])
            images[image_id] = {
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": name,
                "xys": xys,
                "point3D_ids": point3D_ids
            }
    return images

def plot_camera_poses(camera_positions:np.ndarray, camera_directions:np.ndarray, ax: plt.Axes, label: str, color: str = 'r') -> None:
    cur = 0
    for i in range(len(camera_positions)):
        camera_position = camera_positions[i]
        camera_direction = camera_directions[i]
        if cur == 0:
            ax.quiver(camera_position[0], camera_position[1], camera_position[2],
                      camera_direction[0], camera_direction[1], camera_direction[2],
                      length=0.5, color=color, arrow_length_ratio=0.5, label=label)
            cur +=1
        else:
            ax.quiver(camera_position[0], camera_position[1], camera_position[2],
                      camera_direction[0], camera_direction[1], camera_direction[2],
                      length=0.5, color=color, arrow_length_ratio=0.5)

def extract_camera_poses_from_images(images: dict) -> Tuple[np.ndarray, np.ndarray]:
    camera_positions = []
    camera_directions = []
    for image_id, data in images.items():
        R = quaternion_to_rotation_matrix(data['qvec'])
        t = np.array(data['tvec']).reshape((3, 1))
        camera_position = -R.T @ t
        camera_direction = R.T @ np.array([0, 0, 1])
        camera_positions.append(camera_position)
        camera_directions.append(camera_direction)
    camera_positions = np.array(camera_positions)
    camera_directions = np.array(camera_directions)
    return camera_positions, camera_directions

def read_points3d_bin(file_path):
    points3d = {}
    with open(file_path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_points):
            point_id = struct.unpack("<Q", f.read(8))[0]
            xyz = struct.unpack("<3d", f.read(24))
            rgb = struct.unpack("<3B", f.read(3))
            error = struct.unpack("<d", f.read(8))[0]
            track_length = struct.unpack("<Q", f.read(8))[0]
            track = []
            for _ in range(track_length):
                image_id = struct.unpack("<i", f.read(4))[0]
                point2D_idx = struct.unpack("<i", f.read(4))[0]
                track.append((image_id, point2D_idx))
            points3d[point_id] = {
                "xyz": xyz,
                "rgb": rgb,
                "error": error,
                "track": track,
            }
    return points3d

# 特徴点のマッチングを行う関数
def match_descriptors(descriptors1, descriptors2, normType=cv2.NORM_L2, crossCheck=True, distance_threshold=0.8):
    """
    特徴点のマッチングのパラメータ:
    - normType: 特徴点の距離の計算に使用するノルムの種類
    - crossCheck: クロスチェックを行うかどうか
    - distance_threshold: マッチングの距離の閾値
    """
    logger.info("Match Features...")
    # descriptors1 = normalize_descriptors(descriptors1.astype(np.float32))
    # descriptors2 = normalize_descriptors(descriptors2.astype(np.float32)) 
    descriptors1 = descriptors1.astype(np.float32)
    descriptors2 = descriptors2.astype(np.float32)
    
    # BFMのインスタンスを作成
    # bf = cv2.BFMatcher(normType=normType, crossCheck=crossCheck)
    # matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    # good_matches = []
    # if distance_threshold is not None:
    #     # good_matches = [m for m in matches if m.distance < distance_threshold* max([m.distance for m in matches])]
    #     for m, n in matches:
    #         if m.distance < distance_threshold * n.distance:
    #             good_matches.append(m)
    # else :
    #     good_matches = [m for m in matches]
    # logger.info(matches[0].distance)
    
    
    # FLANNのインデックスパラメータと検索パラメータ
    index_params = dict(algorithm=1, trees=50)  # 1はFLANN_INDEX_KDTREE
    search_params = dict(checks=500)  # チェックの回数
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)    
    logger.info(f"Found {len(matches)} ")
    good_matches = []
    # 距離の閾値でフィルタリング
    for m, n in matches:
        if m.distance < distance_threshold * n.distance:
            good_matches.append(m)
    good_matches = sorted(good_matches, key=lambda x: x.distance)
    logger.info(f"Found {len(good_matches)} good matches")
    return good_matches

def feature_id_to_points3d_id(points3d, image_feature_start_indices):
    feature_id_to_point3d_id = {}
    for point3d_id, point_data in points3d.items():
        for image_id, feature_id in point_data['track']:
            global_feature_id = image_feature_start_indices[image_id] + feature_id
            feature_id_to_point3d_id[global_feature_id] = point3d_id
    return feature_id_to_point3d_id

def solve_pnp(object_points, image_points, camera_matrix, dist_coeffs):
    """
    PnP問題を解く関数
    - object_points: 3D空間上の点の座標
    - image_points: 画像上の点の座標
    - camera_matrix: カメラ行列
    - dist_coeffs: 歪み係数
    """
    _, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image_points, camera_matrix, dist_coeffs)
    R, _ = cv2.Rodrigues(rvec)
    camera_position = -R.T @ tvec
    camera_direction = R.T @ np.array([0, 0, 1])
    return camera_position, camera_direction

def extract_points_from_matches(matches,base_points3d, keypoints, feature_id_to_points3d_id_dict):
    object_points = []
    image_points = []
    for m in matches:
        point3d_id = feature_id_to_points3d_id_dict.get(m.trainIdx, None)
        if point3d_id is not None:
            object_points.append(base_points3d[point3d_id]['xyz'])
            image_points.append(keypoints[m.queryIdx][:2])
    object_points = np.array(object_points, dtype=np.float32)
    image_points = np.array(image_points, dtype=np.float32)
    logger.info(f"found {len(object_points)} object")
    return object_points, image_points
        

def main():
    ### camera parameters
    fx, fy, cx, cy = 9231, 9231, 512, 512
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1))  # 歪みなしの場合

    ### source data
    #### base model
    base_db_path = './Ryugu_Data/Ryugu_CLAHE/database.db'
    base_points3d_path = './Ryugu_Data/Ryugu_CLAHE/sparse/0/points3D.bin'
    base_images_bin_path = './Ryugu_Data/Ryugu_CLAHE/sparse/0/images.bin'
    #### additional model
    additional_db_path = './Ryugu_Data/Ryugu_CLAHE/database4.db'
    additional_images_path = './Ryugu_Data/Ryugu_CLAHE/Input4'
    additional_images_bin_path = './Ryugu_Data/Ryugu_CLAHE/sparse/c/images.bin'
    
    ### plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ### ベースモデルの読み込み
    base_images = read_images_bin(base_images_bin_path)
    base_points3d = read_points3d_bin(base_points3d_path)
    base_descriptors, base_image_feature_start_indices = extract_descriptors_list_from_db(base_db_path)
    feature_id_to_points3d_id_dict = feature_id_to_points3d_id(base_points3d, base_image_feature_start_indices)
    base_camera_pose= extract_camera_poses_from_images(base_images)
    #### ベースモデルのカメラ位置をプロット
    plot_camera_poses(base_camera_pose[0], base_camera_pose[1], ax, "BOX-A", 'r')
    
    ### 追加モデルの読み込み
    additional_images_bin = read_images_bin(additional_images_bin_path)
    additional_images_file = [os.path.join(additional_images_path, f) for f in os.listdir(additional_images_path) if f.endswith('.jpeg')]    
    # additional_camera_pose= extract_camera_poses_from_images(additional_images_bin)
    # plot_camera_poses(additional_camera_pose[0], additional_camera_pose[1], ax, "BOX-C", 'b')    
    additional_descriptors = extract_descriptors_dict_from_db(additional_db_path)
    additioanl_keypoints = extract_keypoints_dict_from_db(additional_db_path)
    
    additional_camera_positions = []
    additioanl_camera_directions = []
    for i in range(len(additional_images_file)):
        image_file = additional_images_file[i]
        descriptors = np.vstack(additional_descriptors[i + 1])
        keypoints = np.vstack(additioanl_keypoints[i + 1])
        
        ### ベースモデルの画像と追加モデルの画像の特徴点マッチング
        matches = match_descriptors(descriptors, base_descriptors)
        
        ### マッチングした特徴点から3D-2D対応点を抽出
        object_points, image_points = extract_points_from_matches(matches, base_points3d, keypoints, feature_id_to_points3d_id_dict)
        
        ### PnP問題を解く
        if len(object_points) >= 6:
            camera_position, camera_direction = solve_pnp(object_points, image_points, camera_matrix, dist_coeffs)
            additional_camera_positions.append(camera_position)
            additioanl_camera_directions.append(camera_direction)
        else:
            logger.warning(f"Not enough points for pose estimation ({len(object_points)} points) in {image_file}")
            
    #### 追加モデルのカメラ位置をプロット
    plot_camera_poses(np.array(additional_camera_positions), np.array(additioanl_camera_directions), ax, "BOX-C", 'b')    
    
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_zlim(-4, 4)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

    logger.info("Processed all images in Input")
    
if __name__ == "__main__":
    main()
