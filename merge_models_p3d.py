
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
import datetime

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
now = datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=9), 'JST')).strftime("%Y-%m-%d_%H-%M-%S")
logger = Log(f"./log/{now}.log")


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

def extract_points_from_matches(matches,base_points3d, additional_points3d,feature_id_to_points3d_id_dict, additional_feature_id_to_points3d_id_dict):
    base_object_points = []
    additional_object_points = []

    for m in matches:
        base_point3d_id = feature_id_to_points3d_id_dict.get(m.trainIdx, None)
        additional_point3d_id = additional_feature_id_to_points3d_id_dict.get(m.queryIdx, None)
        if base_point3d_id is not None and additional_point3d_id is not None:
            base_object_points.append(base_points3d[base_point3d_id]['xyz'])
            additional_object_points.append(additional_points3d[additional_point3d_id]['xyz'])
    base_object_points = np.array(base_object_points, dtype=np.float32)
    additional_object_points = np.array(additional_object_points, dtype=np.float32)
    return base_object_points, additional_object_points
        
def estimate_transformation_matrix(base_object_points, additional_object_points):
    # 対応する3D点群の数が一致していることを確認
    assert base_object_points.shape == additional_object_points.shape, "対応する3D点の数が一致していません"
    
    # 各点群の重心を計算
    base_centroid = np.mean(base_object_points, axis=0)
    additional_centroid = np.mean(additional_object_points, axis=0)

    # 重心を基準に各点群を中心に揃える
    base_centered = base_object_points - base_centroid
    additional_centered = additional_object_points - additional_centroid

    # 各点群の距離の二乗和を計算してスケールを推定
    base_scale = np.sqrt(np.sum(base_centered ** 2))
    additional_scale = np.sqrt(np.sum(additional_centered ** 2))
    scale = base_scale / additional_scale  # スケールの計算を逆にする

    # スケールを反映した点群を作成
    additional_centered_scaled = additional_centered * scale

    # 回転行列をSVDで推定
    H = np.dot(additional_centered_scaled.T, base_centered)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # 回転行列の行列式が負の場合、反転行列を修正
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # 平行移動ベクトルを計算
    t = base_centroid.T - np.dot(R, additional_centroid.T * scale)

    logger.info(f"Estimated Rotation Matrix: {R}")
    logger.info(f"Estimated Translation Matrix: {t}")
    logger.info(f"Estimated Scale: {scale}")
    return R, t, scale

def transform_camera_poses(camera_positions, camera_directions, R, t, scale):
    """
    カメラの位置と方向に座標変換を適用する。
    
    Parameters:
    camera_positions (numpy array): 複数のカメラの位置ベクトル (N, 3)
    camera_directions (numpy array): 複数のカメラの方向ベクトル (N, 3)
    R (numpy array): 回転行列 (3x3)
    t (numpy array): 平行移動ベクトル (3,)
    scale (float): スケール因子
    
    Returns:
    transformed_positions (numpy array): 変換後のカメラ位置 (N, 3)
    transformed_directions (numpy array): 変換後のカメラ方向 (N, 3)
    """
    # カメラ位置が (N, 3, 1) の場合、(N, 3) に変換
    if camera_positions.shape[-1] == 1:
        camera_positions = camera_positions.squeeze(-1)  # (N, 3)
    # カメラ位置に回転、スケール、平行移動を適用
    transformed_positions = scale * np.dot(camera_positions, R.T) + t
    
    # カメラ方向に回転のみを適用（方向はスケールや平行移動を適用しない）
    transformed_directions = np.dot(camera_directions, R.T)
    
    return transformed_positions, transformed_directions

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
    additional_points3d_path = './Ryugu_Data/Ryugu_CLAHE/sparse/c/points3D.bin'
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
    additional_points3d = read_points3d_bin(additional_points3d_path)   
    additional_camera_pose= extract_camera_poses_from_images(additional_images_bin)
    # plot_camera_poses(additional_camera_pose[0], additional_camera_pose[1], ax, "BOX-C", 'b')    
    # additional_descriptors = extract_descriptors_dict_from_db(additional_db_path)
    additioanl_keypoints = extract_keypoints_dict_from_db(additional_db_path)
    additional_descriptors, additional_image_feature_start_indices = extract_descriptors_list_from_db(additional_db_path)
    additional_feature_id_to_points3d_id_dict = feature_id_to_points3d_id(additional_points3d, additional_image_feature_start_indices)
    
        
    ### ベースモデルの画像と追加モデルの画像の特徴点マッチング
    matches = match_descriptors(additional_descriptors, base_descriptors)
        
    ### マッチングした特徴点から3D-2D対応点を抽出
    base_object_points, additional_object_points = extract_points_from_matches(matches=matches, base_points3d=base_points3d, additional_points3d=additional_points3d,feature_id_to_points3d_id_dict=feature_id_to_points3d_id_dict, additional_feature_id_to_points3d_id_dict=additional_feature_id_to_points3d_id_dict)
    R, t , scale= estimate_transformation_matrix(base_object_points, additional_object_points)

    additional_camera_positions, additioanl_camera_directions = transform_camera_poses(additional_camera_pose[0], additional_camera_pose[1], R, t, scale)
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
    plt.savefig(f"./plot/{now}.jpg")
    plt.show()

    logger.info("Processed all images in Input")
    
if __name__ == "__main__":
    main()
