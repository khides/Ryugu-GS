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

# Databaseから特徴点の記述子を抽出する関数
def extract_features_from_db(db_path):
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
    logger.info(f"Extracted feature start indices: {image_feature_start_indices}")
    return all_descriptors, image_feature_start_indices

# 特徴点の記述子を保存する関数
def save_features_to_npy(features, output_file):
    np.save(output_file, features)
    logger.info(f"Features saved to {output_file}")

# points3D.binから3Dポイントを読み込む関数
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

# NPYファイルから特徴点を読み込む関数
def load_features_from_npy(features_file):
    logger.info(f"Loading features from {features_file}")
    return np.load(features_file)

# Databaseから画像のIDを取得する関数
def fetch_image_id(db_path, image_name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT image_id FROM images WHERE name=?", (image_name,))
    image_id = cursor.fetchone()
    conn.close()
    return image_id[0] if image_id else None

# 特徴点の検出と記述子の計算を行う関数
def detect_features(image_path, nfeatures=10000, nOctaveLayers=6, contrastThreshold=0.004, edgeThreshold=5, sigma=1.2):
    """
    SIFTのパラメータ:
    - nfeatures: 検出する特徴点の最大数
    - nOctaveLayers: 各オクターブのレイヤー数
    - contrastThreshold: 対比閾値
    - edgeThreshold: エッジ閾値
    - sigma: 初期画像のガウスフィルタのシグマ
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create(nfeatures=nfeatures, nOctaveLayers=nOctaveLayers, contrastThreshold=contrastThreshold, edgeThreshold=edgeThreshold, sigma=sigma)
    keypoints, descriptors = sift.detectAndCompute(image, None)
    logger.info(f"Detected {len(keypoints)} features from {image_path}")
    return keypoints, descriptors

# 特徴点のマッチングを行う関数
def match_features(descriptors1, descriptors2, normType=cv2.NORM_L2, crossCheck=True, distance_threshold=0.85):
    """
    特徴点のマッチングのパラメータ:
    - normType: 特徴点の距離の計算に使用するノルムの種類
    - crossCheck: クロスチェックを行うかどうか
    - distance_threshold: マッチングの距離の閾値
    """
    descriptors1 = descriptors1.astype(np.float32)
    descriptors2 = descriptors2.astype(np.float32)

    # FLANNのインデックスパラメータと検索パラメータ
    index_params = dict(algorithm=1, trees=10)  # 1はFLANN_INDEX_KDTREE
    search_params = dict(checks=100)  # チェックの回数
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    knn_matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in knn_matches:
        if m.distance < distance_threshold * n.distance:
            good_matches.append(m)
    # if distance_threshold is not None:
    #     good_matches = [m for m in good_matches if m.distance < distance_threshold]
    good_matches = sorted(good_matches, key=lambda x: x.distance)
    logger.info(f"Found {len(good_matches)} good matches")
    return good_matches

# バイナリファイルを読み込む関数 (cameras.bin)
def read_cameras_bin(file_path):
    cameras = {}
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(40)
            if not data:
                break
            camera_id, model, width, height = struct.unpack('<iiQQ', data[:24])
            params = struct.unpack('<4d', data[24:])
            cameras[camera_id] = {'model': model, 'width': width, 'height': height, 'params': params}
    return cameras

# バイナリファイルを読み込む関数 (images.bin)
def read_images_bin(file_path):
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

# カメラパラメータを更新する関数
def update_cameras_bin(camera_file, new_camera_data):
    with open(camera_file, 'ab') as f:
        f.write(struct.pack('<i', new_camera_data['camera_id']))
        f.write(struct.pack('<iiQQ4d', new_camera_data['model'], new_camera_data['width'], new_camera_data['height'], *new_camera_data['params']))

# 画像情報を更新する関数
def update_images_bin(images_file, new_image_data):
    with open(images_file, 'ab') as f:
        f.write(struct.pack('<i4d3diH', new_image_data['image_id'], *new_image_data['qvec'], *new_image_data['tvec'], new_image_data['camera_id'], len(new_image_data['name'])))
        f.write(new_image_data['name'].encode('utf-8'))
        for xy, point3D_id in zip(new_image_data['xys'], new_image_data['point3D_ids']):
            point3D_id = int(point3D_id)  # Ensure point3D_id is an integer
            f.write(struct.pack('<ddq', xy[0], xy[1], point3D_id))

# 回転行列をクォータニオンに変換する関数
def quaternion_from_matrix(matrix):
    q = quaternion.from_rotation_matrix(matrix)
    return [q.x, q.y, q.z, q.w]

# Quaternionを回転行列に変換する関数
def quaternion_to_rotation_matrix(q):
    q = np.quaternion(q[0], q[1], q[2], q[3])
    return quaternion.as_rotation_matrix(q)

# メイン処理
def main():
    # 特徴点の記述子をデータベースから抽出し保存する
    db_path = './Ryugu_Data/Ryugu_mask_3-1/database.db'
    output_file = './Ryugu_Data/Ryugu_mask_3-1/sparse/0/features.npy'

    # 特徴点の記述子を抽出
    features, image_feature_start_indices = extract_features_from_db(db_path)
        
    # 特徴点の記述子を保存
    save_features_to_npy(features, output_file)
    
    # 既存の3Dポイントと特徴点を読み込む
    points3d_bin_path = './Ryugu_Data/Ryugu_mask_3-1/sparse/0/points3D.bin'
    pre_points3d = read_points3d_bin(points3d_bin_path)
    pre_features = load_features_from_npy(output_file)
    
    # 既存のcameras.binとimages.binを読み込む
    # cameras = read_cameras_bin('./Ryugu_Data/Ryugu_mask_3-1/sparse/0/cameras.bin')
    images = read_images_bin('./Ryugu_Data/Ryugu_mask_3-1/sparse/0/images.bin')

    # Input2ディレクトリ内のすべての画像を処理
    input_dir = './Ryugu_Data/Ryugu_mask_3-1/Input2'
    image_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.jpeg')]

    # 3Dプロットの準備
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 既存のカメラ位置をプロット
 # カメラ位置の計算とプロットデータの準備
    camera_positions = []
    for image_id, data in images.items():
        R = quaternion_to_rotation_matrix(data['qvec'])
        t = np.array(data['tvec']).reshape((3, 1))
        camera_position = -R.T @ t
        camera_positions.append(camera_position)

    camera_positions = np.array(camera_positions)
    cur = 0
    for image_id, data in images.items():
        R = quaternion_to_rotation_matrix(data['qvec'])
        t = np.array(data['tvec']).reshape((3, 1))
        camera_position = -R.T @ t
        camera_direction = R.T @ np.array([0, 0, 1])
        if cur == 0:
            ax.quiver(camera_position[0], camera_position[1], camera_position[2],
                      camera_direction[0], camera_direction[1], camera_direction[2],
                      length=0.5, color='r', arrow_length_ratio=0.5, label="BOX-A")
            cur +=1
        else:
            ax.quiver(camera_position[0], camera_position[1], camera_position[2],
                      camera_direction[0], camera_direction[1], camera_direction[2],
                      length=0.5, color='r', arrow_length_ratio=0.5)

    # 各画像に対してカメラポーズ推定を実行し、結果をプロット
    for image_file in image_files:
        # 新しい画像の特徴点を検出
        keypoints, descriptors = detect_features(image_file)

        # 特徴点のマッチング
        matches = match_features(descriptors, pre_features)

        # 3DポイントのIDを取得して、対応する3Dポイントの座標を取得
        feature_id_to_point3d_id = {}
        for point3d_id, point_data in pre_points3d.items():
            for image_id, feature_id in point_data['track']:
                global_feature_id = image_feature_start_indices[image_id] + feature_id
                feature_id_to_point3d_id[global_feature_id] = point3d_id

        object_points = []
        image_points = []
        for m in matches:
            point3d_id = feature_id_to_point3d_id.get(m.trainIdx, None)
            if point3d_id is not None:
                object_points.append(pre_points3d[point3d_id]['xyz'])
                image_points.append(keypoints[m.queryIdx].pt)
        object_points = np.array(object_points, dtype=np.float32)
        image_points = np.array(image_points, dtype=np.float32)

        # 十分な対応点がある場合のみカメラポーズを推定
        if len(object_points) >= 6:
            # カメラの内部パラメータ
            fx, fy, cx, cy = 9231, 9231, 512, 512
            camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            dist_coeffs = np.zeros((4, 1))  # 歪みなしの場合

            # solvePnPでカメラポーズを推定
            _, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
            R, _ = cv2.Rodrigues(rvec)

            new_image_name = os.path.basename(image_file)
            new_image_id = fetch_image_id(db_path, new_image_name)

            if new_image_id is None:
                new_image_id = max(images.keys()) + 1        

            new_image_data = {
                'image_id': new_image_id,
                'camera_id': 1,
                'name': new_image_name,
                'qvec': quaternion_from_matrix(R),
                'tvec': tvec.flatten().tolist(),
                'xys': image_points.tolist(),
                'point3D_ids': [int(id) for id in feature_id_to_point3d_id.values()]  # Ensure point3D_ids are integers
            }

            new_images_file = './Ryugu_Data/Ryugu_mask_3-1/sparse/2/images.bin'
            new_images_dir = os.path.dirname(new_images_file)

            if os.path.exists(new_images_dir):
                shutil.rmtree(new_images_dir)

            shutil.copytree('./Ryugu_Data/Ryugu_mask_3-1/sparse/0', new_images_dir)
            update_images_bin(new_images_file, new_image_data)
            logger.info(f"Updated images.bin with new data for {new_image_name}")

            # 3Dポイントをプロット
            ax.scatter(object_points[:, 0], object_points[:, 1], object_points[:, 2], c='gray', marker='o')

            # カメラの位置をプロット
            camera_position = -R.T @ tvec

            # カメラの向きを矢印で表示
            camera_direction = R.T @ np.array([0, 0, 1])
            ax.quiver(camera_position[0], camera_position[1], camera_position[2],
                    camera_direction[0], camera_direction[1], camera_direction[2], length=0.5, color='b', arrow_length_ratio=0.5, label="BOX-C")
        else:
            logger.warning(f"Not enough points for pose estimation in {image_file}")

    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_zlim(-4, 4)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

    logger.info("Processed all images in Input2")

if __name__ == "__main__":
    main()
