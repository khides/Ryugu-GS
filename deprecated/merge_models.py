import matplotlib.pyplot as plt
import logging
import datetime
import struct
import sqlite3
import numpy as np
import quaternion
import cv2
from typing import Any, List
import os
from matplotlib.figure import Figure
from matplotlib.axes import Axes

class Logger():
    def __init__(self, filename: str) -> None:
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
        
class Model: 
    def __init__(self, model_path: str, name: str, logger: Logger) -> None:
        self.name = name
        self.logger = logger
        self.model_paah = model_path
        self.db_path = f"{model_path}/database.db"
        self.images_path = f"{model_path}/Input"
        self.points3d_path = f"{model_path}/sparse/0/points3D.bin"
        self.image_bin_path = f"{model_path}/sparse/0/images.bin"
        self.images_bin : dict = None
        self.images : list = None
        self.points3d: dict = None
        self.keypoints: dict = None
        self.descriptors: np.ndarray = None
        self.image_feature_start_indices: dict = None
        self.feature_id_to_point3d_id: dict = None
        self.camera_positions: np.ndarray = None
        self.camera_directions: np.ndarray = None

    def quaternion_to_rotation_matrix(q: Any) -> np.ndarray:
        q = np.quaternion(q[0], q[1], q[2], q[3])
        return quaternion.as_rotation_matrix(q)

    def read_images_from_bin(self) -> None:
        images = {}
        with open(self.image_bin_path, "rb") as f:
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
        self.images_bin = images
        self.read_camera_poses_from_images()
    
    def read_images_file(self) -> None:
        self.images = [os.path.join(self.images_path, f) for f in os.listdir(self.images_path) if f.endswith('.jpeg')]
        
    def read_points3d_from_bin(self) -> None:
        points3d = {}
        with open(self.points3d_path, "rb") as f:
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
        self.points3d = points3d 

    def read_keypoints_from_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
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
        self.keypoints = all_keypoints

    def read_descriptors_from_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
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
        self.descriptors = all_descriptors
        self.image_feature_start_indices = image_feature_start_indices
                            
    def get_feature_id_to_points3d_id(self) -> None:
        feature_id_to_point3d_id = {}
        for point3d_id, point_data in self.points3d.items():
            for image_id, feature_id in point_data['track']:
                global_feature_id = self.image_feature_start_indices[image_id] + feature_id
                feature_id_to_point3d_id[global_feature_id] = point3d_id
        self.feature_id_to_point3d_id = feature_id_to_point3d_id
    
    def read_camera_poses_from_images(self)-> None: 
        camera_positions = []
        camera_directions = []
        for image_id, data in self.images_bin.items():
            R = self.quaternion_to_rotation_matrix(data['qvec'])
            t = np.array(data['tvec']).reshape((3, 1))
            camera_position = -R.T @ t
            camera_direction = R.T @ np.array([0, 0, 1])
            camera_positions.append(camera_position)
            camera_directions.append(camera_direction)
        camera_positions = np.array(camera_positions)
        camera_directions = np.array(camera_directions)
        self.camera_positions = camera_positions
        self.camera_directions = camera_directions
        
    def read_model(self):
        self.read_images_from_bin()
        self.read_images_file()
        self.read_points3d_from_bin()
        # self.read_camera_poses_from_images()
        self.read_keypoints_from_db()
        self.read_descriptors_from_db()
        self.get_feature_id_to_points3d_id()
        self.logger.info(f"Model {self.name} is read.")
        

class ModelMerger:
    def __init__(self, query_model: Model, train_model: Model, logger: Logger) -> None:
        self.now = datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=9), 'JST')).strftime("%Y-%m-%d_%H-%M-%S")
        self.query_model = query_model
        self.train_model = train_model
        self.train_object_points: np.ndarray = None
        self.query_object_points: np.ndarray = None
        self.logger = logger
        self.is_plot = False
        self.fig : Figure= None
        self.ax: Axes = None
        self.matches: List[Any] = None
        self.R: np.ndarray = None
        self.t: np.ndarray = None
        self.scale: float = None
        self.query_camera_positions: np.ndarray = None
        self.query_camera_directions: np.ndarray = None
        self.merge_model: Model = None
        
    def plot_setup(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.is_plot = True

    # 特徴点のマッチングを行う関数
    def match_descriptors(self, normType:int =cv2.NORM_L2, crossCheck: bool=True, distance_threshold: float=0.8) ->None:
        """
        特徴点のマッチングのパラメータ:
        - normType: 特徴点の距離の計算に使用するノルムの種類
        - crossCheck: クロスチェックを行うかどうか
        - distance_threshold: マッチングの距離の閾値
        """
        self.logger.info("Match Features...")
        # descriptors1 = normalize_descriptors(descriptors1.astype(np.float32))
        # descriptors2 = normalize_descriptors(descriptors2.astype(np.float32)) 
        query_descriptors = self.query_model.descriptors.astype(np.float32)
        train_descriptors = self.train_model.descriptors.astype(np.float32)
        
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
        matches = flann.knnMatch(queryDescriptors=query_descriptors, trainDescriptors=train_descriptors, k=2)    
        self.logger.info(f"Found {len(matches)} ")
        good_matches = []
        # 距離の閾値でフィルタリング
        for m, n in matches:
            if m.distance < distance_threshold * n.distance:
                good_matches.append(m)
        good_matches = sorted(good_matches, key=lambda x: x.distance)
        self.logger.info(f"Found {len(good_matches)} good matches")
        self.matches = good_matches
    
    def extract_points_from_matches(self) -> None:
        train_object_points = []
        query_object_points = []

        for m in self.matches:
            train_point3d_id = self.train_model.feature_id_to_point3d_id.get(m.trainIdx, None)
            query_point3d_id = self.query_model.feature_id_to_point3d_id.get(m.queryIdx, None)
            
            if train_point3d_id is not None and query_point3d_id is not None:
                train_object_points.append(self.train_model.points3d[train_point3d_id]['xyz'])
                query_object_points.append(self.query_model.points3d[query_point3d_id]['xyz'])
        train_object_points = np.array(train_object_points, dtype=np.float32)
        query_object_points = np.array(query_object_points, dtype=np.float32)
        self.train_object_points = train_object_points
        self.query_object_points = query_object_points
            
    def estimate_transformation_matrix(self):
        # 対応する3D点群の数が一致していることを確認
        assert self.train_object_points.shape == self.query_object_points.shape, "対応する3D点の数が一致していません"
        
        # 各点群の重心を計算
        train_centroid = np.mean(self.train_object_points, axis=0)
        query_centroid = np.mean(self.query_object_points, axis=0)

        # 重心を基準に各点群を中心に揃える
        train_centered = self.train_object_points - train_centroid
        query_centered = self.query_object_points - query_centroid

        # 各点群の距離の二乗和を計算してスケールを推定
        train_scale = np.sqrt(np.sum(train_centered ** 2))
        query_scale = np.sqrt(np.sum(query_centered ** 2))
        scale = train_scale / query_scale  # スケールの計算を逆にする

        # スケールを反映した点群を作成
        query_centered_scaled = query_centered * scale

        # 回転行列をSVDで推定
        H = np.dot(query_centered_scaled.T, train_centered)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        # 回転行列の行列式が負の場合、反転行列を修正
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = np.dot(Vt.T, U.T)

        # 平行移動ベクトルを計算
        t = train_centroid.T - np.dot(R, query_centroid.T * scale)

        self.logger.info(f"Estimated Rotation Matrix: {R}")
        self.logger.info(f"Estimated Translation Matrix: {t}")
        self.logger.info(f"Estimated Scale: {scale}")
        self.R = R
        self.t = t
        self.scale = scale

    def transform_query_camera_pose(self):
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
        if self.query_model.camera_positions.shape[-1] == 1:
            camera_positions = camera_positions.squeeze(-1)  # (N, 3)
        else :
            camera_positions = self.query_model.camera_positions
            
        # カメラ位置に回転、スケール、平行移動を適用
        transformed_positions = self.scale * np.dot(camera_positions, self.R.T) + self.t
        
        # カメラ方向に回転のみを適用（方向はスケールや平行移動を適用しない）
        transformed_directions = np.dot(self.query_model.camera_directions, self.R.T)
        self.query_camera_positions = transformed_positions
        self.query_camera_directions = transformed_directions
        
    def plot_camera_poses(self, camera_positions:np.ndarray, camera_directions:np.ndarray, label: str, color: str = 'r') -> None:
        cur = 0
        for i in range(len(camera_positions)):
            camera_position = camera_positions[i]
            camera_direction = camera_directions[i]
            if cur == 0:
                self.ax.quiver(camera_position[0], camera_position[1], camera_position[2],
                        camera_direction[0], camera_direction[1], camera_direction[2],
                        length=0.5, color=color, arrow_length_ratio=0.5, label=label)
                cur +=1
            else:
                self.ax.quiver(camera_position[0], camera_position[1], camera_position[2],
                        camera_direction[0], camera_direction[1], camera_direction[2],
                        length=0.5, color=color, arrow_length_ratio=0.5)
    def plot(self):
        self.plot_camera_poses(self.train_model.camera_positions, self.train_model.camera_directions, self.train_model.name, color='r')
        self.plot_camera_poses(self.query_camera_positions, self.query_camera_directions, self.query_model.name, color='b')
        self.ax.set_box_aspect([1, 1, 1])
        self.ax.set_xlim(-4, 4)
        self.ax.set_ylim(-4, 4)
        self.ax.set_zlim(-4, 4)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.legend()
        plt.savefig(f"./plot/{self.now}.jpg")
        plt.show()
        
    def merge(self):
        self.plot_setup()
        self.match_descriptors()
        self.extract_points_from_matches()
        self.estimate_transformation_matrix()
        self.transform_query_camera_pose()
        self.plot()
        self.logger.info("Processed all images in Input")

