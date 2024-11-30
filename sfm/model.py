import struct
import sqlite3
import numpy as np
import quaternion
import cv2
from typing import Any, List
import os
from logger import Logger
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from sklearn.model_selection import train_test_split
import shutil
import json

class Model: 
    def __init__(self, model_path: str, name: str, logger: Logger) -> None:
        self.name = name
        self.logger = logger
        self.model_paah = model_path
        self.db_path = f"{model_path}/database.db"
        self.images_path = f"{model_path}/images"
        self.points3d_path = f"{model_path}/sparse/0/points3D.bin"
        self.image_bin_path = f"{model_path}/sparse/0/images.bin"
        self.camera_bin_path = f"{model_path}/sparse/0/cameras.bin"
        self.pcd_ply_path = f"{model_path}/sparse/0/points3D.ply"
        self.images_bin : dict = {}
        self.cameras_bin: dict = {}
        self.images : list = None
        self.points3d: dict = {}
        self.keypoints: dict = None
        self.descriptors: np.ndarray = None
        self.pcd: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
        self.image_feature_start_indices: dict = None
        self.feature_id_to_point3d_id: dict = None
        self.camera_pose: dict = {}
        self.camera_positions: np.ndarray = None
        self.camera_directions: np.ndarray = None
        self.camera_ups: np.ndarray = None
        self.camera_rights: np.ndarray = None
        self.train_json: dict = None
        self.test_json: dict = None
        self.train_path = f"{model_path}/train"
        self.test_path = f"{model_path}/test"
        self.train_json_path = f"{model_path}/transforms_train.json"
        self.test_json_path = f"{model_path}/transforms_test.json"

    def quaternion_to_rotation_matrix(self, q: Any) -> np.ndarray:
        """
        クォータ二オンを回転行列に変換する\\
        params:
        - q: クォータニオン
        
        returns:
        - R: 回転行列
        """
        q = np.quaternion(q[0], q[1], q[2], q[3])
        return quaternion.as_rotation_matrix(q)
    
    def read_cameras_from_bin(self):
        cameras = {}
        with open(self.camera_bin_path, 'rb') as f:
            # カメラの数を読み取る
            num_cameras = struct.unpack('<Q', f.read(8))[0]            
            for _ in range(num_cameras):
                # カメラID
                camera_id = struct.unpack('<I', f.read(4))[0]
                
                # モデルID（PINHOLEモデルを仮定）
                model_id = struct.unpack('<I', f.read(4))[0]
                models = ["SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV", "OPENCV_FISHEYE"]
                model = models[model_id] if model_id < len(models) else f"Unknown({model_id})"
                
                # 幅と高さ
                width = struct.unpack('<I', f.read(4))[0]
                _ = struct.unpack('<I', f.read(4))[0]  # 無視される値
                height = struct.unpack('<I', f.read(4))[0]
                _ = struct.unpack('<I', f.read(4))[0]  # 無視される値
                
                # パラメータ（PINHOLEモデルの場合、fx, fy, cx, cyの4つを仮定）
                if model == "SIMPLE_PINHOLE" :
                    num_params = 3
                elif model == "PINHOLE":
                    num_params = 4
                else: # その他のモデルの場合、パラメータ数は不明
                    raise ValueError(f"Unknown camera model: {model}")
                param_data = f.read(8 * num_params)
                params = struct.unpack('<' + 'd' * num_params, param_data)
                
                # 結果を保存
                cameras[camera_id] = {
                    'model': model,
                    'width': width,
                    'height': height,
                    'params': params,
                }
        self.cameras_bin = cameras
        # self.logger.info(f"Cameras are read. {self.cameras_bin}")

    def read_images_from_bin(self) -> None:
        """
        images.binファイルを読み込む
        """
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
        # self.logger.info(f"Images are read. {self.images_bin}")
        self.read_camera_poses_from_images()
    
    def read_images_file(self) -> None:
        """
        Inputフォルダ内のjpegファイルを読み込む
        """
        self.images = [os.path.join(self.images_path, f) for f in os.listdir(self.images_path) if f.endswith('.jpeg')]
        
    def read_points3d_from_bin(self) -> None:
        """
        points3D.binファイルを読み込む
        """
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
        """
        database.dbからkeypointsを読み込む
        """
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
        """
        database.dbからdescriptorsを読み込む
        """
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
        """
        feature_idからpoint3d_idを取得する
        """
        feature_id_to_point3d_id = {}
        for point3d_id, point_data in self.points3d.items():
            for image_id, feature_id in point_data['track']:
                global_feature_id = self.image_feature_start_indices[image_id] + feature_id
                feature_id_to_point3d_id[global_feature_id] = point3d_id
        self.feature_id_to_point3d_id = feature_id_to_point3d_id
    
    def read_camera_poses_from_images(self)-> None: 
        """
        images.binからカメラの位置と向きを取得する
        """
        camera_positions = []
        camera_directions = []
        camera_ups = []
        camera_rights = []
        index = 0
        for image_id, data in self.images_bin.items():
            R = self.quaternion_to_rotation_matrix(data['qvec'])
            t = np.array(data['tvec']).reshape((3, ))
            
            camera_position = - R.T @ t
            camera_direction = R.T @ np.array([0, 0, 1])
            camera_up = R.T @ np.array([0, -1, 0])
            camera_right = R.T @ np.array([1, 0, 0])
            
            self.camera_pose.setdefault(data['name'], {
                "position": camera_position,
                "direction": camera_direction,
                "up": camera_up,
                "right": camera_right
            })
            camera_positions.append(camera_position)
            camera_directions.append(camera_direction)
            camera_ups.append(camera_up)
            camera_rights.append(camera_right)
        
        self.camera_positions = np.array(camera_positions)
        self.camera_directions = np.array(camera_directions)
        self.camera_ups = np.array(camera_ups)
        self.camera_rights = np.array(camera_rights)
    
    def read_pcd_from_ply(self) -> None:
        """
        points3D.plyファイルを読み込む
        """
        self.pcd = o3d.io.read_point_cloud(self.pcd_ply_path)
                
    def read_model(self, estimate_type: str) -> None:
        """
        モデルを読み込む
        """
        self.read_cameras_from_bin()
        self.read_images_from_bin()
        if estimate_type == "cpd":
            self.read_images_file()
        self.read_points3d_from_bin()
        # self.read_camera_poses_from_images()
        self.read_keypoints_from_db()
        self.read_descriptors_from_db()
        self.read_pcd_from_ply()
        if estimate_type != "cpd":
            self.get_feature_id_to_points3d_id()
        self.logger.info(f"Model {self.name} is read.")
        
    def update_cameras_bin(self, scale: float) -> None:
        """
        cameras_binを更新する
        """
        for camera_id, camera_data in self.cameras_bin.items():
            camera_data['width'] = int(round(camera_data['width'] * scale))
            camera_data['height'] = int(round(camera_data['height'] * scale))
            camera_data['params'] = [param * scale for param in camera_data['params']]
    
    def update_images_bin(self,
                        #   R_init = None, t_init = None,mean = None, B_reg = None, t_reg = None
                          ) -> None:
        """
        images_binを更新する
        """
        # for image_id, image_data in self.images_bin.items():
        #     # qvec と tvec を取得
        #     qvec = np.quaternion(image_data["qvec"][0], image_data["qvec"][1], image_data["qvec"][2], image_data["qvec"][3])
        #     tvec = np.array(image_data["tvec"]).reshape(-1, 1)  # 必ず (3, 1) に整形

        #     # 初期回転と並進の適用
        #     # if R_init is not None and t_init is not None:
        #     #     qrot_init = quaternion.from_rotation_matrix(R_init)
        #     #     qvec = qrot_init * qvec  # 四元数の回転を適用
        #     #     tvec = R_init @ tvec + t_init.reshape(-1, 1)  # 並進を適用

        #     # # 平均減算
        #     # if mean is not None:
        #     #     tvec = tvec - mean.reshape(-1, 1)

        #     # 正規化変換の適用
        #     # if B_reg is not None and t_reg is not None:
        #     #     qrot_reg = quaternion.from_rotation_matrix(B_reg)
        #     #     qvec = qrot_reg * qvec  # 四元数の回転を適用
        #     #     tvec = B_reg @ tvec + t_reg.reshape(-1, 1)  # 並進を適用

        #     # 結果を保存
        #     image_data["qvec"] = [qvec.w, qvec.x, qvec.y, qvec.z]  # クォータニオンをリスト形式で保存
        #     image_data["tvec"] = tvec.flatten().tolist()  # tvec を 1 次元リストに変換して保存
            
        for image_name, pose_data in self.camera_pose.items():
            # カメラの位置と向き情報を取得
            camera_position = pose_data["position"].reshape((3, 1))
            camera_direction = pose_data["direction"].flatten()
            camera_up = pose_data["up"].flatten()
            camera_right = pose_data["right"].flatten()
            
            # 回転行列 R を計算
            R_matrix = np.array([camera_right, -camera_up, camera_direction])  # (3, 3) 行列
            
           # カメラの前方方向のベクトル（z軸）を定義
            # z_axis = np.array([0, 0, 1])

            # # z軸と異なる場合、回転行列 R を計算
            # if not np.allclose(camera_direction, z_axis):
            #     rotation_axis = np.cross(camera_direction, z_axis)  # camera_direction -> z軸への回転軸
            #     if np.linalg.norm(rotation_axis) > 0:  # 有効な回転軸のみ処理
            #         rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)  # 正規化
            #     rotation_angle = np.arccos(np.dot(camera_direction, z_axis) /
            #                             (np.linalg.norm(camera_direction) * np.linalg.norm(z_axis)))
            #     R_matrix, _ = cv2.Rodrigues(rotation_axis * rotation_angle)  # 逆回転用に軸を逆に設定
            # else:
            #     R_matrix = np.eye(3)  # z軸と一致する場合は単位行列

            # 回転行列をクォータニオンに変換
            qvec = quaternion.from_rotation_matrix(R_matrix)

            # 逆変換で並進ベクトル tvec を計算
            tvec = - R_matrix @ camera_position

            # images_bin 内の該当する画像に位置と向きを設定
            for image_id, image_data in self.images_bin.items():
                if image_data["name"] == image_name:
                    image_data["qvec"] = [qvec.w, qvec.x, qvec.y, qvec.z]  # クォータニオンをリスト形式で格納
                    image_data["tvec"] = tvec.flatten().tolist()  # 並進ベクトルをリストに変換して格納
                    break
        
    def update_points3d(self, R_init = None, t_init = None, s_init = None ,mean = None, B_reg = None, t_reg = None) -> None:
        """
        points3dを更新する
        """
        # すべてのxyzを抽出し、変換を一括で適用
        all_xyz = np.array([point_data["xyz"] for point_data in self.points3d.values()])  # (N, 3)の形状
        transformed_xyz = all_xyz
        if R_init is not None and t_init is not None:
            if s_init:
                transformed_xyz = s_init * np.dot(transformed_xyz, R_init.T) + t_init
            else :
                transformed_xyz = np.dot(transformed_xyz, R_init.T) + t_init  # 一括で変換を適用
        if mean is not None:
            transformed_xyz = transformed_xyz - mean
        if B_reg is not None and t_reg is not None:
            transformed_xyz = np.dot(transformed_xyz, B_reg) + t_reg  # 一括で変換を適用
            transformed_xyz = transformed_xyz[:,:3]

        # 変換されたxyzをpoints3dに戻す
        for i, (point_id, point_data) in enumerate(self.points3d.items()):
            point_data["xyz"] = tuple(transformed_xyz[i])  # 更新
            
    def generate_camera_poses_json(self):
        """
        Generates a JSON structure containing camera poses.
        """
         # カメラの内部パラメータ
        fx = 9231  # 水平方向の焦点距離
        fy = 9231  # 垂直方向の焦点距離 (ここでは使用しない)
        cx = 512   # 水平方向のセンサー中心
        cy = 512   # 垂直方向のセンサー中心 (ここでは使用しない)
        # 水平画角の計算
        camera_angle_x = 2 * np.arctan(cx / fx)

        def create_transform_matrix(qvec, tvec):
            """
            Creates a 4x4 transformation matrix from a quaternion and translation vector.
            """
            rotation_matrix = self.quaternion_to_rotation_matrix(qvec)
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rotation_matrix
            transform_matrix[:3, 3] = tvec
            return transform_matrix
        def create_camera_pose(input_data, mode="train"):
            """
            Creates a camera pose JSON object.
            """
            frames = []
            for entry in input_data:
                file_path = entry["name"]
                qvec = entry["qvec"]
                tvec = entry["tvec"]

                # Compute the transformation matrix
                transform_matrix = create_transform_matrix(qvec, tvec).tolist()
                if mode == "train":
                    file_path = f"./train/{file_path}".rpartition('.')[0]
                else:
                    file_path = f"./test/{file_path}".rpartition('.')[0]

                frame = {
                    "file_path":file_path,
                    "transform_matrix": transform_matrix
                }
                frames.append(frame)

            camera_poses = {
                "camera_angle_x": camera_angle_x,
                "frames": frames
            }
            return camera_poses
        def copy_images_to_split_dirs( train_data, test_data):
            """
            Copies images from the source directory to train and test directories based on split.
            """
            for entry in self.images_bin.values():
                src_path = os.path.join(self.images_path, os.path.basename(entry["name"]))
                if not os.path.exists(src_path):
                    self.logger.warning(f"Warning: {src_path} does not exist.")
                    continue
                
                # Determine destination directory
                if entry in train_data:
                    dest_dir = self.train_path
                else:
                    dest_dir = self.test_path
                
                # Create destination directory if it doesn't exist
                os.makedirs(dest_dir, exist_ok=True)

                # Copy the image
                dest_path = os.path.join(dest_dir, os.path.basename(entry["name"]))
                shutil.copy(src_path, dest_path)
                # print(f"Copied {src_path} to {dest_path}")

        train_data, test_data = train_test_split(list(self.images_bin.values()), test_size=0.2, random_state=42)
        copy_images_to_split_dirs( train_data, test_data)
        self.train_json = create_camera_pose(train_data, mode="train")
        self.test_json = create_camera_pose(test_data, mode="test") 
    
    def write_camera_poses_json(self):
        """
        Writes the camera poses JSON to a file.
        """
        with open(self.test_json_path, "w") as f:
            json.dump(self.test_json, f, indent=4)
        with open(self.train_json_path, "w") as f:
            json.dump(self.train_json, f, indent=4)
        self.logger.info(f"Camera poses JSON written to {self.test_json_path} and {self.train_json_path}")
 
    def write_points3d_to_bin(self) -> None:
        """
        points3D.binにself.points3dの内容を書き込む
        """
        os.makedirs(os.path.dirname(self.points3d_path), exist_ok=True)
        with open(self.points3d_path, "wb") as f:
            # 最初に3Dポイント数を書き込む
            f.write(struct.pack("<Q", len(self.points3d)))
            for point_id, point_data in self.points3d.items():
                # 各ポイントのデータを書き込む
                f.write(struct.pack("<Q", point_id))
                f.write(struct.pack("<3d", *point_data["xyz"]))
                f.write(struct.pack("<3B", *point_data["rgb"]))
                f.write(struct.pack("<d", point_data["error"]))
                f.write(struct.pack("<Q", len(point_data["track"])))
                for image_id, point2D_idx in point_data["track"]:
                    f.write(struct.pack("<i", image_id))
                    f.write(struct.pack("<i", point2D_idx))
    
    def write_images_to_bin(self) -> None:
        """
        images.binにself.images_binの内容を書き込む
        """
        os.makedirs(os.path.dirname(self.image_bin_path), exist_ok=True)
        with open(self.image_bin_path, "wb") as f:
            # 登録された画像の数を書き込む
            f.write(struct.pack("Q", len(self.images_bin)))
            for image_id, image_data in self.images_bin.items():
                # 各画像のデータを書き込む
                f.write(struct.pack("I", image_id))
                f.write(struct.pack("4d", *image_data["qvec"]))
                f.write(struct.pack("3d", *image_data["tvec"]))
                f.write(struct.pack("I", image_data["camera_id"]))
                
                # 画像名を書き込む（NULL文字で終了）
                f.write(image_data["name"].encode("utf-8"))
                f.write(b'\x00')
                
                # 2Dポイントの数を書き込む
                f.write(struct.pack("Q", len(image_data["xys"])))
                for xy, point3D_id in zip(image_data["xys"], image_data["point3D_ids"]):
                    f.write(struct.pack("2d", *xy))
                    # point3D_idの範囲チェック
                    if not (0 <= point3D_id < (1 << 64)):
                        raise ValueError(f"point3D_id {point3D_id} is out of range for uint64")
                    
                    f.write(struct.pack("Q", point3D_id))
    
    def write_cameras_bin(self):
        os.makedirs(os.path.dirname(self.camera_bin_path), exist_ok=True)
        # モデル名からモデルIDへのマッピング
        models = {
            "SIMPLE_PINHOLE": 0,
            "PINHOLE": 1,
            "SIMPLE_RADIAL": 2,
            "RADIAL": 3,
            "OPENCV": 4,
            "OPENCV_FISHEYE": 5
        }

        with open(self.camera_bin_path, "wb") as f:
            # カメラの数を書き込む（64ビット符号なし整数）
            f.write(struct.pack('<Q', len(self.cameras_bin)))

            for camera_id, camera_data in self.cameras_bin.items():
                # カメラID（32ビット符号なし整数）
                f.write(struct.pack('<I', camera_id))
                
                # モデルID（32ビット符号なし整数）
                model_id = models.get(camera_data['model'], -1)
                if model_id == -1:
                    raise ValueError(f"Unknown camera model: {camera_data['model']}")
                f.write(struct.pack('<I', model_id))
                
                # 幅と高さ（32ビット符号なし整数）
                f.write(struct.pack('<I', camera_data['width']))
                f.write(struct.pack('<I', 0))  # 予約領域（常に0）
                f.write(struct.pack('<I', camera_data['height']))
                f.write(struct.pack('<I', 0))  # 予約領域（常に0）
                
                # カメラパラメータ（64ビット浮動小数点数の配列）
                params = camera_data['params']
                num_params = len(params)
                f.write(struct.pack('<' + 'd' * num_params, *params))
                
    def write_pcd_to_ply(self) -> None:
        """
        points3D.plyにself.pcdの内容を書き込む
        """
        os.makedirs(os.path.dirname(self.pcd_ply_path), exist_ok=True)
        o3d.io.write_point_cloud(self.pcd_ply_path, self.pcd)
        
    def write_model(self) -> None:
        """
        モデルを書き込む
        """
        self.write_cameras_bin()
        self.write_images_to_bin()
        self.write_points3d_to_bin()
        # self.write_camera_poses_json()
        # self.write_pcd_to_ply()
        self.logger.info(f"Model {self.name} is written.")
        
    