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

class Model: 
    def __init__(self, model_path: str, name: str, logger: Logger) -> None:
        self.name = name
        self.logger = logger
        self.model_paah = model_path
        self.db_path = f"{model_path}/database.db"
        self.images_path = f"{model_path}/Input"
        self.points3d_path = f"{model_path}/sparse/0/points3D.bin"
        self.image_bin_path = f"{model_path}/sparse/0/images.bin"
        self.pcd_ply_path = f"{model_path}/sparse/0/points3D.ply"
        self.images_bin : dict = {}
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
        index = 0
        for image_id, data in self.images_bin.items():
            R = self.quaternion_to_rotation_matrix(data['qvec'])
            t = np.array(data['tvec']).reshape((3, ))
            
            camera_position = - R.T @ t
            camera_direction = R.T @ np.array([0, 0, 1])
            self.camera_pose.setdefault(data['name'], {
                "position": camera_position,
                "direction": camera_direction
            })
            camera_positions.append(camera_position)
            camera_directions.append(camera_direction)
            if index == 0:
                print(data['name'])
                print("R ",R)
                print("qvec ",data['qvec'])
                print("tvec ",data['tvec'])
                print("camera_position ",camera_position)
                print("camera_direction ",camera_direction)
                index += 1
        
        camera_positions = np.array(camera_positions)
        camera_directions = np.array(camera_directions)
        self.camera_positions = camera_positions
        self.camera_directions = camera_directions
    
    def read_pcd_from_ply(self) -> None:
        """
        points3D.plyファイルを読み込む
        """
        self.pcd = o3d.io.read_point_cloud(self.pcd_ply_path)
                
    def read_model(self):
        """
        モデルを読み込む
        """
        self.read_images_from_bin()
        self.read_images_file()
        self.read_points3d_from_bin()
        # self.read_camera_poses_from_images()
        self.read_keypoints_from_db()
        self.read_descriptors_from_db()
        self.read_pcd_from_ply()
        # self.get_feature_id_to_points3d_id()
        self.logger.info(f"Model {self.name} is read.")
        
    def update_images_bin(self) -> None:
        """
        images_binを更新する
        """
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # name = ""
        index = 0
        for image_name, pose_data in self.camera_pose.items():
            name = image_name
            # カメラの位置と向き情報を取得
            camera_position = pose_data["position"].reshape((3, 1))
            camera_direction = pose_data["direction"].flatten()
            
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

            # 回転行列をクォータニオンに変換
            qvec = quaternion.from_rotation_matrix(R_matrix)

            # 逆変換で並進ベクトル tvec を計算
            tvec = - R_matrix @ camera_position
            if index == 0:
                print("camera_direction ",camera_direction)
                R = self.quaternion_to_rotation_matrix([qvec.w, qvec.x, qvec.y, qvec.z])
                t = np.array(tvec.flatten().tolist()).reshape((3, ))
                camera_position = - R.T @ t
                camera_direction = R.T @ np.array([0, 0, 1])
                print("name ",name)
                print("R ",R)
                print("qvec ",qvec)
                print("tvec ",tvec)
                print("camera_position ",camera_position)
                print("camera_direction ",camera_direction)
                index += 1

            # images_bin 内の該当する画像に位置と向きを設定
            for image_id, image_data in self.images_bin.items():
                if image_data["name"] == image_name:
                    image_data["qvec"] = [qvec.w, qvec.x, qvec.y, qvec.z]  # クォータニオンをリスト形式で格納
                    image_data["tvec"] = tvec.flatten().tolist()  # 並進ベクトルをリストに変換して格納
                    break
            # break
        # camera_positions = []
        # camera_directions = []
        # for image_id, data in self.images_bin.items():
        #     R = self.quaternion_to_rotation_matrix(data['qvec'])
        #     t = np.array(data['tvec']).reshape((3, 1))
        #     camera_position = - R @ t
        #     camera_direction = R @ np.array([0, 0, 1])
        #     self.camera_pose.setdefault(data['name'], {
        #         "position": camera_position,
        #         "direction": camera_direction
        #     })
        #     camera_positions.append(camera_position)
        #     camera_directions.append(camera_direction)
        # for i in range(len(camera_positions)):
        #     camera_position = camera_positions[i]
        #     camera_direction = camera_directions[i]
        #     ax.quiver(camera_position[0], camera_position[1], camera_position[2],
        #                 camera_direction[0], camera_direction[1], camera_direction[2],
        #                 length=0.5, color="b")
        
        # ax.set_box_aspect([1, 1, 1])
        # ax.set_xlim(-4, 4)
        # ax.set_ylim(-4, 4)
        # ax.set_zlim(-4, 4)
        # plt.show()
    
    def update_points3d(self, R_init = None, t_init = None,mean = None, B_reg = None, t_reg = None) -> None:
        """
        points3dを更新する
        """
        # すべてのxyzを抽出し、変換を一括で適用
        all_xyz = np.array([point_data["xyz"] for point_data in self.points3d.values()])  # (N, 3)の形状
        transformed_xyz = all_xyz
        if R_init is not None and t_init is not None:
            transformed_xyz = np.dot(transformed_xyz, R_init.T) + t_init  # 一括で変換を適用
        if mean is not None:
            transformed_xyz = transformed_xyz - mean
        if B_reg is not None and t_reg is not None:
            transformed_xyz = np.dot(transformed_xyz, B_reg) + t_reg  # 一括で変換を適用
            transformed_xyz = transformed_xyz[:,:3]

        # 変換されたxyzをpoints3dに戻す
        for i, (point_id, point_data) in enumerate(self.points3d.items()):
            point_data["xyz"] = tuple(transformed_xyz[i])  # 更新
            
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
        self.write_images_to_bin()
        self.write_points3d_to_bin()
        self.write_pcd_to_ply()
        self.logger.info(f"Model {self.name} is written.")