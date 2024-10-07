import struct
import sqlite3
import numpy as np
import quaternion
import cv2
from typing import Any, List
import os
from logger import Logger

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

    def quaternion_to_rotation_matrix(self, q: Any) -> np.ndarray:
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