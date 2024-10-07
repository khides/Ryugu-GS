import struct
import sqlite3
import numpy as np
import quaternion
import cv2
from typing import Any, List
import os
import matplotlib.pyplot as plt
import datetime
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from logger import Logger
from sfm.model import Model

class ModelMerger:
    def __init__(self, query_model: Model, train_model: Model, logger: Logger) -> None:
        self.now = datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=9), 'JST')).strftime("%Y-%m-%d_%H-%M-%S")
        self.show_plot = False
        self.save_plot = False
        self.query_model = query_model
        self.train_model = train_model
        self.train_object_points: np.ndarray = None
        self.query_object_points: np.ndarray = None
        self.logger = logger
        self.fig : Figure= None
        self.ax: Axes = None
        self.matches: List[Any] = None
        self.R: np.ndarray = None
        self.t: np.ndarray = None
        self.scale: float = None
        self.query_camera_positions: np.ndarray = None
        self.query_camera_directions: np.ndarray = None
        self.merge_model: Model = None
        
    def plot_setup(self, show_plot = True, save_plot = True) -> None:
        self.show_plot = show_plot
        self.save_plot = save_plot
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

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
            camera_positions = self.query_model.camera_positions.squeeze(-1)  # (N, 3)
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
        if self.save_plot:
            plt.savefig(f"./plot/{self.now}.jpg")
        if self.show_plot:
            plt.show()
        else :plt.close()
        
    def merge(self, show_plot = True, save_plot = True):
        self.plot_setup(show_plot=show_plot, save_plot=save_plot)
        self.match_descriptors()
        self.extract_points_from_matches()
        self.estimate_transformation_matrix()
        self.transform_query_camera_pose()
        self.plot()
        self.logger.info("Processed all images in Input")

