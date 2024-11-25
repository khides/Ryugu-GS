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
import open3d as o3d
from scipy.optimize import minimize
from pycpd import AffineRegistration, RigidRegistration
from scipy.spatial.transform import Rotation as R

class ModelMerger:
    def __init__(self, query_model: Model, train_model: Model, merge_model_path: str, merge_model_name: str, logger: Logger) -> None:
        self.merge_model_path = merge_model_path
        self.merge_model_name = merge_model_name
        self.now = datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=9), 'JST')).strftime("%Y-%m-%d_%H-%M-%S")
        self.show_plot = False
        self.save_plot = False
        self.query_model = query_model
        self.train_model = train_model
        self.train_object_points: np.ndarray = None
        self.query_object_points: np.ndarray = None
        self.logger = logger
        self.fig: Figure = None
        self.ax: Axes = None
        self.matches: List[Any] = None
        self.R: np.ndarray = None
        self.t: np.ndarray = None
        self.scale: float = None
        self.affine_matrix: np.ndarray = None
        self.query_camera_positions_transformed: np.ndarray = None
        self.query_camera_directions_transformed: np.ndarray = None
        self.query_object_points_transformed: np.ndarray = None
        self.B_reg: np.ndarray = None
        self.t_reg: np.ndarray = None
        self.s_reg: float = None
        self.query_object_points_down: np.ndarray = None
        self.train_object_points_down: np.ndarray = None
        self.query_object_points_down_transformed: np.ndarray = None
        self.R_init: np.ndarray = None
        self.t_init: np.ndarray = None
        self.query_object_points_pretreeted: np.ndarray = None
        self.query_camera_positions_pretreeted: np.ndarray = None
        self.query_camera_directions_pretreeted: np.ndarray = None
        self.query_object_points_down_pretreeted: np.ndarray = None
        self.train_object_points_down_pretreeted: np.ndarray = None
        self.train_camera_positions: np.ndarray = None
        self.train_camera_directions: np.ndarray = None
        self.new_model: Model = None
        
    def plot_setup(self, show_plot=True, save_plot=True) -> None:
        """
        プロットの設定を行う関数\\
        params:
        - show_plot: プロットを表示するかどうか
        - save_plot: プロットを保存するかどうか
        """
        self.show_plot = show_plot
        self.save_plot = save_plot
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

    def read_points(self) -> None:
        """
        3D点群を読み込む関数
        """
        self.train_object_points = np.asarray(self.train_model.pcd.points)
        self.query_object_points = np.asarray(self.query_model.pcd.points)
        
    def match_descriptors(self, normType:int =cv2.NORM_L2, crossCheck: bool=True, distance_threshold: float=0.8) ->None:
        """
        特徴点のマッチングを行う関数 \\
        params:
        - normType: 特徴点の距離の計算に使用するノルムの種類
        - crossCheck: クロスチェックを行うかどうか
        - distance_threshold: マッチングの距離の閾値
        """
        # descriptors1 = normalize_descriptors(descriptors1.astype(np.float32))
        # descriptors2 = normalize_descriptors(descriptors2.astype(np.float32)) 
        query_descriptors = self.query_model.descriptors.astype(np.float32)
        train_descriptors = self.train_model.descriptors.astype(np.float32)
        self.logger.info(f"Query Descriptors: {len(query_descriptors)}")
        self.logger.info(f"Train Descriptors: {len(train_descriptors)}")
        
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
        
        self.logger.info("Match Features...")
        matches = flann.knnMatch(queryDescriptors=query_descriptors, trainDescriptors=train_descriptors, k=2)    
        self.logger.info(f"Found {len(matches)}  matches")
        
        good_matches = []
        # 距離の閾値でフィルタリング
        for m, n in matches:
            if m.distance < distance_threshold * n.distance:
                good_matches.append(m)
        good_matches = sorted(good_matches, key=lambda x: x.distance)
        self.logger.info(f"Found {len(good_matches)} good matches")
        self.matches = good_matches
    
    def extract_points_from_matches(self) -> None:
        '''
        マッチングした特徴点から3D点を抽出する
        '''
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
        self.logger.info(f"Extracted {len(train_object_points)} 3D points")
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
        """
        # カメラ位置が (N, 3, 1) の場合、(N, 3) に変換
        if self.query_model.camera_positions.shape[-1] == 1:
            camera_positions = self.query_model.camera_positions.squeeze(-1)  # (N, 3)
        else :
            camera_positions = self.query_model.camera_positions
            
        if self.query_object_points.shape[-1] == 1:
            query_object_points = self.query_object_points.squeeze(-1)  # (N, 3)
        else :
            query_object_points = self.query_object_points

        # カメラ位置に回転、スケール、平行移動を適用
        transformed_positions = self.scale * np.dot(camera_positions, self.R.T) + self.t
        
        # カメラ方向に回転のみを適用（方向はスケールや平行移動を適用しない）
        transformed_directions = np.dot(self.query_model.camera_directions, self.R.T)
        
        # point3dに回転、スケール、平行移動を適用
        query_object_points_transformed = self.scale * np.dot(query_object_points, self.R.T) + self.t
        
        self.query_camera_positions_transformed = transformed_positions
        self.query_camera_directions_transformed = transformed_directions
        self.query_object_points_transformed = query_object_points_transformed
    
    def estimate_affine_matrix_with_ransac(self, ransac_threshold: float = 3.0, confidence: float = 0.99) -> None:
        """
        RANSACを用いて2つの3D点群の座標変換（アフィン変換）を推定する \\
        :params
        - ransac_threshold: RANSACのしきい値。大きくすると外れ値を許容しやすくなる。デフォルトは3.0 
        - confidence: 推定に対する信頼度。デフォルトは0.99。
        """
        # 座標点が十分にあるかを確認
        if len(self.train_object_points) < 4 or len(self.query_object_points) < 4:
            self.logger.error("少なくとも4つの対応点が必要です。")
            raise ValueError("少なくとも4つの対応点が必要です。")

        # RANSACを使ってアフィン変換を推定
        success, transformation_matrix, inliers = cv2.estimateAffine3D(
            src=self.query_object_points, 
            dst=self.train_object_points,
            ransacThreshold=ransac_threshold,
            confidence=confidence
            )

        if success:
            self.affine_matrix = transformation_matrix
            self.logger.info(f"Estimated Affine Matrix: {transformation_matrix}")
        else:
            self.logger.error("RANSACによる座標変換の推定に失敗しました。")
            raise RuntimeError("RANSACによる座標変換の推定に失敗しました。")
        
    def transform_query_camera_pose_with_affine(self):
        """
        点群にアフィン変換を適用する
        """
         # カメラ位置が (N, 3, 1) の場合、(N, 3) に変換
        if self.query_model.camera_positions.shape[-1] == 1:
            camera_positions = self.query_model.camera_positions.squeeze(-1)  # (N, 3)
        else :
            camera_positions = self.query_model.camera_positions
        
        if self.query_object_points.shape[-1] == 1:
            query_object_points = self.query_object_points.squeeze(-1)  # (N, 3)
        else :
            query_object_points = self.query_object_points
        
        camera_positions = self.query_camera_positions_pretreeted
        camera_directions = self.query_camera_directions_pretreeted
        camera_ups = self.query_camera_ups_pretreeted
        camera_rights = self.query_camera_rights_pretreeted
        
        query_object_points = self.query_object_points_pretreeted
        # # アフィン変換の回転・スケーリング部分（3x3行列）と並進部分を分離
        # R_affine = self.affine_matrix[:3, :3]  # 回転・スケーリング行列
        # t_affine = self.affine_matrix[:3, 3]   # 並進ベクトル

        # カメラ位置に回転、スケール、平行移動を適用
        # camera_positions_homogeneous = np.hstack((camera_positions, np.ones((camera_positions.shape[0], 1))))
        # transformed_positions = np.dot(camera_positions_homogeneous, self.affine_matrix.T) 
        transformed_positions = np.dot(camera_positions, self.B_reg) + self.t_reg
        
        # カメラ方向に回転のみを適用（スケールや平行移動は適用しない）
        # transformed_directions = np.dot(self.query_model.camera_directions, R_affine.T)
        # transformed_directions_normalized = transformed_directions / np.linalg.norm(transformed_directions, axis=1, keepdims=True)
        transformed_directions = np.dot(camera_directions, self.B_reg)
        transformed_directions_normalized = transformed_directions / np.linalg.norm(transformed_directions, axis=1, keepdims=True)
        
        transformed_ups = np.dot(camera_ups, self.B_reg)
        transformed_ups_normalized = transformed_ups / np.linalg.norm(transformed_ups, axis=1, keepdims=True)
        
        transformed_rights = np.dot(camera_rights, self.B_reg)
        transformed_rights_normalized = transformed_rights / np.linalg.norm(transformed_rights, axis=1, keepdims=True)
        
        # point3dに回転、スケール、平行移動を適用
        # query_object_points_homogeneous = np.hstack((query_object_points, np.ones((query_object_points.shape[0], 1))))
        # transformed_object_points = np.dot(query_object_points_homogeneous, self.affine_matrix.T) 
        transformed_object_points = np.dot(query_object_points, self.B_reg) + self.t_reg

        # 結果を保存
        self.query_camera_positions_transformed = transformed_positions  # (N, 3) の位置座標
        self.query_camera_directions_transformed = transformed_directions_normalized       # (N, 3) の方向ベクトル
        self.query_camera_ups_transformed = transformed_ups_normalized
        self.query_camera_rights_transformed = transformed_rights_normalized
        self.query_object_points_transformed = transformed_object_points[:,:3] # (N, 3) の3D点群
        
    def estimate_transformation_matrix_with_icp(self, threshold: float = 0.1) -> None:
        """
        ICPを用いて2つの3D点群の座標変換（アフィン変換）を推定する \\
        params: 
        - threshold: ICPの収束条件となる距離の閾値
        """
        # ICPの初期変換行列を設定
        trans_init = np.eye(4)
        # ICPの設定 
        reg_p2p = o3d.pipelines.registration.registration_icp(
            self.query_model.pcd, self.train_model.pcd, threshold,trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        self.affine_matrix = reg_p2p.transformation
        self.logger.info(f"Estimated Affine Matrix: {self.affine_matrix}")

        # self.query_object_points = np.asarray(self.query_model.pcd.points)
        # self.train_object_points = np.asarray(self.train_model.pcd.points)
        
        # # KDTreeを使って最近傍点を見つける関数
        # def find_nearest_neighbors(source_points, target_points):
        #     target_pcd = o3d.geometry.PointCloud()
        #     target_pcd.points = o3d.utility.Vector3dVector(target_points)
            
        #     # KDTreeを作成
        #     target_tree = o3d.geometry.KDTreeFlann(target_pcd)
            
        #     nearest_points = []
        #     for point in source_points:
        #         # source_pointに最も近いtarget_pointを探す
        #         [_, idx, _] = target_tree.search_knn_vector_3d(point, 1)
        #         nearest_points.append(target_points[idx[0]])
            
        #     return np.array(nearest_points)

        
        # # 損失関数: 双方向での対応点を見つけ、それぞれの距離を最小化する
        # def affine_icp_loss(params, source_points, target_points):
        #     scale = params[0]
        #     rotation_angles = params[1:4]  # 回転角度
        #     translation_vector = params[4:7]  # 平行移動

        #     # 回転行列の計算
        #     rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz(rotation_angles)

        #     # スケール、回転、平行移動を適用して点群を変換
        #     transformed_source = scale * (source_points @ rotation_matrix.T) + translation_vector

        #     # 対応点（source_points → target_pointsの対応）
        #     nearest_target_to_source = find_nearest_neighbors(transformed_source, target_points)

        #     # 対応点（target_points → source_pointsの対応）
        #     nearest_source_to_target = find_nearest_neighbors(target_points, transformed_source)

        #     # 対応点間の距離の二乗和を計算（双方向の誤差）
        #     loss1 = np.sum(np.linalg.norm(transformed_source - nearest_target_to_source, axis=1)**2)
        #     loss2 = np.sum(np.linalg.norm(target_points - nearest_source_to_target, axis=1)**2)
            
        #     return loss1 + loss2
        
        # # 初期パラメータ（スケール=1、回転なし、平行移動なし）
        # initial_params = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # # 点群をnumpy配列に変換
        # source_points = np.asarray(self.query_model.pcd.points)
        # target_points = np.asarray(self.train_model.pcd.points)
        
        # # 最適化 (BFGS法)
        # result = minimize(affine_icp_loss, initial_params, args=(source_points, target_points))
        # optimal_params = result.x
        # scale = optimal_params[0]
        # rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz(optimal_params[1:4])
        # translation_vector = optimal_params[4:7]
        
        # self.scale = scale
        # self.R = rotation_matrix
        # self.t = translation_vector
        
    def estimate_transformation_matrix_with_cpd(self, threshold: float = 0.1) -> None:
        """
        CPDを用いて2つの3D点群の座標変換（アフィン変換）を推定する\\
        params:
        - threshold: CPDの収束条件となる距離の閾値
        """
        self.logger.info("Estimating Affine Matrix with CPD...")
        query = self.query_object_points_down_pretreeted
        train = self.train_object_points_down
        # CPDによるアフィン変換
        # reg = AffineRegistration(X=train, Y=query)
        # TY,(B_reg, t_reg) = reg.register()
        # self.B_reg = B_reg
        # self.t_reg = t_reg
        reg = RigidRegistration(X=train, Y=query)
        TY, (s, R, t) = reg.register()
        
        self.logger.info(f"Estimated R: {R}")
        self.logger.info(f"Estimated t: {t}")
        self.logger.info(f"Estimated s: {s}")
        
        self.query_object_points_down_transformed = TY
        self.affine_matrix = np.eye(4)
        self.B_reg = s * R
        self.t_reg = t
        self.s_reg = s
        # self.affine_matrix[:3, :3] = B_reg
        # self.affine_matrix[:3, 3] = t_reg
        
        # self.logger.info(f"Estimated Affine Matrix: {self.affine_matrix}")
        self.logger.info(f"Estimated B: {self.B_reg}")
        self.logger.info(f"Estimated t: {self.t_reg}")
    
    def pretreet(self, query_positions: np.ndarray, train_positions: np.ndarray) -> None:
        '''
        点群を前処理する関数 \\
        params:
        - query_positions: クエリのカメラ位置
        - train_positions: トレーニングのカメラ位置
        '''
         # 各点群の重心を計算
        train_centroid = np.mean(train_positions, axis=0)
        query_centroid = np.mean(query_positions, axis=0)

        # 重心を基準に各点群を中心に揃える
        train_centered = train_positions - train_centroid
        query_centered = query_positions - query_centroid

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
        self.R_init = R

        # 平行移動ベクトルを計算
        t = train_centroid.T - np.dot(R, query_centroid.T * scale)
        self.t_init = t

        self.logger.info(f"Estimated Pretreetment Rotation Matrix: {R}")
        self.logger.info(f"Estimated Pretreetment Translation Matrix: {t}")
        self.logger.info(f"Estimated Pretreetment Scale: {scale}")
        
        self.query_object_points_down_pretreeted = np.dot(self.query_object_points_down, self.R_init.T) + self.t_init
        self.query_object_points_pretreeted = np.dot(self.query_object_points, self.R_init.T) + self.t_init
        
         # カメラ位置が (N, 3, 1) の場合、(N, 3) に変換
        if self.query_model.camera_positions.shape[-1] == 1:
            camera_positions = self.query_model.camera_positions.squeeze(-1)  # (N, 3)
        else :
            camera_positions = self.query_model.camera_positions
        self.query_camera_positions_pretreeted = np.dot(camera_positions, self.R_init.T) + self.t_init  
        self.query_camera_directions_pretreeted = np.dot(self.query_model.camera_directions, self.R_init.T)
        self.query_camera_ups_pretreeted = np.dot(self.query_model.camera_ups, self.R_init.T)
        self.query_camera_rights_pretreeted = np.dot(self.query_model.camera_rights, self.R_init.T)
        
        if self.train_model.camera_positions.shape[-1] == 1:
            train_camera_positions = self.train_model.camera_positions.squeeze(-1)
        else :
            train_camera_positions = self.train_model.camera_positions
            
        
        self.train_mean = np.mean(self.train_object_points, axis=0)
        self.train_object_points = self.train_object_points - self.train_mean
        self.train_object_points_down = self.train_object_points_down - self.train_mean
        self.train_camera_positions = train_camera_positions - self.train_mean
        self.train_camera_directions = self.train_model.camera_directions
        self.train_camera_ups = self.train_model.camera_ups
        self.train_camera_rights = self.train_model.camera_rights
        
        self.query_mean = np.mean(self.query_object_points_pretreeted, axis=0)
        # self.query_object_points = self.query_object_points - query_mean
        self.query_object_points_down_pretreeted = self.query_object_points_down_pretreeted - self.query_mean
        self.query_object_points_pretreeted = self.query_object_points_pretreeted - self.query_mean
        self.query_camera_positions_pretreeted = self.query_camera_positions_pretreeted - self.query_mean
        # self.query_camera_directions_pretreeted = self.query_camera_directions_pretreeted - np.mean(self.query_camera_directions_pretreeted, axis=0)
        
    def voxel_down_sample(self, voxel_size: float = 0.01, distance_threshold: float = 1.0e-5) -> None:
        """
        点群を外れ値除去してからダウンサンプリングする \\
        params:
        - voxel_size: ボクセルのサイズ
        - distance_threshold: 外れ値とみなす最近点までの距離のしきい値
        """

        # KDTreeを使って最寄点までの距離を計算
        query_kdtree = o3d.geometry.KDTreeFlann(self.query_model.pcd)
        train_kdtree = o3d.geometry.KDTreeFlann(self.train_model.pcd)

        def filter_outliers(points, kdtree, threshold):
            """
            はずれ値を除去する関数\\
            params:
            - points: 3D点群
            - kdtree: KDTree
            - threshold: しきい値
            """
            self.logger.info(f"Filtering outliers with threshold: {threshold}")
            filtered_points = []
            for point in points:
                [_, idx, dists] = kdtree.search_knn_vector_3d(point, 2)  # 自分自身を含む2点を検索
                if len(dists) > 1 and dists[1] <= threshold:  # 自分自身を除いた最近距離がしきい値以下なら保持
                    filtered_points.append(point)
            return np.array(filtered_points)

        # 外れ値除去
        filtered_query_points = filter_outliers(self.query_object_points, query_kdtree, distance_threshold)
        filtered_train_points = filter_outliers(self.train_object_points, train_kdtree, distance_threshold)

        # 外れ値除去後の点群をPCDに設定
        query_pcd_filtered = o3d.geometry.PointCloud()
        train_pcd_filtered = o3d.geometry.PointCloud()
        query_pcd_filtered.points = o3d.utility.Vector3dVector(filtered_query_points)
        train_pcd_filtered.points = o3d.utility.Vector3dVector(filtered_train_points)
        
        query_hull, _ = query_pcd_filtered.compute_convex_hull()
        query_volume = query_hull.get_volume()
        train_hull, _ = train_pcd_filtered.compute_convex_hull()
        train_volume = train_hull.get_volume()
        self.logger.info(f"Query Object Volume: {query_volume}")
        self.logger.info(f"Train Object Volume: {train_volume}")

        # ダウンサンプリング処理
        query_pcd_down = query_pcd_filtered.voxel_down_sample(voxel_size=voxel_size)
        train_pcd_down = train_pcd_filtered.voxel_down_sample(voxel_size=voxel_size)
        
        query_hull_down, _ = query_pcd_down.compute_convex_hull()
        query_volume_down = query_hull_down.get_volume()
        train_hull_down, _ = train_pcd_down.compute_convex_hull()
        train_volume_down = train_hull_down.get_volume()
        self.logger.info(f"Query Object Volume Down: {query_volume_down}")
        self.logger.info(f"Train Object Volume Down: {train_volume_down}")

        # ダウンサンプリング後の点群をnumpy配列に変換して保持
        self.query_object_points_down = np.asarray(query_pcd_down.points)
        self.train_object_points_down = np.asarray(train_pcd_down.points)
    
    def plot_points(self, points_list: List[np.ndarray], label_list: List[str], color_list: List[str], title: str, show_plot: bool, save_plot: bool, scale:float = 0.3, center:List[float] = [0,0,0]) -> None:
        self.plot_setup(show_plot=show_plot, save_plot=save_plot)
        for i in range(len(points_list)):
            self.ax.scatter(points_list[i][:, 0], points_list[i][:, 1], points_list[i][:, 2], color=color_list[i], label=label_list[i], s=0.01) 
        self.ax.set_box_aspect([1, 1, 1])
        self.ax.set_xlim(center[0]-scale, center[0]+scale)
        self.ax.set_ylim(center[1]-scale, center[1]+scale)
        self.ax.set_zlim(center[2]-scale, center[2]+scale)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.legend()
        self.ax.set_title(title)
        if show_plot:
            plt.show()   
        if save_plot:
            plt.savefig(f"./plot/{self.now}.{title.replace(' ', '_')}.jpg")
    
    def plot_poses(self, camera_positions_list:List[np.ndarray], camera_directions_list:List[np.ndarray], label_list: List[str], color_list: List[str], title: str, show_plot: bool, save_plot: bool, scale = 4) -> None:
        
        def plot_camera_poses(camera_positions:np.ndarray, camera_directions:np.ndarray, label: str, color: str = 'r') -> None:
            """
            カメラの位置と方向をプロットする\\
            params:
            - camera_positions: カメラの位置のリスト
            - camera_directions: カメラの方向のリスト
            - label: ラベル
            - color: 色
            """
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
                    
        self.plot_setup(show_plot=show_plot, save_plot=save_plot)
        for i in range(len(camera_positions_list)):
            plot_camera_poses(camera_positions=camera_positions_list[i], 
                                   camera_directions=camera_directions_list[i], 
                                   label=label_list[i], 
                                   color=color_list[i])            
        self.ax.set_box_aspect([1, 1, 1])
        self.ax.set_xlim(-scale, scale)
        self.ax.set_ylim(-scale, scale)
        self.ax.set_zlim(-scale, scale)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.legend()
        self.ax.set_title(title)
        if save_plot:
            self.fig.savefig(f"./plot/{self.now}.{title.replace(' ', '_')}.jpg")
        if show_plot:
            plt.show()   

    def merge_model (self):
        ## トレーニングモデルのカメラ位置、方向、3D点群を更新
        index = 0
        for pose in self.train_model.camera_pose:
            self.train_model.camera_pose[pose]["position"] = self.train_camera_positions[index]
            self.train_model.camera_pose[pose]["direction"] = self.train_camera_directions[index]
            self.train_model.camera_pose[pose]["up"] = self.train_camera_ups[index]
            self.train_model.camera_pose[pose]["right"] = self.train_camera_rights[index]
            index += 1
        self.train_model.update_images_bin(mean=self.train_mean)
        self.train_model.update_points3d(mean=self.train_mean)
        self.train_model.pcd.points = o3d.utility.Vector3dVector(self.train_object_points)
        
        ## クエリモデルのカメラ位置、方向、3D点群を更新
        index = 0
        for pose in self.query_model.camera_pose:
            self.query_model.camera_pose[pose]["position"] = self.query_camera_positions_transformed[index]
            self.query_model.camera_pose[pose]["direction"] = self.query_camera_directions_transformed[index]
            self.query_model.camera_pose[pose]["up"] = self.query_camera_ups_transformed[index]
            self.query_model.camera_pose[pose]["right"] = self.query_camera_rights_transformed[index]
            index += 1
        # self.query_model.update_cameras_bin(scale=self.s_reg)
        self.query_model.update_images_bin(R_init=self.R_init, t_init=self.t_init,mean=self.query_mean, B_reg=self.B_reg, t_reg=self.t_reg)
        self.query_model.update_points3d(R_init=self.R_init, t_init=self.t_init,mean=self.query_mean, B_reg=self.B_reg, t_reg=self.t_reg)
        self.query_model.pcd.points = o3d.utility.Vector3dVector(self.query_object_points_transformed)
        
        ## モデルをマージ
        self.new_model = Model( model_path=self.merge_model_path, name=self.merge_model_name, logger=self.logger)
        self.new_model.camera_positions =np.vstack([self.train_camera_positions, self.query_camera_positions_transformed])
        self.new_model.camera_directions = np.vstack([self.train_camera_directions, self.query_camera_directions_transformed])
        self.new_model.camera_ups = np.vstack([self.train_camera_ups, self.query_camera_ups_transformed])
        self.new_model.camera_rights = np.vstack([self.train_camera_rights, self.query_camera_rights_transformed])
        new_object_points = np.vstack([self.train_object_points, self.query_object_points_transformed])
        self.new_model.pcd = self.train_model.pcd + self.query_model.pcd
        
        self.new_model.cameras_bin.update(self.train_model.cameras_bin)
        max_camera_id = max(self.new_model.cameras_bin.keys(), default=0)
        self.new_model.points3d.update(self.train_model.points3d)
        max_point_id = max(self.new_model.points3d.keys(), default=0)
        min_point_id = min(self.new_model.points3d.keys(), default=0)
        self.new_model.images_bin.update(self.train_model.images_bin)
        max_image_id = max(self.new_model.images_bin.keys(), default=0)
        min_image_id = min(self.new_model.images_bin.keys(), default=0)
        
        for camera_id, camera_data in self.query_model.cameras_bin.items():
            new_camera_id = max_camera_id + camera_id 
            if new_camera_id not in self.new_model.cameras_bin:                
                self.new_model.cameras_bin[new_camera_id] = {
                    "model": camera_data["model"],
                    "width": camera_data["width"],
                    "height": camera_data["height"],
                    "params": camera_data["params"]
                }
            else :
                self.logger.error(f"Camera ID {new_camera_id} is already exist.")
        
        for point_id, point_data in self.query_model.points3d.items():
            new_point_id = max_point_id + point_id 
            new_track = []
            for image_id, point2D_idx in point_data["track"]:
                new_image_id = max_image_id + image_id  # 新しいimage_idを作成
                new_track.append((new_image_id, point2D_idx))
                
            if new_point_id not in self.new_model.points3d:
                # points3dに新しいエントリを追加し、trackと新しいpoint_idで更新
                self.new_model.points3d[new_point_id] = {
                    "xyz": point_data["xyz"],
                    "rgb": point_data["rgb"],
                    "error": point_data["error"],
                    "track": new_track
                }
            else :
                self.logger.error(f"Point ID {new_point_id} is already exist.")
        
        for image_id, image_data in self.query_model.images_bin.items():
            new_image_id = max_image_id + image_id 
            new_point3D_ids = []
            for point3D_id in image_data["point3D_ids"]:
                new_point_id = point3D_id
                if point3D_id != 18446744073709551615:
                    new_point_id = max_point_id + point3D_id   # 新しいIDを設定
                new_point3D_ids.append(new_point_id)
            new_camera_id = max_camera_id + image_data["camera_id"]   # 新しいcamera_idを作成
            if new_image_id not in self.new_model.images_bin:
                self.new_model.images_bin[new_image_id] = {
                    "qvec": image_data["qvec"],
                    "tvec": image_data["tvec"],
                    "camera_id": new_camera_id,
                    "name": image_data["name"],
                    "xys": image_data["xys"],
                    "point3D_ids": new_point3D_ids
                }
            else :
                self.logger.error(f"Image ID {new_image_id} is already exist.")
        
        # self.new_model.generate_camera_poses_json()
        self.new_model.write_model()
        self.logger.info(f"Model Merged: {self.new_model.name}")

    def culc_volume(self) -> None:
        query_pcd_filtered = o3d.geometry.PointCloud()
        train_pcd_filtered = o3d.geometry.PointCloud()
        query_pcd_filtered.points = o3d.utility.Vector3dVector(self.query_object_points_down_transformed)
        train_pcd_filtered.points = o3d.utility.Vector3dVector(self.train_object_points_down)
        
        query_hull, _ = query_pcd_filtered.compute_convex_hull()
        query_volume = query_hull.get_volume()
        train_hull, _ = train_pcd_filtered.compute_convex_hull()
        train_volume = train_hull.get_volume()
        self.logger.info(f"Transformed Query Object Volume: {query_volume}")
        self.logger.info(f"Train Object Volume: {train_volume}")
    
    def merge(self, estimate_type, show_plot = True, save_plot = True):
        """
        2つのモデルをマージする\\
        params:
        - estimate_type: 3D点群の座標変換を推定する手法
        - show_plot: プロットを表示するかどうか
        - save_plot: プロットを保存するかどうか
        """
        ## read points
        self.read_points()
        
        ## plot initial points and poses
        self.plot_poses(camera_positions_list=[self.train_model.camera_positions,self.query_model.camera_positions],
                        camera_directions_list=[self.train_model.camera_directions, self.query_model.camera_directions],
                        label_list=[self.train_model.name, self.query_model.name],
                        color_list=['r', 'b'],
                        title="initial Camera Poses",
                        show_plot=show_plot,
                        save_plot=False)
        self.plot_points(points_list=[self.train_object_points, self.query_object_points],
                         label_list=[self.train_model.name, self.query_model.name],
                         color_list=['r', 'b'],
                         title="initial Object Points",
                         show_plot=show_plot,
                         save_plot=False,
                        #  scale = 0.3
                         scale=1,
                        #  center=[0.7,-0.3,0.3]
                        center = [-0.7, 0, 0.5]
                        # center=[-1, 0,1]
                         )
        
        ## pretreetment
        query_positions = np.array([self.query_model.camera_pose["hyb2_onc_20180824_072322_tvf_l2a.fit.png"]["position"].squeeze(),
                                    self.query_model.camera_pose["hyb2_onc_20180824_083942_tvf_l2a.fit.png"]["position"].squeeze(),
                                    self.query_model.camera_pose["hyb2_onc_20180824_095602_tvf_l2a.fit.png"]["position"].squeeze()])
        # train_positions = np.array([self.query_model.camera_pose["hyb2_onc_20180824_073628_tvf_l2a.fit.png"]["position"].squeeze(),
        #                             self.query_model.camera_pose["hyb2_onc_20180824_083942_tvf_l2a.fit.png"]["position"].squeeze(),
        #                             self.query_model.camera_pose["hyb2_onc_20180824_095602_tvf_l2a.fit.png"]["position"].squeeze()])
        train_positions = np.array([self.train_model.camera_pose["hyb2_onc_20180710_060508_tvf_l2a.fit.png"]["position"].squeeze(),
                                    self.train_model.camera_pose["hyb2_onc_20180710_064228_tvf_l2a.fit.png"]["position"].squeeze(),
                                    self.train_model.camera_pose["hyb2_onc_20180710_073100_tvf_l2a.fit.png"]["position"].squeeze()])
        query_directions = np.array([self.query_model.camera_pose["hyb2_onc_20180824_072322_tvf_l2a.fit.png"]["direction"].squeeze(),
                                    self.query_model.camera_pose["hyb2_onc_20180824_083942_tvf_l2a.fit.png"]["direction"].squeeze(),
                                    self.query_model.camera_pose["hyb2_onc_20180824_095602_tvf_l2a.fit.png"]["direction"].squeeze()])
        # train_directions = np.array([self.query_model.camera_pose["hyb2_onc_20180824_073628_tvf_l2a.fit.png"]["direction"].squeeze(),
        #                             self.query_model.camera_pose["hyb2_onc_20180824_083942_tvf_l2a.fit.png"]["direction"].squeeze(),
        #                             self.query_model.camera_pose["hyb2_onc_20180824_095602_tvf_l2a.fit.png"]["direction"].squeeze()])
        train_directions = np.array([self.train_model.camera_pose["hyb2_onc_20180710_060508_tvf_l2a.fit.png"]["direction"].squeeze(),
                                    self.train_model.camera_pose["hyb2_onc_20180710_064228_tvf_l2a.fit.png"]["direction"].squeeze(),
                                    self.train_model.camera_pose["hyb2_onc_20180710_073100_tvf_l2a.fit.png"]["direction"].squeeze()])
        self.plot_poses(camera_positions_list=[train_positions, query_positions],
                        camera_directions_list=[train_directions, query_directions],
                        label_list=[self.train_model.name, self.query_model.name],
                        color_list=['r', 'b'],
                        title="base Camera Poses of pretreetment",
                        show_plot=show_plot,
                        save_plot=False)
            
        # ダウンサンプリングで点群を軽量化
        self.voxel_down_sample(voxel_size=0.01, distance_threshold=1.0e-5)
        
        # Pretreetment
        self.pretreet(query_positions=query_positions, train_positions=train_positions)
        transformed_query_positions = np.dot(query_positions, self.R_init.T) + self.t_init
        trainformed_query_directions = np.dot(query_directions, self.R_init.T)
        self.plot_poses(camera_positions_list=[train_positions, transformed_query_positions],
                        camera_directions_list=[train_directions, trainformed_query_directions],
                        label_list=[self.train_model.name, self.query_model.name],
                        color_list=['r', 'b'],
                        title="Pretreeted matched Camera Poses",
                        show_plot=show_plot,
                        save_plot=False)
        self.plot_points(points_list=[self.train_object_points_down],
                         label_list=[self.train_model.name],
                         color_list=['r'],
                         title="Train Object Points down sampled",
                         show_plot=show_plot,
                         save_plot=False,
                         scale=0.3,
                        #  scale=1,
                        #  center=[0.7,-0.3,0.3]
                        # center=[-1, 0,1]
                        )
        self.plot_points(points_list=[self.query_object_points_down_pretreeted],
                         label_list=[self.query_model.name],
                         color_list=['b'],
                         title="Pretreeted matched Query Object Points down sampled",
                         show_plot=show_plot,
                         save_plot=False,
                         scale=0.3,
                        #  scale=1,
                        #  center=[0.7,-0.3,0.3]
                        # center=[-1, 0,1]
                        )
        self.plot_points(points_list=[self.train_object_points_down, self.query_object_points_down_pretreeted],
                         label_list=[self.train_model.name, self.query_model.name],
                         color_list=['r', 'b'],
                         title="Pretreeted Object Points down sampled",
                         show_plot=show_plot,
                         save_plot=False,
                         scale=0.3,
                        #  scale=1,
                        #  center=[0.7,-0.3,0.3]
                        # center=[-1, 0,1]
                        )
        self.plot_poses(camera_positions_list=[self.train_model.camera_positions, self.query_camera_positions_pretreeted],
                        camera_directions_list=[self.train_model.camera_directions, self.query_camera_directions_pretreeted],
                        label_list=[self.train_model.name, self.query_model.name],
                        color_list=['r', 'b'],
                        title="Pretreeted Camera Poses",
                        show_plot=show_plot,
                        save_plot=False)
        
        ## estimate transformation matrix
        if estimate_type == 'icp':
            self.estimate_transformation_matrix_with_icp()
            # self.transform_query_camera_pose()
            self.transform_query_camera_pose_with_affine()
        elif estimate_type == 'cpd':
            self.estimate_transformation_matrix_with_cpd()
            self.transform_query_camera_pose_with_affine()
        elif estimate_type == 'ransac':
            self.match_descriptors()
            self.extract_points_from_matches()
            self.estimate_affine_matrix_with_ransac()
            self.transform_query_camera_pose_with_affine()
        else :
            self.match_descriptors()
            self.extract_points_from_matches()
            self.estimate_transformation_matrix()
            self.transform_query_camera_pose()
           
        ## plot down sampled points
        self.plot_points(points_list=[self.train_object_points_down],
                         label_list=[self.train_model.name],
                         color_list=['r'],
                         title="Train Object Points down sampled",
                         show_plot=show_plot,
                         save_plot=False,
                         scale=0.3,
                        #  scale=1,
                        #  center=[0.7,-0.3,0.3]
                        # center=[-1, 0,1]
                        )        
        self.plot_points(points_list=[self.query_object_points_down_transformed],
                         label_list=[self.query_model.name],
                         color_list=['b'],
                         title="Transformed Query Object Points down sampled",
                         show_plot=show_plot,
                         save_plot=False,
                         scale=0.3,
                        #  scale=1,
                        #  center=[0.7,-0.3,0.3]
                        # center=[-1, 0,1]
                        )        
        self.plot_points(points_list=[self.train_object_points_down, self.query_object_points_down_transformed],
                         label_list=[self.train_model.name, self.query_model.name],
                         color_list=['r', 'b'],
                         title="Merged Object Points down sampled",
                         show_plot=show_plot,
                         save_plot=False,
                         scale=0.3,
                        #  scale=1,
                        #  center=[0.7,-0.3,0.3]
                        # center=[-1, 0,1]
                        )        
        
        ## culculate volume
        self.culc_volume()
        
        ## plot all points
        self.plot_points(points_list=[self.train_object_points],
                         label_list=[self.train_model.name], 
                         color_list=['r'],
                         title="Train Object Points",
                         show_plot=show_plot,
                         save_plot=False,
                         scale=0.3,
                        #  scale=1,
                        #  center=[0.7,-0.3,0.3]
                        # center=[-1, 0,1]
                        )        
        self.plot_points(points_list=[self.query_object_points_transformed], 
                         label_list=[self.query_model.name], 
                         color_list=['b'],
                         title="Transformed Query Object Points",
                         show_plot=show_plot,
                         save_plot=False,
                         scale=0.3,
                        #  scale=1,
                        #  center=[0.7,-0.3,0.3]
                        # center=[-1, 0,1]
                        )        
        self.plot_points(points_list=[self.train_object_points, self.query_object_points_transformed],
                            label_list=[self.train_model.name, self.query_model.name],
                            color_list=['r', 'b'],
                            title="Merged Object Points",
                            show_plot=show_plot,
                            save_plot=False,
                         scale=0.3,
                        #  scale=1,
                        #  center=[0.7,-0.3,0.3]
                        # center=[-1, 0,1]
                        )            
        ## plot camera poses
        self.plot_poses(camera_positions_list=[self.train_camera_positions, self.query_camera_positions_transformed],
                        camera_directions_list=[self.train_camera_directions, self.query_camera_directions_transformed],
                        label_list=[self.train_model.name, self.query_model.name],
                        color_list=['r', 'b'],
                        title="Merged Camera Poses",
                        show_plot=show_plot,
                        save_plot=save_plot) 
        self.logger.info("Processed all images in Input")
        
        ## merge model
        self.merge_model()
        self.logger.info("Model Merged")
    


