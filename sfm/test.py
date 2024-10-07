import numpy as np
import cv2

# ダミーの3Dポイントの対応を生成
np.random.seed(42)  # 再現性のためにシードを設定

# 基本の3Dポイント群（10点のランダムな3Dポイント）
base_object_points = np.random.rand(10, 3) * 10  # 10x3のランダムな3D座標

# 回転行列 (45度回転) と並進ベクトル (x=2, y=3, z=4) を適用したポイント群
theta = np.radians(45)  # 45度
rotation_matrix = np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta), np.cos(theta), 0],
    [0, 0, 1]
])  # z軸周りの45度回転

translation_vector = np.array([2, 3, 4])  # 並進ベクトル

# 変換後の追加3Dポイント群
additional_object_points = np.dot(base_object_points, rotation_matrix.T) + translation_vector

# # RANSACを用いて座標変換（回転行列と並進ベクトル）を推定
# rotation_matrix_est, translation_vector_est, inliers = estimate_transformation_ransac(
#     base_object_points, additional_object_points
# )

# 推定されたtransformation_matrixをそのまま使う関数
def apply_affine_transformation(points, transformation_matrix):
    # 同次座標系にするため、pointsに1列の1を追加
    ones_column = np.ones((points.shape[0], 1))
    points_homogeneous = np.hstack((points, ones_column))  # (N, 4)
    
    # アフィン変換行列を適用する (4x3のtransformation_matrixに対して4xNのpoints_homogeneousを掛ける)
    transformed_points = np.dot(points_homogeneous, transformation_matrix.T)  # (N, 3)
    
    return transformed_points

# RANSACによって推定されたアフィン変換行列
success, transformation_matrix, inliers = cv2.estimateAffine3D(
    additional_object_points, base_object_points
)

if success:
    # 推定されたアフィン変換行列を追加の3Dポイント群に適用
    transformed_additional_points_affine = apply_affine_transformation(additional_object_points, transformation_matrix)

    # 結果の表示
    print("元の3Dポイント群:\n", base_object_points)
    print("推定されたアフィン変換後のポイント:\n", transformed_additional_points_affine)

    # 誤差を計算
    error_affine = np.linalg.norm(base_object_points - transformed_additional_points_affine, axis=1)
    print("\nポイントごとの誤差（アフィン変換）:\n", error_affine)
    print(f"平均誤差: {np.mean(error_affine)}")
else:
    print("RANSACによる座標変換の推定に失敗しました")
