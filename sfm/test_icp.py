import open3d as o3d
import numpy as np

# 点群の生成（例: 単純な立方体を作成）
pcd1 = o3d.geometry.PointCloud()
np.random.seed(42)
points = np.random.uniform(-1, 1, (10, 3))
print(points)
pcd1.points = o3d.utility.Vector3dVector(points)

# 2つ目の点群（pcd2）は、pcd1に回転と平行移動を加えたもの
pcd2 = pcd1.translate([0.5, 0.5, 0.5])
R = pcd1.get_rotation_matrix_from_xyz((np.pi / 4, np.pi / 4, np.pi / 4))  # 45度回転
pcd2.rotate(R, center=(0, 0, 0))
print(np.asarray(pcd2.points))

# o3d.visualization.draw_geometries([pcd1, pcd2])

# ICPアルゴリズムを実行
threshold = 0.1  # 距離閾値
trans_init = np.eye(4)  # 初期変換行列
reg_p2p = o3d.pipelines.registration.registration_icp(
    pcd1, pcd2, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint()
)

# ICPの結果として得られた変換行列
print("Transformation matrix:")
print(reg_p2p.transformation)

# pcd1を得られた変換で変換
pcd1.transform(reg_p2p.transformation)

print(np.asarray(pcd1.points))
print(np.asarray(pcd2.points))

# 2つの点群を可視化（位置合わせの前後）
print("Visualizing point clouds before and after alignment...")
# o3d.visualization.draw_geometries([pcd1, pcd2], window_name="Aligned Point Clouds")
