import open3d as o3d
import numpy as np

# ポイントクラウドを読み込む関数
def load_point_cloud(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

# ダウンサンプリングと特徴量の計算
def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=voxel_size * 2, max_nn=30))
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

# RANSACで初期マッチング
def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    return result

# ICP（Iterative Closest Point）でマッチング精度を向上
def refine_registration(source, target, voxel_size, result_ransac):
    distance_threshold = voxel_size * 0.4
    result_icp = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result_icp

# ポイントクラウドを表示する
def draw_registration_result(source, target, transformation):
    source_temp = source.copy()
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target])

# メイン処理
if __name__ == "__main__":
    # ファイルパスを指定
    source_file = "source_point_cloud.ply"  # 変換する側のモデル
    target_file = "target_point_cloud.ply"  # 座標系の基準になるモデル

    # ポイントクラウドの読み込み
    source_pcd = load_point_cloud(source_file)
    target_pcd = load_point_cloud(target_file)

    # ボクセルサイズを設定
    voxel_size = 0.05  # モデルの解像度に応じて調整

    # ダウンサンプリングと特徴量計算
    source_down, source_fpfh = preprocess_point_cloud(source_pcd, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target_pcd, voxel_size)

    # RANSACによる初期マッチング
    result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)

    # マッチング結果をICPでさらに精密に調整
    result_icp = refine_registration(source_pcd, target_pcd, voxel_size, result_ransac)

    # 座標変換行列の表示
    print("RANSAC transformation:")
    print(result_ransac.transformation)

    print("ICP transformation:")
    print(result_icp.transformation)

    # 結果を表示
    draw_registration_result(source_pcd, target_pcd, result_icp.transformation)
