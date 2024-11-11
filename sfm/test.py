import quaternion
import numpy as np
import cv2
def quaternion_to_rotation_matrix(q) -> np.ndarray:
    q = np.quaternion(q[0], q[1], q[2], q[3])
    return quaternion.as_rotation_matrix(q)

v = [ 0.4108291,  -0.46848082  ,0.7821414 ]
print(f"v: {v}")

z_axis = np.array([0, 0, 1])
# カメラ方向が z 軸と異なる場合、回転行列 R を計算
if not np.allclose(v, z_axis):
    rotation_axis = np.cross(z_axis, v)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)  # 正規化
    rotation_angle = np.arccos(np.dot(z_axis, v) / (np.linalg.norm(z_axis) * np.linalg.norm(v)))
    R, _ = cv2.Rodrigues(rotation_axis * rotation_angle)
    print(f"R: {R}")
else:
    R = np.eye(3)  # z 軸と一致する場合は単位行列
# 回転行列をクォータニオンに変換
qvec = quaternion.from_rotation_matrix(R)
print(f"qvec: {qvec}")

qvec = [qvec.w, qvec.x, qvec.y, qvec.z]
print(f"qvec: {qvec}")
R = quaternion_to_rotation_matrix(qvec)
print(f"R: {R}")
v = R @ np.array([0, 0, 1])
print(f"v: {v}")