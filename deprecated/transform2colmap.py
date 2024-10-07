import json
import os
import numpy as np
from scipy.spatial.transform import Rotation as R

# JSONファイルの読み込み
def load_transforms(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

json_path = './Ryugu_Data/Ryugu_camera_2/blender/transforms_train.json'
data = load_transforms(json_path)

def save_colmap_cameras_and_images(data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'cameras.txt'), 'w') as cam_file, \
         open(os.path.join(output_dir, 'images.txt'), 'w') as img_file:

        # 書式設定
        cam_format = '1 PINHOLE {w} {h} {fx} {fy} {cx} {cy}\n'
        img_format = '{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {file_path}\n'

        # カメラ情報を書き込み（例として1つのカメラ）
        w, h = data['frames'][0]['image_resolution']
        fx, fy, cx, cy = 1.0, 1.0, w / 2.0, h / 2.0  # 仮の値、実際には適切な焦点距離と中心点を設定
        cam_file.write(cam_format.format(w=w, h=h, fx=fx, fy=fy, cx=cx, cy=cy))

        # 画像とカメラポーズ情報を書き込み
        for i, frame in enumerate(data['frames']):
            transform_matrix = np.array(frame['transform_matrix'])
            rotation_matrix = transform_matrix[:3, :3]
            translation_vector = transform_matrix[:3, 3]

            # 回転行列からクォータニオンを計算
            rotation = R.from_matrix(rotation_matrix)
            qx, qy, qz, qw = rotation.as_quat()

            img_file.write(img_format.format(
                image_id=i+1,
                qw=qw, qx=qx, qy=qy, qz=qz,
                tx=translation_vector[0],
                ty=translation_vector[1],
                tz=translation_vector[2],
                camera_id=1,
                file_path=frame['file_path']
            ))

output_dir = '../Ryugu_Data/Ryugu_camera_2/colmap_data'
save_colmap_cameras_and_images(data, output_dir)
