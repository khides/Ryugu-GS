from matplotlib import pyplot as plt 
import sqlite3
import numpy as np 
from PIL import Image, ImageDraw


def extract_features_from_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT image_id, data FROM descriptors")
    descriptors = cursor.fetchall()
    
    image_feature_start_indices = {}
    for image_id, desc in descriptors:
        desc_array = np.frombuffer(desc, dtype=np.uint8).reshape(-1, 128).astype(np.float32)
        image_feature_start_indices[image_id] = desc_array.shape[0]
        
    conn.close()
    return  image_feature_start_indices

def extract_keypoints_from_db(db_path):
    conn = sqlite3.connect(db_path)
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
    return all_keypoints

if __name__ == "__main__":
    db_path1 = "./Ryugu_Data/Ryugu_mask_3/database.db"
    # db_path2 = "./Ryugu_Data/Ryugu_CLAHE/database.db"
    db_path2 = "./Ryugu_Data/Ryugu_CLAHE/database4.db"
    # n_features1 = extract_features_from_db(db_path1)
    # n_features2 = extract_features_from_db(db_path2)
    # # 辞書のキーを取得
    # labels = list(n_features1.keys())

    # # 辞書の値を取得
    # values1 = list(n_features1.values())
    # values2 = list(n_features2.values())

    # # バーの位置を設定
    # x = np.arange(len(labels))
    # width = 0.35  # バーの幅

    # # ヒストグラムの作成
    # fig, ax = plt.subplots()
    # rects1 = ax.bar(x - width/2, values1, width, label='Default', color="b")
    # rects2 = ax.bar(x + width/2, values2, width, label='CLAHE', color="r")
    # ax.legend()
    # plt.show()
    
    
    
    keypointsdict = extract_keypoints_from_db(db_path2)
    keypoints = np.vstack(keypointsdict[1])
    coordinates = keypoints[:,:2]

    # image_path = "./Ryugu_Data/Ryugu_CLAHE/Input/hyb2_onc_20180710_060508_tvf_l2a.fit.jpeg"
    image_path = "./Ryugu_Data/Ryugu_CLAHE/Input4/hyb2_onc_20180720_071727_tvf_l2a.fit.jpeg"
    image = Image.open(image_path)
    image = image.convert("RGB")

    # 描画用のオブジェクトを作成
    draw = ImageDraw.Draw(image)

    # 点の描画
    radius = 1.5              # 点の半径
    for coord in coordinates:
        x, y = coord
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255,0,0), outline=(255,0,0))

    # 画像の保存
    output_path = "output_image_with_points.jpg"
    image.save(output_path)

    # 画像の表示
    image.show()