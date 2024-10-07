import cv2
import os
import numpy as np

def apply_clahe(image):
    # グレースケールに変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # CLAHEを適用
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(16, 16))
    cl1 = clahe.apply(gray)
    return cl1

def process_images(input_dir, output_dir):
    # 入力ディレクトリの全ての画像ファイルを取得
    image_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    
    # 出力ディレクトリが存在しない場合は作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for image_file in image_files:
        # 画像を読み込み
        image_path = os.path.join(input_dir, image_file)
        image = cv2.imread(image_path)
        
        # CLAHEを適用
        processed_image = apply_clahe(image)
        
        # 処理済み画像を保存
        output_path = os.path.join(output_dir, image_file)
        cv2.imwrite(output_path, processed_image)
        print(f"{output_path} に保存しました")

# ディレクトリのパスを指定
input_directory = './Ryugu_Data/Ryugu_mask_3-1/20180720'
output_directory = './Ryugu_Data/Ryugu_CLAHE/Input4'

# 画像を処理
process_images(input_directory, output_directory)
