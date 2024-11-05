import cv2
import os
from omegaconf import OmegaConf
from notice import send_notification

def apply_clahe(image, clipLimit=20.0, tileGridSize=(16, 16)):
    # グレースケールに変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # CLAHEを適用
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(16, 16))
    cl1 = clahe.apply(gray)
    return cl1

def process_images(input_dir, output_dir, clipLimit=20.0, tileGridSize=(16, 16)):
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
        processed_image = apply_clahe(image, clipLimit=clipLimit, tileGridSize=tileGridSize)
        
        # 処理済み画像を保存
        output_path = os.path.join(output_dir, image_file)
        cv2.imwrite(output_path, processed_image)
        print(f"{output_path} に保存しました")

if __name__ == "__main__":
    with open("config.yaml", mode="r") as f:
        conf = OmegaConf.load(f)
    # ディレクトリのパスを指定
    input_directory = conf.masked_path
    output_directory = conf.input_path
    send_notification(
        file = __file__,
        webhook_url=conf.webhook_url,
        method=process_images ,
        input_dir = input_directory,
        output_dir = output_directory,
        clipLimit = conf.clahe_clip_limit,
        
        )
    # # 画像を処理
    # process_images(input_directory, output_directory)
