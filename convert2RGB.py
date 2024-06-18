import os
from PIL import Image

def convert_images_to_rgb(source_directory, output_directory=None):
    if output_directory is None:
        output_directory = source_directory

    # ディレクトリが存在しない場合は作成
    os.makedirs(output_directory, exist_ok=True)

    # ソースディレクトリ内のすべてのJPEGファイルを処理
    for filename in os.listdir(source_directory):
        if filename.lower().endswith(".jpeg") or filename.lower().endswith(".jpg"):
            file_path = os.path.join(source_directory, filename)
            output_path = os.path.join(output_directory, filename)

            # 画像を開いてRGBに変換
            with Image.open(file_path) as img:
                rgb_image = img.convert('RGB')
                rgb_image.save(output_path)

            print(f"Converted and saved {filename} to {output_path}")

if __name__ == "__main__":    
    # 使用例
    source_dir = 'D:/Users/taiko/3D-gaussian-splatting/Ryugu_Data/Ryugu_pole4/Input'
    output_dir = 'D:/Users/taiko/3D-gaussian-splatting/Ryugu_Data/Ryugu_pole4/images'  # 新しいディレクトリに保存する場合
    convert_images_to_rgb(source_dir, output_dir)
