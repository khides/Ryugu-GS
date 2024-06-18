import astropy.io.fits as fits
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

FITSDIR = "./Ryugu_FITS"
JPEGDIR = "./Ryugu_JPEG"


def convert_fits_to_jpeg(fits_file, output_dir):
    # FITSファイルを開く
    with fits.open(fits_file) as hdul:
        # 画像データを含むHDUを取得（ここでは1番目のHDUと仮定）
        data = hdul[1].data

        # データを正規化
        normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))

        # PILを使用して画像を保存
        image = Image.fromarray((normalized_data * 255).astype(np.uint8))
        output_path = os.path.join(output_dir, os.path.basename(fits_file) + '.jpeg')
        image.save(output_path)

def process_directory(fits_dir, jpeg_dir):
    # 出力ディレクトリの作成
    os.makedirs(jpeg_dir, exist_ok=True)

    # ディレクトリ内の全ての.fitファイルに対して処理
    for root, dirs, files in os.walk(fits_dir):
        for file in files:
            if file.endswith('.fit') and 'tvf' in file:
                fits_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, fits_dir)
                output_dir = os.path.join(jpeg_dir, relative_path)
                os.makedirs(output_dir, exist_ok=True)
                convert_fits_to_jpeg(fits_path, output_dir)


if __name__ == "__main__":    
    # メインディレクトリを指定して処理を開始
    process_directory(FITSDIR, JPEGDIR)