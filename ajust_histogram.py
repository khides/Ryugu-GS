import os
import cv2
import numpy as np
from skimage import exposure

def load_images_from_directory(directory):
    images = []
    filenames = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            img = cv2.imread(os.path.join(directory, filename))
            if img is not None:
                images.append(img)
                filenames.append(filename)
    return images, filenames

def match_histograms(source, reference):
    matched = exposure.match_histograms(source, reference, multichannel=True)
    return matched

def save_images_to_directory(images, filenames, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    for img, filename in zip(images, filenames):
        cv2.imwrite(os.path.join(directory, filename), img)

def main(source_dir, reference_image_path, output_dir):
    reference_image = cv2.imread(reference_image_path)
    if reference_image is None:
        print("Reference image not found or could not be loaded.")
        return
    
    images, filenames = load_images_from_directory(source_dir)
    if not images:
        print("No images found in source directory.")
        return
    
    matched_images = []
    for img in images:
        if img.shape == reference_image.shape:
            matched_images.append(match_histograms(img, reference_image))
        else:
            print("Image shape mismatch:", img.shape, reference_image.shape)
            matched_images.append(img)  # Shape mismatchの場合は元の画像をそのまま使う

    save_images_to_directory(matched_images, filenames, output_dir)
    print("All images have been processed and saved to", output_dir)

if __name__ == "__main__":
    source_dir = "./Ryugu_Data/Ryugu_mask_3-1/Input3"  # 画像が保存されているディレクトリ
    reference_image_path = "./Ryugu_Data/Ryugu_mask_3-1/images/hyb2_onc_20180710_060508_tvf_l2a.fit.jpeg"  # 参照画像のパス
    output_dir = "./Ryugu_Data/Ryugu_mask_3-1/Input3"  # 調整された画像を保存するディレクトリ
    
    main(source_dir, reference_image_path, output_dir)
