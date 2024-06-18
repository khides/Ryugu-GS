import os
import cv2
import numpy as np
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Detectron2の設定
def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.015  # スコア閾値を下げる
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg

# マスク生成関数
def generate_mask(image_path, output_path, predictor):
    image = cv2.imread(image_path)
    outputs = predictor(image)
    masks = outputs["instances"].pred_masks.cpu().numpy()
    
    if len(masks) > 0:
        mask_image = np.zeros(image.shape[:2], dtype=np.uint8)
        for mask in masks:
            mask_image = np.maximum(mask_image, mask.astype(np.uint8) * 255)
    else:
        mask_image = np.zeros(image.shape[:2], dtype=np.uint8)
        # mask_image = np.ones(image.shape[:2], dtype=np.uint8) * 255  # マスクが見つからない場合は全白にする

        print("mask not found")
    
    cv2.imwrite(output_path, mask_image)
    print(f"Saved mask image to: {output_path}")
# ディレクトリ内の画像に対してマスクを生成する関数
def generate_masks(input_dir, output_dir, predictor):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".jpeg")  or filename.endswith(".png"):
            image_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".png")
            generate_mask(image_path, output_path, predictor)

if __name__ == "__main__":
    # 入力ディレクトリと出力ディレクトリのパス
    input_directory = "./Ryugu_Data/Ryugu_mask_3/Input"
    output_directory = "./Ryugu_Data/Ryugu_mask_3/Mask"
    
    cfg = setup_cfg()
    predictor = DefaultPredictor(cfg)

    # マスク生成の実行
    generate_masks(input_directory, output_directory, predictor)
