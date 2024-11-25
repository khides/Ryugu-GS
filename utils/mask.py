import os
import cv2
import numpy as np
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from omegaconf import OmegaConf
from notice import send_notification

# Detectron2の設定
def setup_cfg(threshold=0.05):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  # スコア閾値を設定（必要に応じて調整）
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg

# 背景を透明化するマスキング関数
def apply_mask_and_save_with_transparency(input_dir, output_dir, predictor):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpeg") or filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)
            outputs = predictor(image)
            masks = outputs["instances"].pred_masks.cpu().numpy()
            
            # RGBA画像を初期化（背景透明）
            h, w, _ = image.shape
            transparent_image = np.zeros((h, w, 4), dtype=np.uint8)
            
            for mask in masks:
                # マスクがTrueの部分に元の画像のRGB値をコピー
                transparent_image[mask, :3] = image[mask]
                # アルファ値を255（不透明）に設定
                transparent_image[mask, 3] = 255
            
            output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".png")  # 出力形式はPNG
            cv2.imwrite(output_path, transparent_image)
            print(f"Saved transparent image to: {output_path}")
        else:
            print(f"File {filename} has unsupported format")

if __name__ == "__main__":
    with open("config.yaml", mode="r") as f:
        conf = OmegaConf.load(f)
    # 入力ディレクトリと出力ディレクトリのパス
    input_directory = conf.image_path
    output_directory = conf.masked_path
    
    cfg = setup_cfg(threshold=conf.mask_threshold)
    predictor = DefaultPredictor(cfg)
    send_notification(
        file=__file__,
        webhook_url=conf.webhook_url,
        method=apply_mask_and_save_with_transparency,
        input_dir=input_directory,
        output_dir=output_directory,
        predictor=predictor
    )

    # # 背景を透明化するマスキングと保存の実行
    # apply_mask_and_save_with_transparency(input_directory, output_directory, predictor)
