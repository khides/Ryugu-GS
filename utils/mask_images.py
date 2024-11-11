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
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold # スコア閾値を設定（必要に応じて調整）
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg

# 背景マスキング関数
def apply_mask_and_save(input_dir, output_dir, predictor):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpeg") or filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)
            outputs = predictor(image)
            masks = outputs["instances"].pred_masks.cpu().numpy()
            
            masked_image = np.zeros_like(image)  # 背景を黒に設定
            for mask in masks:
                masked_image[mask] = image[mask]

            output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".jpeg")
            cv2.imwrite(output_path, masked_image)
            print(f"Saved masked image to: {output_path}")
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
        file = __file__,
        webhook_url=conf.webhook_url,
        method=apply_mask_and_save ,
        input_dir = input_directory,
        output_dir = output_directory,
        predictor = predictor
    )

    # # マスキングと保存の実行
    # apply_mask_and_save(input_directory, output_directory, predictor)
