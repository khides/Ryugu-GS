#!/usr/bin/env python3
"""
Advanced Image Masking Tool
高度な画像マスキングツール

複数のマスキング手法を提供:
1. Detectron2による物体検出マスキング (AI)
2. 色範囲による閾値マスキング
3. 輪郭検出マスキング
4. 手動マスク適用

Usage:
    python image_masking.py --input-dir ./images --output-dir ./masked --method detectron2
    python image_masking.py --input-file image.jpg --output-file masked.png --method color-range
"""

import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union
import warnings

# Detectron2の依存関係（オプション）
try:
    import torch
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    from detectron2.data import MetadataCatalog
    DETECTRON2_AVAILABLE = True
except ImportError:
    DETECTRON2_AVAILABLE = False
    print("警告: Detectron2が利用できません。AI masking機能は無効です。")
    print("インストール方法: https://detectron2.readthedocs.io/en/latest/tutorials/install.html")


class ImageMaskingTool:
    """画像マスキング処理クラス"""
    
    def __init__(self, method: str = "color-range", **kwargs):
        """
        初期化
        
        Args:
            method: マスキング手法 ("detectron2", "color-range", "contour", "manual")
            **kwargs: 各手法固有のパラメータ
        """
        self.method = method
        self.supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Detectron2設定
        if method == "detectron2":
            if not DETECTRON2_AVAILABLE:
                raise ValueError("Detectron2が必要ですがインストールされていません")
            self.predictor = self._setup_detectron2(**kwargs)
        
        # 色範囲マスキング設定
        elif method == "color-range":
            self.color_lower = kwargs.get('color_lower', (0, 0, 0))
            self.color_upper = kwargs.get('color_upper', (50, 50, 50))
            self.color_space = kwargs.get('color_space', 'BGR')
        
        # 輪郭検出設定
        elif method == "contour":
            self.min_area = kwargs.get('min_area', 1000)
            self.max_area = kwargs.get('max_area', 100000)
            self.threshold_val = kwargs.get('threshold_val', 127)
    
    def _setup_detectron2(self, threshold: float = 0.05, model: str = "mask_rcnn_R_50_FPN_3x") -> DefaultPredictor:
        """Detectron2の設定"""
        cfg = get_cfg()
        config_file = f"COCO-InstanceSegmentation/{model}.yaml"
        cfg.merge_from_file(model_zoo.get_config_file(config_file))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        return DefaultPredictor(cfg)
    
    def is_image_file(self, file_path: Path) -> bool:
        """画像ファイルかどうかをチェック"""
        return file_path.suffix.lower() in self.supported_extensions
    
    def mask_with_detectron2(self, image: np.ndarray, 
                           output_mode: str = "transparent") -> np.ndarray:
        """
        Detectron2による物体検出マスキング
        
        Args:
            image: 入力画像
            output_mode: 出力モード ("transparent", "black_background", "original_background")
            
        Returns:
            マスクされた画像
        """
        if not DETECTRON2_AVAILABLE:
            raise ValueError("Detectron2が利用できません")
        
        outputs = self.predictor(image)
        masks = outputs["instances"].pred_masks.cpu().numpy()
        
        if len(masks) == 0:
            print("警告: オブジェクトが検出されませんでした")
            return image
        
        # 全マスクを統合
        combined_mask = np.any(masks, axis=0)
        
        if output_mode == "transparent":
            # RGBA画像作成（透明背景）
            h, w = image.shape[:2]
            result = np.zeros((h, w, 4), dtype=np.uint8)
            result[combined_mask, :3] = image[combined_mask]
            result[combined_mask, 3] = 255  # アルファ値
            return result
        
        elif output_mode == "black_background":
            # 黒背景
            result = np.zeros_like(image)
            result[combined_mask] = image[combined_mask]
            return result
        
        else:  # original_background
            # 検出されたオブジェクト以外をぼかし
            result = image.copy()
            blurred = cv2.GaussianBlur(image, (21, 21), 0)
            result[~combined_mask] = blurred[~combined_mask]
            return result
    
    def mask_with_color_range(self, image: np.ndarray,
                            invert_mask: bool = False) -> np.ndarray:
        """
        色範囲による閾値マスキング
        
        Args:
            image: 入力画像
            invert_mask: マスクを反転するか
            
        Returns:
            マスクされた画像
        """
        # 色空間変換
        if self.color_space == 'HSV':
            converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.color_space == 'LAB':
            converted = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        else:  # BGR
            converted = image.copy()
        
        # 色範囲でマスク作成
        mask = cv2.inRange(converted, self.color_lower, self.color_upper)
        
        if invert_mask:
            mask = cv2.bitwise_not(mask)
        
        # RGBA画像作成
        h, w = image.shape[:2]
        result = np.zeros((h, w, 4), dtype=np.uint8)
        result[mask > 0, :3] = image[mask > 0]
        result[mask > 0, 3] = 255
        
        return result
    
    def mask_with_contours(self, image: np.ndarray) -> np.ndarray:
        """
        輪郭検出によるマスキング
        
        Args:
            image: 入力画像
            
        Returns:
            マスクされた画像
        """
        # グレースケール変換
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 二値化
        _, binary = cv2.threshold(gray, self.threshold_val, 255, cv2.THRESH_BINARY)
        
        # 輪郭検出
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 面積でフィルタリング
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area <= area <= self.max_area:
                filtered_contours.append(contour)
        
        # マスク作成
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, filtered_contours, 255)
        
        # RGBA画像作成
        result = np.zeros((h, w, 4), dtype=np.uint8)
        result[mask > 0, :3] = image[mask > 0]
        result[mask > 0, 3] = 255
        
        return result
    
    def apply_manual_mask(self, image: np.ndarray, mask_path: Path) -> np.ndarray:
        """
        手動マスクファイルを適用
        
        Args:
            image: 入力画像
            mask_path: マスク画像のパス
            
        Returns:
            マスクされた画像
        """
        if not mask_path.exists():
            raise ValueError(f"マスクファイルが見つかりません: {mask_path}")
        
        # マスク画像読み込み
        mask_image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_image is None:
            raise ValueError(f"マスク画像を読み込めません: {mask_path}")
        
        # サイズ調整
        h, w = image.shape[:2]
        if mask_image.shape != (h, w):
            mask_image = cv2.resize(mask_image, (w, h))
        
        # RGBA画像作成
        result = np.zeros((h, w, 4), dtype=np.uint8)
        result[mask_image > 127, :3] = image[mask_image > 127]
        result[mask_image > 127, 3] = 255
        
        return result
    
    def process_single_image(self, input_path: Path, output_path: Path,
                           mask_path: Optional[Path] = None, **kwargs) -> bool:
        """
        単一画像の処理
        
        Args:
            input_path: 入力画像パス
            output_path: 出力画像パス
            mask_path: 手動マスクパス（手動モード用）
            
        Returns:
            処理成功時True
        """
        try:
            print(f"処理中: {input_path.name}")
            
            # 画像読み込み
            image = cv2.imread(str(input_path))
            if image is None:
                print(f"エラー: 画像を読み込めません - {input_path}")
                return False
            
            # マスキング処理
            if self.method == "detectron2":
                result = self.mask_with_detectron2(image, **kwargs)
            elif self.method == "color-range":
                result = self.mask_with_color_range(image, **kwargs)
            elif self.method == "contour":
                result = self.mask_with_contours(image)
            elif self.method == "manual":
                if mask_path is None:
                    print(f"エラー: 手動モード用のマスクパスが必要です")
                    return False
                result = self.apply_manual_mask(image, mask_path)
            else:
                print(f"エラー: サポートされていない手法 - {self.method}")
                return False
            
            # 出力ディレクトリ作成
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 画像保存（透明対応のためPNG）
            success = cv2.imwrite(str(output_path), result)
            if success:
                print(f"完了: {output_path}")
                return True
            else:
                print(f"エラー: 画像保存に失敗 - {output_path}")
                return False
                
        except Exception as e:
            print(f"処理エラー {input_path}: {e}")
            return False
    
    def process_directory(self, input_dir: Path, output_dir: Path,
                         preserve_structure: bool = True,
                         mask_dir: Optional[Path] = None) -> Tuple[int, int]:
        """
        ディレクトリ内の全画像を処理
        
        Args:
            input_dir: 入力ディレクトリ
            output_dir: 出力ディレクトリ
            preserve_structure: ディレクトリ構造を保持するか
            mask_dir: 手動マスクディレクトリ（手動モード用）
            
        Returns:
            (成功数, 失敗数) のタプル
        """
        if not input_dir.exists():
            print(f"エラー: 入力ディレクトリが存在しません - {input_dir}")
            return 0, 0
        
        # 画像ファイルを再帰的に検索
        image_files = []
        for ext in self.supported_extensions:
            image_files.extend(input_dir.rglob(f"*{ext}"))
            image_files.extend(input_dir.rglob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"画像ファイルが見つかりません: {input_dir}")
            return 0, 0
        
        print(f"見つかった画像ファイル: {len(image_files)}個")
        
        success_count = 0
        failure_count = 0
        
        for image_file in image_files:
            # 出力パス決定
            if preserve_structure:
                rel_path = image_file.relative_to(input_dir)
                output_file = output_dir / rel_path.with_suffix('.png')
            else:
                output_file = output_dir / f"{image_file.stem}.png"
            
            # 手動マスクパス決定
            mask_path = None
            if self.method == "manual" and mask_dir:
                mask_rel_path = image_file.relative_to(input_dir)
                mask_path = mask_dir / mask_rel_path
                if not mask_path.exists():
                    print(f"警告: マスクファイルが見つかりません - {mask_path}")
                    continue
            
            # 処理実行
            if self.process_single_image(image_file, output_file, mask_path):
                success_count += 1
            else:
                failure_count += 1
        
        return success_count, failure_count


def main():
    parser = argparse.ArgumentParser(
        description="高度な画像マスキングツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
マスキング手法:
  detectron2    : AI物体検出による自動マスキング（要Detectron2）
  color-range   : 色範囲による閾値マスキング
  contour       : 輪郭検出マスキング
  manual        : 手動マスクファイル適用

使用例:
  # AI自動マスキング
  python image_masking.py --input-dir ./images --output-dir ./masked --method detectron2
  
  # 色範囲マスキング（黒背景除去）
  python image_masking.py --input-dir ./images --output-dir ./masked --method color-range --color-upper 50,50,50
  
  # 輪郭検出マスキング
  python image_masking.py --input-file image.jpg --output-file masked.png --method contour --min-area 500
        """
    )
    
    # 入力指定（排他的）
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-file",
        type=Path,
        help="入力画像ファイル"
    )
    input_group.add_argument(
        "--input-dir",
        type=Path,
        help="入力ディレクトリ（画像ファイルを再帰的に検索）"
    )
    
    # 出力指定
    parser.add_argument(
        "--output-file",
        type=Path,
        help="出力画像ファイル（--input-fileと組み合わせ）"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./masked_images"),
        help="出力ディレクトリ（デフォルト: ./masked_images）"
    )
    
    # マスキング手法
    parser.add_argument(
        "--method",
        choices=["detectron2", "color-range", "contour", "manual"],
        default="color-range",
        help="マスキング手法（デフォルト: color-range）"
    )
    
    # Detectron2パラメータ
    parser.add_argument(
        "--detection-threshold",
        type=float,
        default=0.05,
        help="物体検出閾値（デフォルト: 0.05）"
    )
    
    parser.add_argument(
        "--output-mode",
        choices=["transparent", "black_background", "original_background"],
        default="transparent",
        help="Detectron2出力モード（デフォルト: transparent）"
    )
    
    # 色範囲マスキングパラメータ
    parser.add_argument(
        "--color-lower",
        type=str,
        default="0,0,0",
        help="色範囲下限値（BGR/HSV/LAB、例: 0,0,0）"
    )
    
    parser.add_argument(
        "--color-upper",
        type=str,
        default="50,50,50",
        help="色範囲上限値（BGR/HSV/LAB、例: 50,50,50）"
    )
    
    parser.add_argument(
        "--color-space",
        choices=["BGR", "HSV", "LAB"],
        default="BGR",
        help="色空間（デフォルト: BGR）"
    )
    
    parser.add_argument(
        "--invert-mask",
        action="store_true",
        help="色範囲マスクを反転"
    )
    
    # 輪郭検出パラメータ
    parser.add_argument(
        "--min-area",
        type=int,
        default=1000,
        help="輪郭最小面積（デフォルト: 1000）"
    )
    
    parser.add_argument(
        "--max-area",
        type=int,
        default=100000,
        help="輪郭最大面積（デフォルト: 100000）"
    )
    
    parser.add_argument(
        "--threshold-val",
        type=int,
        default=127,
        help="二値化閾値（デフォルト: 127）"
    )
    
    # 手動マスク
    parser.add_argument(
        "--mask-file",
        type=Path,
        help="手動マスクファイル（単一ファイル用）"
    )
    
    parser.add_argument(
        "--mask-dir",
        type=Path,
        help="手動マスクディレクトリ（ディレクトリ処理用）"
    )
    
    # その他オプション
    parser.add_argument(
        "--flat-output",
        action="store_true",
        help="出力をフラット構造にする"
    )
    
    args = parser.parse_args()
    
    # 引数検証
    if args.input_file and not args.output_file:
        print("エラー: --input-file を使用する場合は --output-file も指定してください")
        sys.exit(1)
    
    if args.output_file and not args.input_file:
        print("エラー: --output-file は --input-file と組み合わせて使用してください")
        sys.exit(1)
    
    if args.method == "manual" and not args.mask_file and not args.mask_dir:
        print("エラー: 手動モードではマスクファイルまたはディレクトリが必要です")
        sys.exit(1)
    
    # 色範囲パラメータ解析
    try:
        color_lower = tuple(map(int, args.color_lower.split(',')))
        color_upper = tuple(map(int, args.color_upper.split(',')))
    except ValueError:
        print("エラー: 色範囲値は 'R,G,B' 形式で指定してください")
        sys.exit(1)
    
    print("=== Advanced Image Masking Tool ===")
    print(f"マスキング手法: {args.method}")
    
    try:
        # パラメータ設定
        kwargs = {}
        if args.method == "detectron2":
            kwargs.update({
                'threshold': args.detection_threshold,
                'output_mode': args.output_mode
            })
        elif args.method == "color-range":
            kwargs.update({
                'color_lower': color_lower,
                'color_upper': color_upper,
                'color_space': args.color_space,
                'invert_mask': args.invert_mask
            })
        elif args.method == "contour":
            kwargs.update({
                'min_area': args.min_area,
                'max_area': args.max_area,
                'threshold_val': args.threshold_val
            })
        
        # マスキングツール初期化
        masker = ImageMaskingTool(method=args.method, **kwargs)
        
        if args.input_file:
            # 単一ファイル処理
            print(f"入力ファイル: {args.input_file}")
            print(f"出力ファイル: {args.output_file}")
            
            if not args.input_file.exists():
                print(f"エラー: 入力ファイルが存在しません - {args.input_file}")
                sys.exit(1)
            
            if not masker.is_image_file(args.input_file):
                print(f"エラー: サポートされていないファイル形式 - {args.input_file}")
                sys.exit(1)
            
            success = masker.process_single_image(
                args.input_file, 
                args.output_file,
                mask_path=args.mask_file
            )
            
            if success:
                print("処理完了")
            else:
                print("処理失敗")
                sys.exit(1)
        
        else:
            # ディレクトリ一括処理
            print(f"入力ディレクトリ: {args.input_dir}")
            print(f"出力ディレクトリ: {args.output_dir}")
            print(f"ディレクトリ構造保持: {not args.flat_output}")
            
            success_count, failure_count = masker.process_directory(
                args.input_dir,
                args.output_dir,
                preserve_structure=not args.flat_output,
                mask_dir=args.mask_dir
            )
            
            print(f"\n=== 処理結果 ===")
            print(f"成功: {success_count}ファイル")
            print(f"失敗: {failure_count}ファイル")
            print(f"合計: {success_count + failure_count}ファイル")
            
            if failure_count > 0:
                sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n処理が中断されました")
        sys.exit(1)
    except Exception as e:
        print(f"予期しないエラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()