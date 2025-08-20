#!/usr/bin/env python3
"""
CLAHE Histogram Adjustment Tool
画像のヒストグラム調整（CLAHE）を行うスクリプト

Usage:
    python adjust_histogram.py --input-dir ./input_images --output-dir ./output_images
    python adjust_histogram.py --input-file image.jpg --output-file adjusted.jpg
"""

import cv2
import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

class HistogramAdjuster:
    """ヒストグラム調整（CLAHE）を行うクラス"""
    
    def __init__(self, clip_limit: float = 20.0, tile_grid_size: Tuple[int, int] = (16, 16)):
        """
        初期化
        
        Args:
            clip_limit: CLAHE のクリップ制限値
            tile_grid_size: タイルグリッドサイズ
        """
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        CLAHE（Contrast Limited Adaptive Histogram Equalization）を適用
        
        Args:
            image: 入力画像（BGR or グレースケール）
            
        Returns:
            CLAHE適用後のグレースケール画像
        """
        try:
            # 既にグレースケールかチェック
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # CLAHEを適用
            clahe = cv2.createCLAHE(
                clipLimit=self.clip_limit, 
                tileGridSize=self.tile_grid_size
            )
            enhanced = clahe.apply(gray)
            
            return enhanced
            
        except Exception as e:
            print(f"CLAHE適用エラー: {e}")
            return image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def is_image_file(self, file_path: Path) -> bool:
        """画像ファイルかどうかをチェック"""
        return file_path.suffix.lower() in self.supported_extensions
    
    def process_single_image(self, input_path: Path, output_path: Path) -> bool:
        """
        単一画像を処理
        
        Args:
            input_path: 入力画像パス
            output_path: 出力画像パス
            
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
            
            # CLAHE適用
            enhanced_image = self.apply_clahe(image)
            
            # 出力ディレクトリ作成
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 画像保存
            success = cv2.imwrite(str(output_path), enhanced_image)
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
                         preserve_structure: bool = True) -> Tuple[int, int]:
        """
        ディレクトリ内の全画像を処理
        
        Args:
            input_dir: 入力ディレクトリ
            output_dir: 出力ディレクトリ
            preserve_structure: ディレクトリ構造を保持するか
            
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
                output_file = output_dir / rel_path
            else:
                output_file = output_dir / image_file.name
            
            # 処理実行
            if self.process_single_image(image_file, output_file):
                success_count += 1
            else:
                failure_count += 1
        
        return success_count, failure_count

def main():
    parser = argparse.ArgumentParser(
        description="CLAHE（Contrast Limited Adaptive Histogram Equalization）による画像のヒストグラム調整",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # ディレクトリ一括処理
  python adjust_histogram.py --input-dir ./images --output-dir ./enhanced
  
  # 単一ファイル処理
  python adjust_histogram.py --input-file image.jpg --output-file enhanced.jpg
  
  # パラメータ調整
  python adjust_histogram.py --input-dir ./images --output-dir ./enhanced --clip-limit 40.0 --tile-size 8
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
        default=Path("./enhanced_images"),
        help="出力ディレクトリ（デフォルト: ./enhanced_images）"
    )
    
    # CLAHEパラメータ
    parser.add_argument(
        "--clip-limit",
        type=float,
        default=20.0,
        help="CLAHEクリップ制限値（デフォルト: 20.0）"
    )
    
    parser.add_argument(
        "--tile-size",
        type=int,
        default=16,
        help="タイルグリッドサイズ（NxN, デフォルト: 16）"
    )
    
    parser.add_argument(
        "--flat-output",
        action="store_true",
        help="出力をフラット構造にする（ディレクトリ構造を保持しない）"
    )
    
    args = parser.parse_args()
    
    # 引数検証
    if args.input_file and not args.output_file:
        print("エラー: --input-file を使用する場合は --output-file も指定してください")
        sys.exit(1)
    
    if args.output_file and not args.input_file:
        print("エラー: --output-file は --input-file と組み合わせて使用してください")
    
    if args.tile_size <= 0:
        print("エラー: タイルサイズは正の整数である必要があります")
        sys.exit(1)
    
    print("=== CLAHE Histogram Adjustment Tool ===")
    print(f"クリップ制限値: {args.clip_limit}")
    print(f"タイルグリッドサイズ: {args.tile_size}x{args.tile_size}")
    
    try:
        adjuster = HistogramAdjuster(
            clip_limit=args.clip_limit,
            tile_grid_size=(args.tile_size, args.tile_size)
        )
        
        if args.input_file:
            # 単一ファイル処理
            print(f"入力ファイル: {args.input_file}")
            print(f"出力ファイル: {args.output_file}")
            
            if not args.input_file.exists():
                print(f"エラー: 入力ファイルが存在しません - {args.input_file}")
                sys.exit(1)
            
            if not adjuster.is_image_file(args.input_file):
                print(f"エラー: サポートされていないファイル形式 - {args.input_file}")
                sys.exit(1)
            
            success = adjuster.process_single_image(args.input_file, args.output_file)
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
            
            success_count, failure_count = adjuster.process_directory(
                args.input_dir,
                args.output_dir,
                preserve_structure=not args.flat_output
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
