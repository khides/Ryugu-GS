#!/usr/bin/env python3
"""
FITS to JPEG Converter
FITSファイルをJPEG画像に変換するスクリプト

Usage:
    python fits_to_jpeg.py --input-dir ./hayabusa2_data --output-dir ./jpeg_images
    python fits_to_jpeg.py --input-file data.fits --output-file output.jpg
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union
import warnings

try:
    from astropy.io import fits
    from astropy.visualization import ZScaleInterval, ImageNormalize
    from astropy.utils.exceptions import AstropyWarning
except ImportError:
    print("エラー: astropy が必要です。次のコマンドでインストールしてください:")
    print("pip install astropy")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("エラー: Pillow が必要です。次のコマンドでインストールしてください:")
    print("pip install Pillow")
    sys.exit(1)

# astropy の警告を抑制
warnings.filterwarnings('ignore', category=AstropyWarning)


class FitsToJpegConverter:
    """FITSファイルをJPEG画像に変換するクラス"""
    
    def __init__(self, quality: int = 95, stretch: str = "zscale"):
        """
        初期化
        
        Args:
            quality: JPEG品質 (1-100)
            stretch: 画像ストレッチ方法 ("zscale", "minmax", "percentile")
        """
        self.quality = quality
        self.stretch = stretch
        
        if stretch not in ["zscale", "minmax", "percentile"]:
            raise ValueError("stretch must be 'zscale', 'minmax', or 'percentile'")
    
    def read_fits_data(self, fits_path: Path) -> Optional[np.ndarray]:
        """
        FITSファイルからデータを読み取り
        
        Args:
            fits_path: FITSファイルのパス
            
        Returns:
            画像データ配列、読み取りに失敗した場合はNone
        """
        try:
            with fits.open(fits_path) as hdul:
                # 最初の画像データを取得
                for hdu in hdul:
                    if hdu.data is not None and len(hdu.data.shape) >= 2:
                        data = hdu.data
                        
                        # 3次元以上の場合は最初のスライスを使用
                        while len(data.shape) > 2:
                            data = data[0]
                        
                        # NaN値を0に置換
                        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                        
                        return data
                
                print(f"警告: {fits_path} に有効な画像データが見つかりません")
                return None
                
        except Exception as e:
            print(f"エラー: {fits_path} の読み取りに失敗しました - {e}")
            return None
    
    def normalize_data(self, data: np.ndarray) -> np.ndarray:
        """
        データを0-255の範囲に正規化
        
        Args:
            data: 入力データ配列
            
        Returns:
            正規化された8ビット画像データ
        """
        if self.stretch == "zscale":
            # ZScale normalization (天文画像で一般的)
            try:
                interval = ZScaleInterval()
                norm = ImageNormalize(data, interval=interval)
                normalized = norm(data)
            except Exception:
                # ZScaleが失敗した場合はminmaxにフォールバック
                normalized = self._minmax_normalize(data)
        
        elif self.stretch == "minmax":
            normalized = self._minmax_normalize(data)
        
        elif self.stretch == "percentile":
            # 2%と98%でクリッピング
            p2, p98 = np.percentile(data[np.isfinite(data)], [2, 98])
            normalized = np.clip((data - p2) / (p98 - p2), 0, 1)
        
        # 0-255の範囲に変換
        normalized = np.clip(normalized * 255, 0, 255).astype(np.uint8)
        
        return normalized
    
    def _minmax_normalize(self, data: np.ndarray) -> np.ndarray:
        """最小値-最大値による正規化"""
        finite_data = data[np.isfinite(data)]
        if len(finite_data) == 0:
            return np.zeros_like(data)
        
        min_val, max_val = finite_data.min(), finite_data.max()
        if max_val == min_val:
            return np.zeros_like(data)
        
        return (data - min_val) / (max_val - min_val)
    
    def convert_fits_to_jpeg(self, input_path: Path, output_path: Path) -> bool:
        """
        単一のFITSファイルをJPEGに変換
        
        Args:
            input_path: 入力FITSファイルパス
            output_path: 出力JPEGファイルパス
            
        Returns:
            変換成功時True
        """
        print(f"変換中: {input_path.name}")
        
        # FITSデータ読み取り
        data = self.read_fits_data(input_path)
        if data is None:
            return False
        
        # データ正規化
        normalized_data = self.normalize_data(data)
        
        # PIL Imageに変換
        # 画像の向きを調整（通常、天文画像は上下反転している）
        image_array = np.flipud(normalized_data)
        
        try:
            image = Image.fromarray(image_array, mode='L')  # グレースケール
            
            # 出力ディレクトリ作成
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # JPEG保存
            image.save(output_path, "JPEG", quality=self.quality, optimize=True)
            
            print(f"完了: {output_path}")
            return True
            
        except Exception as e:
            print(f"エラー: JPEG保存に失敗しました - {e}")
            return False
    
    def convert_directory(self, input_dir: Path, output_dir: Path, 
                         preserve_structure: bool = True) -> Tuple[int, int]:
        """
        ディレクトリ内の全FITSファイルを変換
        
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
        
        # FITSファイルを再帰的に検索
        fits_extensions = ['.fits', '.fit', '.fts']
        fits_files = []
        
        for ext in fits_extensions:
            fits_files.extend(input_dir.rglob(f"*{ext}"))
            fits_files.extend(input_dir.rglob(f"*{ext.upper()}"))
        
        if not fits_files:
            print(f"FITSファイルが見つかりません: {input_dir}")
            return 0, 0
        
        print(f"見つかったFITSファイル: {len(fits_files)}個")
        
        success_count = 0
        failure_count = 0
        
        for fits_file in fits_files:
            # 出力ファイルパスを決定
            if preserve_structure:
                # 相対パスを保持
                rel_path = fits_file.relative_to(input_dir)
                output_file = output_dir / rel_path.with_suffix('.jpg')
            else:
                # フラット構造
                output_file = output_dir / f"{fits_file.stem}.jpg"
            
            # 変換実行
            if self.convert_fits_to_jpeg(fits_file, output_file):
                success_count += 1
            else:
                failure_count += 1
        
        return success_count, failure_count


def main():
    parser = argparse.ArgumentParser(
        description="FITSファイルをJPEG画像に変換",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # ディレクトリ一括変換
  python fits_to_jpeg.py --input-dir ./hayabusa2_data --output-dir ./jpeg_images
  
  # 単一ファイル変換
  python fits_to_jpeg.py --input-file data.fits --output-file output.jpg
  
  # 画像ストレッチ方法指定
  python fits_to_jpeg.py --input-dir ./data --output-dir ./images --stretch percentile
        """
    )
    
    # 入力指定（排他的）
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-file",
        type=Path,
        help="入力FITSファイル"
    )
    input_group.add_argument(
        "--input-dir", 
        type=Path,
        help="入力ディレクトリ（FITSファイルを再帰的に検索）"
    )
    
    # 出力指定
    parser.add_argument(
        "--output-file",
        type=Path,
        help="出力JPEGファイル（--input-fileと組み合わせ）"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./jpeg_output"),
        help="出力ディレクトリ（デフォルト: ./jpeg_output）"
    )
    
    # 変換オプション
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        choices=range(1, 101),
        help="JPEG品質 (1-100, デフォルト: 95)"
    )
    
    parser.add_argument(
        "--stretch",
        choices=["zscale", "minmax", "percentile"],
        default="zscale",
        help="画像ストレッチ方法（デフォルト: zscale）"
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
        sys.exit(1)
    
    print("=== FITS to JPEG Converter ===")
    print(f"画像ストレッチ: {args.stretch}")
    print(f"JPEG品質: {args.quality}")
    
    try:
        converter = FitsToJpegConverter(
            quality=args.quality,
            stretch=args.stretch
        )
        
        if args.input_file:
            # 単一ファイル変換
            print(f"入力ファイル: {args.input_file}")
            print(f"出力ファイル: {args.output_file}")
            
            if not args.input_file.exists():
                print(f"エラー: 入力ファイルが存在しません - {args.input_file}")
                sys.exit(1)
            
            success = converter.convert_fits_to_jpeg(args.input_file, args.output_file)
            if success:
                print("変換完了")
            else:
                print("変換失敗")
                sys.exit(1)
        
        else:
            # ディレクトリ一括変換
            print(f"入力ディレクトリ: {args.input_dir}")
            print(f"出力ディレクトリ: {args.output_dir}")
            print(f"ディレクトリ構造保持: {not args.flat_output}")
            
            success_count, failure_count = converter.convert_directory(
                args.input_dir,
                args.output_dir,
                preserve_structure=not args.flat_output
            )
            
            print(f"\n=== 変換結果 ===")
            print(f"成功: {success_count}ファイル")
            print(f"失敗: {failure_count}ファイル")
            print(f"合計: {success_count + failure_count}ファイル")
            
            if failure_count > 0:
                sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n変換が中断されました")
        sys.exit(1)
    except Exception as e:
        print(f"予期しないエラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()