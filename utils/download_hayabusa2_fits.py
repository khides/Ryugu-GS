#!/usr/bin/env python3
"""
Hayabusa2 ONC FITS Data Downloader
Download FITS files from JAXA DARTS repository for specified date range

Usage:
    python download_hayabusa2_fits.py --start-date 2005-11-02 --end-date 2005-11-19 --output-dir ./hayabusa2_data
"""

import os
import sys
import argparse
import requests
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urljoin
import time
from typing import List, Optional

class HayabusaDataDownloader:
    """Hayabusa2 ONC FITS データダウンローダー"""
    
    BASE_URL = "https://data.darts.isas.jaxa.jp/pub/pds3/hay-a-onc-2-edr-1.0/hayonc_0001/data/"
    
    def __init__(self, output_dir: str = "./hayabusa2_data", delay: float = 1.0):
        """
        初期化
        
        Args:
            output_dir: 保存先ディレクトリ
            delay: リクエスト間の待機時間（秒）
        """
        self.output_dir = Path(output_dir)
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Ryugu-GS Research Tool/1.0'
        })
        
        # 出力ディレクトリ作成
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_date_directories(self, start_date: datetime, end_date: datetime) -> List[str]:
        """
        指定された日付範囲のディレクトリ名リストを生成
        
        Args:
            start_date: 開始日
            end_date: 終了日
            
        Returns:
            YYYYMMDD形式のディレクトリ名リスト
        """
        date_dirs = []
        current_date = start_date
        
        while current_date <= end_date:
            date_str = current_date.strftime("%Y%m%d")
            date_dirs.append(date_str)
            current_date += timedelta(days=1)
            
        return date_dirs
    
    def check_directory_exists(self, date_dir: str) -> bool:
        """
        指定されたディレクトリがサーバー上に存在するかチェック
        
        Args:
            date_dir: YYYYMMDD形式のディレクトリ名
            
        Returns:
            存在する場合True
        """
        url = urljoin(self.BASE_URL, f"{date_dir}/")
        try:
            response = self.session.head(url, timeout=30)
            return response.status_code == 200
        except requests.RequestException as e:
            print(f"ディレクトリチェックエラー {date_dir}: {e}")
            return False
    
    def get_fits_files_in_directory(self, date_dir: str) -> List[str]:
        """
        指定されたディレクトリ内のFITSファイル一覧を取得
        
        Args:
            date_dir: YYYYMMDD形式のディレクトリ名
            
        Returns:
            FITSファイル名のリスト
        """
        url = urljoin(self.BASE_URL, f"{date_dir}/")
        fits_files = []
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # HTMLパースして.fitsまたは.fitファイルを抽出
            content = response.text
            import re
            
            # href="filename.fits" または href="filename.fit" を探す
            fits_pattern = r'href="([^"]*\.fits?)"'
            matches = re.findall(fits_pattern, content, re.IGNORECASE)
            
            for match in matches:
                if match.lower().endswith(('.fits', '.fit')):
                    fits_files.append(match)
                    
        except requests.RequestException as e:
            print(f"ディレクトリ内容取得エラー {date_dir}: {e}")
            
        return fits_files
    
    def download_file(self, date_dir: str, filename: str) -> bool:
        """
        指定されたファイルをダウンロード
        
        Args:
            date_dir: YYYYMMDD形式のディレクトリ名
            filename: ファイル名
            
        Returns:
            成功した場合True
        """
        url = urljoin(self.BASE_URL, f"{date_dir}/{filename}")
        output_path = self.output_dir / date_dir / filename
        
        # 出力ディレクトリ作成
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 既にファイルが存在する場合はスキップ
        if output_path.exists():
            print(f"スキップ: {output_path} (既に存在)")
            return True
        
        try:
            print(f"ダウンロード中: {url}")
            response = self.session.get(url, timeout=60, stream=True)
            response.raise_for_status()
            
            # ファイルサイズ取得
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # 進捗表示
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\r  進捗: {progress:.1f}% ({downloaded}/{total_size} bytes)", end='')
            
            print(f"\n完了: {output_path}")
            return True
            
        except requests.RequestException as e:
            print(f"ダウンロードエラー {filename}: {e}")
            # 不完全なファイルを削除
            if output_path.exists():
                output_path.unlink()
            return False
        
        except Exception as e:
            print(f"予期しないエラー {filename}: {e}")
            if output_path.exists():
                output_path.unlink()
            return False
    
    def download_date_range(self, start_date: datetime, end_date: datetime) -> None:
        """
        指定された日付範囲のデータをダウンロード
        
        Args:
            start_date: 開始日
            end_date: 終了日
        """
        date_dirs = self.get_date_directories(start_date, end_date)
        print(f"対象期間: {start_date.strftime('%Y-%m-%d')} から {end_date.strftime('%Y-%m-%d')}")
        print(f"チェック対象ディレクトリ: {len(date_dirs)}個")
        
        total_files = 0
        successful_downloads = 0
        
        for date_dir in date_dirs:
            print(f"\n=== {date_dir} の処理 ===")
            
            # ディレクトリ存在チェック
            if not self.check_directory_exists(date_dir):
                print(f"ディレクトリが存在しません: {date_dir}")
                continue
            
            # FITS ファイル一覧取得
            fits_files = self.get_fits_files_in_directory(date_dir)
            
            if not fits_files:
                print(f"FITSファイルが見つかりません: {date_dir}")
                continue
            
            print(f"FITSファイル数: {len(fits_files)}")
            
            # 各FITSファイルをダウンロード
            for filename in fits_files:
                total_files += 1
                
                if self.download_file(date_dir, filename):
                    successful_downloads += 1
                
                # レート制限のための待機
                if self.delay > 0:
                    time.sleep(self.delay)
        
        print(f"\n=== ダウンロード完了 ===")
        print(f"総ファイル数: {total_files}")
        print(f"成功: {successful_downloads}")
        print(f"失敗: {total_files - successful_downloads}")
        print(f"保存先: {self.output_dir}")


def parse_date(date_str: str) -> datetime:
    """日付文字列をdatetimeオブジェクトに変換"""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}. Use YYYY-MM-DD.")


def main():
    parser = argparse.ArgumentParser(
        description="Hayabusa2 ONC FITS データダウンローダー",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--start-date", 
        type=parse_date,
        default="2005-11-02",
        help="開始日 (YYYY-MM-DD形式, デフォルト: 2005-11-02)"
    )
    
    parser.add_argument(
        "--end-date",
        type=parse_date, 
        default="2005-11-19",
        help="終了日 (YYYY-MM-DD形式, デフォルト: 2005-11-19)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./hayabusa2_data",
        help="保存先ディレクトリ (デフォルト: ./hayabusa2_data)"
    )
    
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="リクエスト間の待機時間（秒, デフォルト: 1.0）"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="実際にはダウンロードせず、対象ファイルの確認のみ行う"
    )
    
    args = parser.parse_args()
    
    # 日付範囲チェック
    if args.start_date > args.end_date:
        print("エラー: 開始日は終了日より前である必要があります")
        sys.exit(1)
    
    print("=== Hayabusa2 ONC FITS Data Downloader ===")
    print(f"開始日: {args.start_date.strftime('%Y-%m-%d')}")
    print(f"終了日: {args.end_date.strftime('%Y-%m-%d')}")
    print(f"保存先: {args.output_dir}")
    print(f"待機時間: {args.delay}秒")
    
    if args.dry_run:
        print("*** DRY RUN モード - 実際にはダウンロードしません ***")
    
    try:
        downloader = HayabusaDataDownloader(
            output_dir=args.output_dir,
            delay=args.delay
        )
        
        if args.dry_run:
            # ドライランモード: ファイル一覧のみ表示
            date_dirs = downloader.get_date_directories(args.start_date, args.end_date)
            for date_dir in date_dirs:
                if downloader.check_directory_exists(date_dir):
                    fits_files = downloader.get_fits_files_in_directory(date_dir)
                    print(f"{date_dir}: {len(fits_files)} FITSファイル")
                else:
                    print(f"{date_dir}: ディレクトリなし")
        else:
            # 実際のダウンロード
            downloader.download_date_range(args.start_date, args.end_date)
            
    except KeyboardInterrupt:
        print("\n\nダウンロードが中断されました")
        sys.exit(1)
    except Exception as e:
        print(f"予期しないエラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()