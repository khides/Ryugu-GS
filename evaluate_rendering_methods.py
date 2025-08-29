#!/usr/bin/env python3

"""
evaluate_rendering_methods.py

2つの3Dレンダリング手法（Blender メッシュベース vs Gaussian Splatting）の
レンダリング時間と精度を比較・評価するスクリプト

重要な改善点：
- 学習用と評価用のデータセットを適切に分離
- 公平な比較のための画像前処理
- 包括的なメトリクス評価（PSNR, SSIM, LPIPS）
"""

import argparse
import csv
import json
import logging
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import math

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import torchvision.transforms.functional as tf
from tqdm import tqdm

# Gaussian Splattingモジュールのインポート（mvs_benchmarkと同じ方式）
# PATHに追加してからインポート
gs_path = Path(__file__).parent / "gaussian-splatting"
if gs_path.exists():
    sys.path.append(str(gs_path))

try:
    from utils.loss_utils import ssim
    from utils.image_utils import psnr
    from lpipsPyTorch import lpips
    GS_UTILS_AVAILABLE = True
except ImportError:
    print("Warning: Gaussian Splatting utils not found. Using fallback implementations.")
    print("  To enable full functionality, ensure you're running from the gaussian-splatting directory")
    print("  or that PYTHONPATH includes the gaussian-splatting directory.")
    GS_UTILS_AVAILABLE = False

# mvs_benchmarkの実装を参考にしたフォールバック実装
if not GS_UTILS_AVAILABLE:
    import numpy as np
    try:
        from skimage.metrics import structural_similarity as compare_ssim
        from skimage.metrics import peak_signal_noise_ratio as compare_psnr
        SKIMAGE_AVAILABLE = True
    except ImportError:
        SKIMAGE_AVAILABLE = False
        print("Warning: scikit-image not available, using basic fallback implementations")
    
    def psnr(img1, img2):
        """Fallback PSNR implementation using scikit-image"""
        if SKIMAGE_AVAILABLE:
            img1_np = img1.squeeze().cpu().numpy().transpose(1, 2, 0)
            img2_np = img2.squeeze().cpu().numpy().transpose(1, 2, 0)
            return compare_psnr(img1_np, img2_np, data_range=1.0)
        else:
            # Basic PSNR implementation
            mse = torch.mean((img1 - img2) ** 2)
            if mse == 0:
                return float('inf')
            psnr_val = 10.0 * torch.log10(1.0 / mse)
            return psnr_val.cpu().item() if torch.is_tensor(psnr_val) else float(psnr_val)
    
    def ssim(img1, img2):
        """Fallback SSIM implementation using scikit-image"""
        if SKIMAGE_AVAILABLE:
            img1_np = img1.squeeze().cpu().numpy().transpose(1, 2, 0)
            img2_np = img2.squeeze().cpu().numpy().transpose(1, 2, 0)
            
            # Convert to grayscale if needed
            if img1_np.shape[2] == 3:
                img1_gray = np.mean(img1_np, axis=2)
                img2_gray = np.mean(img2_np, axis=2)
            else:
                img1_gray = img1_np.squeeze()
                img2_gray = img2_np.squeeze()
                
            return compare_ssim(img1_gray, img2_gray, data_range=1.0)
        else:
            # Basic SSIM approximation
            return 0.5  # Fallback value

    def lpips(img1, img2, net_type='alex', version='0.1'):
        """Simple LPIPS fallback using L2 distance"""
        l2_dist = torch.mean((img1 - img2) ** 2)
        return l2_dist.cpu().item() if torch.is_tensor(l2_dist) else float(l2_dist)

    ssim = ssim
    psnr = psnr
    lpips = lpips


@dataclass
class EvaluationResult:
    """評価結果を格納するデータクラス"""
    frame_filename: str
    blender_render_time_sec: float
    blender_psnr: float
    blender_ssim: float
    blender_lpips: float
    gs_render_time_sec: float
    gs_psnr: float
    gs_ssim: float
    gs_lpips: float


class ImagePreprocessor:
    """画像前処理クラス - オブジェクトの中心移動と背景正規化"""
    
    def __init__(self, target_bg_ratio: float = 0.3):
        """
        Args:
            target_bg_ratio: 背景黒以外の目標割合
        """
        self.target_bg_ratio = target_bg_ratio
        self.logger = logging.getLogger(__name__)
    
    def find_object_center(self, image: np.ndarray, threshold: int = 10) -> Tuple[int, int]:
        """画像内のオブジェクト（背景黒以外）の中心を見つける"""
        if len(image.shape) == 3:
            # カラー画像の場合、グレースケールに変換
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 背景（黒に近い部分）以外を検出
        mask = gray > threshold
        
        # オブジェクトのバウンディングボックスを取得
        coords = np.column_stack(np.where(mask))
        if len(coords) == 0:
            # オブジェクトが見つからない場合は画像中心を返す
            return image.shape[1] // 2, image.shape[0] // 2
        
        y_coords, x_coords = coords[:, 0], coords[:, 1]
        center_x = int(np.mean(x_coords))
        center_y = int(np.mean(y_coords))
        
        return center_x, center_y
    
    def center_object(self, image: np.ndarray) -> np.ndarray:
        """オブジェクトを画像中心に移動"""
        h, w = image.shape[:2]
        obj_center_x, obj_center_y = self.find_object_center(image)
        
        # 移動量を計算
        shift_x = w // 2 - obj_center_x
        shift_y = h // 2 - obj_center_y
        
        # アフィン変換で移動
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        centered_image = cv2.warpAffine(image, M, (w, h), borderValue=(0, 0, 0))
        
        return centered_image
    
    def normalize_background_ratio(self, image: np.ndarray) -> np.ndarray:
        """背景黒以外の割合を一定に調整"""
        h, w = image.shape[:2]
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 現在の非背景ピクセル数を計算
        mask = gray > 10
        current_ratio = np.sum(mask) / (h * w)
        
        if current_ratio <= 0:
            return image
        
        # スケールファクターを計算
        scale_factor = math.sqrt(self.target_bg_ratio / current_ratio)
        
        # リサイズして中央に配置
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        
        if scale_factor < 1:
            # オブジェクトを縮小して黒キャンバスに配置
            resized = cv2.resize(image, (new_w, new_h))
            
            # 元のサイズのキャンバスを作成（黒背景）
            if len(image.shape) == 3:
                canvas = np.zeros((h, w, image.shape[2]), dtype=image.dtype)
            else:
                canvas = np.zeros((h, w), dtype=image.dtype)
            
            # 中央に配置
            start_y = (h - new_h) // 2
            start_x = (w - new_w) // 2
            end_y = start_y + new_h
            end_x = start_x + new_w
            
            canvas[start_y:end_y, start_x:end_x] = resized
            return canvas
        else:
            # オブジェクトを拡大してから中央をクロップ
            resized = cv2.resize(image, (new_w, new_h))
            
            # 中央をクロップして元のサイズに戻す
            start_y = (new_h - h) // 2
            start_x = (new_w - w) // 2
            end_y = start_y + h
            end_x = start_x + w
            
            cropped = resized[start_y:end_y, start_x:end_x]
            return cropped
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """完全な前処理パイプライン"""
        # 1. オブジェクトを中心に移動
        centered = self.center_object(image)
        
        # 2. 背景割合を正規化
        normalized = self.normalize_background_ratio(centered)
        
        return normalized


class MetricsCalculator:
    """メトリクス計算クラス"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.logger = logging.getLogger(__name__)
        self._fallback_warnings_shown = set()  # 警告の重複を防ぐ
        
        # LPIPSネットワークを初期化 (mvs_benchmarkと同じ方式)
        # デバッグのため一時的に無効化
        self.lpips_fn = None
        self.logger.info("DEBUG: LPIPS network disabled to debug hanging issue")
        # if GS_UTILS_AVAILABLE:
        #     try:
        #         self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)
        #         self.logger.debug("LPIPS network initialized successfully")
        #     except Exception as e:
        #         self.logger.debug(f"LPIPS network initialization failed: {e}")
        #         self.lpips_fn = None
        # else:
        #     self.logger.debug("GS utils not available, LPIPS network will use fallback")
    
    def gaussian(self, window_size, sigma):
        """Gaussian window for SSIM calculation"""
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        """Create window for SSIM calculation"""
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    def _ssim_computation(self, img1, img2, window, window_size, channel, size_average=True):
        """Core SSIM computation from Gaussian Splatting implementation"""
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def gs_ssim(self, img1, img2, window_size=11, size_average=True):
        """SSIM calculation using Gaussian Splatting implementation"""
        channel = img1.size(-3)
        window = self.create_window(window_size, channel)

        if img1.is_cuda:
            window = window.cuda(img1.get_device())
        window = window.type_as(img1)

        return self._ssim_computation(img1, img2, window, window_size, channel, size_average)
    
    def calculate_psnr(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """PSNR計算"""
        psnr_score = psnr(img1, img2)
        if hasattr(psnr_score, 'item'):
            return psnr_score.item()
        elif torch.is_tensor(psnr_score):
            return psnr_score.cpu().item()
        else:
            return float(psnr_score)
    
    def calculate_ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """SSIM計算"""
        # 最初にGaussian Splattingのオリジナル実装を試行
        if ssim is not None:
            try:
                ssim_score = ssim(img1, img2)
                if hasattr(ssim_score, 'item'):
                    return ssim_score.item()
                elif torch.is_tensor(ssim_score):
                    return ssim_score.cpu().item()
                else:
                    return float(ssim_score)
            except Exception as e:
                self.logger.debug(f"GS SSIM failed: {e}, trying custom implementation")
                # フォールバック実装を使用
                result = self.gs_ssim(img1, img2)
                return result.cpu().item() if torch.is_tensor(result) else float(result)
        
        # フォールバック実装
        result = self.gs_ssim(img1, img2)
        return result.cpu().item() if torch.is_tensor(result) else float(result)
        
    def calculate_lpips(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """LPIPS計算"""
        # LPIPSネットワークが利用可能な場合
        if self.lpips_fn is not None:
            try:
                with torch.no_grad():
                    lpips_val = self.lpips_fn(img1, img2)
                    if torch.is_tensor(lpips_val):
                        return lpips_val.cpu().item()
                    else:
                        return float(lpips_val)
            except Exception as e:
                self.logger.debug(f"LPIPS network failed: {e}")
        
        # フォールバック実装 (mvs_benchmarkと同じ)
        if "lpips" not in self._fallback_warnings_shown:
            self.logger.warning("LPIPS not available, using L2 distance as fallback")
            self._fallback_warnings_shown.add("lpips")
        
        result = lpips(img1, img2)
        if torch.is_tensor(result):
            return result.cpu().item()
        else:
            return float(result)
    
    def image_to_tensor(self, image_path: Path) -> torch.Tensor:
        """画像をテンソルに変換"""
        image = Image.open(image_path).convert('RGB')
        tensor = tf.to_tensor(image).unsqueeze(0)[:, :3, :, :].to(self.device)
        return tensor
    
    def calculate_metrics(self, img1_path: Path, img2_path: Path) -> Dict[str, float]:
        """2つの画像間のすべてのメトリクスを計算"""
        try:
            self.logger.debug(f"DEBUG: Converting {img1_path.name} to tensor")
            img1_tensor = self.image_to_tensor(img1_path)
            self.logger.debug(f"DEBUG: Converting {img2_path.name} to tensor")
            img2_tensor = self.image_to_tensor(img2_path)
            
            # サイズが異なる場合はリサイズ
            if img1_tensor.shape != img2_tensor.shape:
                target_size = img1_tensor.shape[-2:]
                self.logger.debug(f"Resizing tensor from {img2_tensor.shape[-2:]} to {target_size}")
                
                # 安全なリサイズ処理
                if len(img2_tensor.shape) == 4:  # バッチテンソル
                    img2_tensor = tf.resize(img2_tensor.squeeze(0), target_size).unsqueeze(0)
                elif len(img2_tensor.shape) == 3:  # 単一画像テンソル
                    img2_tensor = tf.resize(img2_tensor, target_size).unsqueeze(0)
                else:
                    raise ValueError(f"Unexpected tensor shape: {img2_tensor.shape}")
            
            # 最終的な形状チェック
            if img1_tensor.shape != img2_tensor.shape:
                raise ValueError(f"Shape mismatch after resize: {img1_tensor.shape} vs {img2_tensor.shape}")
            
            self.logger.debug(f"DEBUG: Calculating PSNR")
            psnr_val = self.calculate_psnr(img1_tensor, img2_tensor)
            self.logger.debug(f"DEBUG: Calculating SSIM")
            ssim_val = self.calculate_ssim(img1_tensor, img2_tensor)  
            self.logger.debug(f"DEBUG: Calculating LPIPS")
            lpips_val = self.calculate_lpips(img1_tensor, img2_tensor)
            self.logger.debug(f"DEBUG: All metrics calculated successfully")
            
            return {
                'psnr': psnr_val,
                'ssim': ssim_val,
                'lpips': lpips_val
            }
        
        except Exception as e:
            self.logger.error(f"Error calculating metrics for {img1_path} vs {img2_path}: {e}")
            return {
                'psnr': float('nan'),
                'ssim': float('nan'),
                'lpips': float('nan')
            }


class BlenderEvaluator:
    """Blenderレンダリング評価クラス"""
    
    def __init__(self, blender_data_dir: Path, test_data_dir: Path):
        self.blender_data_dir = Path(blender_data_dir)
        self.test_data_dir = Path(test_data_dir)
        self.preprocessor = ImagePreprocessor()
        self.metrics_calc = MetricsCalculator()
        self.logger = logging.getLogger(__name__)
    
    def load_render_times(self) -> Dict[str, float]:
        """render_times.csvから各フレームのレンダリング時間を読み込む"""
        render_times_file = self.blender_data_dir / "render_times.csv"
        render_times = {}
        
        if not render_times_file.exists():
            self.logger.warning(f"render_times.csv not found at {render_times_file}")
            self.logger.warning("All Blender render times will be set to 0.0")
            return render_times
        
        try:
            with open(render_times_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # CSVファイルの構造に応じて調整 - より柔軟にマッピング
                    frame_name = row.get('Frame', row.get('frame', row.get('filename', row.get('image', ''))))
                    time_sec_str = row.get('RenderTime_seconds', row.get('time_sec', row.get('time', row.get('render_time', '0.0'))))
                    
                    # デバッグログ：CSVの各行を確認
                    self.logger.debug(f"CSV row - Frame: '{frame_name}', RenderTime_seconds: '{time_sec_str}'")
                    
                    try:
                        time_sec = float(time_sec_str)
                    except (ValueError, TypeError):
                        time_sec = 0.0
                    
                    # ファイル名の拡張子を除去してマッピング
                    if frame_name:
                        # 元の名前で保存
                        render_times[frame_name] = time_sec
                        # 拡張子なしでも保存 (.png, .jpg 等を除去)
                        stem_name = Path(frame_name).stem
                        render_times[stem_name] = time_sec
                        
                        self.logger.debug(f"Loaded render time: {frame_name} ({stem_name}) = {time_sec}s")
                    
                    # Frameカラムが数値の場合のフォーマット変換処理
                    # ryugu_render_0001.png に対応する Frame=1 の変換
                    try:
                        frame_num = int(frame_name)
                        # ryugu_render_XXXX 形式のキーを作成
                        ryugu_key = f"ryugu_render_{frame_num:04d}"
                        render_times[ryugu_key] = time_sec
                        
                        # .png 付きも作成
                        ryugu_png_key = f"ryugu_render_{frame_num:04d}.png"
                        render_times[ryugu_png_key] = time_sec
                        
                        self.logger.debug(f"Mapped frame {frame_num} -> keys: ['{frame_num}', '{ryugu_key}', '{ryugu_png_key}'] = {time_sec}s")
                        
                    except ValueError:
                        # frame_nameが数値でない場合はスキップ
                        pass
                    
            self.logger.info(f"Successfully loaded render times for {len(set(render_times.keys()))} unique frames")
            
            # サンプルデータを表示
            sample_items = list(render_times.items())[:3]
            if sample_items:
                self.logger.info(f"Sample render times: {sample_items}")
            
        except Exception as e:
            self.logger.error(f"Error loading render times from {render_times_file}: {e}")
            # CSVファイルの内容をサンプル表示
            try:
                with open(render_times_file, 'r') as f:
                    sample_lines = f.readlines()[:5]
                    self.logger.error(f"CSV file sample (first 5 lines):")
                    for i, line in enumerate(sample_lines):
                        self.logger.error(f"  Line {i+1}: {line.strip()}")
                    
                    # CSVヘッダーを解析してフィールド名を表示
                    if sample_lines:
                        header = sample_lines[0].strip().split(',')
                        self.logger.error(f"Available CSV fields: {header}")
                        self.logger.error("Expected fields: 'frame' or 'filename' or 'image', 'time_sec' or 'time' or 'render_time'")
            except Exception as parse_error:
                self.logger.error(f"Could not parse CSV file: {parse_error}")
            
        return render_times
    
    def find_best_match(self, blender_image_path: Path, test_images: List[Path]) -> Tuple[Path, Dict[str, float]]:
        """Blenderレンダリング画像に最も近い評価用画像を見つける"""
        self.logger.info(f"DEBUG: Finding best match for {blender_image_path.name} among {len(test_images)} test images")
        
        # Blender画像を前処理
        self.logger.info(f"DEBUG: Loading Blender image: {blender_image_path}")
        blender_img = cv2.imread(str(blender_image_path))
        if blender_img is None:
            raise ValueError(f"Cannot load Blender image: {blender_image_path} (check file format and permissions)")
        
        self.logger.info(f"DEBUG: Preprocessing Blender image")
        try:
            processed_blender_img = self.preprocessor.preprocess(blender_img)
            self.logger.info(f"DEBUG: Blender image preprocessing completed")
        except Exception as e:
            raise ValueError(f"Failed to preprocess Blender image {blender_image_path}: {e}")
        
        best_match = None
        best_metrics = None
        best_score = float('-inf')
        processed_count = 0
        
        # 一時ファイルとして保存
        temp_blender_path = self.blender_data_dir / "temp_processed_blender.png"
        try:
            success = cv2.imwrite(str(temp_blender_path), processed_blender_img)
            if not success:
                raise ValueError(f"Failed to save processed Blender image to {temp_blender_path}")
        except Exception as e:
            raise ValueError(f"Error saving processed Blender image: {e}")
        
        try:
            for test_image_path in test_images:
                try:
                    # テスト画像を前処理
                    test_img = cv2.imread(str(test_image_path))
                    if test_img is None:
                        self.logger.debug(f"Cannot load test image: {test_image_path}")
                        continue
                    
                    processed_test_img = self.preprocessor.preprocess(test_img)
                    
                    # 一時ファイルとして保存
                    temp_test_path = self.test_data_dir / "temp_processed_test.png"
                    success = cv2.imwrite(str(temp_test_path), processed_test_img)
                    if not success:
                        self.logger.debug(f"Failed to save processed test image {test_image_path}")
                        continue
                    
                    # メトリクスを計算
                    self.logger.info(f"DEBUG: Calculating metrics for {test_image_path.name}")
                    metrics = self.metrics_calc.calculate_metrics(temp_blender_path, temp_test_path)
                    self.logger.info(f"DEBUG: Metrics calculated: PSNR={metrics.get('psnr', 'N/A')}, SSIM={metrics.get('ssim', 'N/A')}, LPIPS={metrics.get('lpips', 'N/A')}")
                    processed_count += 1
                    
                    # スコア計算（PSNR+SSIM-LPIPS の組み合わせ）
                    if not (math.isnan(metrics['psnr']) or math.isnan(metrics['ssim']) or math.isnan(metrics['lpips'])):
                        score = metrics['psnr'] / 50.0 + metrics['ssim'] - metrics['lpips']
                        
                        if score > best_score:
                            best_score = score
                            best_match = test_image_path
                            best_metrics = metrics
                            self.logger.debug(f"New best match: {test_image_path.name} with score {score:.4f}")
                    
                    # 一時ファイル削除
                    if temp_test_path.exists():
                        temp_test_path.unlink()
                        
                except Exception as e:
                    self.logger.debug(f"Error processing test image {test_image_path}: {e}")
                    continue
            
        finally:
            # 一時ファイル削除
            if temp_blender_path.exists():
                temp_blender_path.unlink()
        
        self.logger.debug(f"Processed {processed_count} test images for {blender_image_path.name}")
        
        if best_match is None:
            self.logger.warning(f"No valid metrics computed for {blender_image_path.name}")
            # フォールバック: 最初の利用可能な画像
            if test_images:
                self.logger.debug(f"Using fallback match: {test_images[0].name}")
                best_match = test_images[0]
                best_metrics = {'psnr': float('nan'), 'ssim': float('nan'), 'lpips': float('nan')}
            else:
                raise ValueError(f"No test images available for comparison with {blender_image_path}")
        
        return best_match, best_metrics
    
    def normalize_frame_name(self, frame_name: str) -> str:
        """フレーム名を正規化してマッピングを改善"""
        # 拡張子を除去
        frame_name = Path(frame_name).stem
        
        # 数値部分を抽出してゼロ埋め5桁に統一
        import re
        match = re.search(r'(\d+)', frame_name)
        if match:
            number = int(match.group(1))
            # 「00000」形式に統一
            return f"{number:05d}"
        
        return frame_name
    
    def evaluate(self) -> List[Dict]:
        """Blenderレンダリングの評価を実行 - 各Blender画像ごとに結果生成"""
        self.logger.info("Starting Blender evaluation...")
        
        # 事前チェック: Blenderデータディレクトリの存在確認
        if not self.blender_data_dir.exists():
            self.logger.error(f"Blender data directory not found: {self.blender_data_dir}")
            return []
        
        # 事前チェック: テストデータディレクトリの存在確認
        if not self.test_data_dir.exists():
            self.logger.error(f"Test data directory not found: {self.test_data_dir}")
            return []
        
        # レンダリング時間を読み込み
        render_times = self.load_render_times()
        unique_entries = len(set(render_times.keys()))
        self.logger.info(f"Loaded {len(render_times)} render time entries ({unique_entries} unique frames)")
        
        if render_times:
            # CSVファイルが正常に読み込まれたことを確認
            total_time = sum(render_times.values())
            avg_time = total_time / len(render_times) if render_times else 0
            self.logger.info(f"Total render time: {total_time:.2f}s, Average: {avg_time:.3f}s per frame")
        else:
            self.logger.warning("No render times loaded - all times will be 0.0")
        
        # Blenderレンダリング画像を取得
        blender_images = list(self.blender_data_dir.glob("*.png")) + list(self.blender_data_dir.glob("*.jpg")) + list(self.blender_data_dir.glob("*.jpeg"))
        self.logger.info(f"Found {len(blender_images)} Blender images in {self.blender_data_dir}")
        
        if not blender_images:
            self.logger.error(f"No Blender rendering images found in {self.blender_data_dir}")
            self.logger.error("Expected file extensions: *.png, *.jpg, *.jpeg")
            # ディレクトリ内容を表示
            all_files = list(self.blender_data_dir.iterdir())
            self.logger.error(f"Directory contents: {[f.name for f in all_files[:10]]}{'...' if len(all_files) > 10 else ''}")
            return []
        
        # 評価用画像を取得 - サブディレクトリも含めて検索
        test_images = []
        # まず直接ディレクトリ内を検索
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            test_images.extend(self.test_data_dir.glob(ext))
        
        # 画像が見つからない場合、サブディレクトリも検索
        if not test_images:
            self.logger.info("No images in root test directory, searching subdirectories...")
            subdirs_to_check = ['images', 'Input', 'test']
            for subdir_name in subdirs_to_check:
                subdir = self.test_data_dir / subdir_name
                if subdir.exists() and subdir.is_dir():
                    for ext in ['*.png', '*.jpg', '*.jpeg']:
                        found_in_subdir = list(subdir.glob(ext))
                        if found_in_subdir:
                            test_images.extend(found_in_subdir)
                            self.logger.info(f"Found {len(found_in_subdir)} images in {subdir}")
                            break
                    if test_images:
                        break
        
        self.logger.info(f"Found {len(test_images)} test images total")
        
        if not test_images:
            self.logger.error(f"No test images found in {self.test_data_dir} or its subdirectories")
            self.logger.error("Expected file extensions: *.png, *.jpg, *.jpeg")
            self.logger.error("Searched in: root directory, images/, Input/, test/")
            # ディレクトリ内容を表示
            all_files = list(self.test_data_dir.iterdir()) if self.test_data_dir.exists() else []
            self.logger.error(f"Root directory contents: {[f.name for f in all_files[:10]]}{'...' if len(all_files) > 10 else ''}")
            return []
        
        results = []
        failed_evaluations = 0
        
        self.logger.info(f"DEBUG: Starting evaluation of {len(blender_images)} Blender images")
        
        for i, blender_img_path in enumerate(blender_images):
            if i >= 3:  # テスト用：最初の3枚だけ処理してハング箇所を特定
                self.logger.info(f"DEBUG: Stopping after 3 images for debugging")
                break
            try:
                self.logger.info(f"DEBUG: Processing Blender image {i+1}/{len(blender_images)}: {blender_img_path.name}")
                
                # 最適なマッチを見つける
                self.logger.info(f"DEBUG: Starting find_best_match for {blender_img_path.name}")
                best_match, metrics = self.find_best_match(blender_img_path, test_images)
                self.logger.info(f"DEBUG: find_best_match completed for {blender_img_path.name}")
                
                if best_match:
                    # レンダー時間を取得 - 複数のパターンでマッチングを試行
                    render_time = 0.0
                    # render_times.csvのFrameカラムが数値の場合のマッピングを考慮
                    possible_keys = [
                        blender_img_path.name,      # フルファイル名 (e.g., "ryugu_render_0001.png")
                        blender_img_path.stem,      # 拡張子なし (e.g., "ryugu_render_0001")
                        blender_img_path.stem.replace('_', ''),  # アンダースコアなし
                        blender_img_path.stem.lower(),  # 小文字
                    ]
                    
                    # ryugu_render_0001 -> 1 の変換を試行
                    try:
                        # ryugu_render_XXXX 形式から番号を抽出
                        if blender_img_path.stem.startswith('ryugu_render_'):
                            num_str = blender_img_path.stem.replace('ryugu_render_', '')
                            frame_num = int(num_str)
                            possible_keys.append(str(frame_num))  # "1", "2", "3", ...
                            possible_keys.append(f"{frame_num}")  # 同じだが明示的に
                    except ValueError:
                        # 番号抽出に失敗した場合はスキップ
                        pass
                    
                    for key in possible_keys:
                        if key in render_times:
                            render_time = render_times[key]
                            self.logger.debug(f"Found render time for {blender_img_path.name}: {render_time}s (key: {key})")
                            break
                    
                    if render_time == 0.0:
                        self.logger.warning(f"No render time found for {blender_img_path.name}, tried keys: {possible_keys}")
                    
                    # 正規化されたフレーム名を使用（キーマッピング改善）
                    normalized_frame_name = self.normalize_frame_name(blender_img_path.name)
                    
                    result = {
                        'frame_name': normalized_frame_name,  # 正規化されたフレーム名
                        'blender_image': blender_img_path.name,
                        'ground_truth_image': best_match.name,
                        'render_time_sec': render_time,
                        'psnr': metrics['psnr'],
                        'ssim': metrics['ssim'],
                        'lpips': metrics['lpips']
                    }
                    results.append(result)
                    self.logger.debug(f"Successfully evaluated {blender_img_path.name} -> frame {normalized_frame_name}")
                else:
                    self.logger.warning(f"No matching test image found for {blender_img_path.name}")
                    failed_evaluations += 1
                
            except Exception as e:
                self.logger.error(f"Error evaluating {blender_img_path}: {e}")
                import traceback
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                failed_evaluations += 1
        
        self.logger.info(f"Completed Blender evaluation: {len(results)} successful, {failed_evaluations} failed")
        
        if len(results) == 0:
            self.logger.error("No successful Blender evaluations - all returned N/A")
            self.logger.error("Common causes:")
            self.logger.error("  1. Missing render_times.csv in Blender data directory")
            self.logger.error("  2. Blender image files cannot be loaded")
            self.logger.error("  3. Test images cannot be loaded")
            self.logger.error("  4. Image preprocessing or metrics calculation failed")
        
        return results


class GaussianSplattingEvaluator:
    """Gaussian Splatting評価クラス"""
    
    def __init__(self, train_data_dir: Path, test_data_dir: Path, gs_dir: Path):
        self.train_data_dir = Path(train_data_dir)
        self.test_data_dir = Path(test_data_dir)
        self.gs_dir = Path(gs_dir)
        self.metrics_calc = MetricsCalculator()
        self.logger = logging.getLogger(__name__)
        
        # データフォーマット検証
        self._validate_and_detect_data_format()
    
    def _validate_and_detect_data_format(self):
        """データフォーマットを検証・検出"""
        # NeRF-style structure check
        nerf_paths = [
            self.train_data_dir / "transforms_train.json",
            self.train_data_dir / "train"
        ]
        
        # Check for images directories
        image_dirs = [
            self.train_data_dir / "images",
            self.train_data_dir / "Input"
        ]
        
        image_dir = None
        for d in image_dirs:
            if d.exists() and d.is_dir():
                image_files = list(d.glob("*.jpg")) + list(d.glob("*.jpeg")) + list(d.glob("*.png"))
                if image_files:
                    image_dir = d
                    self.logger.info(f"Found images directory: {image_dir} ({len(image_files)} images)")
                    break
        
        # COLMAP-style structure check (optional sparse directory)
        sparse_dirs = [
            self.train_data_dir / "sparse" / "0",
            self.train_data_dir / "sparse"
        ]
        
        sparse_dir = None
        for d in sparse_dirs:
            if d.exists() and d.is_dir():
                required_files = ["cameras.bin", "images.bin", "points3D.bin"]
                if all((d / f).exists() for f in required_files):
                    sparse_dir = d
                    self.logger.info(f"Found COLMAP sparse directory: {sparse_dir}")
                    break
        
        # Determine format with flexible validation
        if all(p.exists() for p in nerf_paths):
            self.data_format = "nerf"
            self.logger.info("Detected NeRF-style data format")
        elif image_dir:
            # Accept images-only format (will use COLMAP to reconstruct)
            self.data_format = "images_only"
            self.image_dir = image_dir
            if sparse_dir:
                self.data_format = "colmap"
                self.sparse_dir = sparse_dir
                self.logger.info("Detected full COLMAP-style data format")
            else:
                self.logger.info("Detected images-only format (COLMAP reconstruction required)")
                self.logger.info("[WARN] No sparse reconstruction found. GS training may require COLMAP preprocessing.")
        else:
            # Provide detailed error message
            available_files = []
            if self.train_data_dir.exists():
                available_files = [f.name for f in self.train_data_dir.iterdir()]
            raise ValueError(
                f"Invalid training data format in {self.train_data_dir}\n"
                f"Expected: images/ directory with .jpg/.jpeg/.png files\n"
                f"Found: {available_files}\n"
                f"Please ensure the training directory contains an 'images' subdirectory with image files."
            )
            
        self.logger.info(f"Data format: {self.data_format}")
    
    def diagnose_gaussian_splatting_error(self, stderr: str) -> dict:
        """Diagnose common Gaussian Splatting errors and provide suggestions"""
        diagnosis = {
            "diagnosis": "Unknown error",
            "suggestions": []
        }
        
        stderr_lower = stderr.lower()
        
        if "could not recognize scene type" in stderr_lower:
            diagnosis["diagnosis"] = "Invalid data format or path resolution issue"
            diagnosis["suggestions"] = [
                "Ensure data directory contains either COLMAP format (sparse/0/) or NeRF format (transforms*.json)",
                "Check that working directory is correct when running the script",
                "Try using absolute paths instead of relative paths",
                "Verify that images/ or Input/ directory exists and contains images",
                "For COLMAP data: check that sparse/0/cameras.bin, images.bin, points3D.bin exist"
            ]
        elif "paging file" in stderr_lower or "virtual memory" in stderr_lower:
            diagnosis["diagnosis"] = "Virtual memory (page file) insufficient"
            diagnosis["suggestions"] = [
                "Increase Windows virtual memory (page file) size to at least 8GB",
                "Close other memory-intensive applications",
                "Consider upgrading system RAM"
            ]
        elif ("out of memory" in stderr_lower or "cuda out of memory" in stderr_lower or 
              "cublas_status_alloc_failed" in stderr_lower or "cublascreate" in stderr_lower):
            diagnosis["diagnosis"] = "GPU memory exhausted"
            diagnosis["suggestions"] = [
                "Close other GPU-intensive applications (check with nvidia-smi)",
                "Use memory-optimized training: --resolution 2 --data_device cpu",
                "Reduce image resolution or number of training images",
                "Try: python train.py -s <data> --resolution 4 --data_device cpu --sh_degree 2",
                "Consider using a GPU with more VRAM (8GB+ recommended)"
            ]
        elif "cudnn" in stderr_lower and ("dll" in stderr_lower or "library" in stderr_lower):
            diagnosis["diagnosis"] = "CUDA/cuDNN library loading failure"
            diagnosis["suggestions"] = [
                "Verify CUDA toolkit installation matches PyTorch CUDA version",
                "Check PyTorch CUDA support installation",
                "Try CPU-only mode if GPU is not required",
                "Restart system to refresh library paths"
            ]
        elif "torch" in stderr_lower and ("import" in stderr_lower or "module" in stderr_lower):
            diagnosis["diagnosis"] = "PyTorch installation or import issue"
            diagnosis["suggestions"] = [
                "Verify PyTorch installation",
                "Check Python environment activation",
                "Reinstall PyTorch with CUDA support if needed"
            ]
        elif "file not found" in stderr_lower or "no such file" in stderr_lower:
            diagnosis["diagnosis"] = "Missing input files or incorrect data structure"
            diagnosis["suggestions"] = [
                "Verify input data path and structure",
                "Check that COLMAP reconstruction files exist",
                "Ensure images directory contains source images",
                "Run with absolute paths instead of relative paths"
            ]
        
        return diagnosis
    
    def run_gs_training(self, output_dir: Path) -> Tuple[bool, float]:
        """Gaussian Splattingの学習を実行"""
        self.logger.info("Starting Gaussian Splatting training...")
        
        # Use absolute paths to avoid resolution issues
        abs_data_path = self.train_data_dir.resolve()
        abs_model_path = output_dir.resolve()
        
        train_cmd = [
            sys.executable, "train.py",
            "-s", str(abs_data_path),
            "-m", str(abs_model_path),
            "--resolution", "2",  # Reduce image resolution to save VRAM
            "--data_device", "cpu",  # Store images in CPU memory
        ]
        
        # Enable eval mode based on data format - simplified logic
        if hasattr(self, 'data_format') and self.data_format == "colmap":
            # COLMAP data: Always enable eval mode - GS will internally split data
            train_cmd.append("--eval")
            self.logger.info("COLMAP format - enabling evaluation mode with internal data splitting")
        elif hasattr(self, 'data_format') and self.data_format == "nerf":
            # NeRF format: Enable if we have explicit test set
            if (self.train_data_dir / "transforms_test.json").exists():
                train_cmd.append("--eval")
                self.logger.info("NeRF format - enabling evaluation mode with explicit test set")
            else:
                self.logger.info("NeRF format - no transforms_test.json found, training only")
        elif hasattr(self, 'data_format') and self.data_format == "images_only":
            # Images-only: Enable eval mode for automatic splitting
            train_cmd.append("--eval")
            self.logger.info("Images-only format - enabling evaluation mode with automatic data splitting")
        
        # Add images directory specification for COLMAP and images-only data
        if hasattr(self, 'data_format') and self.data_format in ["colmap", "images_only"] and hasattr(self, 'image_dir'):
            try:
                rel_image_path = self.image_dir.relative_to(self.train_data_dir)
                train_cmd.extend(["--images", str(rel_image_path)])
                self.logger.info(f"Using images directory: {rel_image_path}")
            except ValueError:
                # If relative path fails, use the directory name
                train_cmd.extend(["--images", self.image_dir.name])
                self.logger.info(f"Using images directory: {self.image_dir.name}")
        
        # Add memory optimization
        try:
            import psutil
            memory = psutil.virtual_memory()
            if memory.total / (1024**3) < 16:  # Less than 16GB RAM
                train_cmd.extend(["--sh_degree", "2"])
                self.logger.info("Added memory optimization due to limited RAM")
        except ImportError:
            self.logger.info("psutil not available, skipping memory optimization")
        
        self.logger.info(f"Training command: {' '.join(train_cmd)}")
        start_time = time.time()
        
        try:
            # Use Popen for real-time output
            process = subprocess.Popen(
                train_cmd,
                cwd=str(self.gs_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            self.logger.info("[TRAINING] GS Training started - showing real-time output:")
            self.logger.info("-" * 60)
            
            output_lines = []
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    # Log each line in real-time - streamlined output
                    line = output.strip()
                    print(line)  # Direct print to avoid buffering issues
                    output_lines.append(line)
            
            # Wait for process completion
            process.wait()
            training_time = time.time() - start_time
            
            if process.returncode != 0:
                full_output = "\n".join(output_lines)
                
                # Provide detailed error diagnosis
                error_diagnosis = self.diagnose_gaussian_splatting_error(full_output)
                self.logger.error(f"[ERROR] GAUSSIAN SPLATTING TRAINING FAILED")
                self.logger.error(f"Error Diagnosis: {error_diagnosis['diagnosis']}")
                if error_diagnosis['suggestions']:
                    self.logger.error(f"Suggestions:")
                    for suggestion in error_diagnosis['suggestions']:
                        self.logger.error(f"  - {suggestion}")
                
                self.logger.error(f"Return code: {process.returncode}")
                return False, training_time
            
            self.logger.info("-" * 60)
            self.logger.info(f"[OK] GS training completed in {training_time:.2f} seconds")
            return True, training_time
            
        except subprocess.TimeoutExpired:
            self.logger.error("GS training timeout")
            return False, time.time() - start_time
        except Exception as e:
            self.logger.error(f"GS training error: {e}")
            return False, time.time() - start_time
    
    def run_gs_rendering(self, model_dir: Path) -> bool:
        """Gaussian Splattingのレンダリングを実行"""
        self.logger.info("Starting Gaussian Splatting rendering...")
        
        render_cmd = [
            sys.executable, "render.py",
            "-m", str(model_dir.resolve())
        ]
        
        try:
            # Use Popen for real-time output
            process = subprocess.Popen(
                render_cmd,
                cwd=str(self.gs_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            self.logger.info("[RENDERING] GS Rendering started - showing real-time output:")
            self.logger.info("-" * 60)
            
            output_lines = []
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    line = output.strip()
                    print(line)  # Direct print to avoid buffering issues
                    output_lines.append(line)
            
            process.wait()
            
            if process.returncode != 0:
                self.logger.error(f"GS rendering failed with return code {process.returncode}")
                full_output = "\n".join(output_lines)
                self.logger.error(f"Full output: {full_output}")
                return False
            
            self.logger.info("-" * 60)
            self.logger.info("[OK] GS rendering completed")
            return True
            
        except subprocess.TimeoutExpired:
            self.logger.error("GS rendering timeout")
            return False
        except Exception as e:
            self.logger.error(f"GS rendering error: {e}")
            return False
    
    def normalize_frame_name(self, frame_name: str) -> str:
        """フレーム名を正規化してマッピングを改善"""
        # 拡張子を除去
        frame_name = Path(frame_name).stem
        
        # 数値部分を抽出してゼロ埋め5桁に統一
        import re
        match = re.search(r'(\d+)', frame_name)
        if match:
            number = int(match.group(1))
            # 「00000」形式に統一
            return f"{number:05d}"
        
        return frame_name
    
    def calculate_gs_metrics(self, model_dir: Path) -> List[Dict]:
        """Gaussian Splattingの結果メトリクスを計算 - 適切なテストデータで評価 （正規化済み）"""
        results = []
        
        # 1. まず、GSが生成したレンダリング結果を取得
        test_renders_dir = model_dir / "test" / "ours_30000" / "renders"
        train_renders_dir = model_dir / "train" / "ours_30000" / "renders"
        
        renders_dir = None
        if test_renders_dir.exists():
            renders_dir = test_renders_dir
            self.logger.info("Using GS test renders for evaluation")
        elif train_renders_dir.exists():
            renders_dir = train_renders_dir
            self.logger.warning("Using GS train renders for evaluation (suboptimal)")
        else:
            self.logger.error("No GS rendering results found")
            return results
        
        # 2. 独立したテストデータセットからGT画像を取得
        test_images_dirs = [
            self.test_data_dir / "images",
            self.test_data_dir / "Input", 
            self.test_data_dir
        ]
        
        gt_images_dir = None
        for test_dir in test_images_dirs:
            if test_dir.exists() and test_dir.is_dir():
                gt_files = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.jpeg")) + list(test_dir.glob("*.png"))
                if gt_files:
                    gt_images_dir = test_dir
                    self.logger.info(f"Using independent test dataset: {gt_images_dir} ({len(gt_files)} images)")
                    break
        
        if not gt_images_dir:
            self.logger.error("No independent test images found for GS evaluation")
            return results
        
        # 3. レンダリング結果と独立テストデータでメトリクス計算
        render_images = sorted(renders_dir.glob("*.png"))
        gt_images = {}
        
        # GT画像をマッピング
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            for gt_path in gt_images_dir.glob(ext):
                # ファイル名マッチング（拡張子なし）
                gt_images[gt_path.stem] = gt_path
        
        self.logger.info(f"Found {len(render_images)} renders and {len(gt_images)} GT images")
        
        for render_img_path in tqdm(render_images, desc="Calculating GS metrics"):
            render_name = render_img_path.stem
            
            # 最適なGT画像を見つける（名前マッチまたは最適化マッチング）
            gt_img_path = None
            
            # 直接名前マッチング
            if render_name in gt_images:
                gt_img_path = gt_images[render_name]
            else:
                # 部分マッチング（例：render_001 vs 001）
                for gt_name, gt_path in gt_images.items():
                    if gt_name in render_name or render_name in gt_name:
                        gt_img_path = gt_path
                        break
                
                # それでも見つからない場合は最初のGT画像を使用
                if gt_img_path is None and gt_images:
                    gt_img_path = list(gt_images.values())[0]
                    self.logger.warning(f"No GT match for {render_name}, using {gt_img_path.name}")
            
            if gt_img_path and gt_img_path.exists():
                metrics = self.metrics_calc.calculate_metrics(render_img_path, gt_img_path)
                
                # 正規化されたフレーム名を使用（Blender結果とマッチングするため）
                normalized_frame_name = self.normalize_frame_name(render_img_path.name)
                
                result = {
                    'frame_name': normalized_frame_name,  # 正規化されたフレーム名
                    'original_render_name': render_img_path.stem,  # 元のレンダー名も保持
                    'gt_image': gt_img_path.name,  # GT画像名も記録
                    'render_time_sec': 0.1,  # 個別フレーム時間は概算
                    'psnr': metrics['psnr'],
                    'ssim': metrics['ssim'],
                    'lpips': metrics['lpips']
                }
                results.append(result)
                self.logger.debug(f"GS metrics calculated: {render_img_path.stem} -> frame {normalized_frame_name}")
            else:
                self.logger.warning(f"No GT image found for render: {render_name}")
                # メトリクスなしでもフレーム情報を保持
                normalized_frame_name = self.normalize_frame_name(render_img_path.name)
                result = {
                    'frame_name': normalized_frame_name,
                    'original_render_name': render_img_path.stem,
                    'gt_image': 'N/A',
                    'render_time_sec': 0.1,
                    'psnr': float('nan'),
                    'ssim': float('nan'),
                    'lpips': float('nan')
                }
                results.append(result)
        
        self.logger.info(f"Calculated metrics for {len(results)} image pairs")
        return results
    
    def evaluate(self, output_dir: Path) -> Tuple[List[Dict], float]:
        """Gaussian Splattingの完全な評価を実行"""
        # 学習実行
        training_success, training_time = self.run_gs_training(output_dir)
        
        if not training_success:
            return [], training_time
        
        # レンダリング実行
        rendering_success = self.run_gs_rendering(output_dir)
        
        if not rendering_success:
            return [], training_time
        
        # メトリクス計算
        results = self.calculate_gs_metrics(output_dir)
        
        return results, training_time


class RenderingMethodsEvaluator:
    """メインの評価オーケストレーター"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = self._setup_logging()
        
        # パスの設定
        self.train_data_dir = Path(config['train_data_dir'])
        self.test_data_dir = Path(config['test_data_dir'])
        self.blender_data_dir = Path(config['blender_data_dir'])
        self.gs_dir = Path(config['gaussian_splatting_dir'])
        self.output_dir = Path(config['output_dir'])
        
        # 出力ディレクトリ作成
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def normalize_frame_name(self, frame_name: str) -> str:
        """フレーム名を正規化してマッピングを改善"""
        # 拡張子を除去
        frame_name = Path(frame_name).stem
        
        # 数値部分を抽出してゼロ埋め5桁に統一
        import re
        match = re.search(r'(\d+)', frame_name)
        if match:
            number = int(match.group(1))
            # 「00000」形式に統一
            return f"{number:05d}"
        
        return frame_name
    
    def _setup_logging(self) -> logging.Logger:
        """ログ設定"""
        # Clear any existing handlers
        logging.getLogger().handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # File handler
        file_handler = logging.FileHandler('evaluation.log')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        # Console handler  
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        
        # Configure logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _validate_environment(self) -> bool:
        """実行前の環境検証"""
        self.logger.info("Validating evaluation environment...")
        
        validation_passed = True
        
        # 1. 必須ディレクトリの存在確認
        required_dirs = {
            'train_data_dir': self.train_data_dir,
            'test_data_dir': self.test_data_dir,
            'blender_data_dir': self.blender_data_dir,
            'gaussian_splatting_dir': self.gs_dir
        }
        
        for name, path in required_dirs.items():
            if not path.exists():
                self.logger.error(f"Required directory not found: {name} = {path}")
                validation_passed = False
            else:
                self.logger.info(f"[OK] Found {name}: {path}")
        
        if not validation_passed:
            return False
        
        # 2. Blenderデータの詳細チェック
        blender_images = list(self.blender_data_dir.glob("*.png")) + list(self.blender_data_dir.glob("*.jpg")) + list(self.blender_data_dir.glob("*.jpeg"))
        if not blender_images:
            self.logger.error(f"No Blender images found in {self.blender_data_dir}")
            # ディレクトリ内容を詳細表示
            all_files = list(self.blender_data_dir.iterdir())
            self.logger.error(f"Directory contains {len(all_files)} files:")
            for f in all_files[:10]:  # 最初の10ファイルを表示
                self.logger.error(f"  - {f.name}")
            if len(all_files) > 10:
                self.logger.error(f"  ... and {len(all_files) - 10} more files")
            validation_passed = False
        else:
            self.logger.info(f"[OK] Found {len(blender_images)} Blender images")
        
        # render_times.csvの存在確認（警告のみ）
        render_times_file = self.blender_data_dir / "render_times.csv"
        if not render_times_file.exists():
            self.logger.warning(f"render_times.csv not found - render times will be 0.0")
        else:
            self.logger.info(f"[OK] Found render_times.csv")
        
        # 3. テストデータのチェック - サブディレクトリも含めて検索
        test_images = []
        # まず直接ディレクトリ内を検索
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            test_images.extend(self.test_data_dir.glob(ext))
        
        # 画像が見つからない場合、サブディレクトリも検索
        if not test_images:
            self.logger.info("No images in root directory, searching subdirectories...")
            subdirs_to_check = ['images', 'Input', 'test']
            for subdir_name in subdirs_to_check:
                subdir = self.test_data_dir / subdir_name
                if subdir.exists() and subdir.is_dir():
                    for ext in ['*.png', '*.jpg', '*.jpeg']:
                        found_in_subdir = list(subdir.glob(ext))
                        if found_in_subdir:
                            test_images.extend(found_in_subdir)
                            self.logger.info(f"Found {len(found_in_subdir)} images in {subdir}")
                            break
                    if test_images:
                        break
        
        if not test_images:
            self.logger.error(f"No test images found in {self.test_data_dir}")
            # サブディレクトリもチェック
            subdirs = [d for d in self.test_data_dir.iterdir() if d.is_dir()]
            if subdirs:
                self.logger.error(f"Found subdirectories: {[d.name for d in subdirs]}")
                self.logger.error("Tip: Images should be directly in test_data_dir, not in subdirectories")
            validation_passed = False
        else:
            self.logger.info(f"[OK] Found {len(test_images)} test images")
        
        # 4. Gaussian Splattingの必要ファイルチェック
        gs_train_py = self.gs_dir / "train.py"
        if not gs_train_py.exists():
            self.logger.error(f"train.py not found in Gaussian Splatting directory: {gs_train_py}")
            validation_passed = False
        else:
            self.logger.info(f"[OK] Found train.py in Gaussian Splatting directory")
        
        if validation_passed:
            self.logger.info("[OK] Environment validation passed")
        else:
            self.logger.error("[ERROR] Environment validation failed")
            self.logger.error("Please fix the issues above before running evaluation")
        
        return validation_passed
    
    def run_evaluation(self) -> bool:
        """完全な評価を実行"""
        self.logger.info("Starting comprehensive rendering methods evaluation...")
        
        # 事前検証
        if not self._validate_environment():
            return False
        
        # 1. Blender評価
        self.logger.info("Phase 1: Blender evaluation")
        blender_evaluator = BlenderEvaluator(self.blender_data_dir, self.test_data_dir)
        blender_results = blender_evaluator.evaluate()
        
        if not blender_results:
            self.logger.error("Blender evaluation returned no results - check logs above")
        
        # 2. Gaussian Splatting評価
        self.logger.info("Phase 2: Gaussian Splatting evaluation")
        gs_output_dir = self.output_dir / "gs_model"
        gs_evaluator = GaussianSplattingEvaluator(
            self.train_data_dir, self.test_data_dir, self.gs_dir
        )
        gs_results, gs_training_time = gs_evaluator.evaluate(gs_output_dir)
        
        if not gs_results:
            self.logger.error("Gaussian Splatting evaluation returned no results - check logs above")
        
        # 3. 結果の集計と出力
        self.logger.info("Phase 3: Results aggregation")
        self._save_results(blender_results, gs_results, gs_training_time)
        
        # 4. サマリーの表示
        self._print_summary(blender_results, gs_results, gs_training_time)
        
        # 5. 最終ステータス
        if not blender_results and not gs_results:
            self.logger.error("Both evaluations failed - no results to report")
            return False
        elif not blender_results:
            self.logger.warning("Only Gaussian Splatting evaluation succeeded")
        elif not gs_results:
            self.logger.warning("Only Blender evaluation succeeded")
        else:
            self.logger.info("Both evaluations completed successfully")
        
        return True
    
    def _save_results(self, blender_results: List[Dict], gs_results: List[Dict], gs_training_time: float):
        """結果をCSVファイルに保存 - 正規化されたフレーム名でマッピング"""
        output_file = self.output_dir / "comparison_results.csv"
        
        # 結果を正規化されたフレーム名でマッピング
        blender_dict = {r['frame_name']: r for r in blender_results}  # Blenderは既に正規化済み
        gs_dict = {}
        for r in gs_results:
            # GS結果のフレーム名も正規化
            normalized_name = self.normalize_frame_name(r['frame_name'])
            gs_dict[normalized_name] = r
        
        # すべてのフレームを収集
        all_frames = set(blender_dict.keys()) | set(gs_dict.keys())
        
        self.logger.info(f"Frame mapping summary:")
        self.logger.info(f"  Blender frames: {sorted(blender_dict.keys())[:5]}{'...' if len(blender_dict) > 5 else ''}")
        self.logger.info(f"  GS frames: {sorted(gs_dict.keys())[:5]}{'...' if len(gs_dict) > 5 else ''}")
        self.logger.info(f"  Total unique frames: {len(all_frames)}")
        
        with open(output_file, 'w', newline='') as f:
            fieldnames = [
                'frame_filename',
                'blender_render_time_sec',
                'blender_psnr',
                'blender_ssim',
                'blender_lpips',
                'gs_render_time_sec',
                'gs_psnr',
                'gs_ssim',
                'gs_lpips'
            ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for frame in sorted(all_frames):
                blender_data = blender_dict.get(frame, {})
                gs_data = gs_dict.get(frame, {})
                
                row = {
                    'frame_filename': frame,
                    'blender_render_time_sec': blender_data.get('render_time_sec', 'N/A'),
                    'blender_psnr': blender_data.get('psnr', 'N/A'),
                    'blender_ssim': blender_data.get('ssim', 'N/A'),
                    'blender_lpips': blender_data.get('lpips', 'N/A'),
                    'gs_render_time_sec': gs_data.get('render_time_sec', 'N/A'),
                    'gs_psnr': gs_data.get('psnr', 'N/A'),
                    'gs_ssim': gs_data.get('ssim', 'N/A'),
                    'gs_lpips': gs_data.get('lpips', 'N/A')
                }
                writer.writerow(row)
                
                # デバッグ情報: マッチング状態をログ出力
                if frame in blender_dict and frame in gs_dict:
                    self.logger.debug(f"Frame {frame}: Both Blender and GS data available")
                elif frame in blender_dict:
                    self.logger.debug(f"Frame {frame}: Only Blender data available")
                elif frame in gs_dict:
                    self.logger.debug(f"Frame {frame}: Only GS data available")
        
        self.logger.info(f"Results saved to: {output_file}")
        
        # 詳細な統計情報も保存
        stats_file = self.output_dir / "evaluation_stats.json"
        stats = {
            'gs_training_time_sec': gs_training_time,
            'blender_results_count': len(blender_results),
            'gs_results_count': len(gs_results),
            'total_evaluation_frames': len(all_frames)
        }
        
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def _print_summary(self, blender_results: List[Dict], gs_results: List[Dict], gs_training_time: float):
        """評価結果のサマリーを表示"""
        print("\n" + "="*80)
        print("RENDERING METHODS EVALUATION SUMMARY")
        print("="*80)
        
        # 評価状態の表示
        print(f"\nEVALUATION STATUS:")
        print(f"  Blender results: {len(blender_results)} {'([OK] Success)' if blender_results else '([ERROR] Failed)'}")
        print(f"  GS results: {len(gs_results)} {'([OK] Success)' if gs_results else '([ERROR] Failed)'}")
        
        if not blender_results:
            print(f"\n[WARNING] BLENDER EVALUATION FAILED:")
            print(f"  - All Blender metrics will show as 'N/A' in the CSV")
            print(f"  - Check that blender_data_dir contains *.png/*.jpg files")
            print(f"  - Check that test_data_dir contains ground truth images")
            print(f"  - Review error logs above for specific issues")
        
        if not gs_results:
            print(f"\n[WARNING] GAUSSIAN SPLATTING EVALUATION FAILED:")
            print(f"  - All GS metrics will show as 'N/A' in the CSV")
            print(f"  - Check training data format and directory structure")
            print(f"  - Review error logs above for specific issues")
        
        # Blenderサマリー
        if blender_results:
            avg_blender_time = np.mean([r['render_time_sec'] for r in blender_results])
            avg_blender_psnr = np.nanmean([r['psnr'] for r in blender_results])
            avg_blender_ssim = np.nanmean([r['ssim'] for r in blender_results])
            avg_blender_lpips = np.nanmean([r['lpips'] for r in blender_results])
            
            print(f"\nBLENDER (Mesh-based) RESULTS:")
            print(f"  Frames evaluated: {len(blender_results)}")
            print(f"  Average render time: {avg_blender_time:.3f} sec/frame")
            print(f"  Average PSNR: {avg_blender_psnr:.2f} dB")
            print(f"  Average SSIM: {avg_blender_ssim:.4f}")
            print(f"  Average LPIPS: {avg_blender_lpips:.4f}")
        
        # Gaussian Splattingサマリー
        if gs_results:
            avg_gs_time = np.mean([r['render_time_sec'] for r in gs_results])
            avg_gs_psnr = np.nanmean([r['psnr'] for r in gs_results])
            avg_gs_ssim = np.nanmean([r['ssim'] for r in gs_results])
            avg_gs_lpips = np.nanmean([r['lpips'] for r in gs_results])
            
            print(f"\nGAUSSIAN SPLATTING RESULTS:")
            print(f"  Total training time: {gs_training_time:.1f} sec")
            print(f"  Frames evaluated: {len(gs_results)}")
            print(f"  Average render time: {avg_gs_time:.3f} sec/frame")
            print(f"  Average PSNR: {avg_gs_psnr:.2f} dB")
            print(f"  Average SSIM: {avg_gs_ssim:.4f}")
            print(f"  Average LPIPS: {avg_gs_lpips:.4f}")
        
        # 全データの総合比較（両方成功した場合のみ）
        if blender_results and gs_results:
            print(f"\n" + "="*80)
            print("OVERALL COMPARISON SUMMARY")
            print("="*80)
            
            # Blender全体平均
            blender_avg_accuracy = np.nanmean([r['psnr'] for r in blender_results])
            blender_avg_render_time = np.mean([r['render_time_sec'] for r in blender_results])
            
            # Gaussian Splatting全体平均  
            gs_avg_accuracy = np.nanmean([r['psnr'] for r in gs_results])
            gs_avg_render_time = np.mean([r['render_time_sec'] for r in gs_results])
            
            print(f"\nFINAL RESULTS - AVERAGE ACCURACY (PSNR):")
            print(f"  Blender (Mesh-based):     {blender_avg_accuracy:.2f} dB")
            print(f"  Gaussian Splatting:       {gs_avg_accuracy:.2f} dB")
            
            print(f"\nFINAL RESULTS - AVERAGE RENDER TIME:")
            print(f"  Blender (Mesh-based):     {blender_avg_render_time:.3f} sec/frame")
            print(f"  Gaussian Splatting:       {gs_avg_render_time:.3f} sec/frame")
            print(f"  GS Training Time:         {gs_training_time:.1f} sec (one-time)")
            
            # 優位性の判定
            if blender_avg_accuracy > gs_avg_accuracy:
                accuracy_winner = "Blender"
                accuracy_diff = blender_avg_accuracy - gs_avg_accuracy
            else:
                accuracy_winner = "Gaussian Splatting"
                accuracy_diff = gs_avg_accuracy - blender_avg_accuracy
                
            if blender_avg_render_time < gs_avg_render_time:
                speed_winner = "Blender"
                speed_ratio = gs_avg_render_time / blender_avg_render_time
            else:
                speed_winner = "Gaussian Splatting"
                speed_ratio = blender_avg_render_time / gs_avg_render_time
            
            print(f"\nCOMPARISON ANALYSIS:")
            print(f"  Accuracy Winner:    {accuracy_winner} (+{accuracy_diff:.2f} dB advantage)")
            print(f"  Speed Winner:       {speed_winner} ({speed_ratio:.1f}x faster)")
            print("="*80)
        elif blender_results or gs_results:
            print("\n" + "="*80)
        else:
            print("\n❌ BOTH EVALUATIONS FAILED - Please check the error messages above")
            print("="*80)
        
        # 全データの統計サマリー（平均値と標準偏差）を追加
        print("\n" + "="*80)
        print("DETAILED STATISTICS SUMMARY")
        print("="*80)
        
        if blender_results:
            # Blenderの統計計算
            blender_psnr_values = [r['psnr'] for r in blender_results if not math.isnan(r['psnr'])]
            blender_ssim_values = [r['ssim'] for r in blender_results if not math.isnan(r['ssim'])]
            blender_lpips_values = [r['lpips'] for r in blender_results if not math.isnan(r['lpips'])]
            blender_time_values = [r['render_time_sec'] for r in blender_results]
            
            blender_psnr_mean = np.mean(blender_psnr_values) if blender_psnr_values else float('nan')
            blender_psnr_std = np.std(blender_psnr_values) if blender_psnr_values else float('nan')
            blender_ssim_mean = np.mean(blender_ssim_values) if blender_ssim_values else float('nan')
            blender_ssim_std = np.std(blender_ssim_values) if blender_ssim_values else float('nan')
            blender_lpips_mean = np.mean(blender_lpips_values) if blender_lpips_values else float('nan')
            blender_lpips_std = np.std(blender_lpips_values) if blender_lpips_values else float('nan')
            blender_time_mean = np.mean(blender_time_values)
            blender_time_std = np.std(blender_time_values)
            
            print(f"\nBLENDER DETAILED STATISTICS:")
            print(f"  PSNR:        Mean = {blender_psnr_mean:.2f} ± {blender_psnr_std:.2f} dB")
            print(f"  SSIM:        Mean = {blender_ssim_mean:.4f} ± {blender_ssim_std:.4f}")
            print(f"  LPIPS:       Mean = {blender_lpips_mean:.4f} ± {blender_lpips_std:.4f}")
            print(f"  Render Time: Mean = {blender_time_mean:.3f} ± {blender_time_std:.3f} sec/frame")
        
        if gs_results:
            # Gaussian Splattingの統計計算
            gs_psnr_values = [r['psnr'] for r in gs_results if not math.isnan(r['psnr'])]
            gs_ssim_values = [r['ssim'] for r in gs_results if not math.isnan(r['ssim'])]
            gs_lpips_values = [r['lpips'] for r in gs_results if not math.isnan(r['lpips'])]
            gs_time_values = [r['render_time_sec'] for r in gs_results]
            
            gs_psnr_mean = np.mean(gs_psnr_values) if gs_psnr_values else float('nan')
            gs_psnr_std = np.std(gs_psnr_values) if gs_psnr_values else float('nan')
            gs_ssim_mean = np.mean(gs_ssim_values) if gs_ssim_values else float('nan')
            gs_ssim_std = np.std(gs_ssim_values) if gs_ssim_values else float('nan')
            gs_lpips_mean = np.mean(gs_lpips_values) if gs_lpips_values else float('nan')
            gs_lpips_std = np.std(gs_lpips_values) if gs_lpips_values else float('nan')
            gs_time_mean = np.mean(gs_time_values)
            gs_time_std = np.std(gs_time_values)
            
            print(f"\nGAUSSIAN SPLATTING DETAILED STATISTICS:")
            print(f"  PSNR:        Mean = {gs_psnr_mean:.2f} ± {gs_psnr_std:.2f} dB")
            print(f"  SSIM:        Mean = {gs_ssim_mean:.4f} ± {gs_ssim_std:.4f}")
            print(f"  LPIPS:       Mean = {gs_lpips_mean:.4f} ± {gs_lpips_std:.4f}")
            print(f"  Render Time: Mean = {gs_time_mean:.3f} ± {gs_time_std:.3f} sec/frame")
            print(f"  Training Time: {gs_training_time:.1f} sec (one-time setup)")
        
        print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and compare Blender mesh-based rendering vs Gaussian Splatting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python evaluate_rendering_methods.py
  
Expected directory structure:
  project_root/
  ├── data_input/
  │   ├── BOX-A_train/       # Training dataset (for GS only)
  │   └── BOX-A_test/        # Evaluation dataset (for both methods)
  ├── blender_data/
  │   ├── *.png              # Blender rendered images
  │   └── render_times.csv   # Rendering time data
  └── gaussian-splatting/    # GS implementation directory
        """
    )
    
    parser.add_argument("--train-data", type=str, default="data_input/BOX-A_train",
                        help="Training dataset directory (for GS)")
    parser.add_argument("--test-data", type=str, default="data_input/BOX-A_test", 
                        help="Test dataset directory (for evaluation)")
    parser.add_argument("--blender-data", type=str, default="blender_data",
                        help="Blender rendering data directory")
    parser.add_argument("--gs-dir", type=str, default="gaussian-splatting",
                        help="Gaussian Splatting implementation directory")
    parser.add_argument("--output", type=str, default="evaluation_results",
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    # 設定
    config = {
        'train_data_dir': args.train_data,
        'test_data_dir': args.test_data,
        'blender_data_dir': args.blender_data,
        'gaussian_splatting_dir': args.gs_dir,
        'output_dir': args.output
    }
    
    # 入力ディレクトリの存在チェック（出力ディレクトリは除外）
    input_dirs = {k: v for k, v in config.items() if k != 'output_dir'}
    missing_dirs = []
    
    for key, path in input_dirs.items():
        if not Path(path).exists():
            missing_dirs.append((key, path))
    
    if missing_dirs:
        print("Error: The following required directories were not found:")
        for key, path in missing_dirs:
            print(f"  - {key}: {path}")
        
        print("\nPlease ensure you have:")
        print("  1. Training data: data_input/BOX-A_train/ (with images/ subdirectory)")
        print("  2. Test data: data_input/BOX-A_test/ (with images/ subdirectory)")
        print("  3. Blender data: blender_data/ (with *.png files and render_times.csv)")
        print("  4. Gaussian Splatting: gaussian-splatting/ (with train.py)")
        print("\nAvailable datasets detected:")
        print("  - data_input/merged/: COLMAP format")
        print("  - data_input/nerf_blender_qiita/: NeRF format")
        print("\nSee EVALUATION_GUIDE.md for detailed setup instructions.")
        return 1
    
    try:
        # 評価実行
        evaluator = RenderingMethodsEvaluator(config)
        success = evaluator.run_evaluation()
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        return 1
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())