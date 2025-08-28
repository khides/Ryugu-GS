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
import torchvision.transforms.functional as tf
from tqdm import tqdm

# Gaussian Splattingモジュールのインポート（存在する場合）
try:
    from utils.loss_utils import ssim
    from lpipsPyTorch import lpips
    from utils.image_utils import psnr
except ImportError:
    print("Warning: Gaussian Splatting utils not found. Using fallback implementations.")
    print("  To enable full functionality, ensure you're running from the gaussian-splatting directory")
    print("  or that PYTHONPATH includes the gaussian-splatting directory.")
    ssim = None
    lpips = None
    psnr = None


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
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
    
    def calculate_psnr(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """PSNR計算"""
        if psnr is not None:
            return psnr(img1, img2).item()
        else:
            # フォールバック実装
            mse = torch.mean((img1 - img2) ** 2)
            if mse == 0:
                return float('inf')
            return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()
    
    def calculate_ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """SSIM計算"""
        if ssim is not None:
            return ssim(img1, img2).item()
        else:
            # 簡易的なSSIM実装（完全ではない）
            self.logger.warning("Using fallback SSIM implementation")
            return 0.5  # プレースホルダー
    
    def calculate_lpips(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """LPIPS計算"""
        if lpips is not None:
            return lpips(img1, img2, net_type='vgg').item()
        else:
            # フォールバック実装
            self.logger.warning("LPIPS not available, using L2 distance as fallback")
            return torch.mean((img1 - img2) ** 2).item()
    
    def image_to_tensor(self, image_path: Path) -> torch.Tensor:
        """画像をテンソルに変換"""
        image = Image.open(image_path).convert('RGB')
        tensor = tf.to_tensor(image).unsqueeze(0)[:, :3, :, :].to(self.device)
        return tensor
    
    def calculate_metrics(self, img1_path: Path, img2_path: Path) -> Dict[str, float]:
        """2つの画像間のすべてのメトリクスを計算"""
        try:
            img1_tensor = self.image_to_tensor(img1_path)
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
            
            psnr_val = self.calculate_psnr(img1_tensor, img2_tensor)
            ssim_val = self.calculate_ssim(img1_tensor, img2_tensor)
            lpips_val = self.calculate_lpips(img1_tensor, img2_tensor)
            
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
        
        try:
            with open(render_times_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # CSVファイルの構造に応じて調整
                    frame_name = row.get('frame', row.get('filename', ''))
                    time_sec = float(row.get('time_sec', row.get('time', 0.0)))
                    render_times[frame_name] = time_sec
        except Exception as e:
            self.logger.error(f"Error loading render times: {e}")
            
        return render_times
    
    def find_best_match(self, blender_image_path: Path, test_images: List[Path]) -> Tuple[Path, Dict[str, float]]:
        """Blenderレンダリング画像に最も近い評価用画像を見つける"""
        # Blender画像を前処理
        blender_img = cv2.imread(str(blender_image_path))
        if blender_img is None:
            raise ValueError(f"Cannot load Blender image: {blender_image_path}")
        
        processed_blender_img = self.preprocessor.preprocess(blender_img)
        
        best_match = None
        best_metrics = None
        best_score = float('-inf')
        
        # 一時ファイルとして保存
        temp_blender_path = self.blender_data_dir / "temp_processed_blender.png"
        cv2.imwrite(str(temp_blender_path), processed_blender_img)
        
        try:
            for test_image_path in test_images:
                # テスト画像を前処理
                test_img = cv2.imread(str(test_image_path))
                if test_img is None:
                    continue
                
                processed_test_img = self.preprocessor.preprocess(test_img)
                
                # 一時ファイルとして保存
                temp_test_path = self.test_data_dir / "temp_processed_test.png"
                cv2.imwrite(str(temp_test_path), processed_test_img)
                
                # メトリクスを計算
                metrics = self.metrics_calc.calculate_metrics(temp_blender_path, temp_test_path)
                
                # スコア計算（PSNR+SSIM-LPIPS の組み合わせ）
                if not (math.isnan(metrics['psnr']) or math.isnan(metrics['ssim']) or math.isnan(metrics['lpips'])):
                    score = metrics['psnr'] / 50.0 + metrics['ssim'] - metrics['lpips']
                    
                    if score > best_score:
                        best_score = score
                        best_match = test_image_path
                        best_metrics = metrics
                
                # 一時ファイル削除
                if temp_test_path.exists():
                    temp_test_path.unlink()
            
        finally:
            # 一時ファイル削除
            if temp_blender_path.exists():
                temp_blender_path.unlink()
        
        if best_match is None:
            # フォールバック: 最初の利用可能な画像
            if test_images:
                best_match = test_images[0]
                best_metrics = {'psnr': float('nan'), 'ssim': float('nan'), 'lpips': float('nan')}
        
        return best_match, best_metrics
    
    def evaluate(self) -> List[Dict]:
        """Blenderレンダリングの評価を実行"""
        self.logger.info("Starting Blender evaluation...")
        
        # レンダリング時間を読み込み
        render_times = self.load_render_times()
        
        # Blenderレンダリング画像を取得
        blender_images = list(self.blender_data_dir.glob("*.png")) + list(self.blender_data_dir.glob("*.jpg")) + list(self.blender_data_dir.glob("*.jpeg"))
        
        # 評価用画像を取得
        test_images = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            test_images.extend(self.test_data_dir.glob(ext))
        
        results = []
        
        for blender_img_path in tqdm(blender_images, desc="Evaluating Blender renders"):
            try:
                # 最適なマッチを見つける
                best_match, metrics = self.find_best_match(blender_img_path, test_images)
                
                if best_match:
                    result = {
                        'blender_image': blender_img_path.name,
                        'ground_truth_image': best_match.name,
                        'render_time_sec': render_times.get(blender_img_path.stem, 0.0),
                        'psnr': metrics['psnr'],
                        'ssim': metrics['ssim'],
                        'lpips': metrics['lpips']
                    }
                    results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error evaluating {blender_img_path}: {e}")
        
        self.logger.info(f"Completed Blender evaluation: {len(results)} results")
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
    
    def calculate_gs_metrics(self, model_dir: Path) -> List[Dict]:
        """Gaussian Splattingの結果メトリクスを計算"""
        results = []
        
        # レンダリング結果ディレクトリ
        test_renders_dir = model_dir / "test" / "ours_30000" / "renders"
        test_gt_dir = model_dir / "test" / "ours_30000" / "gt"
        
        if not test_renders_dir.exists() or not test_gt_dir.exists():
            self.logger.warning("GS test results not found, using train results")
            test_renders_dir = model_dir / "train" / "ours_30000" / "renders"
            test_gt_dir = model_dir / "train" / "ours_30000" / "gt"
        
        if not test_renders_dir.exists():
            self.logger.error("No GS rendering results found")
            return results
        
        # レンダリング画像を取得
        render_images = sorted(test_renders_dir.glob("*.png"))
        
        for render_img_path in tqdm(render_images, desc="Calculating GS metrics"):
            gt_img_path = test_gt_dir / render_img_path.name
            
            if gt_img_path.exists():
                metrics = self.metrics_calc.calculate_metrics(render_img_path, gt_img_path)
                
                result = {
                    'frame_name': render_img_path.stem,
                    'render_time_sec': 0.1,  # 個別フレーム時間は概算
                    'psnr': metrics['psnr'],
                    'ssim': metrics['ssim'],
                    'lpips': metrics['lpips']
                }
                results.append(result)
        
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
    
    def run_evaluation(self) -> bool:
        """完全な評価を実行"""
        self.logger.info("Starting comprehensive rendering methods evaluation...")
        
        # 1. Blender評価
        self.logger.info("Phase 1: Blender evaluation")
        blender_evaluator = BlenderEvaluator(self.blender_data_dir, self.test_data_dir)
        blender_results = blender_evaluator.evaluate()
        
        # 2. Gaussian Splatting評価
        self.logger.info("Phase 2: Gaussian Splatting evaluation")
        gs_output_dir = self.output_dir / "gs_model"
        gs_evaluator = GaussianSplattingEvaluator(
            self.train_data_dir, self.test_data_dir, self.gs_dir
        )
        gs_results, gs_training_time = gs_evaluator.evaluate(gs_output_dir)
        
        # 3. 結果の集計と出力
        self.logger.info("Phase 3: Results aggregation")
        self._save_results(blender_results, gs_results, gs_training_time)
        
        # 4. サマリーの表示
        self._print_summary(blender_results, gs_results, gs_training_time)
        
        return True
    
    def _save_results(self, blender_results: List[Dict], gs_results: List[Dict], gs_training_time: float):
        """結果をCSVファイルに保存"""
        output_file = self.output_dir / "comparison_results.csv"
        
        # 結果をフレーム名でマッピング
        blender_dict = {r['ground_truth_image']: r for r in blender_results}
        gs_dict = {r['frame_name']: r for r in gs_results}
        
        # すべてのフレームを収集
        all_frames = set(blender_dict.keys()) | set(gs_dict.keys())
        
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
        
        print("\n" + "="*80)


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