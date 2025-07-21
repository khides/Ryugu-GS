#!/usr/bin/env python3

import os
import json
import argparse
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from tqdm import tqdm

# Import Gaussian Splatting utilities
import sys
sys.path.append(str(Path(__file__).parent.parent / "gaussian-splatting"))

try:
    from utils.loss_utils import ssim
    from utils.image_utils import psnr
    from lpipsPyTorch import lpips
    GS_UTILS_AVAILABLE = True
except ImportError:
    print("Warning: Gaussian Splatting utilities not found. Installing fallback implementations...")
    GS_UTILS_AVAILABLE = False

# Fallback implementations if Gaussian Splatting utils are not available
if not GS_UTILS_AVAILABLE:
    import numpy as np
    from skimage.metrics import structural_similarity as compare_ssim
    from skimage.metrics import peak_signal_noise_ratio as compare_psnr
    
    def psnr(img1, img2):
        """Fallback PSNR implementation"""
        img1_np = img1.squeeze().cpu().numpy().transpose(1, 2, 0)
        img2_np = img2.squeeze().cpu().numpy().transpose(1, 2, 0)
        return compare_psnr(img1_np, img2_np, data_range=1.0)
    
    def ssim(img1, img2):
        """Fallback SSIM implementation"""
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

    # Simple LPIPS fallback using L2 distance
    def lpips(img1, img2, net_type='alex', version='0.1'):
        """Fallback LPIPS implementation using L2 distance"""
        return torch.mean((img1 - img2) ** 2).item()

class MVSMetricsCalculator:
    """Calculate image quality metrics for MVS rendered images"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Initialize LPIPS network if available
        try:
            self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)
        except:
            print("Warning: LPIPS network not available, using fallback")
            self.lpips_fn = None
    
    def load_images(self, rendered_dir: str, gt_dir: str):
        """Load rendered and ground truth images"""
        rendered_dir = Path(rendered_dir)
        gt_dir = Path(gt_dir)
        
        if not rendered_dir.exists():
            raise FileNotFoundError(f"Rendered images directory not found: {rendered_dir}")
        
        if not gt_dir.exists():
            raise FileNotFoundError(f"Ground truth images directory not found: {gt_dir}")
        
        # Get list of rendered images
        rendered_files = list(rendered_dir.glob("*.png")) + list(rendered_dir.glob("*.jpg"))
        rendered_files.sort()
        
        if len(rendered_files) == 0:
            raise ValueError(f"No images found in rendered directory: {rendered_dir}")
        
        renders = []
        gts = []
        image_names = []
        missing_gt = []
        
        print(f"Loading images from {rendered_dir}...")
        
        for render_file in tqdm(rendered_files):
            # Try to find corresponding ground truth image
            gt_candidates = [
                gt_dir / render_file.name,
                gt_dir / render_file.name.replace("_rendered", ""),
                gt_dir / (render_file.stem.replace("_rendered", "") + ".png"),
                gt_dir / (render_file.stem.replace("_rendered", "") + ".jpg"),
            ]
            
            gt_file = None
            for candidate in gt_candidates:
                if candidate.exists():
                    gt_file = candidate
                    break
            
            if gt_file is None:
                missing_gt.append(render_file.name)
                continue
            
            try:
                # Load images
                render = Image.open(render_file).convert('RGB')
                gt = Image.open(gt_file).convert('RGB')
                
                # Resize ground truth to match rendered image if needed
                if render.size != gt.size:
                    gt = gt.resize(render.size, Image.LANCZOS)
                
                # Convert to tensors
                render_tensor = tf.to_tensor(render).unsqueeze(0).to(self.device)
                gt_tensor = tf.to_tensor(gt).unsqueeze(0).to(self.device)
                
                renders.append(render_tensor)
                gts.append(gt_tensor)
                image_names.append(render_file.name)
                
            except Exception as e:
                print(f"Failed to load {render_file}: {e}")
                continue
        
        if missing_gt:
            print(f"Warning: Could not find ground truth for {len(missing_gt)} images:")
            for name in missing_gt[:5]:  # Show first 5
                print(f"  - {name}")
            if len(missing_gt) > 5:
                print(f"  ... and {len(missing_gt) - 5} more")
        
        if len(renders) == 0:
            raise ValueError("No valid image pairs found")
        
        print(f"Loaded {len(renders)} image pairs")
        return renders, gts, image_names
    
    def calculate_metrics(self, rendered_dir: str, gt_dir: str):
        """Calculate PSNR, SSIM, and LPIPS metrics"""
        # Load images
        renders, gts, image_names = self.load_images(rendered_dir, gt_dir)
        
        # Initialize metric storage
        psnr_scores = []
        ssim_scores = []
        lpips_scores = []
        per_image_results = {}
        
        print("Calculating metrics...")
        
        for render, gt, name in tqdm(zip(renders, gts, image_names), total=len(renders)):
            # Calculate PSNR
            psnr_score = psnr(render, gt).item() if hasattr(psnr(render, gt), 'item') else psnr(render, gt)
            
            # Calculate SSIM
            ssim_score = ssim(render, gt).item() if hasattr(ssim(render, gt), 'item') else ssim(render, gt)
            
            # Calculate LPIPS
            if self.lpips_fn is not None:
                with torch.no_grad():
                    lpips_score = self.lpips_fn(render, gt).item()
            else:
                # Fallback LPIPS
                lpips_score = lpips(render, gt)
            
            # Store scores
            psnr_scores.append(psnr_score)
            ssim_scores.append(ssim_score)
            lpips_scores.append(lpips_score)
            
            per_image_results[name] = {
                "PSNR": float(psnr_score),
                "SSIM": float(ssim_score),
                "LPIPS": float(lpips_score)
            }
        
        # Calculate averages
        avg_psnr = sum(psnr_scores) / len(psnr_scores)
        avg_ssim = sum(ssim_scores) / len(ssim_scores)
        avg_lpips = sum(lpips_scores) / len(lpips_scores)
        
        results = {
            "summary": {
                "num_images": len(renders),
                "PSNR": float(avg_psnr),
                "SSIM": float(avg_ssim),
                "LPIPS": float(avg_lpips)
            },
            "per_image": per_image_results
        }
        
        return results
    
    def print_results(self, results: dict):
        """Print formatted results"""
        summary = results["summary"]
        
        print(f"\n{'='*60}")
        print("MVS BENCHMARK METRICS")
        print(f"{'='*60}")
        print(f"Number of test images: {summary['num_images']}")
        print(f"PSNR ↑: {summary['PSNR']:.2f} dB")
        print(f"SSIM ↑: {summary['SSIM']:.4f}")
        print(f"LPIPS ↓: {summary['LPIPS']:.4f}")
        print(f"{'='*60}\n")
    
    def save_results(self, results: dict, output_file: str):
        """Save results to JSON file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Calculate metrics for MVS rendered images")
    parser.add_argument("-r", "--rendered", type=str, required=True,
                       help="Directory containing rendered images")
    parser.add_argument("-g", "--gt", type=str, required=True,
                       help="Directory containing ground truth images")
    parser.add_argument("-o", "--output", type=str, default="./metrics.json",
                       help="Output JSON file for metrics")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for computation (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.rendered).exists():
        print(f"Error: Rendered images directory not found: {args.rendered}")
        return 1
    
    if not Path(args.gt).exists():
        print(f"Error: Ground truth images directory not found: {args.gt}")
        return 1
    
    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        args.device = "cpu"
    
    try:
        # Calculate metrics
        calculator = MVSMetricsCalculator(args.device)
        results = calculator.calculate_metrics(args.rendered, args.gt)
        
        # Print and save results
        calculator.print_results(results)
        calculator.save_results(results, args.output)
        
        return 0
        
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return 1

if __name__ == "__main__":
    exit(main())