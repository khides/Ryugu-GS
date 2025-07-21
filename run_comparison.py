#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Tuple

class ComparisonBenchmark:
    """Orchestrates comparison between Gaussian Splatting and MVS pipelines"""
    
    def __init__(self, data_path: str, output_dir: str = None, gs_dir: str = None):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir) if output_dir else self.data_path / "comparison_output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gs_dir_override = Path(gs_dir) if gs_dir else None
        
        # Validate input data structure
        self.validate_input_data()
        
        # Initialize results storage
        self.results = {
            "data_path": str(self.data_path),
            "output_dir": str(self.output_dir),
            "gaussian_splatting": {},
            "mvs_benchmark": {},
            "comparison": {}
        }
    
    def validate_input_data(self):
        """Validate that input data has required structure"""
        # Check for NeRF-style structure
        nerf_paths = [
            self.data_path / "colmap",
            self.data_path / "images",
            self.data_path / "transforms_test.json"
        ]
        
        # Check for COLMAP-style structure
        colmap_paths = [
            self.data_path / "sparse" / "0",
            self.data_path / "Input"
        ]
        
        # Test if we have NeRF-style data
        nerf_missing = [p for p in nerf_paths if not p.exists()]
        colmap_missing = [p for p in colmap_paths if not p.exists()]
        
        if len(nerf_missing) == 0:
            # NeRF-style format detected
            self.data_format = "nerf"
            print("Detected NeRF-style data format")
            return
        elif len(colmap_missing) == 0:
            # COLMAP-style format detected
            self.data_format = "colmap"
            print("Detected COLMAP-style data format")
            
            # Check for required COLMAP files
            sparse_dir = self.data_path / "sparse" / "0"
            required_files = ["cameras.bin", "images.bin", "points3D.bin"]
            missing_files = [f for f in required_files if not (sparse_dir / f).exists()]
            
            if missing_files:
                raise FileNotFoundError(f"Missing COLMAP sparse reconstruction files: {missing_files}")
            
            return
        else:
            # Neither format is valid
            print("Error: Invalid input data structure.")
            print("Expected either:")
            print("\nNeRF-style:")
            print("  <data_path>/")
            print("  ├── colmap/              # COLMAP sparse reconstruction")
            print("  ├── images/              # Input images")
            print("  └── transforms_test.json # Test camera poses")
            
            print("\nCOLMAP-style:")
            print("  <data_path>/")
            print("  ├── sparse/0/            # COLMAP sparse reconstruction")
            print("  │   ├── cameras.bin")
            print("  │   ├── images.bin")
            print("  │   └── points3D.bin")
            print("  ├── Input/               # Source images")
            print("  └── database.db          # COLMAP database (optional)")
            
            raise FileNotFoundError("Invalid input data structure")
    
    def find_gaussian_splatting_directory(self) -> Path:
        """Find the Gaussian Splatting directory relative to the current location"""
        # If user provided override, use that
        if self.gs_dir_override and self.gs_dir_override.exists():
            if (self.gs_dir_override / "train.py").exists():
                print(f"Using user-specified Gaussian Splatting directory: {self.gs_dir_override}")
                return self.gs_dir_override.resolve()
            else:
                print(f"Warning: User-specified directory {self.gs_dir_override} does not contain train.py")
        
        # Try different possible locations for gaussian-splatting directory
        script_dir = Path(__file__).parent  # Directory where this script is located
        
        candidates = [
            # Same directory as this script
            script_dir / "gaussian-splatting",
            # Parent directory
            script_dir.parent / "gaussian-splatting",
            # Relative to data path
            self.data_path.parent / "gaussian-splatting",
            # Direct relative path
            Path("./gaussian-splatting"),
            Path("../gaussian-splatting"),
        ]
        
        for candidate in candidates:
            if candidate.exists() and (candidate / "train.py").exists():
                print(f"Found Gaussian Splatting directory: {candidate}")
                return candidate.resolve()
        
        # If not found, try to find it in the current working directory tree
        cwd = Path.cwd()
        for candidate in [
            cwd / "gaussian-splatting",
            cwd.parent / "gaussian-splatting"
        ]:
            if candidate.exists() and (candidate / "train.py").exists():
                print(f"Found Gaussian Splatting directory: {candidate}")
                return candidate.resolve()
        
        print("Warning: Could not automatically locate gaussian-splatting directory")
        print("Searched locations:")
        for candidate in candidates:
            print(f"  - {candidate}")
        
        return None
    
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
        elif "ページング ファイルが小さすぎる" in stderr or "paging file" in stderr_lower:
            diagnosis["diagnosis"] = "Virtual memory (page file) insufficient"
            diagnosis["suggestions"] = [
                "Increase Windows virtual memory (page file) size to at least 8GB",
                "Close other memory-intensive applications",
                "Consider upgrading system RAM",
                "Run: Control Panel → System → Advanced → Performance Settings → Advanced → Virtual Memory → Change"
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
                "Check if PyTorch is installed with correct CUDA support: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118",
                "Try CPU-only mode if GPU is not required",
                "Restart system to refresh library paths"
            ]
        elif "torch" in stderr_lower and ("import" in stderr_lower or "module" in stderr_lower):
            diagnosis["diagnosis"] = "PyTorch installation or import issue"
            diagnosis["suggestions"] = [
                "Verify PyTorch installation: pip install torch torchvision",
                "Check Python environment activation",
                "Reinstall PyTorch with CUDA support if needed",
                "Try: conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia"
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
    
    def run_command_with_timing(self, cmd: list, description: str, 
                               cwd: str = None) -> Tuple[float, int, str, str]:
        """Run command and measure execution time"""
        print(f"\n{'='*60}")
        print(f"Running: {description}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Use Popen for real-time output
            process = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Capture output while displaying it in real-time
            output_lines = []
            print(f"🔄 {description} started - showing real-time output:")
            print("─" * 60)
            
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
                    output_lines.append(output.strip())
            
            # Wait for process completion
            process.wait()
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Combine all output
            full_output = "\n".join(output_lines)
            
            print("─" * 60)
            if process.returncode != 0:
                print(f"❌ Error in {description}:")
                print(f"Return code: {process.returncode}")
                return elapsed_time, process.returncode, full_output, full_output
            
            print(f"✓ {description} completed in {elapsed_time:.2f}s")
            return elapsed_time, process.returncode, full_output, ""
            
        except subprocess.TimeoutExpired:
            print(f"Timeout: {description} exceeded 2 hours")
            return 7200.0, -1, "", "Timeout exceeded"
        except Exception as e:
            print(f"Failed to run {description}: {e}")
            return 0.0, -1, "", str(e)
    
    def run_gaussian_splatting_pipeline(self) -> bool:
        """Run the Gaussian Splatting training and evaluation pipeline"""
        print(f"\n🚀 STARTING GAUSSIAN SPLATTING PIPELINE")
        
        gs_output_dir = self.output_dir / "gaussian_splatting"
        gs_output_dir.mkdir(exist_ok=True)
        
        # Find Gaussian Splatting directory
        gs_dir = self.find_gaussian_splatting_directory()
        if gs_dir is None:
            print("Error: Could not find gaussian-splatting directory")
            self.results["gaussian_splatting"]["error"] = "Gaussian Splatting directory not found"
            return False
        
        try:
            # Step 1: Training with memory-optimized settings
            # Use absolute paths to avoid path resolution issues when changing working directory
            abs_data_path = self.data_path.resolve()
            abs_model_path = (gs_output_dir / "model").resolve()
            
            train_cmd = [
                sys.executable, "train.py",
                "-s", str(abs_data_path),
                "-m", str(abs_model_path),
                "--resolution", "2",  # Reduce image resolution to save VRAM
                "--data_device", "cpu",  # Store images in CPU memory
            ]
            
            # Add additional memory optimization if system has limited VRAM
            try:
                import psutil
                memory = psutil.virtual_memory()
                if memory.total / (1024**3) < 16:  # Less than 16GB RAM
                    train_cmd.extend(["--sh_degree", "2"])  # Reduce spherical harmonics degree
                    print("ℹ️  Added memory optimization due to limited RAM")
            except ImportError:
                print("ℹ️  psutil not available, skipping automatic memory optimization")
                print("    You can install it with: pip install psutil")
            
            train_time, train_ret, _, train_stderr = self.run_command_with_timing(
                train_cmd, "Gaussian Splatting Training", 
                cwd=str(gs_dir)
            )
            
            if train_ret != 0:
                # Provide detailed error diagnosis
                error_diagnosis = self.diagnose_gaussian_splatting_error(train_stderr)
                print(f"\n❌ GAUSSIAN SPLATTING TRAINING FAILED")
                print(f"Error Diagnosis: {error_diagnosis['diagnosis']}")
                if error_diagnosis['suggestions']:
                    print(f"Suggestions:")
                    for suggestion in error_diagnosis['suggestions']:
                        print(f"  • {suggestion}")
                
                self.results["gaussian_splatting"]["training"] = {
                    "success": False,
                    "time_seconds": train_time,
                    "error": train_stderr,
                    "diagnosis": error_diagnosis
                }
                return False
            
            # Step 2: Rendering
            render_cmd = [
                sys.executable, "render.py",
                "-m", str(abs_model_path)
            ]
            
            render_time, render_ret, _, render_stderr = self.run_command_with_timing(
                render_cmd, "Gaussian Splatting Rendering",
                cwd=str(gs_dir)
            )
            
            if render_ret != 0:
                self.results["gaussian_splatting"]["rendering"] = {
                    "success": False,
                    "time_seconds": render_time,
                    "error": render_stderr
                }
                return False
            
            # Step 3: Metrics calculation
            metrics_cmd = [
                sys.executable, "metrics.py",
                "-m", str(abs_model_path)
            ]
            
            metrics_time, _, _, _ = self.run_command_with_timing(
                metrics_cmd, "Gaussian Splatting Metrics",
                cwd=str(gs_dir)
            )
            
            # Parse metrics results (attempt to find results.json)
            metrics_file = gs_output_dir / "model" / "results.json"
            gs_metrics = {}
            if metrics_file.exists():
                try:
                    with open(metrics_file) as f:
                        gs_metrics = json.load(f)
                except Exception as e:
                    print(f"Warning: Could not parse GS metrics: {e}")
            
            # Store results
            self.results["gaussian_splatting"] = {
                "success": True,
                "training_time_seconds": train_time,
                "rendering_time_seconds": render_time,
                "metrics_time_seconds": metrics_time,
                "total_time_seconds": train_time + render_time + metrics_time,
                "metrics": gs_metrics
            }
            
            print("✓ Gaussian Splatting pipeline completed successfully")
            return True
            
        except Exception as e:
            print(f"Error in Gaussian Splatting pipeline: {e}")
            self.results["gaussian_splatting"]["error"] = str(e)
            return False
    
    def run_mvs_benchmark_pipeline(self) -> bool:
        """Run the MVS benchmark pipeline"""
        print(f"\n🏗️  STARTING MVS BENCHMARK PIPELINE")
        
        mvs_output_dir = self.output_dir / "mvs_benchmark"
        
        try:
            # Step 1: Run MVS reconstruction
            mvs_cmd = [
                sys.executable, "mvs_benchmark/run_mvs.py",
                "-s", str(self.data_path),
                "-o", str(mvs_output_dir)
            ]
            
            mvs_time, mvs_ret, _, mvs_stderr = self.run_command_with_timing(
                mvs_cmd, "MVS Reconstruction"
            )
            
            if mvs_ret != 0:
                self.results["mvs_benchmark"]["reconstruction"] = {
                    "success": False,
                    "time_seconds": mvs_time,
                    "error": mvs_stderr
                }
                return False
            
            # Find generated mesh file
            mesh_candidates = [
                mvs_output_dir / "scene_textured.obj",
                mvs_output_dir / "scene.obj",
                mvs_output_dir / "mesh_textured.obj"
            ]
            
            mesh_file = None
            for candidate in mesh_candidates:
                if candidate.exists():
                    mesh_file = candidate
                    break
            
            if mesh_file is None:
                print("Error: No textured mesh found after MVS reconstruction")
                return False
            
            # Step 2: Render novel views
            render_output_dir = mvs_output_dir / "rendered_images"
            render_cmd = [
                sys.executable, "mvs_benchmark/render_mvs.py",
                "-m", str(mesh_file),
                "-t", str(self.data_path / "transforms_test.json"),
                "-o", str(render_output_dir)
            ]
            
            render_time, render_ret, _, render_stderr = self.run_command_with_timing(
                render_cmd, "MVS Novel View Rendering"
            )
            
            if render_ret != 0:
                self.results["mvs_benchmark"]["rendering"] = {
                    "success": False,
                    "time_seconds": render_time,
                    "error": render_stderr
                }
                return False
            
            # Step 3: Calculate metrics
            # Find ground truth test images based on data format
            if hasattr(self, 'data_format') and self.data_format == "colmap":
                # For COLMAP format, use Input directory
                gt_dir = self.data_path / "Input"
            else:
                # For NeRF format, try test directory first
                gt_dir = self.data_path / "test"
                if not gt_dir.exists():
                    # Try alternative locations
                    alt_gt_dirs = [
                        self.data_path / "images" / "test",
                        self.data_path / "images",
                    ]
                    for alt_dir in alt_gt_dirs:
                        if alt_dir.exists():
                            gt_dir = alt_dir
                            break
            
            metrics_output = mvs_output_dir / "metrics.json"
            metrics_cmd = [
                sys.executable, "mvs_benchmark/metrics_mvs.py",
                "-r", str(render_output_dir),
                "-g", str(gt_dir),
                "-o", str(metrics_output)
            ]
            
            metrics_time, _, _, _ = self.run_command_with_timing(
                metrics_cmd, "MVS Metrics Calculation"
            )
            
            # Parse MVS benchmark stats
            benchmark_stats = {}
            stats_file = mvs_output_dir / "benchmark_stats.json"
            if stats_file.exists():
                try:
                    with open(stats_file) as f:
                        benchmark_stats = json.load(f)
                except Exception as e:
                    print(f"Warning: Could not parse MVS benchmark stats: {e}")
            
            # Parse metrics
            mvs_metrics = {}
            if metrics_output.exists():
                try:
                    with open(metrics_output) as f:
                        mvs_metrics = json.load(f)
                except Exception as e:
                    print(f"Warning: Could not parse MVS metrics: {e}")
            
            # Store results
            total_time = mvs_time + render_time + metrics_time
            self.results["mvs_benchmark"] = {
                "success": True,
                "reconstruction_time_seconds": mvs_time,
                "rendering_time_seconds": render_time,
                "metrics_time_seconds": metrics_time,
                "total_time_seconds": total_time,
                "benchmark_stats": benchmark_stats,
                "metrics": mvs_metrics,
                "mesh_file": str(mesh_file)
            }
            
            print("✓ MVS benchmark pipeline completed successfully")
            return True
            
        except Exception as e:
            print(f"Error in MVS benchmark pipeline: {e}")
            self.results["mvs_benchmark"]["error"] = str(e)
            return False
    
    def generate_comparison_report(self):
        """Generate comparative analysis report"""
        print(f"\n📊 GENERATING COMPARISON REPORT")
        
        gs_success = self.results["gaussian_splatting"].get("success", False)
        mvs_success = self.results["mvs_benchmark"].get("success", False)
        
        if not gs_success or not mvs_success:
            print("Warning: One or both pipelines failed, comparison may be incomplete")
        
        # Extract timing information
        gs_time = self.results["gaussian_splatting"].get("total_time_seconds", 0)
        mvs_time = self.results["mvs_benchmark"].get("total_time_seconds", 0)
        
        # Extract memory information
        gs_memory = 0  # Would need to parse from GS logs
        mvs_memory = 0
        if "benchmark_stats" in self.results["mvs_benchmark"]:
            mvs_stats = self.results["mvs_benchmark"]["benchmark_stats"]
            if "summary" in mvs_stats:
                mvs_memory = mvs_stats["summary"].get("peak_memory_mb", 0)
        
        # Extract quality metrics
        gs_metrics = self.results["gaussian_splatting"].get("metrics", {})
        mvs_metrics = self.results["mvs_benchmark"].get("metrics", {})
        
        # Extract summary metrics
        gs_psnr = self._extract_metric(gs_metrics, "PSNR")
        gs_ssim = self._extract_metric(gs_metrics, "SSIM") 
        gs_lpips = self._extract_metric(gs_metrics, "LPIPS")
        
        mvs_psnr = mvs_metrics.get("summary", {}).get("PSNR", 0)
        mvs_ssim = mvs_metrics.get("summary", {}).get("SSIM", 0)
        mvs_lpips = mvs_metrics.get("summary", {}).get("LPIPS", 0)
        
        # Create comparison
        self.results["comparison"] = {
            "timing": {
                "gaussian_splatting_seconds": gs_time,
                "mvs_benchmark_seconds": mvs_time,
                "gs_formatted": self._format_time(gs_time),
                "mvs_formatted": self._format_time(mvs_time),
                "winner": "Gaussian Splatting" if gs_time < mvs_time else "MVS"
            },
            "memory": {
                "gaussian_splatting_mb": gs_memory,
                "mvs_benchmark_mb": mvs_memory,
                "winner": "Gaussian Splatting" if gs_memory < mvs_memory else "MVS" if mvs_memory > 0 else "Unknown"
            },
            "quality": {
                "PSNR": {
                    "gaussian_splatting": gs_psnr,
                    "mvs_benchmark": mvs_psnr,
                    "winner": "Gaussian Splatting" if gs_psnr > mvs_psnr else "MVS"
                },
                "SSIM": {
                    "gaussian_splatting": gs_ssim,
                    "mvs_benchmark": mvs_ssim,
                    "winner": "Gaussian Splatting" if gs_ssim > mvs_ssim else "MVS"
                },
                "LPIPS": {
                    "gaussian_splatting": gs_lpips,
                    "mvs_benchmark": mvs_lpips,
                    "winner": "Gaussian Splatting" if gs_lpips < mvs_lpips else "MVS"
                }
            }
        }
        
        # Print comparison table
        self.print_comparison_table()
        
        # Save results
        results_file = self.output_dir / "comparison_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nDetailed results saved to: {results_file}")
    
    def _extract_metric(self, metrics_dict: dict, metric_name: str) -> float:
        """Extract metric value from potentially nested dictionary"""
        if not metrics_dict:
            return 0.0
        
        # Try direct access
        if metric_name in metrics_dict:
            return float(metrics_dict[metric_name])
        
        # Try nested access (common in GS results)
        for _, value in metrics_dict.items():
            if isinstance(value, dict) and metric_name in value:
                return float(value[metric_name])
        
        return 0.0
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def print_comparison_table(self):
        """Print formatted comparison table"""
        comp = self.results["comparison"]
        
        print(f"\n{'='*80}")
        print("                         COMPARATIVE ANALYSIS RESULTS")
        print(f"{'='*80}")
        print(f"{'Metric':<15} | {'Gaussian Splatting':<20} | {'MVS Benchmark':<20} | {'Winner':<12}")
        print(f"{'-'*80}")
        
        # Computational Cost
        print(f"{'[COST]':<15} |")
        print(f"{'Total Time':<15} | {comp['timing']['gs_formatted']:<20} | {comp['timing']['mvs_formatted']:<20} | {comp['timing']['winner']:<12}")
        
        if comp['memory']['mvs_benchmark_mb'] > 0:
            print(f"{'Peak Memory':<15} | {comp['memory']['gaussian_splatting_mb']:.1f} MB{'':<9} | {comp['memory']['mvs_benchmark_mb']:.1f} MB{'':<9} | {comp['memory']['winner']:<12}")
        
        print(f"{'-'*80}")
        
        # Quality Metrics
        print(f"{'[QUALITY]':<15} |")
        print(f"{'PSNR ↑':<15} | {comp['quality']['PSNR']['gaussian_splatting']:.2f} dB{'':<12} | {comp['quality']['PSNR']['mvs_benchmark']:.2f} dB{'':<12} | {comp['quality']['PSNR']['winner']:<12}")
        print(f"{'SSIM ↑':<15} | {comp['quality']['SSIM']['gaussian_splatting']:.4f}{'':<16} | {comp['quality']['SSIM']['mvs_benchmark']:.4f}{'':<16} | {comp['quality']['SSIM']['winner']:<12}")
        print(f"{'LPIPS ↓':<15} | {comp['quality']['LPIPS']['gaussian_splatting']:.4f}{'':<16} | {comp['quality']['LPIPS']['mvs_benchmark']:.4f}{'':<16} | {comp['quality']['LPIPS']['winner']:<12}")
        
        print(f"{'='*80}")
        
        # Summary
        quality_wins = [comp['quality']['PSNR']['winner'], comp['quality']['SSIM']['winner'], comp['quality']['LPIPS']['winner']]
        gs_quality_wins = quality_wins.count('Gaussian Splatting')
        mvs_quality_wins = quality_wins.count('MVS')
        
        print(f"\nSUMMARY:")
        print(f"  Speed Winner: {comp['timing']['winner']}")
        if comp['memory']['winner'] != 'Unknown':
            print(f"  Memory Winner: {comp['memory']['winner']}")
        print(f"  Quality Winner: {'Gaussian Splatting' if gs_quality_wins > mvs_quality_wins else 'MVS' if mvs_quality_wins > gs_quality_wins else 'Tie'} ({gs_quality_wins}/3 metrics)")
    
    def run_full_comparison(self):
        """Run the complete comparison benchmark"""
        print(f"🎯 STARTING COMPREHENSIVE COMPARISON")
        print(f"Data Path: {self.data_path}")
        print(f"Output Directory: {self.output_dir}")
        
        total_start_time = time.time()
        
        # Run Gaussian Splatting pipeline
        gs_success = self.run_gaussian_splatting_pipeline()
        
        # Run MVS benchmark pipeline
        mvs_success = self.run_mvs_benchmark_pipeline()
        
        # Generate comparison report
        self.generate_comparison_report()
        
        total_time = time.time() - total_start_time
        
        print(f"\n🏁 COMPARISON COMPLETE")
        print(f"Total benchmark time: {self._format_time(total_time)}")
        
        if not gs_success or not mvs_success:
            print("⚠️  Warning: One or both pipelines failed. See detailed logs above.")
            return False
        
        return True

def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive comparison between Gaussian Splatting and MVS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python run_comparison.py -s ./data_input/nerf_blender_qiita
  python run_comparison.py -s ./data_input/nerf_blender_qiita -o ./comparison_results

Required data structure:
  <data_path>/
  ├── colmap/               # COLMAP sparse reconstruction output
  ├── images/               # Source images
  ├── transforms_test.json  # Test camera poses (NeRF format)
  └── test/                 # Ground truth test images (optional)
        """
    )
    
    parser.add_argument("-s", "--source_path", type=str, required=True,
                       help="Path to input data directory")
    parser.add_argument("-o", "--output", type=str, default=None,
                       help="Output directory for comparison results (default: <source_path>/comparison_output)")
    parser.add_argument("--gaussian-splatting-dir", type=str, default=None,
                       help="Path to gaussian-splatting directory (auto-detected if not specified)")
    
    args = parser.parse_args()
    
    # Validate input
    if not Path(args.source_path).exists():
        print(f"Error: Source path does not exist: {args.source_path}")
        return 1
    
    try:
        # Run comparison
        benchmark = ComparisonBenchmark(args.source_path, args.output, args.gaussian_splatting_dir)
        success = benchmark.run_full_comparison()
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"Comparison failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())