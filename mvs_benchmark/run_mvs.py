#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import sys
import time
import psutil
from pathlib import Path
from typing import Dict, Any, Tuple
import shutil

class PerformanceMonitor:
    """Monitors CPU, memory, and execution time for subprocess calls"""
    
    def __init__(self):
        self.process = None
        self.peak_memory_mb = 0
        self.start_time = 0
        self.end_time = 0
    
    def start_monitoring(self, process: subprocess.Popen):
        """Start monitoring a subprocess"""
        self.process = process
        self.peak_memory_mb = 0
        self.start_time = time.time()
        
    def update_memory_usage(self):
        """Update peak memory usage if process is running"""
        if self.process and self.process.poll() is None:
            try:
                # Get memory usage of main process and all children
                main_process = psutil.Process(self.process.pid)
                memory_mb = main_process.memory_info().rss / (1024 * 1024)
                
                # Include child processes
                for child in main_process.children(recursive=True):
                    try:
                        memory_mb += child.memory_info().rss / (1024 * 1024)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                        
                self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    
    def stop_monitoring(self) -> Tuple[float, float]:
        """Stop monitoring and return (elapsed_time, peak_memory_mb)"""
        self.end_time = time.time()
        elapsed_time = self.end_time - self.start_time
        return elapsed_time, self.peak_memory_mb

class MVSBenchmark:
    """OpenMVS-based benchmark pipeline for comparing with Gaussian Splatting"""
    
    def __init__(self, input_path: str, output_dir: str):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Detect and validate input data format
        self.detect_data_format()
        
        self.benchmark_stats = {
            "input_path": str(self.input_path),
            "output_path": str(self.output_dir),
            "data_format": self.data_format,
            "stages": {}
        }
    
    def detect_data_format(self):
        """Detect the input data format (NeRF-style or COLMAP-style)"""
        # Check for NeRF-style structure (colmap/ + images/)
        nerf_colmap_dir = self.input_path / "colmap"
        nerf_images_dir = self.input_path / "images"
        
        # Check for COLMAP-style structure (sparse/ + Input/)
        colmap_sparse_dir = self.input_path / "sparse"
        colmap_input_dir = self.input_path / "Input"
        colmap_database = self.input_path / "database.db"
        
        if nerf_colmap_dir.exists() and nerf_images_dir.exists():
            # NeRF-style format
            self.data_format = "nerf"
            self.colmap_dir = nerf_colmap_dir
            self.images_dir = nerf_images_dir
            print(f"Detected NeRF-style data format")
            
        elif colmap_sparse_dir.exists() and colmap_input_dir.exists():
            # COLMAP-style format
            self.data_format = "colmap"
            # For COLMAP format, we need to use sparse/0 as the reconstruction
            self.colmap_dir = colmap_sparse_dir / "0"
            self.images_dir = colmap_input_dir
            
            # Validate COLMAP sparse reconstruction
            required_files = ["cameras.bin", "images.bin", "points3D.bin"]
            missing_files = [f for f in required_files if not (self.colmap_dir / f).exists()]
            
            if missing_files:
                raise FileNotFoundError(f"Missing COLMAP sparse reconstruction files: {missing_files}")
            
            print(f"Detected COLMAP-style data format")
            
        else:
            raise FileNotFoundError(
                f"Invalid input data structure. Expected either:\n"
                f"  NeRF-style: {self.input_path}/colmap/ + {self.input_path}/images/\n"
                f"  COLMAP-style: {self.input_path}/sparse/ + {self.input_path}/Input/"
            )
        
        # Validate that directories exist and are accessible
        if not self.colmap_dir.exists():
            raise FileNotFoundError(f"COLMAP directory not found: {self.colmap_dir}")
        
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
    
    def check_openmvs_installation(self):
        """Check if OpenMVS tools are available"""
        required_tools = ["DensifyPointCloud", "ReconstructMesh", "TextureMesh"]
        missing_tools = []
        
        for tool in required_tools:
            if not shutil.which(tool):
                missing_tools.append(tool)
        
        if missing_tools:
            print(f"Error: Missing OpenMVS tools: {missing_tools}")
            print("Please install OpenMVS:")
            print("  Ubuntu/Debian: sudo apt-get install openmvs")
            print("  Or build from source: https://github.com/cdcseacave/openMVS")
            return False
        
        print("✓ All required OpenMVS tools found")
        return True
    
    def run_subprocess_with_monitoring(self, cmd: list, stage_name: str, 
                                     cwd: str = None) -> Tuple[float, float]:
        """Run subprocess with performance monitoring"""
        print(f"Starting {stage_name}...")
        print(f"Command: {' '.join(cmd)}")
        
        monitor = PerformanceMonitor()
        
        try:
            # Start the process
            process = subprocess.Popen(
                cmd, 
                cwd=cwd,
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            
            monitor.start_monitoring(process)
            
            # Monitor memory usage while process is running
            while process.poll() is None:
                monitor.update_memory_usage()
                time.sleep(0.5)  # Check every 0.5 seconds
            
            # Get final results
            stdout, stderr = process.communicate()
            elapsed_time, peak_memory = monitor.stop_monitoring()
            
            # Check if process succeeded
            if process.returncode != 0:
                print(f"Error in {stage_name}:")
                print(f"Return code: {process.returncode}")
                print(f"STDERR: {stderr}")
                raise subprocess.CalledProcessError(process.returncode, cmd, stderr)
            
            print(f"✓ {stage_name} completed in {elapsed_time:.2f}s (Peak memory: {peak_memory:.1f} MB)")
            
            return elapsed_time, peak_memory
            
        except Exception as e:
            print(f"Failed to run {stage_name}: {e}")
            raise
    
    def convert_colmap_to_openmvs(self) -> str:
        """Convert COLMAP output to OpenMVS format"""
        print("Converting COLMAP output to OpenMVS format...")
        
        # Output file for OpenMVS scene
        scene_mvs = self.output_dir / "scene.mvs"
        
        cmd = [
            "InterfaceCOLMAP",
            "-i", str(self.colmap_dir),
            "-o", str(scene_mvs),
            "--image-folder", str(self.images_dir)
        ]
        
        elapsed_time, peak_memory = self.run_subprocess_with_monitoring(
            cmd, "COLMAP to OpenMVS conversion"
        )
        
        self.benchmark_stats["stages"]["conversion"] = {
            "elapsed_time_seconds": elapsed_time,
            "peak_memory_mb": peak_memory,
            "command": " ".join(cmd)
        }
        
        return str(scene_mvs)
    
    def densify_point_cloud(self, scene_mvs: str) -> str:
        """Run DensifyPointCloud to create dense point cloud"""
        dense_mvs = self.output_dir / "scene_dense.mvs"
        
        cmd = [
            "DensifyPointCloud",
            scene_mvs,
            "-o", str(dense_mvs),
            "--resolution-level", "1",  # Higher resolution for better quality
            "--number-views", "4"       # Consider 4 neighboring views
        ]
        
        elapsed_time, peak_memory = self.run_subprocess_with_monitoring(
            cmd, "Dense Point Cloud Generation"
        )
        
        self.benchmark_stats["stages"]["densify"] = {
            "elapsed_time_seconds": elapsed_time,
            "peak_memory_mb": peak_memory,
            "command": " ".join(cmd)
        }
        
        return str(dense_mvs)
    
    def reconstruct_mesh(self, dense_mvs: str) -> str:
        """Run ReconstructMesh to create mesh from dense point cloud"""
        mesh_mvs = self.output_dir / "scene_mesh.mvs"
        
        cmd = [
            "ReconstructMesh",
            dense_mvs,
            "-o", str(mesh_mvs),
            "--smooth", "2",           # Smoothing iterations
            "--thickness-factor", "1", # Control mesh thickness
        ]
        
        elapsed_time, peak_memory = self.run_subprocess_with_monitoring(
            cmd, "Mesh Reconstruction"
        )
        
        self.benchmark_stats["stages"]["mesh"] = {
            "elapsed_time_seconds": elapsed_time,
            "peak_memory_mb": peak_memory,
            "command": " ".join(cmd)
        }
        
        return str(mesh_mvs)
    
    def texture_mesh(self, mesh_mvs: str) -> str:
        """Run TextureMesh to add texture to the mesh"""
        textured_mvs = self.output_dir / "scene_textured.mvs"
        
        cmd = [
            "TextureMesh",
            mesh_mvs,
            "-o", str(textured_mvs),
            "--resolution-level", "0",  # Full resolution textures
            "--outlier-threshold", "6e-2"
        ]
        
        elapsed_time, peak_memory = self.run_subprocess_with_monitoring(
            cmd, "Mesh Texturing"
        )
        
        self.benchmark_stats["stages"]["texture"] = {
            "elapsed_time_seconds": elapsed_time,
            "peak_memory_mb": peak_memory,
            "command": " ".join(cmd)
        }
        
        return str(textured_mvs)
    
    def save_benchmark_stats(self):
        """Save benchmark statistics to JSON file"""
        # Calculate total time and peak memory
        total_time = sum(stage["elapsed_time_seconds"] for stage in self.benchmark_stats["stages"].values())
        peak_memory = max(stage["peak_memory_mb"] for stage in self.benchmark_stats["stages"].values())
        
        self.benchmark_stats["summary"] = {
            "total_time_seconds": total_time,
            "total_time_formatted": f"{total_time//3600:02.0f}:{(total_time%3600)//60:02.0f}:{total_time%60:02.0f}",
            "peak_memory_mb": peak_memory,
            "peak_memory_gb": peak_memory / 1024
        }
        
        stats_file = self.output_dir / "benchmark_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.benchmark_stats, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"MVS Benchmark Complete!")
        print(f"Total Time: {self.benchmark_stats['summary']['total_time_formatted']}")
        print(f"Peak Memory: {peak_memory:.1f} MB ({peak_memory/1024:.1f} GB)")
        print(f"Results saved to: {stats_file}")
        print(f"{'='*60}\n")
    
    def run_full_pipeline(self):
        """Execute the complete MVS pipeline"""
        print("Starting OpenMVS benchmark pipeline...")
        
        # Check OpenMVS installation
        if not self.check_openmvs_installation():
            sys.exit(1)
        
        try:
            # Step 1: Convert COLMAP to OpenMVS format
            scene_mvs = self.convert_colmap_to_openmvs()
            
            # Step 2: Densify point cloud
            dense_mvs = self.densify_point_cloud(scene_mvs)
            
            # Step 3: Reconstruct mesh
            mesh_mvs = self.reconstruct_mesh(dense_mvs)
            
            # Step 4: Texture mesh
            textured_mvs = self.texture_mesh(mesh_mvs)
            
            # Save benchmark results
            self.save_benchmark_stats()
            
            print(f"Final textured mesh saved to: {textured_mvs}")
            
        except Exception as e:
            print(f"Pipeline failed: {e}")
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run OpenMVS benchmark pipeline")
    parser.add_argument("-s", "--source_path", type=str, required=True,
                       help="Path to input data directory (should contain 'colmap' and 'images' subdirs)")
    parser.add_argument("-o", "--output", type=str, default=None,
                       help="Output directory (default: <source_path>/mvs_output)")
    
    args = parser.parse_args()
    
    # Set default output directory
    if args.output is None:
        args.output = os.path.join(args.source_path, "mvs_output")
    
    # Validate input path
    if not os.path.exists(args.source_path):
        print(f"Error: Source path does not exist: {args.source_path}")
        sys.exit(1)
    
    # Run the benchmark
    try:
        benchmark = MVSBenchmark(args.source_path, args.output)
        benchmark.run_full_pipeline()
    except Exception as e:
        print(f"Benchmark failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()