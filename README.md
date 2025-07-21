# Ryugu-GS

A specialized 3D reconstruction research project applying Gaussian Splatting technology to create highly accurate 3D models of asteroid **162173 Ryugu** (visited by Japan's Hayabusa2 mission). This project combines advanced computer vision, astronomical data processing, and neural rendering techniques for scientific analysis of planetary surfaces.

## Features

- **Neural 3D Rendering**: Based on the state-of-the-art Gaussian Splatting implementation
- **Multi-model Merging**: Combines multiple 3D reconstruction models using advanced registration techniques  
- **Astronomical Accuracy**: Incorporates real viewing condition data for precise lighting simulation
- **Comparative Benchmarking**: Comprehensive comparison framework between Gaussian Splatting and traditional MVS methods
- **Scientific Workflow**: Extensive logging, visualization, and notification systems for research workflows

## Quick Start

### Environment Setup

```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate myenv

# Install Gaussian Splatting submodules
pip install -e ./gaussian-splatting/submodules/diff-gaussian-rasterization
pip install -e ./gaussian-splatting/submodules/simple-knn

# Install additional dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Train Gaussian Splatting model
cd gaussian-splatting
python train.py -s <path_to_data>

# Render novel views
python render.py -m <model_path>

# Calculate quality metrics
python metrics.py -m <model_path>
```

### Comparative Benchmarking

This repository includes a comprehensive benchmarking system that compares Gaussian Splatting against traditional Multi-View Stereo (MVS) methods:

```bash
# Run complete comparison benchmark
python run_comparison.py -s <data_path>

# NeRF-style data example
python run_comparison.py -s ./data_input/nerf_blender_qiita -o ./benchmark_results

# COLMAP-style data example  
python run_comparison.py -s ./data_input/colmap_asteroid -o ./benchmark_results
```

#### Prerequisites for MVS Benchmark

Install OpenMVS for the traditional reconstruction pipeline:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install openmvs

# Or build from source
git clone https://github.com/cdcseacave/openMVS.git --recursive
# Follow build instructions in OpenMVS repository
```

#### Required Data Structure

The benchmarking system supports two data formats:

**NeRF-style Format:**
```
<data_path>/
├── colmap/               # COLMAP sparse reconstruction output
├── images/               # Source images  
├── transforms_test.json  # Test camera poses (NeRF format)
└── test/                 # Ground truth test images (optional)
```

**COLMAP-style Format:**
```
<data_path>/
├── sparse/0/             # COLMAP sparse reconstruction
│   ├── cameras.bin       # Camera parameters
│   ├── images.bin        # Image metadata and poses
│   ├── points3D.bin      # 3D point cloud
│   └── points3D.ply      # 3D point cloud (optional)
├── Input/                # Source images
└── database.db           # COLMAP database (optional)
```

The system automatically detects the data format and adapts the processing pipeline accordingly.

#### Benchmark Output

The comparison generates detailed performance metrics:

```
============================================================
           | Conventional MVS      | Gaussian Splatting
------------------------------------------------------------
[Cost]
Total Time   | 02:35:10 (HH:MM:SS) | 00:45:20 (HH:MM:SS)
Peak VRAM    | 18.5 GB             | 14.2 GB
------------------------------------------------------------
[Realism]
PSNR ↑       | 21.5 dB             | 24.8 dB
SSIM ↑       | 0.85                | 0.92  
LPIPS ↓      | 0.21                | 0.11
============================================================
```

## Repository Structure

```
Ryugu-GS/
├── gaussian-splatting/          # Core Gaussian Splatting implementation
├── mvs_benchmark/               # MVS comparison pipeline
│   ├── run_mvs.py              # OpenMVS integration with performance monitoring
│   ├── render_mvs.py           # Novel view rendering from meshes
│   └── metrics_mvs.py          # Quality evaluation metrics
├── sfm/                        # Structure from Motion and model merging
├── utils/                      # Data processing utilities
├── data_input/                 # Input datasets
├── run_comparison.py           # Main benchmarking orchestration script
└── CLAUDE.md                   # Development configuration
```

## Scientific Context

This project focuses on **planetary science research** with applications to:

- Asteroid surface analysis and mapping
- Multi-viewpoint 3D reconstruction from spacecraft imagery
- Comparison of classical photogrammetry vs. neural rendering approaches
- Scientific validation through quantitative metrics (PSNR, SSIM, LPIPS)

### Key Technologies

- **Base Framework**: PyTorch + CUDA (11.8)
- **3D Processing**: Open3D, scipy optimization, quaternion mathematics
- **Computer Vision**: OpenCV, COLMAP integration, detectron2
- **Rendering**: PyTorch3D, custom neural splatting
- **Data Formats**: NeRF-compatible transforms, PLY point clouds, FITS astronomical data

## Original Implementation

This repository utilizes the Gaussian Splatting implementation from:

- **Original Repository**: [Gaussian Splatting GitHub](https://github.com/graphdeco-inria/gaussian-splatting)
- **License**: The Gaussian Splatting code is distributed under the terms specified in its `LICENSE` file, located in the `gaussian-splatting/` directory.

## Custom Modifications

1. **Slack Integration**: Notification system for long-running training processes
2. **Configuration Management**: YAML-based hierarchical configuration system
3. **Astronomical Data Processing**: Scripts for handling spacecraft observation data
4. **Model Merging Pipeline**: Advanced registration and alignment of multiple 3D models
5. **Comparative Benchmarking**: Comprehensive evaluation framework against traditional methods
6. **Scientific Logging**: Detailed performance and quality metrics tracking

## Dependencies

### Core Requirements
- Python 3.8+
- PyTorch with CUDA support
- OpenMVS (for benchmarking)

### Optional Dependencies
- PyTorch3D (preferred for mesh rendering)
- Open3D (fallback renderer)
- Detectron2 (for advanced image processing)

See `requirements.txt` and `environment.yml` for complete dependency lists.

## License

This repository contains two distinct parts:

1. **Gaussian Splatting Code**: Licensed under the terms specified by Inria and Max Planck Institut for Informatik, located in the `gaussian-splatting/` directory.
2. **Custom Additions**: Modifications and additional scripts provided in this repository are distributed under the MIT License.

For details about the Gaussian Splatting license, please refer to the `LICENSE` file in the `gaussian-splatting/` directory.

## Citation

If you use this work in your research, please cite both the original Gaussian Splatting paper and acknowledge this asteroid reconstruction application:

```bibtex
@article{kerbl20233d,
  title={3d gaussian splatting for real-time radiance field rendering},
  author={Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  journal={ACM Transactions on Graphics},
  volume={42},
  number={4},
  pages={1--14},
  year={2023},
  publisher={ACM New York, NY, USA}
}
```
