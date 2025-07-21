# Claude Code Settings

This file contains configuration and instructions for Claude Code to better understand and work with this project.

## Project Overview

**Ryugu-GS** is a specialized 3D reconstruction research project that applies Gaussian Splatting technology to create highly accurate 3D models of asteroid **162173 Ryugu** (visited by Japan's Hayabusa2 mission). This project combines advanced computer vision, astronomical data processing, and neural rendering techniques for scientific analysis of planetary surfaces.

### Key Features:
- **Multi-model merging**: Combines multiple 3D reconstruction models (BOX-A, BOX-B, BOX-C) using advanced registration techniques
- **Astronomical accuracy**: Incorporates real viewing condition data (Sun/Earth positions, phase angles) for precise lighting simulation
- **Structure from Motion (SfM)**: Advanced camera pose estimation and 3D point cloud alignment using ICP registration
- **Neural rendering**: Based on the official Gaussian Splatting implementation with custom modifications
- **Scientific workflow**: Extensive logging, visualization, and Slack notifications for long-running training processes

## Technology Stack

- **Base Framework**: PyTorch + CUDA (11.8)
- **3D Processing**: Open3D, scipy optimization, quaternion mathematics
- **Computer Vision**: OpenCV, COLMAP integration, detectron2
- **Data Formats**: NeRF-compatible transforms, PLY point clouds, FITS astronomical data
- **Environment**: Python 3.8, Conda-managed dependencies, Docker deployment

## Development Commands

```bash
# Environment setup
conda env create -f environment.yml
conda activate myenv

# Install Gaussian Splatting submodules
pip install -e ./gaussian-splatting/submodules/diff-gaussian-rasterization
pip install -e ./gaussian-splatting/submodules/simple-knn

# Install project dependencies
pip install -r requirements.txt

# Install OpenMVS for benchmarking (Ubuntu/Debian)
sudo apt-get install openmvs

# Main training (Gaussian Splatting)
cd gaussian-splatting
python train.py -s <path_to_data> --config <config_file>

# Model merging workflow
python -m sfm.merge --config config.yaml

# Rendering and evaluation
python render.py -m <model_path>
python metrics.py -m <model_path>

# Comparative benchmarking (supports both NeRF and COLMAP formats)
python run_comparison.py -s <data_path>

# Individual MVS benchmark components
python mvs_benchmark/run_mvs.py -s <data_path>
python mvs_benchmark/render_mvs.py -m <mesh_path> -t <transforms_file>
python mvs_benchmark/metrics_mvs.py -r <rendered_dir> -g <gt_dir>

# Docker deployment
docker build -t ryugu-gs .
docker run --gpus all -it ryugu-gs
```

## Project Structure

```
Ryugu-GS/
├── CLAUDE.md                           # This configuration file
├── config.example.yaml                 # Configuration template
├── RyuguViewingConditionRoughEstimate.csv # Astronomical viewing data
├── gaussian-splatting/                 # Core Gaussian Splatting implementation
│   ├── train.py                        # Main training script
│   ├── render.py                       # Rendering pipeline
│   ├── scene/                          # Scene representation and cameras
│   └── utils/                          # Utilities for graphics and training
├── sfm/                                # Structure from Motion and model merging
│   ├── model.py                        # 3D model representation
│   ├── model_merger.py                 # Multi-model registration and merging
│   └── merge.py                        # Main merging workflow
├── data_input/                         # Input datasets
│   └── nerf_blender_qiita/             # Example NeRF-format dataset
├── utils/                              # Data processing utilities
├── logger/                             # Custom logging system
├── notice/                             # Slack notification system
├── log/ and plot/                      # Training logs and visualizations
└── Dockerfile                          # Container deployment
```

## Workflow Understanding

1. **Data Preparation**: Convert astronomical observations to NeRF-compatible format
2. **Individual Training**: Train separate Gaussian Splatting models for different viewing conditions
3. **Model Registration**: Use ICP and optimization to align multiple models in 3D space
4. **Model Merging**: Combine registered models for comprehensive asteroid representation
5. **Evaluation**: Generate renders and compute metrics for scientific validation

## Development Notes for Claude

### Code Style & Conventions:
- **Language**: Python with Japanese comments (astronomical/space domain)
- **Configuration**: YAML-based with OmegaConf for hierarchical configs
- **Logging**: Custom Logger class with both file and console output
- **Notifications**: Slack integration for long-running scientific computations
- **3D Math**: Heavy use of quaternions, rotation matrices, and optimization

### Scientific Context:
- This is **planetary science research** - treat with appropriate scientific rigor
- Astronomical coordinate systems and viewing geometry are critical
- Multiple coordinate frame transformations (camera, world, astronomical)
- Model merging requires careful handling of scale, rotation, and translation

### Important Files to Understand:
- `sfm/model_merger.py:19` - Core model registration and merging logic
- `gaussian-splatting/train.py` - Main neural training pipeline  
- `logger/__init__.py:3` - Custom logging for scientific workflows
- `config.example.yaml` - Configuration parameters for different datasets
- `run_comparison.py` - Complete benchmarking orchestration system
- `mvs_benchmark/run_mvs.py` - OpenMVS integration with performance monitoring

### Never Modify:
- Astronomical data in CSV files (scientific accuracy required)
- Core Gaussian Splatting implementation (maintain compatibility)
- Docker CUDA configuration (GPU computing requirements)

### Data Formats Supported:
- **NeRF-style**: `colmap/` + `images/` + `transforms_test.json`
- **COLMAP-style**: `sparse/0/` + `Input/` + `database.db` (optional)
- Automatic format detection and processing pipeline adaptation

### Testing:
- No traditional unit tests - validation through scientific metrics
- Use `metrics.py` for quantitative evaluation
- Visual inspection of renders in `plot/` directory
- Model merging validation through camera pose visualization
- Comparative benchmarking via `run_comparison.py` for method evaluation