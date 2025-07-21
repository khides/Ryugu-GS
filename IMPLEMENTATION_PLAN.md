# Implementation Plan: Comparative Analysis of Conventional MVS and Gaussian Splatting

## 1\. High-Level Goal

The primary objective is to **empirically demonstrate the superiority of Gaussian Splatting (GS) over a conventional 3D reconstruction method** for asteroid modeling. This will be achieved by implementing a classical **Multi-View Stereo (MVS)** pipeline within the existing `Ryugu-GS` repository.

This new MVS pipeline will serve as a **quantitative benchmark** against the current GS implementation, allowing us to compare them on two key axes:

1.  **Computational Cost**: Execution time and peak memory usage.
2.  **Photorealism**: Quality of rendered novel views, measured by standard image metrics (PSNR, SSIM, LPIPS).

## 2\. Background and Rationale

The `Ryugu-GS` repository is currently capable of producing a high-fidelity 3D model using Gaussian Splatting from pre-processed COLMAP data. However, it lacks an integrated "conventional" pipeline to compare against. Without this benchmark, we can only evaluate the absolute quality of the GS model, but we cannot make the core research claim that GS is *better* than the methods it aims to replace.

### The "Conventional Method" Pipeline

The standard photogrammetry pipeline, which we will implement, consists of four main stages:

1.  **Structure-from-Motion (SfM)**: Estimates camera poses and a sparse point cloud.
      - **Your Action**: You have already performed this step outside the repository using COLMAP. **We will use this exact same SfM output as the starting point for both pipelines to ensure a fair comparison.**
2.  **Multi-View Stereo (MVS)**: Creates a dense point cloud from the sparse cloud and images.
      - **Claude's Task**: Implement this stage.
3.  **Meshing**: Converts the dense point cloud into a continuous surface mesh (e.g., `.obj` file).
      - **Claude's Task**: Implement this stage.
4.  **Texturing**: Projects the original images onto the mesh to create a photorealistic, textured model.
      - **Claude's Task**: Implement this stage.

By implementing steps 2, 3, and 4, we can generate a final, textured mesh model that represents the output of a complete conventional pipeline.

### The Comparison Framework

The goal is to create a parallel workflow:

```mermaid
graph TD
    A[Input Data: Images + COLMAP SfM Output] --> B{Gaussian Splatting Pipeline};
    A --> C{Conventional MVS Pipeline (To Be Implemented)};

    B --> D[GS Model (.ply)];
    C --> E[Textured Mesh (.obj)];

    subgraph "Photorealism Comparison"
        F[Render Novel Views from Test Cameras]
        G[Render Novel Views from Test Cameras]
        H[Ground-Truth Test Images]

        D --> F;
        E --> G;

        F --> I{Calculate Metrics (PSNR, SSIM, LPIPS)};
        G --> I;
        H --> I;
    end

    subgraph "Computational Cost Comparison"
        B --> J{Log Time & Memory};
        C --> J;
    end

    I --> K[Final Result: Photorealism Score];
    J --> L[Final Result: Cost Score];

    K & L --> M[Comparative Analysis Report];

```

## 3\. Detailed Implementation Plan

To maintain code organization and avoid disrupting the existing GS workflow, all new code for the conventional benchmark should be placed in a new top-level directory: `mvs_benchmark/`.

### Step 1: Integrate an MVS Library

Instead of writing an MVS algorithm from scratch, we will integrate a robust, industry-standard open-source library. **OpenMVS** is the ideal choice as it is designed to work directly with COLMAP output.

**Task**:

  - Create a new directory `mvs_benchmark/`.
  - Provide instructions in the `README.md` for installing the OpenMVS binaries (e.g., via `apt-get install openmvs` or building from source).
  - The core of the MVS pipeline will involve calling the OpenMVS command-line tools in sequence.

### Step 2: Create the MVS Workflow Script

This script will orchestrate the execution of the OpenMVS tools and measure performance.

**Task**: Create a new file `mvs_benchmark/run_mvs.py`.

This script should:

1.  Accept the same `-s <path_to_data>` command-line argument as `train.py`. The input data directory is expected to contain the `colmap/` subdirectory from the external COLMAP run.
2.  Use Python's `subprocess` module to execute the following OpenMVS commands in order:
    a.  `DensifyPointCloud`: Takes the COLMAP sparse reconstruction and images to produce a dense point cloud.
    b.  `ReconstructMesh`: Converts the dense point cloud into a watertight mesh.
    c.  `TextureMesh`: Projects the original images onto the mesh to generate a final textured `.obj` model.
3.  **Crucially, wrap each `subprocess` call with timing and memory profiling logic**:
      - Record the wall-clock time for each of the three stages.
      - (Advanced) Monitor and log the peak memory usage during each stage.
4.  Log all performance metrics to a file (e.g., `mvs_benchmark/output/benchmark_stats.json`).
5.  The final output (e.g., `scene_textured.obj`, `scene_textured.mtl`, and texture files) should be saved in a predictable output directory.

### Step 3: Implement Rendering for the MVS Mesh

To compare photorealism, we need to render novel views from the generated mesh, using the same test camera poses as the GS evaluation.

**Task**: Create a new file `mvs_benchmark/render_mvs.py`.

This script should:

1.  Load the textured mesh (`.obj`) generated by `run_mvs.py`.
2.  Load the camera poses from the `transforms_test.json` file (the same one used by the GS pipeline).
3.  Use a library like **PyTorch3D** or **Open3D's renderer** to render an image from each test camera pose.
      - PyTorch3D is recommended as it aligns with the project's existing PyTorch dependency.
4.  Save the rendered images to an output directory (e.g., `mvs_benchmark/output/rendered_images/`).

### Step 4: Adapt Metrics Calculation

We need to calculate PSNR, SSIM, and LPIPS for the MVS-rendered images against the ground-truth test images.

**Task**: Create a new file `mvs_benchmark/metrics_mvs.py`.

This script should:

1.  Be based on the existing `metrics.py` script.
2.  Take the directory of MVS-rendered images and the ground-truth test images as input.
3.  Calculate the average PSNR, SSIM, and LPIPS across all test images.
4.  Print the results and save them to a file (e.g., `mvs_benchmark/output/metrics.json`).

## 4\. Final Orchestration Script

To make the entire comparison process seamless, create a single entry-point script that runs both pipelines and generates a final report.

**Task**: Create a new top-level script `run_comparison.py`.

This script should:

1.  Accept a single data source argument `-s <path_to_data>`.
2.  **Execute the Gaussian Splatting Pipeline**:
      - Call `gaussian-splatting/train.py`.
      - Call `render.py`.
      - Call `metrics.py`.
      - Parse the resulting logs and metric files to extract GS training time, memory usage, and quality scores.
3.  **Execute the MVS Benchmark Pipeline**:
      - Call `mvs_benchmark/run_mvs.py`.
      - Call `mvs_benchmark/render_mvs.py`.
      - Call `mvs_benchmark/metrics_mvs.py`.
      - Parse the resulting logs and metric files to extract MVS processing time, memory usage, and quality scores.
4.  **Generate a Comparative Report**:
      - Print a clean, formatted table to the console that summarizes the results side-by-side.

**Example Final Output Table:**

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

## 5\. Summary of Deliverables for Claude

1.  A new directory `mvs_benchmark/` containing:
      - `run_mvs.py`: The main workflow script to execute OpenMVS and profile performance.
      - `render_mvs.py`: A script to render novel views from the generated textured mesh.
      - `metrics_mvs.py`: A script to calculate quality scores for the MVS renderings.
2.  A new top-level script `run_comparison.py` to orchestrate both pipelines and produce a final comparative report.
3.  Updates to the project's main `README.md` to document the new benchmark functionality and the required OpenMVS dependency.