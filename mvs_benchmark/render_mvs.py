#!/usr/bin/env python3

import argparse
import json
import numpy as np
import torch
import trimesh
from pathlib import Path
from typing import Dict, List, Tuple
import cv2

# PyTorch3D imports
try:
    import pytorch3d
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (
        FoVPerspectiveCameras, 
        RasterizationSettings, 
        MeshRenderer, 
        MeshRasterizer,
        SoftPhongShader,
        TexturesUV,
        PointLights
    )
    from pytorch3d.io import load_obj
    PYTORCH3D_AVAILABLE = True
except ImportError:
    PYTORCH3D_AVAILABLE = False
    print("Warning: PyTorch3D not available. Falling back to Open3D rendering.")

# Open3D fallback imports
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

class MVSRenderer:
    """Renders novel views from textured meshes for MVS benchmark evaluation"""
    
    def __init__(self, mesh_path: str, output_dir: str, device: str = "cuda"):
        self.mesh_path = Path(mesh_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        # Check if mesh file exists
        if not self.mesh_path.exists():
            raise FileNotFoundError(f"Mesh file not found: {self.mesh_path}")
        
        # Initialize renderer
        self.renderer = None
        self.mesh = None
        self.setup_renderer()
    
    def setup_renderer(self):
        """Initialize the appropriate renderer (PyTorch3D preferred, Open3D fallback)"""
        if PYTORCH3D_AVAILABLE and torch.cuda.is_available():
            print("Using PyTorch3D renderer")
            self.setup_pytorch3d_renderer()
        elif OPEN3D_AVAILABLE:
            print("Using Open3D renderer (fallback)")
            self.setup_open3d_renderer()
        else:
            raise RuntimeError("Neither PyTorch3D nor Open3D is available for rendering")
    
    def setup_pytorch3d_renderer(self):
        """Setup PyTorch3D renderer"""
        # Load mesh with texture
        verts, faces, aux = load_obj(self.mesh_path)
        
        # Handle textures if available
        if aux.texture_images is not None and len(aux.texture_images) > 0:
            # Use provided texture
            texture_image = aux.texture_images[0]  # Use first texture
            verts_uvs = aux.verts_uvs[None, ...]  # Add batch dimension
            faces_uvs = faces.textures_idx[None, ...]  # Add batch dimension
            textures = TexturesUV(
                maps=texture_image[None, ...], 
                faces_uvs=faces_uvs,
                verts_uvs=verts_uvs
            )
        else:
            # Create simple white texture if no texture available
            print("Warning: No texture found, using white material")
            textures = None
        
        # Create mesh
        self.mesh = Meshes(
            verts=[verts], 
            faces=[faces.verts_idx], 
            textures=textures
        ).to(self.device)
        
        # Setup rasterizer
        raster_settings = RasterizationSettings(
            image_size=512,  # Will be adjusted per camera
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=None
        )
        
        # Setup lights
        lights = PointLights(
            device=self.device,
            location=[[0.0, 0.0, -3.0]],
            ambient_color=[[0.2, 0.2, 0.2]],
            diffuse_color=[[0.6, 0.6, 0.6]],
            specular_color=[[0.2, 0.2, 0.2]]
        )
        
        # Setup renderer
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=raster_settings),
            shader=SoftPhongShader(device=self.device, lights=lights)
        )
    
    def setup_open3d_renderer(self):
        """Setup Open3D renderer (fallback)"""
        # Load mesh
        self.mesh = o3d.io.read_triangle_mesh(str(self.mesh_path))
        if len(self.mesh.vertices) == 0:
            raise ValueError(f"Failed to load mesh from {self.mesh_path}")
        
        print(f"Loaded mesh with {len(self.mesh.vertices)} vertices and {len(self.mesh.triangles)} faces")
        
        # Setup renderer
        self.renderer = o3d.visualization.rendering.OffscreenRenderer(640, 480)
        self.renderer.scene.add_geometry("mesh", self.mesh, o3d.visualization.rendering.MaterialRecord())
    
    def load_camera_poses(self, transforms_file: str) -> List[Dict]:
        """Load camera poses from transforms_test.json"""
        with open(transforms_file, 'r') as f:
            transforms_data = json.load(f)
        
        if 'frames' not in transforms_data:
            raise ValueError(f"Invalid transforms file: {transforms_file}")
        
        return transforms_data['frames']
    
    def nerf_matrix_to_pytorch3d(self, nerf_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert NeRF camera matrix to PyTorch3D camera parameters"""
        # NeRF uses different coordinate system than PyTorch3D
        # NeRF: +X right, +Y up, -Z forward
        # PyTorch3D: +X left, +Y up, +Z forward
        
        c2w = np.array(nerf_matrix)
        
        # Convert to PyTorch3D camera coordinate system
        c2w[:, 1:3] *= -1  # Flip Y and Z axes
        w2c = np.linalg.inv(c2w)
        
        # Extract rotation and translation
        R = w2c[:3, :3]
        T = w2c[:3, 3]
        
        return R, T
    
    def render_pytorch3d(self, camera_poses: List[Dict], image_size: Tuple[int, int] = (800, 800)) -> List[np.ndarray]:
        """Render views using PyTorch3D"""
        if self.renderer is None or self.mesh is None:
            raise RuntimeError("PyTorch3D renderer not properly initialized")
        
        rendered_images = []
        
        for i, frame in enumerate(camera_poses):
            # Get camera parameters
            transform_matrix = np.array(frame['transform_matrix'])
            R, T = self.nerf_matrix_to_pytorch3d(transform_matrix)
            
            # Create camera
            cameras = FoVPerspectiveCameras(
                device=self.device,
                R=torch.tensor(R, dtype=torch.float32).unsqueeze(0),
                T=torch.tensor(T, dtype=torch.float32).unsqueeze(0),
                fov=60.0  # Default field of view
            )
            
            # Update rasterizer settings for this image size
            raster_settings = RasterizationSettings(
                image_size=image_size,
                blur_radius=0.0,
                faces_per_pixel=1
            )
            self.renderer.rasterizer.raster_settings = raster_settings
            
            # Render
            with torch.no_grad():
                images = self.renderer(self.mesh, cameras=cameras)
                image = images[0, ..., :3]  # Remove alpha channel
                image_np = image.cpu().numpy()
                
                # Convert to uint8 and RGB format
                image_np = (image_np * 255).astype(np.uint8)
                rendered_images.append(image_np)
            
            print(f"Rendered frame {i+1}/{len(camera_poses)}")
        
        return rendered_images
    
    def render_open3d(self, camera_poses: List[Dict], image_size: Tuple[int, int] = (800, 800)) -> List[np.ndarray]:
        """Render views using Open3D (fallback)"""
        if self.renderer is None:
            raise RuntimeError("Open3D renderer not properly initialized")
        
        rendered_images = []
        
        # Update renderer size
        width, height = image_size
        self.renderer.setup_camera(60.0, width / height, 0.01, 1000.0)
        
        for i, frame in enumerate(camera_poses):
            # Get camera parameters
            transform_matrix = np.array(frame['transform_matrix'])
            
            # Convert NeRF camera matrix to Open3D format
            # Open3D camera looks down -Z axis
            c2w = transform_matrix.copy()
            c2w[:, 1:3] *= -1  # Flip Y and Z to match Open3D coordinate system
            
            # Set camera pose
            self.renderer.scene.camera.look_at(
                c2w[:3, 3],  # Camera position
                c2w[:3, 3] + c2w[:3, 2],  # Look at point (position + forward)
                -c2w[:3, 1]  # Up vector (negative Y after flip)
            )
            
            # Render
            image = self.renderer.render_to_image()
            image_np = np.asarray(image)
            
            rendered_images.append(image_np)
            print(f"Rendered frame {i+1}/{len(camera_poses)}")
        
        return rendered_images
    
    def render_views(self, transforms_file: str, image_size: Tuple[int, int] = (800, 800)) -> List[str]:
        """Render novel views from test camera poses"""
        print(f"Loading camera poses from: {transforms_file}")
        camera_poses = self.load_camera_poses(transforms_file)
        print(f"Found {len(camera_poses)} test camera poses")
        
        # Render images
        if PYTORCH3D_AVAILABLE and torch.cuda.is_available():
            rendered_images = self.render_pytorch3d(camera_poses, image_size)
        else:
            rendered_images = self.render_open3d(camera_poses, image_size)
        
        # Save rendered images
        output_files = []
        for i, (image, frame) in enumerate(zip(rendered_images, camera_poses)):
            # Get original filename if available, otherwise use index
            if 'file_path' in frame:
                base_name = Path(frame['file_path']).stem
            else:
                base_name = f"frame_{i:04d}"
            
            output_file = self.output_dir / f"{base_name}_rendered.png"
            cv2.imwrite(str(output_file), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            output_files.append(str(output_file))
            
            print(f"Saved: {output_file}")
        
        print(f"\nRendering complete! {len(output_files)} images saved to {self.output_dir}")
        return output_files

def main():
    parser = argparse.ArgumentParser(description="Render novel views from MVS textured mesh")
    parser.add_argument("-m", "--mesh", type=str, required=True,
                       help="Path to textured mesh file (.obj)")
    parser.add_argument("-t", "--transforms", type=str, required=True,
                       help="Path to transforms_test.json file")
    parser.add_argument("-o", "--output", type=str, default="./rendered_images",
                       help="Output directory for rendered images")
    parser.add_argument("--width", type=int, default=800,
                       help="Image width (default: 800)")
    parser.add_argument("--height", type=int, default=600,
                       help="Image height (default: 600)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for rendering (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.mesh).exists():
        print(f"Error: Mesh file not found: {args.mesh}")
        return 1
    
    if not Path(args.transforms).exists():
        print(f"Error: Transforms file not found: {args.transforms}")
        return 1
    
    # Check device availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    try:
        # Create renderer and render views
        renderer = MVSRenderer(args.mesh, args.output, args.device)
        output_files = renderer.render_views(args.transforms, (args.width, args.height))
        
        print(f"Successfully rendered {len(output_files)} views")
        return 0
        
    except Exception as e:
        print(f"Rendering failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())