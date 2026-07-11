"""
BENCHMARK SUITE — Hyperparameter Tuning & GPU Benchmark Engine
============================================================
Evaluates pipeline configurations (inference steps, resolution, LoRA scale)
across speed, VRAM, and visual consistency metrics.
Provides recommendations and auto-apply exporters.
"""

import os
import gc
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image, ImageDraw

log = logging.getLogger("pipeline.benchmark")


class BenchmarkSuite:
    """
    Grid-search parameter sweep and GPU preheating optimizer.
    Evaluates configurations and suggests optimal quality-to-speed ratios.
    """

    def __init__(self, settings_path: str = "config/settings.yaml"):
        self.settings_path = settings_path
        self.settings: Dict[str, Any] = {}
        self.load_settings()

        # Output folder for benchmark runs
        self.output_dir = self.settings.get("benchmark", {}).get("output_dir", "outputs/benchmarks")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def load_settings(self):
        """Load settings from settings.yaml."""
        try:
            from utils.config_helper import load_settings
            self.settings = load_settings() or {}
        except Exception as e:
            log.warning(f"Failed to load settings in benchmark: {e}")
            self.settings = {}

    def get_sweep_parameters(self) -> Tuple[List[int], List[float], List[int]]:
        """Retrieve sweep parameter ranges from configuration with safe defaults."""
        bench_conf = self.settings.get("benchmark", {})
        sweep_conf = bench_conf.get("sweep", {})

        steps = sweep_conf.get("steps", [15, 20, 25, 30])
        lora_scales = sweep_conf.get("lora_scales", [0.5, 0.8, 1.0])
        resolutions = sweep_conf.get("resolutions", [512, 768])

        return steps, lora_scales, resolutions

    def run_sweeps(self, prompt: str = "A warrior in rain", mock: bool = False) -> List[Dict[str, Any]]:
        """
        Run the grid-search parameter sweep.
        
        Args:
            prompt: Text prompt to generate/evaluate.
            mock: If True, uses the fast CPU mock generator (no GPU/model download needed).
            
        Returns:
            List of dictionaries containing metrics for each run.
        """
        steps_range, lora_range, res_range = self.get_sweep_parameters()
        results = []

        log.info(f"=== Starting Benchmark Sweep (Mock Mode: {mock}) ===")
        log.info(f"Sweep Grid: Steps={steps_range}, LoRA={lora_range}, Resolutions={res_range}")

        # Keep track of baseline image at highest settings to calculate structural similarities
        baseline_img: Optional[Image.Image] = None
        max_steps = max(steps_range)
        max_lora = max(lora_range)
        max_res = max(res_range)

        # 1. Generate baseline image first
        log.info(f"Generating high-quality baseline image (steps={max_steps}, lora={max_lora}, res={max_res})...")
        baseline_img, _ = self._run_generation(
            prompt=prompt,
            width=max_res,
            height=max_res,
            steps=max_steps,
            lora_scale=max_lora,
            mock=mock,
            seed=42
        )

        # Save baseline image
        if baseline_img:
            baseline_path = os.path.join(self.output_dir, "baseline_image.png")
            baseline_img.save(baseline_path)
            log.info(f"Baseline image saved to: {baseline_path}")

        # 2. Run GPU Preheat / Cold Start Test
        preheat_time_s = 0.0
        if not mock:
            log.info("Running cold-start preheating test...")
            preheat_time_s = self._run_preheat_test()
            log.info(f"Cold-start model preheat took {preheat_time_s:.2f} seconds.")

        # 3. Run Grid Search Sweeps
        run_count = 0
        total_runs = len(steps_range) * len(lora_range) * len(res_range)

        for res in res_range:
            # Resize baseline to match current sweep resolution for structural metrics
            scaled_baseline = None
            if baseline_img:
                scaled_baseline = baseline_img.resize((res, res), Image.Resampling.LANCZOS)

            for steps in steps_range:
                for lora in lora_range:
                    run_count += 1
                    log.info(f"Sweep [{run_count}/{total_runs}]: Res={res}x{res}, Steps={steps}, LoRA={lora}")

                    # Measure start memory/time
                    start_time = time.time()
                    vram_start = self._get_gpu_memory_usage()

                    # Run generation
                    img, error = self._run_generation(
                        prompt=prompt,
                        width=res,
                        height=res,
                        steps=steps,
                        lora_scale=lora,
                        mock=mock,
                        seed=42
                    )

                    # Measure end memory/time
                    gen_time = time.time() - start_time
                    vram_end = self._get_gpu_memory_usage()
                    vram_peak = max(0.0, vram_end - vram_start)

                    # Compute quality metrics against baseline
                    ssim_score = 1.0
                    edge_score = 1.0
                    color_score = 1.0

                    if img and scaled_baseline and not error:
                        ssim_score, edge_score, color_score = self.evaluate_quality(img, scaled_baseline)

                    # Save sweep result image
                    filename = f"run_res{res}_steps{steps}_lora{int(lora*100)}.png"
                    file_path = os.path.join(self.output_dir, filename)
                    if img:
                        img.save(file_path)

                    # Record run metrics
                    run_data = {
                        "run_id": run_count,
                        "resolution": res,
                        "steps": steps,
                        "lora_scale": lora,
                        "generation_time_s": round(gen_time, 3),
                        "vram_peak_mb": round(vram_peak, 2),
                        "ssim_similarity": round(ssim_score, 4),
                        "edge_similarity": round(edge_score, 4),
                        "color_similarity": round(color_score, 4),
                        "image_name": filename,
                        "error": error
                    }
                    results.append(run_data)

                    # Optional GC to prevent VRAM accumulation
                    gc.collect()
                    self._clear_cuda_cache()

        log.info(f"=== Benchmark Sweep Completed ({len(results)} runs evaluation saved) ===")
        return results

    def _run_generation(self, prompt: str, width: int, height: int, steps: int, 
                        lora_scale: float, mock: bool, seed: int = 42) -> Tuple[Optional[Image.Image], Optional[str]]:
        """Run either real GPU generation or mock CPU generation."""
        if mock:
            return self._run_mock_generation(width, height, steps, lora_scale, seed), None
        
        try:
            # Dynamically import integrated pipeline to run on GPU
            from integrated_pipeline import IntegratedComicPipeline
            pipeline = IntegratedComicPipeline(dry_run=False)
            
            # Temporary override config settings
            pipeline.settings["generation"] = {
                "default_size": {"width": width, "height": height},
                "inference_steps": steps,
                "guidance_scale": 7.5,
                "seed": seed,
                "batch_size": 1,
                "safety_checker": False
            }
            
            # Fetch prompt embeds configuration
            config = {
                "width": width,
                "height": height,
                "num_steps": steps,
                "guidance_scale": 7.5,
                "seed": seed,
                "lora_scale": lora_scale,
                "enable_attention_slicing": True,
                "enable_vae_slicing": True
            }
            
            # Load sdxl backend
            backend = pipeline.backend_selector.select({"layout": {"size_class": "medium", "camera_angle": "medium_shot"}})
            image = backend.generate(prompt, "blurry, low quality", config)
            return image, None
            
        except Exception as e:
            log.error(f"Generation failed: {e}")
            # Fallback to mock representation on error
            return self._run_mock_generation(width, height, steps, lora_scale, seed), str(e)

    def _run_mock_generation(self, width: int, height: int, steps: int, lora_scale: float, seed: int) -> Image.Image:
        """CPU Mock image generator to verify benchmark suite code without GPU/models."""
        # Generate a gradient background
        image = Image.new("RGB", (width, height), color=(30, 30, 45))
        draw = ImageDraw.Draw(image)
        
        # Draw a central circle representing the generated prompt subject
        # Change color/radius slightly based on steps and LoRA scale to simulate variations
        import random
        random.seed(seed + steps)
        
        radius = int(min(width, height) * 0.25 + (steps * 2) * lora_scale)
        cx, cy = width // 2, height // 2
        
        # Color shade changes depending on LoRA scale
        r = int(200 * lora_scale)
        g = int(100 + 50 * lora_scale)
        b = int(255 - 50 * lora_scale)
        
        # Draw outer glowing ring (more steps = smoother ring)
        glow_steps = min(5, steps // 5)
        for i in range(glow_steps, 0, -1):
            offset = i * 8
            draw.ellipse([cx - radius - offset, cy - radius - offset, cx + radius + offset, cy + radius + offset], 
                         fill=None, outline=(r, g, b, 50), width=2)
            
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(r, g, b), outline=(255, 255, 255))
        
        # Add mock prompt text
        draw.text((20, height - 30), f"Mock Generation ({width}x{height}) Steps: {steps}", fill=(255, 255, 255))
        return image

    def _run_preheat_test(self) -> float:
        """Measure cold-start loading time for the model backend selector."""
        start_time = time.time()
        try:
            from integrated_pipeline import IntegratedComicPipeline
            pipeline = IntegratedComicPipeline(dry_run=False)
            # Select backend which triggers lazy load
            pipeline.backend_selector.select({"layout": {"size_class": "medium", "camera_angle": "medium_shot"}})
        except Exception as e:
            log.warning(f"Preheat test failed: {e}")
        return time.time() - start_time

    def evaluate_quality(self, img1: Image.Image, img2: Image.Image) -> Tuple[float, float, float]:
        """
        Evaluate structural, edge, and color similarity between two images.
        Uses pure-python fallback calculations if skimage/OpenCV are missing.
        """
        import numpy as np
        
        # Ensure sizes are identical (resize if needed)
        if img1.size != img2.size:
            img1 = img1.resize(img2.size, Image.Resampling.LANCZOS)
            
        arr1 = np.array(img1.convert("L"), dtype=np.float32)
        arr2 = np.array(img2.convert("L"), dtype=np.float32)

        # --- 1. Compute SSIM (Structural Similarity) ---
        ssim_score = 1.0
        try:
            from skimage.metrics import structural_similarity as ssim
            ssim_score = float(ssim(arr1, arr2, data_range=255.0))
        except ImportError:
            # Simple MSE-based similarity fallback
            mse = np.mean((arr1 - arr2) ** 2)
            if mse == 0:
                ssim_score = 1.0
            else:
                ssim_score = 1.0 - min(1.0, mse / (255.0 ** 2))

        # --- 2. Compute Canny Edge Similarity ---
        edge_score = 1.0
        try:
            import cv2
            # Calculate Canny edges
            edges1 = cv2.Canny(np.array(img1), 100, 200)
            edges2 = cv2.Canny(np.array(img2), 100, 200)
            # Percent matching pixels
            matching = np.sum(edges1 == edges2)
            edge_score = float(matching / edges1.size)
        except ImportError:
            # Simple Sobel Edge fallback using numpy
            # Simple gradient diff
            dx1 = np.diff(arr1, axis=1)[:, :-1]
            dx2 = np.diff(arr2, axis=1)[:, :-1]
            edge_diff = np.mean(np.abs(dx1 - dx2))
            edge_score = 1.0 - min(1.0, edge_diff / 255.0)

        # --- 3. Compute Color Histogram Similarity ---
        color_score = 1.0
        try:
            import cv2
            # Convert PIL to CV2 BGR
            cv_img1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
            cv_img2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)
            
            hsv1 = cv2.cvtColor(cv_img1, cv2.COLOR_BGR2HSV)
            hsv2 = cv2.cvtColor(cv_img2, cv2.COLOR_BGR2HSV)
            
            hist1 = cv2.calcHist([hsv1], [0, 1], None, [50, 60], [0, 180, 0, 256])
            hist2 = cv2.calcHist([hsv2], [0, 1], None, [50, 60], [0, 180, 0, 256])
            
            cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
            
            color_score = float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))
            color_score = max(0.0, color_score)  # Handle negative correlations
        except ImportError:
            # Simple PIL histogram fallback
            hist1 = img1.histogram()
            hist2 = img2.histogram()
            # Calculate correlation
            h1 = np.array(hist1, dtype=np.float32)
            h2 = np.array(hist2, dtype=np.float32)
            cos_sim = np.dot(h1, h2) / (np.linalg.norm(h1) * np.linalg.norm(h2) + 1e-8)
            color_score = float(cos_sim)

        return ssim_score, edge_score, color_score

    def get_recommendation(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Recommendation Engine:
        Finds the run with the best balance of speed, memory, and quality.
        Target score: (SSIM * 0.5 + Color * 0.3 + Edge * 0.2) / GenerationTime.
        """
        if not results:
            return {}

        best_score = -1.0
        best_run = results[0]

        for run in results:
            if run.get("error"):
                continue
                
            q_score = (
                run.get("ssim_similarity", 1.0) * 0.5 + 
                run.get("color_similarity", 1.0) * 0.3 + 
                run.get("edge_similarity", 1.0) * 0.2
            )
            gen_time = max(0.1, run.get("generation_time_s", 1.0))
            
            # Efficiency index: score per second
            efficiency = q_score / gen_time
            
            if efficiency > best_score:
                best_score = efficiency
                best_run = run

        return best_run

    def apply_configuration(self, best_config: Dict[str, Any]) -> bool:
        """
        Auto-Apply configuration exporter.
        Safely writes the recommended step count and resolution to settings.yaml.
        """
        res = best_config.get("resolution", 768)
        steps = best_config.get("steps", 25)
        lora = best_config.get("lora_scale", 0.8)

        log.info(f"Applying recommended settings to settings.yaml: Res={res}x{res}, Steps={steps}, LoRA={lora}")

        try:
            # Read settings file under file locking
            from core.feedback_tuner import lock_file
            import yaml

            with open(self.settings_path, "r+", encoding="utf-8") as f:
                with lock_file(f):
                    data = yaml.safe_load(f) or {}
                    
                    # Update generation parameters
                    data.setdefault("generation", {})
                    data["generation"]["default_size"] = {"width": res, "height": res}
                    data["generation"]["inference_steps"] = steps
                    
                    data.setdefault("models", {})
                    data["models"].setdefault("lora", {})
                    data["models"]["lora"]["adapter_scale"] = lora

                    # Rewind and truncate file
                    f.seek(0)
                    yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
                    f.truncate()
            log.info("Settings file updated successfully!")
            return True
        except Exception as e:
            log.error(f"Failed to auto-apply configurations: {e}")
            return False

    def _get_gpu_memory_usage(self) -> float:
        """Return allocated GPU memory in MB if CUDA is available."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.max_memory_allocated() / (1024 * 1024)
        except Exception:
            pass
        return 0.0

    def _clear_cuda_cache(self):
        """Clears PyTorch CUDA tensor cache."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
