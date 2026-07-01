"""
CONSISTENCY CHECKER
Validates that character looks the same across all panels using color, structure, style, and semantic similarity
Optimized for T4 GPU with model caching and configurable metrics
"""

import numpy as np
from PIL import Image
import os
import cv2
import gc
import sys

from typing import Any

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Global model cache - prevents reloading models for every comparison
_CLIP_MODEL: Any = None
_CLIP_PROCESSOR: Any = None
_DINOV2_MODEL: Any = None
_DINOV2_PROCESSOR: Any = None

class ConsistencyChecker:

    def __init__(self):
        self.reference_features = None
        self.clip_model = None
        self.clip_processor = None
        # Tracks whether reference was set from a character sheet or the first panel
        self.reference_mode = "character_sheet"  # "character_sheet" | "panel"
        
        # Determine dry run status
        self.dry_run = False
        
        # Load configuration for which metrics to use
        try:
            from utils.config_helper import load_settings
            settings = load_settings()
            self.consistency_config = settings.get("consistency", {})
        except:
            # Default to fast settings for T4 GPU
            self.consistency_config = {
                "enable_clip": True,     # Enable CLIP based on methodology
                "enable_dinov2": True,   # Enable DINOv2 based on methodology
                "enable_ssim": True,
                "enable_edge": True,
                "enable_color": True,    # Ensure color is enabled
                "enable_style": True,
                "device": "cpu",         # CPU by default to save VRAM
                "threshold": 0.55        # Methodology threshold
            }
            
        self.device = self.consistency_config.get("device", "cpu")
        if self.device == "cuda" and not (TORCH_AVAILABLE and torch.cuda.is_available()):
            print("[ConsistencyChecker] CUDA is not available. Falling back to CPU.")
            self.device = "cpu"
        
        # Print which metrics are enabled
        print(f"[ConsistencyChecker] Metrics enabled (dry_run={self.dry_run}):")
        print(f"  - CLIP: {self.consistency_config.get('enable_clip', False) if not self.dry_run else False}")
        print(f"  - DINOv2: {self.consistency_config.get('enable_dinov2', False) if not self.dry_run else False}")
        print(f"  - SSIM: {self.consistency_config.get('enable_ssim', True)}")
        print(f"  - Edge: {self.consistency_config.get('enable_edge', True)}")
        print(f"  - Color: {self.consistency_config.get('enable_color', True)}")
        print(f"  - Style: {self.consistency_config.get('enable_style', True)}")

    def _fallback_ssim(self, img1, img2):
        """NumPy/OpenCV implementation of SSIM (Gaussian-based) as a fallback"""
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        # Means
        mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        # Variances and Covariances
        sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
        
        num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        
        ssim_map = num / den
        return float(np.mean(ssim_map))

    def compute_style_gram_matrix(self, img_cv):
        """Compute spatial Gram matrix of color channels + image gradients for texture/style correlation"""
        h, w = 256, 256
        img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Sobel gradients
        sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
        
        img_resized = cv2.resize(img_cv, (w, h)).astype(np.float64) / 255.0
        sx_resized = cv2.resize(sobelx, (w, h)).astype(np.float64) / 255.0
        sy_resized = cv2.resize(sobely, (w, h)).astype(np.float64) / 255.0
        
        # 5-Channel features: R, G, B, SobelX, SobelY
        features = np.zeros((h * w, 5))
        features[:, 0:3] = img_resized.reshape(-1, 3)
        features[:, 3] = sx_resized.reshape(-1)
        features[:, 4] = sy_resized.reshape(-1)
        
        # Gram Matrix
        gram = features.T @ features
        gram /= (h * w)
        return gram

    def compute_aesthetic_score(self, img_cv):
        """Compute an offline aesthetic quality score based on colorfulness, contrast, and sharpness"""
        # 1. Sharpness/Blurriness (Variance of Laplacian)
        img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()
        sharp_score = min(1.0, lap_var / 500.0)
        
        # 2. Contrast
        std_brightness = np.std(img_gray)
        contrast_score = min(1.0, std_brightness / 75.0)
        
        # 3. Colorfulness (Hasler and Suesstrunk metric)
        b, g, r = cv2.split(img_cv)
        rg = np.absolute(r - g)
        yb = np.absolute(0.5 * (r + g) - b)
        std_rg, std_yb = np.std(rg), np.std(yb)
        mean_rg, mean_yb = np.mean(rg), np.mean(yb)
        std_rgyb = np.sqrt(std_rg ** 2 + std_yb ** 2)
        mean_rgyb = np.sqrt(mean_rg ** 2 + mean_yb ** 2)
        colorfulness = std_rgyb + 0.3 * mean_rgyb
        color_score = min(1.0, colorfulness / 80.0)
        
        # Combine metrics into an aesthetic score (weights: 40% sharp, 30% contrast, 30% color)
        aesthetic_val = 0.4 * sharp_score + 0.3 * contrast_score + 0.3 * color_score
        return float(aesthetic_val)

    def compute_clip_image_similarity(self, img1_path, img2_path):
        """Compute visual semantic similarity using CLIP embeddings (runs on CPU/GPU based on config)"""
        if self.dry_run:
            return 0.85
            
        global _CLIP_MODEL, _CLIP_PROCESSOR
        
        # Check if CLIP is enabled in config and torch is available
        if not self.consistency_config.get("enable_clip", False) or not TORCH_AVAILABLE:
            return None
            
        try:
            # Load once globally, reuse forever
            if _CLIP_MODEL is None or _CLIP_PROCESSOR is None:
                from transformers import CLIPProcessor, CLIPModel
                print(f"  [i] Loading CLIP model on {self.device} (once, cached)...")
                _CLIP_MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
                _CLIP_PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                print("  [✓] CLIP model loaded and cached")
            else:
                _CLIP_MODEL = _CLIP_MODEL.to(self.device)
                
            img1 = Image.open(img1_path)
            img2 = Image.open(img2_path)
            
            inputs = _CLIP_PROCESSOR(images=[img1, img2], return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                features = _CLIP_MODEL.get_image_features(**inputs)
                
            if not isinstance(features, torch.Tensor):
                if hasattr(features, "pooler_output"):
                    features = features.pooler_output
                elif isinstance(features, (list, tuple)):
                    features = features[0]
                
            # Normalize embeddings
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            
            # Cosine similarity
            similarity = torch.nn.functional.cosine_similarity(features[0:1], features[1:2]).item()
            
            # Offload if we are using CUDA to free VRAM immediately
            if self.device == "cuda":
                _CLIP_MODEL = _CLIP_MODEL.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                import gc
                gc.collect()
            
            return max(0.0, min(1.0, similarity))
        except Exception as e:
            print(f"[WARNING] CLIP image similarity failed: {e}. Disabling CLIP metric.")
            self.consistency_config["enable_clip"] = False
            return None

    def compute_dinov2_similarity(self, img1_path, img2_path):
        """Compute visual structure similarity using DINOv2 features (runs on CPU/GPU based on config)"""
        if self.dry_run:
            return 0.85
            
        global _DINOV2_MODEL, _DINOV2_PROCESSOR
        
        # Check if DINOv2 is enabled in config and torch is available
        if not self.consistency_config.get("enable_dinov2", False) or not TORCH_AVAILABLE:
            return None
            
        try:
            # Load once globally, reuse forever
            if _DINOV2_MODEL is None or _DINOV2_PROCESSOR is None:
                from transformers import AutoImageProcessor, AutoModel
                print(f"  [i] Loading DINOv2 model on {self.device} (once, cached)...")
                _DINOV2_PROCESSOR = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
                _DINOV2_MODEL = AutoModel.from_pretrained("facebook/dinov2-base").to(self.device)
                print("  [✓] DINOv2 model loaded and cached")
            else:
                _DINOV2_MODEL = _DINOV2_MODEL.to(self.device)
                
            img1 = Image.open(img1_path).convert("RGB")
            img2 = Image.open(img2_path).convert("RGB")
            
            inputs1 = _DINOV2_PROCESSOR(images=img1, return_tensors="pt").to(self.device)
            inputs2 = _DINOV2_PROCESSOR(images=img2, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs1 = _DINOV2_MODEL(**inputs1)
                outputs2 = _DINOV2_MODEL(**inputs2)
                
                # Use pooler_output for global image representation
                features1 = outputs1.pooler_output
                features2 = outputs2.pooler_output
                
                # Normalize embeddings
                features1 = features1 / features1.norm(p=2, dim=-1, keepdim=True)
                features2 = features2 / features2.norm(p=2, dim=-1, keepdim=True)
                
                similarity = torch.nn.functional.cosine_similarity(features1, features2).item()
                
            # Offload if we are using CUDA to free VRAM immediately
            if self.device == "cuda":
                _DINOV2_MODEL = _DINOV2_MODEL.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                import gc
                gc.collect()
            
            return max(0.0, min(1.0, similarity))
        except Exception as e:
            print(f"[WARNING] DINOv2 similarity failed: {e}. Disabling DINOv2 metric.")
            self.consistency_config["enable_dinov2"] = False
            return None

    def extract_features(self, image_path):
        """Extract image features for comparison"""
        if self.dry_run:
            return {
                'histogram': np.zeros((8, 8)),
                'pixels': np.zeros(128 * 128),
                'mean_brightness': 128.0,
                'size': (768, 768),
                'img_gray': np.zeros((768, 768)),
                'edge_density': 0.1,
                'gram_matrix': np.zeros((5, 5)),
                'aesthetic_score': 0.8,
                'path': image_path,
                'reference_path': image_path
            }

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")
            
        # Read file as bytes to handle Unicode paths on Windows safely
        try:
            with open(image_path, 'rb') as f:
                file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
            img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        except Exception as e:
            raise ValueError(f"Could not read image at {image_path}: {e}")
            
        if img_cv is None:
            raise ValueError(f"Could not decode image at {image_path}")
            
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        
        img_pil = Image.open(image_path).convert('L')
        img_resized = img_pil.resize((128, 128))
        pixels = np.array(img_resized).flatten()
        
        img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Compute Canny edge density
        edges = cv2.Canny(img_gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Style Gram matrix
        gram_matrix = self.compute_style_gram_matrix(img_cv)
        
        # Aesthetic score
        aesthetic_score = self.compute_aesthetic_score(img_cv)
        
        return {
            'histogram': hist,
            'pixels': pixels,
            'mean_brightness': np.mean(pixels),
            'size': img_pil.size,
            'img_gray': img_gray,
            'edge_density': edge_density,
            'gram_matrix': gram_matrix,
            'aesthetic_score': aesthetic_score,
            'path': image_path
        }

    def set_reference(self, reference_image_path):
        """Set the reference character image (character sheet mode)."""
        self.reference_features = self.extract_features(reference_image_path)
        self.reference_mode = "character_sheet"
        print(f"[SUCCESS] Reference set (character sheet): {reference_image_path}")

    def set_reference_from_panel(self, panel_image_path):
        """
        Set the reference from the FIRST generated panel instead of a pre-rendered
        character sheet. Used in Story-Weaver / reference-free mode.
        All subsequent panels are compared against this anchor panel.
        """
        self.reference_features = self.extract_features(panel_image_path)
        self.reference_mode = "panel"
        print(f"[SUCCESS] Reference set (panel 1 anchor): {panel_image_path}")

    def check_consistency(self, image_path, threshold=None):
        """Check if image matches reference character using configurable metrics"""
        if self.dry_run:
            return {
                'consistent': True,
                'score': 0.85,
                'color_score': 0.85,
                'struct_score': 0.85,
                'ssim_score': 0.85,
                'edge_score': 0.85,
                'style_score': 0.85,
                'aesthetic_score': 0.85,
                'clip_img_score': 0.85,
                'dinov2_score': 0.85,
                'reference_mode': self.reference_mode,
                'metrics_used': 6
            }

        if self.reference_features is None:
            raise ValueError("No reference set. Call set_reference() first.")
            
        # Use threshold from config if not provided
        if threshold is None:
            threshold = self.consistency_config.get("threshold", 0.55)
            
        features = self.extract_features(image_path)
        
        # Initialize scores
        color_score = 0.0
        ssim_score = 0.0
        edge_score = 0.0
        style_score = 0.0
        legacy_struct_score = 0.0
        
        # 1. Color similarity (HSV correlation) - Always compute (lightweight)
        if self.consistency_config.get("enable_color", True):
            color_score = cv2.compareHist(
                self.reference_features['histogram'], 
                features['histogram'], 
                cv2.HISTCMP_CORREL
            )
            color_score = max(0.0, min(1.0, color_score))
        
        # 2. Grayscale thumbnail correlation - Always compute (lightweight)
        p1 = self.reference_features['pixels']
        p2 = features['pixels']
        std1, std2 = np.std(p1), np.std(p2)
        if std1 > 0 and std2 > 0:
            legacy_struct_score = np.corrcoef(p1, p2)[0, 1]
        else:
            legacy_struct_score = 0.0
        legacy_struct_score = max(0.0, min(1.0, legacy_struct_score))
        
        # 3. Structural Similarity Index (SSIM) - Optional but recommended
        if self.consistency_config.get("enable_ssim", True):
            ref_gray = self.reference_features['img_gray']
            feat_gray = features['img_gray']
            if ref_gray.shape != feat_gray.shape:
                feat_gray = cv2.resize(feat_gray, (ref_gray.shape[1], ref_gray.shape[0]))
                
            try:
                from skimage.metrics import structural_similarity as ssim
                ssim_res = ssim(ref_gray, feat_gray)
                ssim_score = float(ssim_res[0] if isinstance(ssim_res, tuple) else ssim_res)
            except Exception:
                ssim_score = self._fallback_ssim(ref_gray, feat_gray)
            ssim_score = max(0.0, min(1.0, ssim_score))
        
        # 4. Edge Density Similarity - Optional
        if self.consistency_config.get("enable_edge", True):
            ref_density = self.reference_features['edge_density']
            feat_density = features['edge_density']
            edge_diff = abs(ref_density - feat_density)
            edge_score = max(0.0, min(1.0, 1.0 - edge_diff * 5.0))
        
        # 5. Art Style Gram Matrix similarity - Optional
        if self.consistency_config.get("enable_style", True):
            ref_gram = self.reference_features['gram_matrix']
            feat_gram = features['gram_matrix']
            gram_diff = np.mean((ref_gram - feat_gram) ** 2)
            style_score = max(0.0, min(1.0, 1.0 - gram_diff * 10.0))
        
        # 6. CLIP Image-to-Image Cosine Similarity (Semantic) - Optional, heavy
        clip_img_score = None
        if self.consistency_config.get("enable_clip", False):
            try:
                clip_img_score = self.compute_clip_image_similarity(self.reference_features['path'], features['path'])
            except Exception as e:
                pass
                
        # 7. DINOv2 Structural Similarity - Optional, heavy
        dinov2_score = None
        if self.consistency_config.get("enable_dinov2", False):
            try:
                dinov2_score = self.compute_dinov2_similarity(self.reference_features['path'], features['path'])
            except Exception as e:
                pass
        
        # Combine the scores with dynamic weights based on available metrics
        available_metrics = 0
        total_weight = 0
        
        # Build weighted sum based on what's available
        overall_score = 0.0
        
        if self.consistency_config.get("enable_color", True):
            overall_score += color_score * 0.25
            total_weight += 0.25
            available_metrics += 1
            
        if self.consistency_config.get("enable_ssim", True):
            overall_score += ssim_score * 0.30
            total_weight += 0.30
            available_metrics += 1
            
        if self.consistency_config.get("enable_style", True):
            overall_score += style_score * 0.20
            total_weight += 0.20
            available_metrics += 1
            
        if self.consistency_config.get("enable_edge", True):
            overall_score += edge_score * 0.15
            total_weight += 0.15
            available_metrics += 1
            
        # Add heavy metrics if available
        if clip_img_score is not None:
            overall_score += clip_img_score * 0.05
            total_weight += 0.05
            available_metrics += 1
            
        if dinov2_score is not None:
            overall_score += dinov2_score * 0.05
            total_weight += 0.05
            available_metrics += 1
        
        # Normalize by total weight
        if total_weight > 0:
            overall_score = overall_score / total_weight
        
        overall_score = max(0.0, min(1.0, overall_score))

        ref_label = "Panel Anchor" if self.reference_mode == "panel" else "Character Sheet"

        return {
            'consistent': overall_score >= threshold,
            'score': float(overall_score),
            'color_score': float(color_score) if self.consistency_config.get("enable_color", True) else None,
            'struct_score': float(legacy_struct_score),
            'ssim_score': float(ssim_score),
            'edge_score': float(edge_score),
            'style_score': float(style_score),
            'aesthetic_score': float(features['aesthetic_score']),
            'clip_img_score': float(clip_img_score) if clip_img_score is not None else None,
            'dinov2_score': float(dinov2_score) if dinov2_score is not None else None,
            'reference_mode': ref_label,
            'metrics_used': available_metrics
        }

    def validate_panels(self, panel_paths, reference_path):
        """Validate all panels against reference"""
        self.set_reference(reference_path)
        
        results = {}
        for path in panel_paths:
            results[path] = self.check_consistency(path)
        
        return results
    
    def clear_cache(self):
        """Clear cached models to free GPU memory"""
        global _CLIP_MODEL, _CLIP_PROCESSOR, _DINOV2_MODEL, _DINOV2_PROCESSOR
        
        if _CLIP_MODEL is not None:
            del _CLIP_MODEL
            del _CLIP_PROCESSOR
            _CLIP_MODEL = None
            _CLIP_PROCESSOR = None
            
        if _DINOV2_MODEL is not None:
            del _DINOV2_MODEL
            del _DINOV2_PROCESSOR
            _DINOV2_MODEL = None
            _DINOV2_PROCESSOR = None
            
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            
        print("[✓] Model cache cleared, GPU memory freed")

def get_consistency_checker():
    """Factory function to create a new ConsistencyChecker instance"""
    return ConsistencyChecker()