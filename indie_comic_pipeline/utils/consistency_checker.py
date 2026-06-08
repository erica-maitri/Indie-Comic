"""
CONSISTENCY CHECKER
Validates that character looks the same across all panels using color, structure, style, and semantic similarity
"""

import numpy as np
from PIL import Image
import os
import cv2
import torch

class ConsistencyChecker:

    def __init__(self):
        self.reference_features = None
        self.clip_model = None
        self.clip_processor = None
        self.device = None

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
        """Compute visual semantic similarity using CLIP embeddings"""
        if self.clip_model is None or self.clip_processor is None:
            from transformers import CLIPProcessor, CLIPModel
            
            # Use CPU or CUDA depending on torch support
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        
        inputs = self.clip_processor(images=[img1, img2], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            features = self.clip_model.get_image_features(**inputs)
            
        # Normalize embeddings
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        
        # Cosine similarity
        similarity = torch.nn.functional.cosine_similarity(features[0:1], features[1:2]).item()
        return max(0.0, min(1.0, similarity))

    def compute_dinov2_similarity(self, img1_path, img2_path):
        """Compute visual structure similarity using DINOv2 features"""
        if not hasattr(self, 'dinov2_model') or self.dinov2_model is None:
            from transformers import AutoImageProcessor, AutoModel
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.dinov2_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
            self.dinov2_model = AutoModel.from_pretrained("facebook/dinov2-base").to(self.device)
            
        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")
        
        inputs1 = self.dinov2_processor(images=img1, return_tensors="pt").to(self.device)
        inputs2 = self.dinov2_processor(images=img2, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs1 = self.dinov2_model(**inputs1)
            outputs2 = self.dinov2_model(**inputs2)
            
            # Use pooler_output for global image representation
            features1 = outputs1.pooler_output
            features2 = outputs2.pooler_output
            
            # Normalize embeddings
            features1 = features1 / features1.norm(p=2, dim=-1, keepdim=True)
            features2 = features2 / features2.norm(p=2, dim=-1, keepdim=True)
            
            similarity = torch.nn.functional.cosine_similarity(features1, features2).item()
            
        return max(0.0, min(1.0, similarity))

    def extract_features(self, image_path):
        """Extract image features for comparison"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")
            
        img_cv = cv2.imread(image_path)
        if img_cv is None:
            raise ValueError(f"Could not read image at {image_path}")
            
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
        """Set the reference character image"""
        self.reference_features = self.extract_features(reference_image_path)
        print(f"[SUCCESS] Reference set: {reference_image_path}")

    def check_consistency(self, image_path, threshold=0.60):
        """Check if image matches reference character using color, structure, style, and semantic similarity"""
        if self.reference_features is None:
            raise ValueError("No reference set. Call set_reference() first.")
            
        features = self.extract_features(image_path)
        
        # 1. Color similarity (correlation of HSV histograms)
        color_score = cv2.compareHist(
            self.reference_features['histogram'], 
            features['histogram'], 
            cv2.HISTCMP_CORREL
        )
        color_score = max(0.0, min(1.0, color_score))
        
        # 2. Grayscale thumbnail correlation (legacy struct_score)
        p1 = self.reference_features['pixels']
        p2 = features['pixels']
        std1, std2 = np.std(p1), np.std(p2)
        if std1 > 0 and std2 > 0:
            legacy_struct_score = np.corrcoef(p1, p2)[0, 1]
        else:
            legacy_struct_score = 0.0
        legacy_struct_score = max(0.0, min(1.0, legacy_struct_score))
        
        # 3. Structural Similarity Index (SSIM)
        ref_gray = self.reference_features['img_gray']
        feat_gray = features['img_gray']
        if ref_gray.shape != feat_gray.shape:
            feat_gray = cv2.resize(feat_gray, (ref_gray.shape[1], ref_gray.shape[0]))
            
        try:
            from skimage.metrics import structural_similarity as ssim
            ssim_score = ssim(ref_gray, feat_gray)
        except Exception:
            ssim_score = self._fallback_ssim(ref_gray, feat_gray)
        ssim_score = max(0.0, min(1.0, ssim_score))
        
        # 4. Edge Density Similarity
        ref_density = self.reference_features['edge_density']
        feat_density = features['edge_density']
        edge_diff = abs(ref_density - feat_density)
        edge_score = max(0.0, min(1.0, 1.0 - edge_diff * 5.0))
        
        # 5. Art Style Gram Matrix similarity
        ref_gram = self.reference_features['gram_matrix']
        feat_gram = features['gram_matrix']
        gram_diff = np.mean((ref_gram - feat_gram) ** 2)
        style_score = max(0.0, min(1.0, 1.0 - gram_diff * 10.0)) # Scale diff to similarity
        
        # 6. CLIP Image-to-Image Cosine Similarity (Semantic)
        clip_img_score = 0.0
        clip_computed = False
        try:
            clip_img_score = self.compute_clip_image_similarity(self.reference_features['path'], features['path'])
            clip_computed = True
        except Exception as e:
            pass
            
        # 7. DINOv2 Structural Similarity (Vision Transformer representation)
        dinov2_score = 0.0
        dinov2_computed = False
        try:
            dinov2_score = self.compute_dinov2_similarity(self.reference_features['path'], features['path'])
            dinov2_computed = True
        except Exception as e:
            pass
            
        # Combine the scores into a weighted overall score
        if clip_computed and dinov2_computed:
            overall_score = (
                0.15 * color_score + 
                0.15 * clip_img_score + 
                0.20 * dinov2_score + 
                0.15 * ssim_score + 
                0.15 * style_score + 
                0.20 * edge_score
            )
        elif clip_computed:
            overall_score = (
                0.2 * color_score + 
                0.2 * clip_img_score + 
                0.2 * ssim_score + 
                0.2 * style_score + 
                0.2 * edge_score
            )
        elif dinov2_computed:
            overall_score = (
                0.2 * color_score + 
                0.2 * dinov2_score + 
                0.2 * ssim_score + 
                0.2 * style_score + 
                0.2 * edge_score
            )
        else:
            overall_score = (
                0.3 * color_score + 
                0.3 * ssim_score + 
                0.2 * style_score + 
                0.2 * edge_score
            )
            
        overall_score = max(0.0, min(1.0, overall_score))
        
        return {
            'consistent': overall_score >= threshold,
            'score': float(overall_score),
            'color_score': float(color_score),
            'struct_score': float(legacy_struct_score),
            'ssim_score': float(ssim_score),
            'edge_score': float(edge_score),
            'style_score': float(style_score),
            'aesthetic_score': float(features['aesthetic_score']),
            'clip_img_score': float(clip_img_score) if clip_computed else None,
            'dinov2_score': float(dinov2_score) if dinov2_computed else None
        }

    def validate_panels(self, panel_paths, reference_path):
        """Validate all panels against reference"""
        self.set_reference(reference_path)
        
        results = {}
        for path in panel_paths:
            results[path] = self.check_consistency(path)
        
        return results

def get_consistency_checker():
    return ConsistencyChecker()

