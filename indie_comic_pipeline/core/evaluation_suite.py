import os
import torch
import numpy as np
import cv2
from PIL import Image
from typing import Dict, Tuple, List, Optional, Any

class ModelEvaluator:
    """Comprehensive evaluation suite for image, text, and layout models."""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        
        # Lazy loading models
        self.clip_model = None
        self.clip_processor = None
        self.dinov2_model = None
        self.dinov2_processor = None
        self.fid_metric = None
        
    def _load_clip(self):
        if self.clip_model is None:
            from transformers import CLIPProcessor, CLIPModel
            print(f"[ModelEvaluator] Loading CLIP model to {self.device}...")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def _load_dinov2(self):
        if self.dinov2_model is None:
            from transformers import AutoImageProcessor, AutoModel
            print(f"[ModelEvaluator] Loading DINOv2 model to {self.device}...")
            self.dinov2_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
            self.dinov2_model = AutoModel.from_pretrained("facebook/dinov2-base").to(self.device)
            
    def _load_fid(self):
        if self.fid_metric is None:
            from torchmetrics.image.fid import FrechetInceptionDistance
            print(f"[ModelEvaluator] Loading FID metric to {self.device}...")
            self.fid_metric = FrechetInceptionDistance(feature=64).to(self.device)
            
    def compute_fid(self, generated_img: Image.Image, reference_img: Image.Image) -> Optional[float]:
        """Compute Fréchet Inception Distance between two images."""
        try:
            import torchvision.transforms as transforms
            self._load_fid()
            assert self.fid_metric is not None
            
            transform = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor()
            ])
            
            gen_tensor = transform(generated_img).unsqueeze(0).to(self.device)
            ref_tensor = transform(reference_img).unsqueeze(0).to(self.device)
            
            # FID requires a batch size > 1 to compute covariance. If we are evaluating 
            # single images, duplicate them to bypass the error.
            if gen_tensor.shape[0] == 1:
                gen_tensor = gen_tensor.repeat(2, 1, 1, 1)
                ref_tensor = ref_tensor.repeat(2, 1, 1, 1)
            
            # Torchmetrics FID expects ByteTensor [0, 255]
            gen_tensor = (gen_tensor * 255).byte()
            ref_tensor = (ref_tensor * 255).byte()
            
            self.fid_metric.reset()
            self.fid_metric.update(gen_tensor, real=False)
            self.fid_metric.update(ref_tensor, real=True)
            
            return self.fid_metric.compute().item()
        except Exception as e:
            print(f"[ModelEvaluator] FID Error: {e}")
            return None

    def compute_aesthetic_score(self, img: Image.Image) -> float:
        """Compute an offline aesthetic quality score based on colorfulness, contrast, and sharpness"""
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # 1. Sharpness/Blurriness (Variance of Laplacian)
        img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()
        sharp_score = min(1.0, lap_var / 500.0)
        
        # 2. Contrast
        std_brightness = float(np.std(img_gray))  # type: ignore
        contrast_score = min(1.0, std_brightness / 75.0)
        
        # 3. Colorfulness
        b, g, r = cv2.split(img_cv)
        rg = np.absolute(r - g)
        yb = np.absolute(0.5 * (r + g) - b)
        colorfulness = np.sqrt(np.std(rg)**2 + np.std(yb)**2) + 0.3 * np.sqrt(np.mean(rg)**2 + np.mean(yb)**2)
        color_score = min(1.0, colorfulness / 80.0)
        
        return float(0.4 * sharp_score + 0.3 * contrast_score + 0.3 * color_score)
        
    def compute_clip_image_similarity(self, img1: Image.Image, img2: Image.Image) -> Optional[float]:
        try:
            self._load_clip()
            assert self.clip_model is not None
            assert self.clip_processor is not None
            inputs = self.clip_processor(images=[img1, img2], return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                features = self.clip_model.get_image_features(**inputs)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            similarity = torch.nn.functional.cosine_similarity(features[0:1], features[1:2]).item()
            return max(0.0, min(1.0, similarity))
        except Exception as e:
            print(f"[ModelEvaluator] CLIP Image Error: {e}")
            return None

    def compute_clip_text_alignment(self, img: Image.Image, prompt: str) -> Optional[float]:
        try:
            self._load_clip()
            assert self.clip_model is not None
            assert self.clip_processor is not None
            inputs = self.clip_processor(text=[prompt], images=img, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
            score = outputs.logits_per_image.item() / 100.0
            return max(0.0, min(1.0, score))
        except Exception as e:
            print(f"[ModelEvaluator] CLIP Text Error: {e}")
            return None

    def compute_dinov2_similarity(self, img1: Image.Image, img2: Image.Image) -> Optional[float]:
        try:
            self._load_dinov2()
            assert self.dinov2_processor is not None
            assert self.dinov2_model is not None
            i1 = img1.convert("RGB")
            i2 = img2.convert("RGB")
            inputs1 = self.dinov2_processor(images=i1, return_tensors="pt").to(self.device)
            inputs2 = self.dinov2_processor(images=i2, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                f1 = self.dinov2_model(**inputs1).pooler_output
                f2 = self.dinov2_model(**inputs2).pooler_output
                
            f1 = f1 / f1.norm(p=2, dim=-1, keepdim=True)
            f2 = f2 / f2.norm(p=2, dim=-1, keepdim=True)
            similarity = torch.nn.functional.cosine_similarity(f1, f2).item()
            return max(0.0, min(1.0, similarity))
        except Exception as e:
            print(f"[ModelEvaluator] DINOv2 Error: {e}")
            return None

    def compute_bleu(self, generated_text: str, reference_text: str) -> Optional[float]:
        try:
            import nltk
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            
            reference = [reference_text.split()]
            candidate = generated_text.split()
            
            smoothie = SmoothingFunction().method4
            return sentence_bleu(reference, candidate, smoothing_function=smoothie)
        except Exception as e:
            print(f"[ModelEvaluator] BLEU Error: {e}")
            return None

    def compute_iou(self, predicted_bbox: Tuple[int, int, int, int], ground_truth_bbox: Tuple[int, int, int, int]) -> float:
        try:
            x1 = max(predicted_bbox[0], ground_truth_bbox[0])
            y1 = max(predicted_bbox[1], ground_truth_bbox[1])
            x2 = min(predicted_bbox[2], ground_truth_bbox[2])
            y2 = min(predicted_bbox[3], ground_truth_bbox[3])
            
            if x2 < x1 or y2 < y1:
                return 0.0
            
            intersection = (x2 - x1) * (y2 - y1)
            pred_area = (predicted_bbox[2] - predicted_bbox[0]) * (predicted_bbox[3] - predicted_bbox[1])
            gt_area = (ground_truth_bbox[2] - ground_truth_bbox[0]) * (ground_truth_bbox[3] - ground_truth_bbox[1])
            union = pred_area + gt_area - intersection
            
            return intersection / union if union > 0 else 0.0
        except Exception as e:
            print(f"[ModelEvaluator] IoU Error: {e}")
            return 0.0

    def compute_ssim(self, generated_img: Image.Image, reference_img: Image.Image) -> Optional[float]:
        try:
            from torchmetrics.image import StructuralSimilarityIndexMeasure
            import torchvision.transforms as transforms
            
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ])
            gen_tensor = transform(generated_img).unsqueeze(0).to(self.device)
            ref_tensor = transform(reference_img).unsqueeze(0).to(self.device)
            
            ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
            return ssim_metric(gen_tensor, ref_tensor).item()
        except Exception as e:
            print(f"[ModelEvaluator] SSIM Error: {e}")
            return None

    def compute_psnr(self, generated_img: Image.Image, reference_img: Image.Image) -> Optional[float]:
        try:
            from torchmetrics.image import PeakSignalNoiseRatio
            import torchvision.transforms as transforms
            
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ])
            gen_tensor = transform(generated_img).unsqueeze(0).to(self.device)
            ref_tensor = transform(reference_img).unsqueeze(0).to(self.device)
            
            psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
            return psnr_metric(gen_tensor, ref_tensor).item()
        except Exception as e:
            print(f"[ModelEvaluator] PSNR Error: {e}")
            return None

            
    def free_memory(self):
        """Free loaded models from VRAM/RAM"""
        self.clip_model = None
        self.clip_processor = None
        self.dinov2_model = None
        self.dinov2_processor = None
        self.fid_metric = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        import gc
        gc.collect()
