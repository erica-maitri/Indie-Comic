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
        
        return 0.4 * sharp_score + 0.3 * contrast_score + 0.3 * color_score
        
    def compute_clip_image_similarity(self, img1: Image.Image, img2: Image.Image) -> Optional[float]:
        try:
            self._load_clip()
            assert self.clip_model is not None
            assert self.clip_processor is not None
            inputs = self.clip_processor(images=[img1, img2], return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                features = self.clip_model.get_image_features(**inputs)
            if not isinstance(features, torch.Tensor):
                if hasattr(features, "pooler_output"):
                    features = features.pooler_output
                elif isinstance(features, (list, tuple)):
                    features = features[0]
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

    def compute_lpips(self, img1: Image.Image, img2: Image.Image) -> Optional[float]:
        """Compute LPIPS perceptual similarity distance (lower is better, 0.0 is identical)."""
        try:
            import lpips  # type: ignore
            import torchvision.transforms as transforms
            if not hasattr(self, 'lpips_model') or self.lpips_model is None:
                print("[ModelEvaluator] Loading LPIPS model...")
                self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
            
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            t1 = transform(img1).unsqueeze(0).to(self.device)
            t2 = transform(img2).unsqueeze(0).to(self.device)
            with torch.no_grad():
                dist = self.lpips_model(t1, t2)
            return dist.item()
        except Exception as e:
            print(f"[ModelEvaluator] LPIPS Error: {e}")
            return None

    def compute_siglip_similarity(self, img1: Image.Image, img2: Image.Image) -> Optional[float]:
        """Compute SigLIP image similarity (higher is better)."""
        try:
            if not hasattr(self, 'siglip_model') or self.siglip_model is None:
                from transformers import AutoProcessor, AutoModel
                print(f"[ModelEvaluator] Loading SigLIP model to {self.device}...")
                self.siglip_processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
                self.siglip_model = AutoModel.from_pretrained("google/siglip-base-patch16-224").to(self.device)
            
            inputs = self.siglip_processor(images=[img1, img2], return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                features = self.siglip_model.get_image_features(**inputs)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            similarity = torch.nn.functional.cosine_similarity(features[0:1], features[1:2]).item()
            return max(0.0, min(1.0, similarity))
        except Exception as e:
            print(f"[ModelEvaluator] SigLIP Error: {e}")
            return None

    def compute_dinov3_similarity(self, img1: Image.Image, img2: Image.Image) -> Optional[float]:
        """Compute DINOv3 (DINOv2 with registers) similarity (higher is better)."""
        try:
            if not hasattr(self, 'dinov3_model') or self.dinov3_model is None:
                from transformers import AutoImageProcessor, AutoModel
                print(f"[ModelEvaluator] Loading DINOv3 (DINOv2-with-registers) model to {self.device}...")
                self.dinov3_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-with-registers-base")
                self.dinov3_model = AutoModel.from_pretrained("facebook/dinov2-with-registers-base").to(self.device)
            
            inputs1 = self.dinov3_processor(images=img1.convert("RGB"), return_tensors="pt").to(self.device)
            inputs2 = self.dinov3_processor(images=img2.convert("RGB"), return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                f1 = self.dinov3_model(**inputs1).pooler_output
                f2 = self.dinov3_model(**inputs2).pooler_output
                
            f1 = f1 / f1.norm(p=2, dim=-1, keepdim=True)
            f2 = f2 / f2.norm(p=2, dim=-1, keepdim=True)
            similarity = torch.nn.functional.cosine_similarity(f1, f2).item()
            return max(0.0, min(1.0, similarity))
        except Exception as e:
            print(f"[ModelEvaluator] DINOv3 Error: {e}")
            return None

    def compute_detection_metrics(self, pred_boxes: List[Tuple[int, int, int, int]], gt_boxes: List[Tuple[int, int, int, int]], iou_threshold: float = 0.5) -> Dict[str, float]:
        """Calculate detection metrics (Accuracy, Precision, Recall, F1) for speech bubbles (Grounding DINO)."""
        if not pred_boxes and not gt_boxes:
            return {"Precision": 1.0, "Recall": 1.0, "F1": 1.0, "Accuracy": 1.0}
        if not pred_boxes or not gt_boxes:
            return {"Precision": 0.0, "Recall": 0.0, "F1": 0.0, "Accuracy": 0.0}
            
        tp = 0
        fp = 0
        matched_gt = set()
        
        for p_box in pred_boxes:
            best_iou = 0.0
            best_gt_idx = -1
            for idx, g_box in enumerate(gt_boxes):
                if idx in matched_gt:
                    continue
                iou_score = self.compute_iou(p_box, g_box)
                if iou_score > best_iou:
                    best_iou = iou_score
                    best_gt_idx = idx
            
            if best_iou >= iou_threshold and best_gt_idx != -1:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1
                
        fn = len(gt_boxes) - len(matched_gt)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        
        return {"Precision": precision, "Recall": recall, "F1": f1, "Accuracy": accuracy}

    def compute_segmentation_metrics(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
        """Calculate segmentation metrics (IoU, Dice/F1) for character masks (SAM 2.1)."""
        pred = pred_mask.astype(bool)
        gt = gt_mask.astype(bool)
        
        intersection = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        pred_sum = pred.sum()
        gt_sum = gt.sum()
        
        iou = intersection / union if union > 0 else (1.0 if pred_sum == 0 and gt_sum == 0 else 0.0)
        dice = 2.0 * intersection / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else (1.0 if pred_sum == 0 and gt_sum == 0 else 0.0)
        
        tp = intersection
        fp = pred_sum - tp
        fn = gt_sum - tp
        
        precision = tp / pred_sum if pred_sum > 0 else (1.0 if gt_sum == 0 else 0.0)
        recall = tp / gt_sum if gt_sum > 0 else (1.0 if pred_sum == 0 else 0.0)
        
        return {"IoU": iou, "Dice": dice, "F1": dice, "Precision": precision, "Recall": recall}
            
    def free_memory(self):
        """Free loaded models from VRAM/RAM"""
        self.clip_model = None
        self.clip_processor = None
        self.dinov2_model = None
        self.dinov2_processor = None
        self.fid_metric = None
        self.siglip_model = None
        self.siglip_processor = None
        self.dinov3_model = None
        self.dinov3_processor = None
        if hasattr(self, 'lpips_model'):
            self.lpips_model = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        import gc
        gc.collect()
