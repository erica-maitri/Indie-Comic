import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
from PIL import Image

log = logging.getLogger("pipeline.user_preference_critic")


class UserPreferenceModel(nn.Module):
    """
    Lightweight linear regression network to map CLIP features to user ratings.
    """
    def __init__(self, input_dim: int = 512):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x))


class UserPreferenceCritic:
    """
    Dynamic Critic that extracts CLIP image features and trains a local regression
    model on user star ratings to predict visual preference.
    """
    def __init__(self, 
                 model_path: str = "outputs/user_preference_model.pt",
                 device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self.input_dim = 512  # Projection size of openai/clip-vit-base-patch32
        
        self.model = UserPreferenceModel(input_dim=self.input_dim).to(self.device)
        self._is_trained = False
        
        # Lazy loaded CLIP resources
        self._clip_model = None
        self._clip_processor = None
        
        self.load_model()

    def load_model(self) -> bool:
        """Load model weights if they exist."""
        if os.path.exists(self.model_path):
            try:
                # Load safely on the configured device
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self._is_trained = True
                log.info(f"Loaded user preference model weights from {self.model_path}")
                return True
            except Exception as e:
                log.warning(f"Failed to load user preference model from {self.model_path}: {e}")
        return False

    def save_model(self):
        """Save model weights to disk."""
        try:
            Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), self.model_path)
            self._is_trained = True
            log.info(f"Saved user preference model weights to {self.model_path}")
        except Exception as e:
            log.error(f"Failed to save user preference model: {e}")

    def is_trained(self) -> bool:
        return self._is_trained

    def _lazy_load_clip(self):
        """Lazy load CLIP model on demand to save VRAM/RAM."""
        if self._clip_model is None or self._clip_processor is None:
            from transformers import CLIPProcessor, CLIPModel
            log.info(f"Loading CLIP model for preference features on {self.device}...")
            self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def extract_features(self, image: Image.Image) -> torch.Tensor:
        """Extract normalized CLIP image embeddings."""
        self._lazy_load_clip()
        assert self._clip_processor is not None
        assert self._clip_model is not None
        
        # Preprocess and feed to CLIP
        inputs = self._clip_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            features = self._clip_model.get_image_features(**inputs)
            
        if not isinstance(features, torch.Tensor):
            if hasattr(features, "pooler_output"):
                features = features.pooler_output
            elif isinstance(features, (list, tuple)):
                features = features[0]

        # L2 Normalize the projection embeddings
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features

    def predict(self, image: Image.Image) -> float:
        """
        Predict the user rating score for an image.
        Returns a float between 0.0 and 1.0 (representing 1-5 stars normalized).
        If the model is not trained, returns a default neutral score of 0.5.
        """
        if not self._is_trained:
            return 0.5
            
        try:
            features = self.extract_features(image)
            self.model.eval()
            with torch.no_grad():
                pred = self.model(features)
            return float(pred.item())
        except Exception as e:
            log.warning(f"Error predicting preference score: {e}")
            return 0.5

    def train_from_feedback_file(self, 
                                 feedback_file: str = "outputs/comics/rlhf_feedback.json",
                                 panels_dir: str = "outputs/panels",
                                 epochs: int = 50,
                                 lr: float = 0.01,
                                 min_records: int = 3) -> bool:
        """
        Train/update the preference model based on logged user ratings.
        
        Args:
            feedback_file: Path to rlhf_feedback.json
            panels_dir: Folder containing generated panel images
            epochs: Training epochs
            lr: Learning rate
            min_records: Minimum feedback entries required to train
            
        Returns:
            True if training completed successfully, False otherwise
        """
        if not os.path.exists(feedback_file):
            log.info(f"No feedback file found at {feedback_file}. Skipping training.")
            return False
            
        try:
            with open(feedback_file, "r", encoding="utf-8") as f:
                feedback_data = json.load(f)
        except Exception as e:
            log.warning(f"Could not load feedback file {feedback_file}: {e}")
            return False

        panels_feedback = feedback_data.get("panels", [])
        if len(panels_feedback) < min_records:
            log.info(f"Insufficient feedback records ({len(panels_feedback)}/{min_records}). Skipping training.")
            return False

        log.info(f"Starting preference training on {len(panels_feedback)} panel feedback records...")
        
        training_data: List[Tuple[torch.Tensor, float]] = []
        
        # Build training set
        for entry in panels_feedback:
            panel_id = entry.get("panel_id")
            rating = entry.get("rating")
            if panel_id is None or rating is None:
                continue
                
            # Try to resolve panel image
            img_path = os.path.join(panels_dir, f"panel_{panel_id:03d}_final.png")
            if not os.path.exists(img_path):
                # Try fallback names
                img_path = os.path.join(panels_dir, f"panel_{panel_id}.png")
                if not os.path.exists(img_path):
                    log.warning(f"Could not locate image for panel {panel_id} at {panels_dir}")
                    continue
                    
            try:
                # Load image and extract CLIP features
                image = Image.open(img_path).convert("RGB")
                features = self.extract_features(image) # 1 x 512 shape
                
                # Normalize rating (1 to 5 stars mapped to 0.0 to 1.0)
                norm_rating = (float(rating) - 1.0) / 4.0
                training_data.append((features.squeeze(0), norm_rating))
            except Exception as e:
                log.warning(f"Error loading image or extracting features for panel {panel_id}: {e}")
                
        if len(training_data) < min_records:
            log.warning("Not enough valid image feedback records found. Skipping training.")
            return False
            
        # Run standard training loop
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        criterion = nn.MSELoss()
        
        X = torch.stack([item[0] for item in training_data]).to(self.device)
        y = torch.tensor([item[1] for item in training_data], dtype=torch.float32).unsqueeze(1).to(self.device)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            predictions = self.model(X)
            loss = criterion(predictions, y)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                log.debug(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")
                
        self.save_model()
        log.info(f"Successfully trained user preference critic model on {len(training_data)} samples.")
        return True
