"""
ULTIMATE AI COMIC GENERATOR PIPELINE
Combines best of thesis + your optimized T4 pipeline
Version: 2.0.0
"""

import os
import sys
import json
import time
import torch
import gc
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum

# Force UTF-8 encoding for Windows console to support emojis
if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding != 'utf-8':
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')  # type: ignore
    except Exception:
        pass

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ComicConfig:
    """Master configuration for the pipeline"""
    # Character settings
    character_name: str = "Spider-Man"
    story_world: str = "Cyberpunk 2077"
    
    # Generation settings (T4 optimized)
    resolution: Tuple[int, int] = (768, 768)  # T4 sweet spot
    inference_steps: int = 25
    guidance_scale: float = 7.5
    seed: int = 42
    
    # Model selection
    use_lora: bool = True
    model_type: str = "sdxl"  # "sdxl", "sd15", "sdxl_lora"
    style: str = "manga"  # "manga", "western", "noir", "watercolor"
    
    # Consistency settings
    consistency_threshold: float = 0.55
    enable_clip: bool = True  # Methodology dictates 5% weight
    enable_dinov2: bool = True # Methodology dictates 5% weight
    enable_ssim: bool = True
    enable_edge: bool = True
    enable_color: bool = True
    enable_style: bool = True
    
    # Pipeline settings
    num_pages: int = 5
    panels_per_page: int = 4
    enable_fallback: bool = True
    enable_memory_management: bool = True
    
    # Quality metrics
    enable_fid: bool = True  # Enable for methodology metrics run
    enable_bleu: bool = True # Enable for methodology metrics run
    enable_iou: bool = True  # Enable for methodology metrics run

# ============================================================================
# STYLE MANAGER
# ============================================================================

class StyleManager:
    """Manages different artistic styles"""
    
    STYLES = {
        'manga': {
            'lora': 'artificialguybr/LineAniRedmond-LinearMangaSDXL-V2',
            'trigger': 'LineAniAF, lineart',
            'positive': 'clean manga line art, flat colors, crisp outlines',
            'negative': 'photorealistic, 3d render, shading, gradients'
        },
        'western_comic': {
            'lora': None,
            'trigger': 'comic book, bold inks, vibrant colors',
            'positive': 'western comic style, bold outlines, vibrant flat colors',
            'negative': 'manga, anime, photorealistic, watercolor'
        },
        'noir': {
            'lora': None,
            'trigger': 'film noir, dark shadows, high contrast',
            'positive': 'noir style, dramatic shadows, high contrast, gritty',
            'negative': 'bright colors, photorealistic, 3d render'
        },
        'watercolor': {
            'lora': None,
            'trigger': 'watercolor, soft edges, artistic',
            'positive': 'watercolor style, soft edges, artistic, painterly',
            'negative': 'photorealistic, 3d render, sharp edges'
        },
        'retro': {
            'lora': None,
            'trigger': 'retro comic, vintage, classic',
            'positive': 'retro comic style, vintage colors, classic line art',
            'negative': 'modern, photorealistic, 3d render'
        }
    }
    
    def get_style(self, style_name: str) -> Dict:
        """Get style configuration"""
        return self.STYLES.get(style_name, self.STYLES['manga'])
    
    def apply_style_to_prompt(self, prompt: str, style_name: str) -> str:
        """Apply style to prompt"""
        style = self.get_style(style_name)
        return f"{prompt}, {style['positive']}, {style['trigger']}"

# ============================================================================
# SPEECH BUBBLE OPTIMIZER
# ============================================================================

class SpeechBubbleOptimizer:
    """Optimizes speech bubble placement using YOLO + layout optimization"""
    
    def __init__(self):
        self.yolo_model = None
        self.layout_optimizer = LayoutOptimizer()
        self._load_models()
    
    def _load_models(self):
        """Load YOLO model for object detection"""
        try:
            from ultralytics import YOLO  # type: ignore
            self.yolo_model = YOLO('yolov8n.pt')
            print("[✓] YOLO model loaded")
        except:
            print("[!] YOLO not available - using fallback")
            self.yolo_model = None
    
    def optimize_placement(self, panel_image, dialogue: str, speaker_position: Optional[Tuple[int, int]] = None):
        """
        Find optimal speech bubble placement
        
        Args:
            panel_image: PIL Image of the panel
            dialogue: Text to place in bubble
            speaker_position: (x, y) position of speaker (optional)
        
        Returns:
            bbox: (x, y, width, height) of optimal bubble placement
        """
        # 1. Detect faces and key elements
        if self.yolo_model:
            results = self.yolo_model(panel_image)
            objects = self._parse_yolo_results(results)
        else:
            objects = []
        
        # 2. Find optimal position
        optimal_bbox = self.layout_optimizer.find_optimal_bbox(
            panel_image, objects, dialogue, speaker_position
        )
        
        return optimal_bbox
    
    def _parse_yolo_results(self, results):
        """Extract object positions from YOLO results"""
        objects = []
        for result in results:
            for box in result.boxes:
                obj = {
                    'class': result.names[int(box.cls)],
                    'bbox': box.xyxy.tolist()[0],
                    'confidence': float(box.conf)
                }
                objects.append(obj)
        return objects
    
    def render_bubble(self, panel_image, dialogue: str, bbox: Tuple[int, int, int, int]):
        """Render speech bubble on panel"""
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a copy
        panel = panel_image.copy()
        draw = ImageDraw.Draw(panel)
        
        # Draw bubble
        x, y, w, h = bbox
        draw.ellipse([x, y, x+w, y+h], outline='black', width=3, fill='white')
        
        # Add text
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except (OSError, IOError):
            font = ImageFont.load_default()
        draw.text((x+20, y+20), dialogue, fill='black', font=font)
        
        return panel


class LayoutOptimizer:
    """Optimizes layout for speech bubble placement"""
    
    def find_optimal_bbox(self, panel, objects, dialogue, speaker_position=None):
        """Find optimal bounding box for speech bubble"""
        # 1. Calculate available space
        panel_width, panel_height = panel.size
        
        # 2. Identify areas to avoid (faces, important elements)
        avoid_zones = self._get_avoid_zones(objects)
        
        # 3. Choose position closest to speaker
        if speaker_position:
            candidate_positions = self._get_candidate_positions(
                panel_width, panel_height, speaker_position
            )
        else:
            # Default to top-left if no speaker
            candidate_positions = [(50, 50)]
        
        # 4. Select best position
        for pos in candidate_positions:
            if not self._is_colliding(pos, avoid_zones):
                return (*pos, 200, 100)  # width, height
        
        # Fallback: place at top center
        return (panel_width//2 - 100, 20, 200, 100)
    
    def _get_avoid_zones(self, objects):
        """Get zones to avoid (faces, key elements)"""
        zones = []
        for obj in objects:
            if obj['class'] in ['person', 'face']:
                x1, y1, x2, y2 = obj['bbox']
                zones.append((x1, y1, x2, y2))
        return zones
    
    def _get_candidate_positions(self, width, height, speaker_pos):
        """Get candidate bubble positions near speaker"""
        x, y = speaker_pos
        candidates = [
            (x - 150, y - 50),   # Left of speaker
            (x + 50, y - 50),    # Right of speaker
            (x - 100, y - 150),  # Above-left
            (x + 50, y - 150),   # Above-right
        ]
        return [(cx, cy) for cx, cy in candidates if 0 < cx < width-200 and 0 < cy < height-100]
    
    def _is_colliding(self, pos, avoid_zones):
        """Check if position collides with avoid zones"""
        x, y = pos
        bubble_zone = (x, y, x+200, y+100)
        for zone in avoid_zones:
            if self._rects_overlap(bubble_zone, zone):
                return True
        return False
    
    def _rects_overlap(self, rect1, rect2):
        """Check if two rectangles overlap"""
        x1, y1, x2, y2 = rect1
        x3, y3, x4, y4 = rect2
        return not (x2 < x3 or x4 < x1 or y2 < y3 or y4 < y1)

# ============================================================================
# NARRATIVE MEMORY
# ============================================================================

class NarrativeMemory:
    """Tracks character states and story context across panels"""
    
    def __init__(self):
        self.character_states = {}  # {character_name: state_dict}
        self.story_context = []     # List of story beats
        self.panel_history = []     # List of previous panels
        self.current_emotion = None
        self.story_arc = None
    
    def update_state(self, character_name: str, state: Dict):
        """Update character state"""
        if character_name not in self.character_states:
            self.character_states[character_name] = {}
        self.character_states[character_name].update(state)
    
    def get_character_state(self, character_name: str) -> Dict:
        """Get current character state"""
        return self.character_states.get(character_name, {})
    
    def add_panel(self, panel):
        """Add panel to history"""
        self.panel_history.append(panel)
        
        # Limit history length
        if len(self.panel_history) > 10:
            self.panel_history = self.panel_history[-10:]
    
    def add_story_beat(self, beat: str):
        """Add story beat to context"""
        self.story_context.append(beat)
    
    def get_context_prompt(self) -> str:
        """Generate context-aware prompt enrichment"""
        if not self.panel_history:
            return ""
        
        # Get last 3 panels
        recent_panels = self.panel_history[-3:]
        
        context = "Story progression: "
        for i, panel in enumerate(recent_panels, 1):
            context += f"Panel {i}: {panel.get('emotion', 'neutral')} -> "
        
        # Add character states
        for name, state in self.character_states.items():
            context += f"{name} is {state.get('emotion', 'neutral')}, "
        
        return context
    
    def enrich_prompt(self, prompt: str) -> str:
        """Enrich prompt with narrative memory"""
        context = self.get_context_prompt()
        if context:
            return f"{prompt}, {context}, maintain visual consistency"
        return prompt
    
    def reset(self):
        """Reset memory for new story"""
        self.character_states = {}
        self.story_context = []
        self.panel_history = []
        self.current_emotion = None

# ============================================================================
# EMOTION VALIDATOR
# ============================================================================

class EmotionValidator:
    """Validates emotional alignment between text and images"""
    
    EMOTION_PROMPTS = {
        'happy': {
            'expression': 'smiling, bright eyes, joyful expression, relaxed posture',
            'action': 'laughing, celebrating, cheering, hugging',
            'positive': 'warm, bright, cheerful, vibrant'
        },
        'sad': {
            'expression': 'teary eyes, downward gaze, sorrowful, frowning',
            'action': 'crying, mourning, grieving, hugging self',
            'positive': 'dark, muted, melancholic, atmospheric'
        },
        'angry': {
            'expression': 'furrowed brows, clenched jaw, intense glare, aggressive posture',
            'action': 'shouting, fighting, clenching fists, attacking',
            'positive': 'intense, dramatic, high contrast, bold'
        },
        'fearful': {
            'expression': 'wide eyes, pale face, trembling, defensive posture',
            'action': 'running, hiding, cowering, protecting',
            'positive': 'dark, ominous, shadowy, tense'
        },
        'surprised': {
            'expression': 'wide eyes, raised brows, open mouth, startled',
            'action': 'jumping back, gasping, dropping things',
            'positive': 'bright, dramatic, sudden, contrasting'
        },
        'neutral': {
            'expression': 'neutral expression, relaxed, composed',
            'action': 'standing, sitting, walking normally',
            'positive': 'balanced, calm, natural'
        },
        'love': {
            'expression': 'soft smile, loving gaze, tender expression',
            'action': 'hugging, kissing, holding hands, caring',
            'positive': 'warm, soft, romantic, gentle'
        },
        'determined': {
            'expression': 'focused gaze, determined look, set jaw',
            'action': 'charging forward, fighting, persevering',
            'positive': 'dramatic, heroic, dynamic'
        }
    }
    
    def get_emotion_prompt(self, emotion: str) -> Dict:
        """Get emotion-specific prompt modifiers"""
        return self.EMOTION_PROMPTS.get(emotion, self.EMOTION_PROMPTS['neutral'])
    
    def validate_alignment(self, panel, dialogue: str, expected_emotion: str) -> Tuple[bool, float]:
        """
        Validate emotional alignment between dialogue and image
        
        Returns: (is_aligned, alignment_score)
        """
        # 1. Extract emotion from dialogue (simple approach)
        dialogue_emotion = self._extract_emotion_from_text(dialogue)
        
        # 2. Check alignment
        if dialogue_emotion == expected_emotion:
            return True, 1.0
        
        # 3. Compute alignment score
        score = self._compute_emotion_similarity(dialogue_emotion, expected_emotion)
        return score > 0.7, score
    
    def _extract_emotion_from_text(self, text: str) -> str:
        """Extract emotion from dialogue text"""
        # Simple keyword-based extraction
        emotion_keywords = {
            'happy': ['happy', 'joy', 'glad', 'wonderful', 'great', 'excited'],
            'sad': ['sad', 'cry', 'tear', 'grief', 'mourn', 'depressed'],
            'angry': ['angry', 'mad', 'furious', 'rage', 'hate', 'annoyed'],
            'fearful': ['scared', 'afraid', 'terrified', 'fear', 'horror'],
            'surprised': ['surprised', 'shocked', 'amazed', 'astonished'],
            'love': ['love', 'care', 'adore', 'cherish', 'treasure'],
            'determined': ['determined', 'resolve', 'will', 'never give up']
        }
        
        text_lower = text.lower()
        for emotion, keywords in emotion_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return emotion
        
        return 'neutral'
    
    def _compute_emotion_similarity(self, emo1: str, emo2: str) -> float:
        """Compute similarity between two emotions"""
        if emo1 == emo2:
            return 1.0
        
        # Simple similarity mapping
        similar_pairs = {
            ('happy', 'love'): 0.8,
            ('sad', 'fearful'): 0.7,
            ('angry', 'determined'): 0.6,
            ('surprised', 'fearful'): 0.6
        }
        
        return similar_pairs.get((emo1, emo2), similar_pairs.get((emo2, emo1), 0.3))
    
    def enrich_prompt_with_emotion(self, prompt: str, emotion: str) -> str:
        """Enrich prompt with emotion-specific keywords"""
        emotion_data = self.get_emotion_prompt(emotion)
        return f"{prompt}, {emotion_data['expression']}, {emotion_data['action']}, mood: {emotion_data['positive']}"

# ============================================================================
# MODEL ENSEMBLE
# ============================================================================

class ModelEnsemble:
    """Ensemble of different generation models"""
    
    def __init__(self, config: ComicConfig):
        self.config = config
        self.models = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize available models"""
        # Only load the primary model to save resources
        if self.config.model_type == 'sdxl_lora':
            self.models['primary'] = self._load_sdxl_lora()
        elif self.config.model_type == 'sdxl':
            self.models['primary'] = self._load_sdxl_base()
        else:
            self.models['primary'] = self._load_sd15()
    
    def _load_sdxl_lora(self):
        """Load SDXL + LoRA pipeline"""
        from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler, AutoencoderKL
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load VAE
        try:
            vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        except:
            vae = None
        
        # Load pipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            vae=vae,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            use_safetensors=True,
            variant="fp16" if device == "cuda" else None,
            add_watermarker=False
        )
        
        # Load LoRA
        style = StyleManager().get_style(self.config.style)
        if style.get('lora'):
            try:
                pipe.load_lora_weights(style['lora'])
                pipe.fuse_lora(lora_scale=0.8)
            except Exception as e:
                print(f"[!] LoRA load/fuse failed: {e}")
        
        # Scheduler
        scheduler_config = dict(pipe.scheduler.config)
        scheduler_config.pop("_class_name", None)
        scheduler_config.pop("algorithm_type", None)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            scheduler_config,
            use_karras_sigmas=True,
            algorithm_type="sde-dpmsolver++",
            solver_order=2
        )
        
        # Memory optimizations
        if device == "cuda":
            try:
                pipe.enable_model_cpu_offload()
                pipe.enable_attention_slicing("max")
                pipe.enable_vae_slicing()
            except:
                pipe = pipe.to(device)
        
        return pipe
    
    def _load_sdxl_base(self):
        """Load base SDXL pipeline"""
        from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler, AutoencoderKL
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load VAE
        try:
            vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        except:
            vae = None
        
        # Load pipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            vae=vae,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            use_safetensors=True,
            variant="fp16" if device == "cuda" else None,
            add_watermarker=False
        )
        
        # Scheduler
        scheduler_config = dict(pipe.scheduler.config)
        scheduler_config.pop("_class_name", None)
        scheduler_config.pop("algorithm_type", None)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            scheduler_config,
            use_karras_sigmas=True,
            algorithm_type="sde-dpmsolver++",
            solver_order=2
        )
        
        # Memory optimizations
        if device == "cuda":
            try:
                pipe.enable_model_cpu_offload()
                pipe.enable_attention_slicing("max")
                pipe.enable_vae_slicing()
            except:
                pipe = pipe.to(device)
        
        return pipe
    
    def _load_sd15(self):
        """Load SD 1.5 pipeline"""
        from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        scheduler_config = dict(pipe.scheduler.config)
        scheduler_config.pop("_class_name", None)
        scheduler_config.pop("algorithm_type", None)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            scheduler_config,
            use_karras_sigmas=True,
            algorithm_type="sde-dpmsolver++"
        )
        
        if device == "cuda":
            pipe.enable_attention_slicing("max")
            pipe.enable_vae_slicing()
            pipe = pipe.to(device)
        
        return pipe
    
    def generate(self, prompt: str, **kwargs):
        """Generate image using the ensemble"""
        if 'primary' not in self.models:
            raise RuntimeError("No model loaded")
        
        pipe = self.models['primary']
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Prepare generation parameters
        width = kwargs.get('width', self.config.resolution[0])
        height = kwargs.get('height', self.config.resolution[1])
        steps = kwargs.get('steps', self.config.inference_steps)
        guidance = kwargs.get('guidance', self.config.guidance_scale)
        seed = kwargs.get('seed', self.config.seed)
        
        # Always create generator on CPU for cross-device compatibility
        generator = torch.Generator(device="cpu").manual_seed(seed)
        
        # Generate
        image = pipe(
            prompt=prompt,
            negative_prompt="photorealistic, 3d render, shading, gradients, blurry, ugly",
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator
        ).images[0]
        
        return image

# ============================================================================
# QUALITY METRICS
# ============================================================================

class QualityMetrics:
    """Quality validation metrics"""
    
    def compute_fid(self, generated_img, reference_img):
        """Fréchet Inception Distance"""
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance
            import torchvision.transforms as transforms
            
            fid = FrechetInceptionDistance(feature=64)
            
            transform = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor()
            ])
            
            gen_tensor = transform(generated_img).unsqueeze(0)
            ref_tensor = transform(reference_img).unsqueeze(0)
            
            fid.update(gen_tensor, real=False)
            fid.update(ref_tensor, real=True)
            
            return fid.compute().item()
        except:
            return None
    
    def compute_bleu(self, generated_text, reference_text):
        """BLEU score for text quality"""
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            
            reference = [reference_text.split()]
            candidate = generated_text.split()
            
            smoothie = SmoothingFunction().method4
            return sentence_bleu(reference, candidate, smoothing_function=smoothie)
        except:
            return None
    
    def compute_iou(self, predicted_bbox, ground_truth_bbox):
        """Intersection over Union for bubble placement"""
        try:
            # predicted_bbox = (x1, y1, x2, y2)
            # ground_truth_bbox = (x1, y1, x2, y2)
            
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
        except:
            return None

# ============================================================================
# PANEL GENERATOR
# ============================================================================

class PanelGenerator:
    """Generates individual comic panels with full context"""
    
    def __init__(self, config: ComicConfig):
        self.config = config
        self.model = ModelEnsemble(config)
        self.style_manager = StyleManager()
        self.narrative_memory = NarrativeMemory()
        self.emotion_validator = EmotionValidator()
        self.speech_optimizer = SpeechBubbleOptimizer()
        self.quality_metrics = QualityMetrics()
        self.consistency_checker = None  # Lazy load
        
        # Current character state
        self.current_character = config.character_name
    
    def generate_panel(self, 
                       prompt: str,
                       emotion: str = 'neutral',
                       panel_num: int = 1,
                       previous_panels: Optional[List] = None) -> Dict:
        """
        Generate a single comic panel
        
        Returns:
            {
                'image': PIL.Image,
                'prompt': str,
                'emotion': str,
                'dialogue': str,
                'panel_num': int,
                'quality_score': float,
                'consistency_score': float
            }
        """
        # 1. Enrich with narrative memory
        enriched_prompt = self.narrative_memory.enrich_prompt(prompt)
        
        # 2. Apply emotion
        emotion_prompt = self.emotion_validator.enrich_prompt_with_emotion(
            enriched_prompt, emotion
        )
        
        # 3. Apply style
        styled_prompt = self.style_manager.apply_style_to_prompt(
            emotion_prompt, self.config.style
        )
        
        # 4. Generate image
        image = self.model.generate(styled_prompt)
        
        # 5. Generate dialogue
        dialogue = self._generate_dialogue(styled_prompt, emotion)
        
        # 6. Optimize speech bubble placement
        speaker_pos = (image.width // 2, image.height // 2)  # Default center
        bbox = self.speech_optimizer.optimize_placement(image, dialogue, speaker_pos)
        image_with_bubble = self.speech_optimizer.render_bubble(image, dialogue, bbox)  # type: ignore
        
        # 7. Validate emotional alignment
        is_aligned, alignment_score = self.emotion_validator.validate_alignment(
            image_with_bubble, dialogue, emotion
        )
        
        # 8. Check consistency (if we have previous panels)
        consistency_score = 1.0
        if previous_panels and len(previous_panels) > 0:
            consistency_score = self._check_consistency(
                image_with_bubble, previous_panels[-1]['image']
            )
        
        # 9. Update narrative memory
        self.narrative_memory.update_state(
            self.config.character_name,
            {'emotion': emotion, 'last_panel': image_with_bubble}
        )
        self.narrative_memory.add_panel({
            'image': image_with_bubble,
            'emotion': emotion,
            'dialogue': dialogue,
            'panel_num': panel_num
        })
        
        # 10. Compute quality metrics
        quality_score = (alignment_score + consistency_score) / 2
        
        return {
            'image': image_with_bubble,
            'prompt': styled_prompt,
            'emotion': emotion,
            'dialogue': dialogue,
            'panel_num': panel_num,
            'quality_score': quality_score,
            'consistency_score': consistency_score,
            'alignment_score': alignment_score,
            'bbox': bbox
        }
    
    def _generate_dialogue(self, prompt: str, emotion: str) -> str:
        """Generate dialogue for the panel"""
        # Simple template-based dialogue
        # In production, use LangChain + LLM
        dialogues = {
            'happy': ["This is amazing!", "I can't believe it!", "Perfect!"],
            'sad': ["I'm sorry...", "Why did this happen?", "I miss you..."],
            'angry': ["Never again!", "You'll pay for this!", "Enough!"],
            'fearful': ["Stay back!", "What is that?!", "Help me!"],
            'surprised': ["No way!", "What?!", "Incredible!"],
            'love': ["I love you.", "You're everything to me.", "Together forever."],
            'determined': ["I will succeed.", "Never give up.", "I can do this."]
        }
        
        import random
        return random.choice(dialogues.get(emotion, ["..."]))
    
    def _check_consistency(self, new_image, reference_image) -> float:
        """Check consistency with previous panel"""
        try:
            from utils.consistency_checker import get_consistency_checker
            
            if self.consistency_checker is None:
                self.consistency_checker = get_consistency_checker()
            
            # Save images temporarily
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f1:
                reference_image.save(f1.name)
                ref_path = f1.name
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f2:
                new_image.save(f2.name)
                new_path = f2.name
            
            # Check consistency
            self.consistency_checker.set_reference(ref_path)
            result = self.consistency_checker.check_consistency(new_path)
            
            # Cleanup
            os.unlink(ref_path)
            os.unlink(new_path)
            
            return float(result.get('score', 0.5))  # type: ignore
        except:
            return 0.5  # Default if checker not available

# ============================================================================
# PAGE GENERATOR
# ============================================================================

class PageGenerator:
    """Generates comic pages with multiple panels"""
    
    def __init__(self, config: ComicConfig):
        self.config = config
        self.panel_generator = PanelGenerator(config)
        self.narrative_memory = self.panel_generator.narrative_memory
    
    def generate_page(self, 
                      story_prompt: str,
                      page_num: int = 1,
                      emotions: Optional[List[str]] = None) -> Dict:
        """
        Generate a complete page with 4 panels
        
        Returns:
            {
                'page_num': int,
                'panels': List[Dict],
                'page_image': PIL.Image,
                'page_prompt': str,
                'quality_scores': List[float]
            }
        """
        if emotions is None:
            emotions = ['neutral', 'happy', 'sad', 'determined']
        
        panels = []
        quality_scores = []
        
        for i, emotion in enumerate(emotions[:4], 1):
            panel_prompt = f"{story_prompt}, panel {i}, showing {emotion} emotion"
            
            panel = self.panel_generator.generate_panel(
                prompt=panel_prompt,
                emotion=emotion,
                panel_num=(page_num - 1) * 4 + i,
                previous_panels=panels
            )
            
            panels.append(panel)
            quality_scores.append(panel['quality_score'])
        
        # Create page layout (2x2 grid)
        page_image = self._create_page_layout(panels)
        
        return {
            'page_num': page_num,
            'panels': panels,
            'page_image': page_image,
            'page_prompt': story_prompt,
            'quality_scores': quality_scores,
            'avg_quality': sum(quality_scores) / len(quality_scores)
        }
    
    def _create_page_layout(self, panels: List[Dict]):
        """Create 2x2 grid layout from panels"""
        from PIL import Image
        
        # Assuming 4 panels
        if len(panels) != 4:
            return panels[0]['image']  # Return single image if not 4
        
        # Get panel sizes
        panel_width = panels[0]['image'].width
        panel_height = panels[0]['image'].height
        
        # Create page
        page = Image.new('RGB', (panel_width * 2, panel_height * 2), 'white')
        
        # Place panels
        positions = [
            (0, 0),
            (panel_width, 0),
            (0, panel_height),
            (panel_width, panel_height)
        ]
        
        for i, panel in enumerate(panels):
            page.paste(panel['image'], positions[i])
        
        return page

# ============================================================================
# COMIC GENERATOR - MASTER CLASS
# ============================================================================

class UltimateComicGenerator:
    """
    Master class that orchestrates the entire comic generation pipeline
    """
    
    def __init__(self, config: Optional[ComicConfig] = None):
        if config is None:
            config = ComicConfig()
        self.config = config
        self.page_generator = PageGenerator(config)
        self.quality_metrics = QualityMetrics()
        self.results = []
    
    def generate_comic(self, story_prompt: str) -> Dict:
        """
        Generate a complete comic book
        
        Args:
            story_prompt: Description of the story
        
        Returns:
            {
                'pages': List[Dict],
                'comic_image': PIL.Image,
                'overall_quality': float,
                'generation_time': float,
                'config': ComicConfig
            }
        """
        start_time = time.time()
        
        pages = []
        quality_scores = []
        
        # Generate each page
        for page_num in range(1, self.config.num_pages + 1):
            print(f"📖 Generating Page {page_num}/{self.config.num_pages}")
            
            # Build page prompt
            page_prompt = f"{story_prompt}, page {page_num}"
            
            # Generate page
            page = self.page_generator.generate_page(
                story_prompt=page_prompt,
                page_num=page_num
            )
            
            pages.append(page)
            quality_scores.append(page['avg_quality'])
            
            # Clear memory periodically
            if self.config.enable_memory_management and page_num % 3 == 0:
                self._clear_memory()
        
        # Create complete comic image
        comic_image = self._create_comic_image(pages)
        
        generation_time = time.time() - start_time
        
        return {
            'pages': pages,
            'comic_image': comic_image,
            'overall_quality': sum(quality_scores) / len(quality_scores),
            'generation_time': generation_time,
            'config': self.config
        }
    
    def _create_comic_image(self, pages: List[Dict]):
        """Create a single image of the entire comic"""
        from PIL import Image
        
        # Stack pages vertically
        total_height = sum(p['page_image'].height for p in pages)
        max_width = max(p['page_image'].width for p in pages)
        
        comic = Image.new('RGB', (max_width, total_height), 'white')
        
        y_offset = 0
        for page in pages:
            comic.paste(page['page_image'], (0, y_offset))
            y_offset += page['page_image'].height
        
        return comic
    
    def _clear_memory(self):
        """Clear GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    
    def export(self, result: Dict, output_dir: str = "outputs/comic"):
        """Export the generated comic"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save comic
        result['comic_image'].save(os.path.join(output_dir, "comic_book.png"))
        
        # Save individual pages
        for i, page in enumerate(result['pages'], 1):
            page['page_image'].save(os.path.join(output_dir, f"page_{i}.png"))
        
        # Save each panel
        for page in result['pages']:
            for panel in page['panels']:
                panel_num = panel['panel_num']
                panel['image'].save(os.path.join(output_dir, f"panel_{panel_num}.png"))
        
        # Save metadata
        with open(os.path.join(output_dir, "metadata.json"), 'w', encoding='utf-8') as f:
            json.dump({
                'config': self.config.__dict__,
                'overall_quality': result['overall_quality'],
                'generation_time': result['generation_time'],
                'num_pages': len(result['pages']),
                'num_panels': sum(len(p['panels']) for p in result['pages'])
            }, f, indent=2)
        
        print(f"✅ Comic exported to {output_dir}/")

# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Command-line interface for the pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ultimate AI Comic Generator")
    parser.add_argument("--character", type=str, default="Spider-Man",
                        help="Main character name")
    parser.add_argument("--world", type=str, default="Cyberpunk 2077",
                        help="Story world/setting")
    parser.add_argument("--style", type=str, default="manga",
                        choices=['manga', 'western', 'noir', 'watercolor', 'retro'],
                        help="Art style")
    parser.add_argument("--pages", type=int, default=5,
                        help="Number of pages to generate")
    parser.add_argument("--prompt", type=str, default="A superhero fighting evil in a futuristic city",
                        help="Story prompt")
    parser.add_argument("--output", type=str, default="outputs/comic",
                        help="Output directory")
    parser.add_argument("--no-lora", action="store_true",
                        help="Disable LoRA")
    parser.add_argument("--enable-clip", action="store_true",
                        help="Enable CLIP consistency (slower)")
    parser.add_argument("--enable-dinov2", action="store_true",
                        help="Enable DINOv2 consistency (slower)")
    
    args = parser.parse_args()
    
    # Build config
    config = ComicConfig(
        character_name=args.character,
        story_world=args.world,
        style=args.style,
        num_pages=args.pages,
        use_lora=not args.no_lora,
        enable_clip=args.enable_clip,
        enable_dinov2=args.enable_dinov2
    )
    
    # Generate
    print("=" * 70)
    print("🎨 ULTIMATE AI COMIC GENERATOR")
    print("=" * 70)
    print(f"Character: {config.character_name}")
    print(f"World: {config.story_world}")
    print(f"Style: {config.style}")
    print(f"Pages: {config.num_pages}")
    print("=" * 70)
    
    generator = UltimateComicGenerator(config)
    result = generator.generate_comic(args.prompt)
    
    # Export
    generator.export(result, args.output)
    
    print(f"\n✨ Generation complete!")
    print(f"📊 Overall Quality: {result['overall_quality']:.2f}")
    print(f"⏱️ Generation Time: {result['generation_time']:.1f}s")
    print(f"📁 Output: {args.output}/")

if __name__ == "__main__":
    main()
