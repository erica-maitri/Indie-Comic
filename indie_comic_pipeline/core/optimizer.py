"""
SYSTEM BACKPROPAGATION OPTIMIZER — Phase 8
==========================================
Analyzes RLHF feedback logs and automatically computes settings 
optimizations for prompt tuning, quality critic weights, layout gutters,
and backend thresholds.
"""

import os
import json
import logging
from typing import Dict, Any, List
from core.feedback import RLHFFeedbackLoop

log = logging.getLogger("pipeline.optimizer")


class SystemOptimizer:
    """
    Phase 8: System Backpropagation Optimizer.
    
    Reads RLHF telemetry logs and calculates parameter adjustments to backpropagate
    user style and quality preferences into the generator's active configuration.
    """

    def __init__(self, feedback_loop: RLHFFeedbackLoop, 
                 settings_path: str = "config/settings.yaml"):
        self.feedback_loop = feedback_loop
        self.settings_path = settings_path

    def optimize_system_parameters(self) -> Dict[str, Any]:
        """
        Analyze accumulated RLHF feedback and compute parameter adjustments.
        
        Returns:
            Dict containing parameter delta suggestions:
            - quality_critic_threshold_delta: suggested pass/fail threshold
            - lora_scale_adjustment: recommended LoRA scaling modifier
            - guidance_scale_adjustment: recommended guidance scale modifier
            - critic_weight_shifts: recommended quality critic weight shifts
            - positive_terms_to_add: style keywords to add to positive prompts
            - negative_terms_to_add: style keywords to add to negative prompts
        """
        summary = self.feedback_loop.get_feedback_summary()
        avg_rating = summary.get("average_panel_rating", 0.0)
        total_rated = summary.get("total_panels_rated", 0)
        
        adjustments = {
            "quality_critic_threshold_delta": 0.0,
            "lora_scale_adjustment": 0.0,
            "guidance_scale_adjustment": 0.0,
            "critic_weight_shifts": {},
            "positive_terms_to_add": [],
            "negative_terms_to_add": []
        }
        
        if total_rated < 3:
            log.info("System Optimizer: Insufficient feedback records to backpropagate optimizations.")
            return adjustments
            
        log.info(f"System Optimizer: Analyzing {total_rated} panels. Average rating = {avg_rating}/5.0")
        
        # 1. If average rating is low, tighten quality critic parameters to enforce retries
        if avg_rating < 3.0:
            log.info("  Low overall ratings detected. Enforcing stricter critic thresholds.")
            adjustments["quality_critic_threshold_delta"] = +0.05
            adjustments["guidance_scale_adjustment"] = +0.5
        elif avg_rating > 4.5:
            # High ratings; can relax threshold slightly to speed up generation (fewer regenerations)
            log.info("  Excellent overall ratings. Relaxing critic thresholds to accelerate throughput.")
            adjustments["quality_critic_threshold_delta"] = -0.03
            
        # 2. Analyze qualitative comments for specific keywords to adjust weights
        panels = self.feedback_loop.data.get("panels", [])
        consistency_complaints = 0
        detail_complaints = 0
        clutter_complaints = 0
        
        for r in panels:
            comment = r.get("comment", "").lower()
            rating = r.get("rating", 5)
            
            if rating <= 3:
                if any(k in comment for k in ["consistency", "face", "weird character", "looks different", "wardrobe"]):
                    consistency_complaints += 1
                if any(k in comment for k in ["aesthetic", "low quality", "blurry", "ugly", "weird colors"]):
                    detail_complaints += 1
                if any(k in comment for k in ["bubble", "read", "text", "clutter", "crowded", "overlap"]):
                    clutter_complaints += 1
                    
        total_complaints = consistency_complaints + detail_complaints + clutter_complaints
        if total_complaints > 0:
            log.info(f"  Feedback breakdown: Consistency={consistency_complaints}, Quality={detail_complaints}, Readability={clutter_complaints}")
            
            # Suggest shifting quality critic weights towards problematic areas
            shifts = {}
            if consistency_complaints > detail_complaints and consistency_complaints > clutter_complaints:
                shifts["visual_consistency"] = +0.05
                shifts["aesthetic_quality"] = -0.02
                shifts["readability"] = -0.03
                adjustments["lora_scale_adjustment"] = +0.05  # slightly higher LoRA to lock identity
            elif detail_complaints > consistency_complaints and detail_complaints > clutter_complaints:
                shifts["aesthetic_quality"] = +0.05
                shifts["visual_consistency"] = -0.02
                shifts["readability"] = -0.03
            elif clutter_complaints > consistency_complaints:
                shifts["readability"] = +0.05
                shifts["visual_consistency"] = -0.02
                shifts["aesthetic_quality"] = -0.03
                
            adjustments["critic_weight_shifts"] = shifts

        # 3. Analyze qualitative comments for template tuning
        pos_to_add = []
        neg_to_add = []
        if detail_complaints > 0:
            pos_to_add.extend(["sharp focus", "detailed line art", "vibrant colors"])
            neg_to_add.extend(["blurry", "low quality", "noisy"])
        if consistency_complaints > 0:
            pos_to_add.extend(["consistent character features", "same outfit"])
        if clutter_complaints > 0:
            pos_to_add.extend(["clean background", "uncluttered"])
            neg_to_add.extend(["cluttered", "messy"])
            
        adjustments["positive_terms_to_add"] = list(set(pos_to_add))
        adjustments["negative_terms_to_add"] = list(set(neg_to_add))
            
        return adjustments

    def apply_optimizations(self, adjustments: Dict[str, Any] = None) -> bool:
        """
        Applies calculated adjustments directly back to config/settings.yaml.
        
        Args:
            adjustments: Optional pre-calculated adjustments dictionary.
                         If None, calculate from current feedback.
                         
        Returns:
            bool: True if settings file was modified and saved successfully, False otherwise.
        """
        if adjustments is None:
            adjustments = self.optimize_system_parameters()
            
        if not os.path.exists(self.settings_path):
            log.warning(f"Settings file not found at {self.settings_path}. Cannot apply optimizations.")
            return False
            
        try:
            import yaml
            with open(self.settings_path, "r", encoding="utf-8") as f:
                settings = yaml.safe_load(f) or {}
                
            modified = False
            
            # Apply quality threshold
            threshold_delta = adjustments.get("quality_critic_threshold_delta", 0.0)
            if threshold_delta != 0.0:
                if "quality_critic" not in settings:
                    settings["quality_critic"] = {}
                q_critic = settings["quality_critic"]
                
                old_thresh = q_critic.get("threshold", 0.55)
                new_thresh = max(0.1, min(0.95, old_thresh + threshold_delta))
                q_critic["threshold"] = round(new_thresh, 3)
                
                old_strict = q_critic.get("strict_threshold", 0.70)
                new_strict = max(0.1, min(0.95, old_strict + threshold_delta))
                if new_strict < new_thresh:
                    new_strict = new_thresh
                q_critic["strict_threshold"] = round(new_strict, 3)
                
                # Also update consistency threshold if present
                if "consistency" in settings:
                    settings["consistency"]["threshold"] = round(new_thresh, 3)
                    settings["consistency"]["strict_threshold"] = round(new_strict, 3)
                
                log.info(f"Updated quality critic threshold from {old_thresh} to {new_thresh}")
                modified = True
                
            # Apply guidance scale
            guidance_delta = adjustments.get("guidance_scale_adjustment", 0.0)
            if guidance_delta != 0.0:
                if "generation" not in settings:
                    settings["generation"] = {}
                old_guidance = settings["generation"].get("guidance_scale", 7.5)
                new_guidance = max(1.0, min(15.0, old_guidance + guidance_delta))
                settings["generation"]["guidance_scale"] = round(new_guidance, 2)
                log.info(f"Updated guidance scale from {old_guidance} to {new_guidance}")
                modified = True
                
            # Apply LoRA adapter scale
            lora_delta = adjustments.get("lora_scale_adjustment", 0.0)
            if lora_delta != 0.0:
                if "models" not in settings:
                    settings["models"] = {}
                if "lora" not in settings["models"]:
                    settings["models"]["lora"] = {}
                old_lora = settings["models"]["lora"].get("adapter_scale", 0.8)
                new_lora = max(0.0, min(1.5, old_lora + lora_delta))
                settings["models"]["lora"]["adapter_scale"] = round(new_lora, 2)
                log.info(f"Updated LoRA adapter scale from {old_lora} to {new_lora}")
                modified = True
                
            # Apply critic weight shifts
            shifts = adjustments.get("critic_weight_shifts", {})
            if shifts:
                if "quality_critic" not in settings:
                    settings["quality_critic"] = {}
                if "weights" not in settings["quality_critic"]:
                    settings["quality_critic"]["weights"] = {}
                weights = settings["quality_critic"]["weights"]
                
                for k, shift in shifts.items():
                    if k in weights:
                        weights[k] = max(0.0, min(1.0, weights[k] + shift))
                
                # Normalize weights to sum to 1.0
                total = sum(weights.values())
                if total > 0:
                    for k in weights:
                        weights[k] = round(weights[k] / total, 3)
                log.info(f"Updated quality critic weights: {weights}")
                modified = True
                
            # Mutate style positive/negative terms
            pos_add = adjustments.get("positive_terms_to_add", [])
            neg_add = adjustments.get("negative_terms_to_add", [])
            
            if pos_add or neg_add:
                if "style" not in settings:
                    settings["style"] = {}
                
                if pos_add:
                    if "positive_terms" not in settings["style"]:
                        settings["style"]["positive_terms"] = []
                    pos_terms = settings["style"]["positive_terms"]
                    added_any = False
                    for term in pos_add:
                        if term not in pos_terms:
                            pos_terms.append(term)
                            added_any = True
                    if added_any:
                        log.info(f"Added positive style terms: {pos_add}")
                        modified = True
                        
                if neg_add:
                    if "negative_terms" not in settings["style"]:
                        settings["style"]["negative_terms"] = []
                    neg_terms = settings["style"]["negative_terms"]
                    added_any = False
                    for term in neg_add:
                        if term not in neg_terms:
                            neg_terms.append(term)
                            added_any = True
                    if added_any:
                        log.info(f"Added negative style terms: {neg_add}")
                        modified = True
            
            if modified:
                with open(self.settings_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(settings, f, default_flow_style=False, sort_keys=False)
                log.info(f"Successfully saved mutated settings back to {self.settings_path}")
                return True
                
            return False
        except Exception as e:
            log.error(f"Error applying optimizations to {self.settings_path}: {e}")
            return False
