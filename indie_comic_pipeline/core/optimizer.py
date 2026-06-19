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
            - quality_critic_threshold: suggested pass/fail threshold
            - aesthetic_weight: recommended critic weight for aesthetics
            - consistency_weight: recommended critic weight for consistency
            - lora_scale_adjustment: recommended LoRA scaling modifier
        """
        summary = self.feedback_loop.get_feedback_summary()
        avg_rating = summary.get("average_panel_rating", 0.0)
        total_rated = summary.get("total_panels_rated", 0)
        
        adjustments = {
            "quality_critic_threshold_delta": 0.0,
            "lora_scale_adjustment": 0.0,
            "guidance_scale_adjustment": 0.0,
            "critic_weight_shifts": {}
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
            
        return adjustments
