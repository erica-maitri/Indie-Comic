"""
COMIC CRITIC PIPELINE — Phase 6
==================================
Evaluates generated panels across 5 quality dimensions and implements
the reject → regenerate loop for quality-gated output.

Dimensions:
1. Visual Consistency (identity preservation)
2. Emotional Engagement (text-image emotion alignment)
3. Narrative Coherence (story flow continuity)
4. Aesthetic Quality (visual quality score)
5. Readability (bubble placement, text clarity)
"""

import logging
from typing import Dict, Any, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from core.memory import StorySectionMemory

log = logging.getLogger("pipeline.quality_critic")


class QualityCritic:
    """
    Phase 6: COMIC Critic Pipeline.

    Evaluates panels across 5 dimensions and provides a composite quality score.
    If a panel falls below the threshold, it recommends re-generation with
    adjusted parameters.
    """

    def __init__(self, threshold: float = 0.55,
                 strict_threshold: float = 0.70,
                 max_retries: int = 2,
                 weights: Optional[Dict[str, float]] = None,
                 user_pref_model_path: str = "outputs/user_preference_model.pt"):
        self.threshold = threshold
        self.strict_threshold = strict_threshold
        self.max_retries = max_retries
        self.user_pref_model_path = user_pref_model_path
        self.dry_run = False

        # Dimension weights (must sum to 1.0)
        self.weights = weights or {
            "visual_consistency": 0.30,
            "emotional_engagement": 0.15,
            "narrative_coherence": 0.20,
            "aesthetic_quality": 0.25,
            "readability": 0.10,
        }

        self._consistency_checker = None
        self._user_pref_critic = None

    def _get_user_pref_critic(self):
        """Lazy-load user preference critic."""
        if self._user_pref_critic is None:
            try:
                from core.user_preference_critic import UserPreferenceCritic
                self._user_pref_critic = UserPreferenceCritic(model_path=self.user_pref_model_path)
            except Exception as e:
                log.warning(f"Failed to load user preference critic: {e}")
        return self._user_pref_critic

    def evaluate(self, panel_result: Dict[str, Any],
                 memory: "StorySectionMemory") -> Dict[str, Any]:
        """
        Evaluate a generated panel across all 5 dimensions.

        Args:
            panel_result: Result dict from PanelEngine.generate_panel()
            memory: Story Section Memory blackboard

        Returns:
            Evaluation dict with dimension scores, composite score,
            pass/fail verdict, and adjustment recommendations
        """
        panel_id = panel_result.get("panel_id", 0)

        scores = {
            "visual_consistency": self._eval_visual_consistency(panel_result, memory),
            "emotional_engagement": self._eval_emotional_engagement(panel_result, memory),
            "narrative_coherence": self._eval_narrative_coherence(panel_result, memory),
            "aesthetic_quality": self._eval_aesthetic_quality(panel_result),
            "readability": self._eval_readability(panel_result),
        }

        # Lazy evaluate user preference score if model is trained
        user_pref_critic = self._get_user_pref_critic()
        if user_pref_critic and user_pref_critic.is_trained():
            image = panel_result.get("image")
            if image:
                scores["user_preference"] = user_pref_critic.predict(image)

        # Compute weighted composite score (handling dynamic user_preference weight redistribution)
        current_weights = dict(self.weights)
        if "user_preference" in scores:
            current_weights["user_preference"] = 0.20
            sum_other = sum(self.weights.values())
            scale = 0.80 / sum_other if sum_other > 0 else 1.0
            for k in self.weights:
                current_weights[k] = self.weights[k] * scale

        composite = sum(
            scores[dim] * current_weights.get(dim, 0.2)
            for dim in scores if dim in current_weights
        )

        # Determine verdict
        if composite >= self.strict_threshold:
            verdict = "excellent"
        elif composite >= self.threshold:
            verdict = "pass"
        else:
            verdict = "fail"

        # Build adjustment recommendations for re-generation
        adjustments = {}
        if verdict == "fail":
            adjustments = self._compute_adjustments(scores, panel_result)

        evaluation = {
            "panel_id": panel_id,
            "scores": scores,
            "composite_score": round(composite, 4),
            "verdict": verdict,
            "adjustments": adjustments,
        }

        log_msg = (
            f"  Critic evaluation panel {panel_id}: "
            f"composite={composite:.3f} [{verdict}] "
            f"(vis={scores['visual_consistency']:.2f}, "
            f"emo={scores['emotional_engagement']:.2f}, "
            f"nar={scores['narrative_coherence']:.2f}, "
            f"aes={scores['aesthetic_quality']:.2f}, "
            f"read={scores['readability']:.2f}"
        )
        if "user_preference" in scores:
            log_msg += f", pref={scores['user_preference']:.2f})"
        else:
            log_msg += ")"
        log.info(log_msg)

        return evaluation

    def should_regenerate(self, evaluation: Dict[str, Any]) -> bool:
        """Check if the panel should be regenerated."""
        return evaluation.get("verdict") == "fail"


    # ─────────────────────────────────────────────────────────────────────
    # Dimension Evaluators
    # ─────────────────────────────────────────────────────────────────────

    def _eval_visual_consistency(self, panel_result: Dict[str, Any],
                                 memory: "StorySectionMemory") -> float:
        """
        Evaluate visual consistency with anchor/previous panels.
        Uses the consistency checker if available, otherwise heuristic.
        """
        # If no anchor established yet (panel 1), perfect score
        if not memory.anchor_panel_id:
            return 0.85

        image_path = panel_result.get("image_path", "")
        anchor_features = memory.get_anchor_features()

        if anchor_features and image_path:
            checker = self._get_consistency_checker()
            if checker:
                try:
                    ref_path = anchor_features.get("reference_path")
                    if ref_path:
                        current_ref = getattr(checker, 'reference_features', None)
                        if current_ref is None or current_ref.get('path') != ref_path:
                            checker.set_reference(ref_path)
                    res = checker.check_consistency(image_path)
                    score = res.get('score', 0.0) if isinstance(res, dict) else res
                    return max(0.0, min(1.0, score))
                except Exception as e:
                    log.warning(f"Consistency check failed: {e}")

        # Heuristic fallback based on generation parameters
        weights = panel_result.get("weights", {})
        lora_scale = weights.get("lora_scale", 0.8)
        # Higher LoRA = more style consistency
        return 0.5 + lora_scale * 0.3

    def _eval_emotional_engagement(self, panel_result: Dict[str, Any],
                                   memory: "StorySectionMemory") -> float:
        """
        Evaluate emotional alignment between the generated image
        and the intended emotion beat.
        """
        # Heuristic: check if the prompt contains emotion-relevant terms
        prompt = panel_result.get("prompt", "").lower()

        # Strong emotions are more engaging
        context_panel = None
        for p in memory.panel_history:
            if p.panel_id == panel_result.get("panel_id"):
                context_panel = p
                break

        if context_panel:
            emotion = context_panel.emotion
            high_engagement = {
                "contained_fire", "fracture", "breakthrough", "triumph",
                "overflow", "spark", "momentum", "ache",
            }
            if emotion in high_engagement:
                return 0.8
            return 0.65

        return 0.6  # Default moderate engagement

    def _eval_narrative_coherence(self, panel_result: Dict[str, Any],
                                  memory: "StorySectionMemory") -> float:
        """
        Evaluate narrative flow continuity.
        Checks if the panel logically follows from recent panels.
        """
        panel_id = panel_result.get("panel_id", 0)
        recent = memory.get_recent_panels(3)

        if not recent or panel_id == 1:
            return 0.8  # First panel: good baseline

        # Check for emotional arc continuity
        current_beat_idx = memory.current_beat_index
        total_beats = len(memory.arc_beats)

        if total_beats > 0:
            progress = current_beat_idx / total_beats
            # Coherence is high when we're following the planned arc
            return 0.65 + 0.25 * (1.0 - abs(progress - panel_id / memory.total_panels))

        return 0.65

    def _eval_aesthetic_quality(self, panel_result: Dict[str, Any]) -> float:
        """
        Evaluate visual quality of the generated image.
        """
        image = panel_result.get("image")
        if image is None:
            return 0.5

        # Basic image quality heuristics
        w, h = image.size

        # Resolution quality
        resolution_score = min(1.0, (w * h) / (1024 * 1024))

        # Variance check (all-black or all-white images score low)
        try:
            import numpy as np
            arr = np.array(image)
            variance = arr.std() / 128.0  # Normalize
            variance_score = min(1.0, variance)
        except ImportError:
            variance_score = 0.7

        return 0.3 * resolution_score + 0.7 * variance_score

    def _eval_readability(self, panel_result: Dict[str, Any]) -> float:
        """
        Evaluate readability: does the panel have clear composition?
        """
        # Heuristic: based on image complexity and generation quality
        image = panel_result.get("image")
        if image is None:
            return 0.5

        try:
            import numpy as np
            arr = np.array(image.convert("L"))
            # Edge density as a proxy for readability
            # Too many edges = cluttered, too few = empty
            gradient_x = np.abs(np.diff(arr.astype(float), axis=1))
            gradient_y = np.abs(np.diff(arr.astype(float), axis=0))
            edge_density = (gradient_x.mean() + gradient_y.mean()) / 2.0 / 128.0

            # Sweet spot: moderate edge density
            if 0.05 < edge_density < 0.3:
                return 0.8
            elif 0.02 < edge_density < 0.5:
                return 0.6
            else:
                return 0.4
        except ImportError:
            return 0.65

    # ─────────────────────────────────────────────────────────────────────
    # Adjustment Recommendations
    # ─────────────────────────────────────────────────────────────────────

    def _compute_adjustments(self, scores: Dict[str, float],
                             panel_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute parameter adjustments to improve a failed panel.
        """
        adjustments = {}

        # Low visual consistency → increase guidance
        if scores.get("visual_consistency", 1.0) < 0.5:
            adjustments["guidance_scale_delta"] = +1.0
            adjustments["prompt_append"] = "consistent character design, same art style"

        # Low aesthetic quality → increase steps
        if scores.get("aesthetic_quality", 1.0) < 0.5:
            adjustments["steps_delta"] = +5
            adjustments["prompt_append"] = adjustments.get("prompt_append", "") + ", highly detailed, sharp lines"

        # Low readability → simplify
        if scores.get("readability", 1.0) < 0.4:
            adjustments["negative_append"] = "cluttered, busy background, too many details"

        # Low emotional engagement → emphasize emotion
        if scores.get("emotional_engagement", 1.0) < 0.4:
            adjustments["prompt_append"] = adjustments.get("prompt_append", "") + ", expressive emotion, dramatic"

        return adjustments

    def _get_consistency_checker(self):
        """Lazy-load consistency checker."""
        if self._consistency_checker is None:
            try:
                from utils.consistency_checker import ConsistencyChecker
                self._consistency_checker = ConsistencyChecker()
            except ImportError:
                pass
        return self._consistency_checker
