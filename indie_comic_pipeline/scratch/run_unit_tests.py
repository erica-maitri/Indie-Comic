import os
import sys
import time
import shutil
import tempfile
import unittest
from PIL import Image
from pathlib import Path

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.memory import StorySectionMemory, CharacterState, SceneState, PanelRecord, LayoutDirective
from core.agents.agent_coordinator import AgentCoordinator
from core.advanced_attention import AdvancedAttentionManager
from core.quality_critic import QualityCritic
from core.layout_engine import MangaFlowLayoutEngine
from core.feedback import RLHFFeedbackLoop
from core.optimizer import SystemOptimizer
from comic_exporter import ComicExporter


class TestComicPipeline(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.memory = StorySectionMemory()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_memory_blackboard(self):
        """Test StorySectionMemory registration, serialisation, and retrieval."""
        print("\nTesting StorySectionMemory...")
        
        # Test character registration
        char = self.memory.register_character("Hero", costume_desc="Red cape", emotion="determined")
        self.assertEqual(char.name, "Hero")
        self.assertEqual(char.costume_desc, "Red cape")
        
        # Test character update
        self.memory.update_character("Hero", emotion="triumphant", last_action="flew")
        hero = self.memory.get_character("Hero")
        self.assertEqual(hero.emotion, "triumphant")
        self.assertEqual(hero.last_action, "flew")

        # Test scene update
        self.memory.update_scene(location="Neo-Tokyo", weather="neon-rain", lighting="dim")
        scene = self.memory.get_scene()
        self.assertEqual(scene.location, "Neo-Tokyo")
        self.assertEqual(scene.weather, "neon-rain")
        
        # Test layout directives
        directive = LayoutDirective(panel_id=1, size_class="large", camera_angle="close_up")
        self.memory.set_layout_directive(1, directive)
        retrieved_directive = self.memory.get_layout_directive(1)
        self.assertEqual(retrieved_directive.size_class, "large")

        # Test serialisation & deserialisation
        checkpoint_path = os.path.join(self.tmp_dir, "checkpoint.json")
        self.memory.save_checkpoint(checkpoint_path)
        
        new_memory = StorySectionMemory.load_checkpoint(checkpoint_path)
        self.assertEqual(new_memory.get_character("Hero").costume_desc, "Red cape")
        self.assertEqual(new_memory.get_scene().location, "Neo-Tokyo")
        self.assertEqual(new_memory.get_layout_directive(1).camera_angle, "close_up")
        print("  [PASSED] StorySectionMemory tests")

    def test_agent_coordinator(self):
        """Test Multi-Agent Coordinator and beat generation."""
        print("Testing AgentCoordinator...")
        coordinator = AgentCoordinator(self.memory)
        
        # Mock Story Intake output
        story_config = {
            "title": "Neon Sunset",
            "characters": [{"name": "Akira", "costume": "Leather jacket"}],
            "setting": {"location": "Mega-city alleyway", "lighting": "cyberpunk pink"},
            "mood_journey": "despair to hope",
            "recurring_motif": "broken circuit",
            "panels": [
                {"panel": 1, "visual": "Akira looking at the sky", "dialogue": "Is there anyone out there?", "emotion_beat": "lonely"},
                {"panel": 2, "visual": "A drone lights up the alley", "dialogue": "Intruder detected.", "emotion_beat": "startled"},
            ],
            "_metadata": {
                "character": "Akira",
                "world": "Mega-city",
                "emotion": "lonely"
            }
        }
        
        coordinator.run_planning(story_config)
        
        # Check if memory has been populated by coordinator planning
        self.assertEqual(self.memory.total_panels, 2)
        self.assertEqual(self.memory.recurring_motif, "broken_circuit" if "broken_circuit" in self.memory.recurring_motif else "broken circuit")
        self.assertIn("Akira", self.memory.characters)
        self.assertEqual(self.memory.characters["Akira"].costume_desc, "Leather jacket")
        
        # Check context retrieval for generation
        context = coordinator.get_generation_context(1)
        self.assertEqual(context["panel_id"], 1)
        self.assertEqual(context["panel_emotion_beat"], "lonely")
        self.assertEqual(context["panel_dialogue"], "Is there anyone out there?")
        print("  [PASSED] AgentCoordinator tests")

    def test_advanced_attention_manager(self):
        """Test L1-Heat, L2-Attn, L3-STE lifecycle management."""
        print("Testing AdvancedAttentionManager...")
        manager = AdvancedAttentionManager(enabled=True)
        self.assertTrue(manager.enabled)
        
        status = manager.get_status()
        self.assertEqual(status["L1_heat_diffusion"]["alpha"], 0.03)
        self.assertEqual(status["L2_attention_cache"]["layers_cached"], 0)
        self.assertFalse(status["L3_spatiotemporal"]["anchor_captured"])
        
        # Test panel start/end cycle
        manager.on_panel_start(1, is_anchor=True)
        manager.on_panel_end()
        
        status = manager.get_status()
        self.assertIsNotNone(status)
        print("  [PASSED] AdvancedAttentionManager tests")

    def test_quality_critic(self):
        """Test Phase 6 Quality Critic scoring and auto-adjustments."""
        print("Testing QualityCritic...")
        critic = QualityCritic(threshold=0.6, strict_threshold=0.8)
        
        # Mock panel result
        img = Image.new("RGB", (512, 512), (128, 128, 128))
        panel_result = {
            "panel_id": 2,
            "image": img,
            "image_path": os.path.join(self.tmp_dir, "temp_panel.png"),
            "prompt": "An amazing superhero flying through the neon skies",
            "weights": {"lora_scale": 0.8}
        }
        img.save(panel_result["image_path"])
        
        # Set anchor in memory so it attempts visual consistency check
        anchor_img = Image.new("RGB", (512, 512), (100, 100, 100))
        anchor_path = os.path.join(self.tmp_dir, "anchor_panel.png")
        anchor_img.save(anchor_path)
        
        self.memory.set_anchor(1, {"reference_path": anchor_path, "mean_brightness": 128, "aesthetic_score": 0.7})
        
        # Run evaluation
        evaluation = critic.evaluate(panel_result, self.memory)
        
        self.assertEqual(evaluation["panel_id"], 2)
        self.assertIn("scores", evaluation)
        self.assertIn("composite_score", evaluation)
        self.assertIn("verdict", evaluation)
        
        self.assertIn(evaluation["verdict"], ["excellent", "pass", "fail"])
        print("  [PASSED] QualityCritic tests")

    def test_mangaflow_layout_engine(self):
        """Test Phase 7 page layout and geometry constraints."""
        print("Testing MangaFlowLayoutEngine...")
        engine = MangaFlowLayoutEngine(page_width=800, page_height=1200)
        
        # Generate 3 dummy panel images
        img1 = Image.new("RGB", (400, 300), (255, 0, 0))
        img2 = Image.new("RGB", (400, 300), (0, 255, 0))
        img3 = Image.new("RGB", (800, 400), (0, 0, 255))
        
        panels = [
            {"panel_id": 1, "image": img1, "page_num": 1},
            {"panel_id": 2, "image": img2, "page_num": 1},
            {"panel_id": 3, "image": img3, "page_num": 1},
        ]
        
        page_image = engine.layout_page(panels, 1)
        self.assertEqual(page_image.size, (800, 1200))
        print("  [PASSED] MangaFlowLayoutEngine tests")

    def test_comic_exporter(self):
        """Test Multi-format Export (CBZ, HTML scrollbook)."""
        print("Testing ComicExporter...")
        exporter = ComicExporter(output_dir=self.tmp_dir)
        
        img = Image.new("RGB", (800, 1200), (240, 240, 240))
        pages = [
            {
                "page_num": 1,
                "page_image": img,
                "panels": [{"panel_id": 1}, {"panel_id": 2}]
            }
        ]
        
        cbz_path = exporter.export_cbz(pages, title="TestComic")
        html_path = exporter.export_web_comic(pages, os.path.join(self.tmp_dir, "web_comic.html"))
        
        self.assertTrue(os.path.exists(cbz_path))
        self.assertTrue(os.path.exists(html_path))
        self.assertTrue(cbz_path.endswith(".cbz"))
        self.assertTrue(html_path.endswith(".html"))
        print("  [PASSED] ComicExporter tests")

    def test_feedback_and_optimizer(self):
        """Test RLHF feedback storage, parameter adjustments and config mutation."""
        print("Testing Feedback and Optimizer...")
        feedback_path = os.path.join(self.tmp_dir, "test_rlhf.json")
        feedback = RLHFFeedbackLoop(feedback_path=feedback_path)
        
        # Add feedback records (we need >=3 rated panels to trigger optimization)
        feedback.add_panel_feedback(panel_id=1, rating=2, comment="Consistency is bad, character looks different.", prompt_used="A wanderer walking", generation_backend="sdxl")
        feedback.add_panel_feedback(panel_id=2, rating=2, comment="Low quality and blurry, ugly colors.", prompt_used="A wanderer sitting", generation_backend="sdxl")
        feedback.add_panel_feedback(panel_id=3, rating=3, comment="Aesthetic style is nice but text is cluttered, overlap.", prompt_used="A wanderer speaking", generation_backend="sdxl")
        
        # Verify feedback works
        summary = feedback.get_feedback_summary()
        self.assertEqual(summary["total_panels_rated"], 3)
        self.assertAlmostEqual(summary["average_panel_rating"], 2.33, places=2)
        
        # Create a mock settings.yaml file
        settings_yaml = os.path.join(self.tmp_dir, "mock_settings.yaml")
        mock_config = {
            "generation": {
                "guidance_scale": 7.5
            },
            "models": {
                "lora": {
                    "adapter_scale": 0.8
                }
            },
            "quality_critic": {
                "threshold": 0.55,
                "strict_threshold": 0.70,
                "weights": {
                    "visual_consistency": 0.30,
                    "aesthetic_quality": 0.25,
                    "readability": 0.10,
                    "emotional_engagement": 0.15,
                    "narrative_coherence": 0.20
                }
            },
            "style": {
                "positive_terms": ["minimalist line art"],
                "negative_terms": ["photorealistic"]
            }
        }
        
        import yaml
        with open(settings_yaml, "w", encoding="utf-8") as f:
            yaml.dump(mock_config, f)
            
        optimizer = SystemOptimizer(feedback_loop=feedback, settings_path=settings_yaml)
        adjusts = optimizer.optimize_system_parameters()
        
        # Verify calculated deltas
        self.assertTrue(adjusts["quality_critic_threshold_delta"] > 0.0) # low rating delta is positive
        self.assertTrue(adjusts["guidance_scale_adjustment"] > 0.0)
        self.assertIn("visual_consistency", adjusts["critic_weight_shifts"])
        
        # Verify prompt styling keywords are captured
        self.assertIn("sharp focus", adjusts["positive_terms_to_add"])
        self.assertIn("blurry", adjusts["negative_terms_to_add"])
        
        # Apply optimizations
        success = optimizer.apply_optimizations(adjusts)
        self.assertTrue(success)
        
        # Read back mutated settings
        with open(settings_yaml, "r", encoding="utf-8") as f:
            updated_settings = yaml.safe_load(f)
            
        # Verify thresholds, parameters and mutated prompt terms
        self.assertGreater(updated_settings["quality_critic"]["threshold"], 0.55)
        self.assertGreater(updated_settings["generation"]["guidance_scale"], 7.5)
        self.assertIn("sharp focus", updated_settings["style"]["positive_terms"])
        self.assertIn("blurry", updated_settings["style"]["negative_terms"])
        
        # Check sum of normalized quality critic weights is 1.0
        total_weight = sum(updated_settings["quality_critic"]["weights"].values())
        self.assertAlmostEqual(total_weight, 1.0, places=2)
        
        print("  [PASSED] Feedback and Optimizer tests")


if __name__ == "__main__":
    unittest.main()
