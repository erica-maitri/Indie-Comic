import sys
import os
import unittest
from collections import OrderedDict
from PIL import Image

# Add project root and pipeline dir to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "indie_comic_pipeline"))

# Mock the compel library before importing SDXLBackend
from unittest.mock import MagicMock
mock_compel_module = MagicMock()
mock_compel_module.ReturnedEmbeddingsType = MagicMock()
sys.modules['compel'] = mock_compel_module

from indie_comic_pipeline.core.backends.sdxl_backend import SDXLBackend

class MockPipe:
    def __init__(self):
        self.device = "cpu"
        from unittest.mock import MagicMock
        self.tokenizer = MagicMock()
        self.tokenizer_2 = MagicMock()
        self.text_encoder = MagicMock()
        self.text_encoder_2 = MagicMock()
    def __call__(self, **kwargs):
        class MockResult:
            images = [Image.new("RGB", (10, 10))]
        return MockResult()

class TestPromptEmbeddingCache(unittest.TestCase):

    def test_cache_hits_and_eviction(self):
        backend = SDXLBackend()
        backend._pipe = MockPipe()
        backend.device = "cpu"
        backend._max_cache_size = 3  # Set a small limit for testing eviction

        compel_calls = []
        def mock_compel(text):
            compel_calls.append(text)
            return (f"embeds_{text}", f"pooled_{text}")

        # Set the mock compel constructor to return our mock_compel function
        mock_compel_module.Compel = MagicMock(return_value=mock_compel)

        # Generate panel 1 - expect misses for both prompt and negative prompt
        res1 = backend.generate(
            prompt="a hero in neon red",
            negative_prompt="blurry",
            config={"num_steps": 1}
        )
        self.assertIsNotNone(res1)
        self.assertEqual(len(compel_calls), 2)
        self.assertEqual(compel_calls, ["a hero in neon red", "blurry"])

        # Generate panel 2 with exact same prompts - expect 100% cache hits (no new calls)
        res2 = backend.generate(
            prompt="a hero in neon red",
            negative_prompt="blurry",
            config={"num_steps": 1}
        )
        self.assertEqual(len(compel_calls), 2)  # Should still be 2 (no new calls)

        # Generate with new prompts to trigger eviction
        backend.generate(prompt="prompt_3", negative_prompt="", config={"num_steps": 1})
        self.assertEqual(len(compel_calls), 3)  # prompt_3 is called
        
        # Cache current keys should be: ['a hero in neon red', 'blurry', 'prompt_3']
        # 'a hero in neon red' and 'blurry' were accessed, so they moved to end.
        self.assertEqual(list(backend._embeds_cache.keys()), ["a hero in neon red", "blurry", "prompt_3"])

        # Add another new prompt to exceed limit of 3
        backend.generate(prompt="prompt_4", negative_prompt="", config={"num_steps": 1})
        self.assertEqual(len(compel_calls), 4)

        # 'a hero in neon red' (the oldest/least recently used) should be evicted
        self.assertEqual(list(backend._embeds_cache.keys()), ["blurry", "prompt_3", "prompt_4"])
        self.assertNotIn("a hero in neon red", backend._embeds_cache)

        # Unload should clear everything
        backend.unload()
        self.assertEqual(len(backend._embeds_cache), 0)
        self.assertIsNone(backend._compel)
        print("Prompt Embedding Cache with LRU Eviction: All checks PASSED!")

if __name__ == "__main__":
    unittest.main()
