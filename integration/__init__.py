"""
integration package — Mood Weaver → Story Weaver → Indie Comic Pipeline.

Public API
----------
    from integration.emotion_router   import EmotionRouter
    from integration.pipeline_launcher import PipelineLauncher

Quick usage::

    from integration.emotion_router import EmotionRouter
    router = EmotionRouter()
    result = router.full_pipeline("I feel anxious about the future")
    print(result["emotion"])          # "fear"
    print(result["route"]["arc_key"]) # "anxious"
"""

from .emotion_router    import EmotionRouter       # noqa: F401
from .pipeline_launcher import PipelineLauncher    # noqa: F401

__all__ = ["EmotionRouter", "PipelineLauncher"]
__version__ = "2.1.0"
