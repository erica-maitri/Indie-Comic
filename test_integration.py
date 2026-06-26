import sys, json
sys.path.insert(0, 'indie_comic_pipeline')

# Test 1: EmotionRouter import and config loading
from integration.emotion_router import EmotionRouter, _ARCS_CONFIG, _ARC_LOOKUP
print('EmotionRouter imported OK')
print(f'Arc config version: {_ARCS_CONFIG.get("version", "not loaded")}')
print(f'Arc keys available: {list(_ARC_LOOKUP.keys())}')

# Test 2: route() without a model
router = EmotionRouter()
test_cases = [
    {'primary_emotion': 'sadness'},
    {'primary_emotion': 'joy'},
    {'primary_emotion': 'anger'},
    {'primary_emotion': 'fear'},
    {'primary_emotion': 'love'},
    {'primary_emotion': 'grief'},
    {'primary_emotion': 'determined'},
    {'primary_emotion': 'surprise'},
    {'primary_emotion': 'tired'},
]
for case in test_cases:
    route = router.route(case)
    print(f"  {case['primary_emotion']:12s} -> arc_key={route.get('arc_key','?'):12s}  journey={route.get('journey','')}")

# Test 3: PipelineLauncher import
from integration.pipeline_launcher import PipelineLauncher, StoryWeaverBridge
print('PipelineLauncher imported OK')

# Test 4: full_pipeline without model (mock path)
result = router.full_pipeline('I feel exhausted and hollow')
print(f'full_pipeline emotion  : {result["emotion"]}')
print(f'full_pipeline arc_key  : {result["route"]["arc_key"]}')
print(f'full_pipeline character: {result["character"]["name"]}')
print(f'full_pipeline world    : {result["character"]["world"]}')

# Test 5: __init__ re-exports
import integration
print(f'integration.__version__ = {integration.__version__}')
print('All checks PASSED')
