[USER INPUT: Emotion / Story Prompt]
       │
       ▼
[PHASE 0: INTELLIGENT STORY INTAKE (WRITER'S ROOM)]
       │  • Action: User prompts (character, world, traits, reference) are passed to local Ollama (default: llama3.2).
       │  • Processing: Runs a single-pass JSON-structured LLM chain to generate both a 'Story Bible' and 
       │                a detailed hierarchical scene graph script (characters, actions, camera, environment).
       │  • Output: Generates an in-memory structured configuration dictionary (story_config).
       │
       ▼
[story_config (In-Memory)]
       │
       ▼
[PHASE 1: NARRATIVE PLANNING LAYER]
       │  • Action: Director Swarm (Multi-Agent System) parses and loads the scene graph into memory.
       │  ├─ Story Director: Registers characters and sets up the base panel sequence.
       │  ├─ Action Director: Parses character actions, physical verbs, and interactions.
       │  ├─ Dialogue Writer: Details dialogue text, tone, and vector speech bubble categories.
       │  ├─ Pose Director: Identifies body language posture/stance mapping constraints.
       │  ├─ Emotion Director: Resolves granular facial taxonomies (eyes, mouth, expressions).
       │  └─ Camera Director: Dictates framing layouts (camera angle, environment variables).
       │
       ▼
[STORY SECTION MEMORY (EXPLICIT RAM BLACKBOARD)]
       │  • Dynamic Cache: Tracks active character states, visual anchors, scene settings, and layout plans.
       │
       ▼
[PHASE 2: SELF-REFERENTIAL VISUAL ANCHORING]
       │  • Step 2.1: Pulls initial context prompts from memory to execute Panel 1 Generation.
       │  • Step 2.2: Isolates Panel 1 to serve as the baseline Primary Visual Anchor (Self-Reference).
       │  • Step 2.3: Runs Identity Embedding Extraction to capture raw facial topology, wardrobe, and style markers.
       │  • Step 2.4: Injects extracted identity tracking tokens directly back into the Story Section Memory cache.
       │
       ▼
[UPDATED STORY SECTION MEMORY (WITH IDENTITY TOKENS)]
       │
       ▼
[PHASE 3 & 4: IN-GENERATION CONSISTENCY & COMPOSABLE CONTROL]
       │  • Action: Sequentially generates panels 2 through N by pulling context and tokens from memory.
       │  │
       │  ├─ [CharCom Inference Compositor]
       │  │     Calculates dynamic model weight blending at runtime: W_total = W_base + Σ(α_i * W_i)
       │  │
       │  └─ [Multi-Backend Diffusion Denoising Stack (SDXL / Flux / Video DiT)]
       │        ├─ Level 1: Dissipative Latent Smoothing (RealDiffusion)
       │        │    Applies a Gaussian smoothing kernel during denoising to suppress high-frequency noise drift.
       │        ├─ Level 2: Shared Attention Matrix Masking (Accelerated TF)
       │        │    Applies cross-prompt masking to lock character identity keys/values across frames.
       │        └─ Level 3: Sequential Latent Prior (DreamingComics)
       │             Blends channel-wise latent statistics (mean, std) toward anchor distribution.
       │
       ▼
[LATENT SPACE IMAGE CANVAS DATA]
       │
       ▼
[PHASE 5: INTEGRATED TEXT-IMAGE GENERATION]
       │  • Action: Processed latents are fed into the DiffSensei MLLM Domain.
       │  • Processing: Merges image matrices and script dialogue inside a single Unified Multimodal Semantic Space.
       │  • Execution: Binds language vectors directly into cross-attention loops to render dynamic expressions/poses.
       │
       ▼
[RAW PANEL RASTER SHEET]
       │
       ▼
[PHASE 6: QUALITY VALIDATION LAYER]
       │  • Action: Raw panel imagery is intercepted by the COMIC Critic Pipeline.
       │  • Processing: Evaluates performance across an evolutionary ring of human-aligned LLM Critics.
       │  │
       │  ├─── [IF Composite Score < Quality Performance Threshold]
       │  │        Adjusts guidance scale parameters, updates prompting weights, and triggers Regeneration Loop.
       │  │
       │  └─── [IF Composite Score >= Quality Performance Threshold]
       │           Approves frame data and caches the cleared panel asset straight to the assembly stack.
       │
       ▼
[APPROVED RASTER PANELS STACK]
       │
       ▼
[PHASE 7: LAYOUT & ASSEMBLY]
       │  • Action: MangaFlow Engine takes the approved panels and layout parameters from memory.
       │  • Processing: Dynamically cuts border geometry channels depending on scene action intensity.
       │  • Execution: Runs typesetting algorithms to lock vector narrative speech bubbles based on focal subjects.
       │
       ▼
[COMPILED MASTER SHEET LAYOUT]
       │
       ▼
[PHASE 8: EXPORT MODULE & ADAPTIVE PARAMETER OPTIMIZATION]
       │  ├─ File Compilation: Compiles raw layout vectors into final reader formats (PDF / CBZ / HTML).
       │  │
       │  └─ Human Alignment Telemetry Loop:
       │       Gathers explicit user interface performance rankings and rating feedback metrics.
       │       │
       │       ▼
       └─► [SYSTEM PARAMETER OPTIMIZATION]
               Executes weight adjustments to fine-tune COMIC LLM Critic evaluations and mutate prompt generation templates.