# Proposed Methodology

This section presents the Indie-Comic pipeline with the Multi-Level Diffusion Consistency Prior (MDCP) approach—a training‑free, zero‑shot sequential comic generation framework that preserves character identity via inference‑time latent intervention, requiring neither per‑character fine‑tuning nor model retraining. MDCP operates on multi‑scale latent trajectory deviations: high‑frequency noise drift ($\Delta_{HF}$), semantic concept forgetting ($\Delta_{semantic}$), and global structural shifting ($\Delta_{structure}$). By decoupling the consistency energy into lightweight operations, the framework achieves strict $O(1)$ GPU VRAM complexity by asynchronously offloading the anchor panel’s cached attention activations to CPU pinned memory. This reframes the sequential generation constraint from an algorithmic memory ceiling to a systems‑level PCIe bandwidth trade‑off. A spatiotemporal statistical correction further permits dynamic pose changes across panel gutters, overcoming the rigid temporal suppression of video‑centric models.

The full pipeline comprises eight phases: (i) story intake and emotion‑conditioned parsing, (ii) multi‑agent panel enrichment via a six‑director blackboard (Story, Action, Dialogue, Pose, Emotion, Camera), (iii) reference‑free identity anchoring, (iv) the unified MDCP generation loop, (v) LLM‑planned dialogue placement, (vi) automated quality gating, (vii) cadence layout engine, and (viii) multi‑format export and feedback tuning. Five optional mitigations (detail injection, regional masking, saliency, Fourier scaling, AdaIN) are disabled by default. The decomposition explicitly addresses narrative segmentation, cross‑panel visual identity, layout, and lettering, with MDCP as the mathematical core for identity preservation.

## Multi-Level Diffusion Consistency Prior (MDCP)

We generate $N$ images from $N$ panel descriptions. The anchor panel ($n=1$) is generated unconstrained. For $n > 1$, MDCP intervenes at each denoising step $t$, steering the latent $z$ toward the anchor’s manifold without gradients or training.

### Consistency Energy and Operator Splitting

We define a joint energy over the latent $z$ as
$$E_{cons}(z) = w_{HF} \phi_{HF}(z) + w_{sem} \phi_{sem}(z) + w_{str} \phi_{str}(z)$$
where $\phi_{HF}$, $\phi_{sem}$, $\phi_{str}$ penalize high-frequency noise, semantic drift, and structural shift, respectively. The weights are fixed hyperparameters ($\alpha=0.03$, $\beta=0.15$, $\gamma=0.08$). Rather than minimising this energy directly, we apply an operator-splitting scheme:
$$T_{MDCP} = T_3 \circ T_2 \circ T_1$$
with $T_1$ for smoothing, $T_2$ for semantic anchoring via shared cross‑attention, and $T_3$ for statistics alignment.

### $T_1$: Physics-Informed Latent Smoothing

High‑frequency flicker is attenuated via a heat‑equation approximation with a normalized Gaussian kernel $G_\sigma$ ($\sigma=\text{size}/3$):
$$u(t+1) = u(t) + \alpha_{eff}(t) (u * G_\sigma - u(t))$$
where $\alpha_{eff}(t)$ ramps linearly from 0 at $t=0.20$ to $\alpha=0.03$ at $t=0.80$ (denoising steps normalised to $[0,1]$). This ensures smoothing only during mid‑denoising, preserving both coarse structure and fine details.

### $T_2$: Shared Cross-Attention Caching with Pinned Memory

Semantic identity is enforced by mixing the anchor’s Key/Value projections into subsequent panels. From the first four cross‑attention modules of the UNet, we cache ($K_{anchor}$, $V_{anchor}$). For each target panel, the attention output becomes
$$\text{out} = (1 - \beta_{adaptive}) \text{Softmax}\left(\frac{Q_{cur} K_{cur}^T}{\sqrt{d}}\right) V_{cur} + \beta_{adaptive} \text{Softmax}\left(\frac{Q_{cur} K_{anchor}^T}{\sqrt{d}}\right) V_{anchor}$$
with a global $\beta=0.15$, or spatially adaptive when a region mask $M$ is available ($\beta_{adaptive} = \text{clamp}(\beta(1+0.5M), 0, 0.4)$). Since only the anchor’s K/V is retained, memory is $O(1)$ in sequence length. Cached tensors are moved to CPU pinned memory (`pin_memory()`) and fetched asynchronously (`non_blocking=True`). The per‑step PCIe payload is about 5.05 MB (four layers, CFG batch of 2), transferring in 0.16 ms on Gen4 x16—negligible against the 120–250 ms UNet step.

### $T_3$: Spatiotemporal Statistics Alignment

To anchor global lighting and contrast without suppressing dynamic camera shifts, we align channel‑wise latent statistics during $t \in [0.30, 0.60]$. From the anchor’s final latents we compute $\mu_{anchor,c}$, $\sigma_{anchor,c}$. For the current latent we compute $\mu_{cur,c}$, $\sigma_{cur,c}$ and clamp the ratio
$$r_c = \text{clamp}(\sigma_{anchor,c} / \sigma_{cur,c}, 0.80, 1.20)$$
A blend weight $\omega_t = \gamma (t - 0.30) / 0.30$ with $\gamma=0.08$ linearly increases the correction. The final update is
$$z_{final,c} = z_c (1 + \omega_t (r_c - 1)) + \omega_t \gamma (\mu_{anchor,c} - \mu_{cur,c})$$
The clamping bounds $[0.80, 1.20]$ and $\omega_t \le 0.08$ restrict the effective multiplicative scale to approximately $[0.984, 1.016]$, allowing large pose changes while keeping global contrast stable.

### Bounded Stability

**Proposition 1.** Under bounded anchor statistics and $0 < \beta, \gamma < 1$, $T_{MDCP}$ is affinely bounded: $||T_{MDCP}(z)|| \le C||z|| + D$ with $C \le 1.02$.
*Proof sketch:* $T_1$ is a normalized convolution with bounded spectral radius; $T_2$ is a convex combination of linear projections; $T_3$ applies bounded affine scaling. Composition preserves affine boundedness, ensuring no latent divergence.

Algorithm 1 (detailed in the Appendix) summarises the per-step update.

## Integration into the Complete Pipeline

We embed MDCP within an eight‑phase pipeline (Figure 1). The pipeline is local‑first (Ollama LLM, SDXL backend), training‑free, and targets a 16 GB GPU. Below we summarise the key phases.

```
INPUT SCRIPT 
→ Phase 0-1: Story Intake & Multi-Agent Enrichment
→ Phase 2: Anchor Generation & Identity Signature Extraction
→ Phases 3-6: Unified MDCP Generation (with optional M1-M5)
→ Phase 7: Cadence Layout & Lettering 
→ Phase 8: Export & Feedback
```
*Figure 1: Overview of the eight-phase pipeline.*

**Phases 0–1: Story Intake and Multi‑Agent Enrichment.**
A BERT‑based emotion classifier assigns one of eight emotional arcs (e.g., anger → calming) and distributes mood stages across $N$ panels. A `story_mode` switch (literal vs. mood_arc) prevents overwriting the user’s plot. A blackboard of six agents enriches flat panel outlines:
- **StoryDirector** initialises characters and raw panels.
- **ActionDirector** expands action verbs into cinematic schemas (verb, mechanics, impact, reaction, timing) via a 23‑verb exaggeration map, and assigns action intensity (ordinal from size class).
- **DialogueWriter** generates dialogue via local LLM (fallback to beat‑indexed lines).
- **PoseDirector** and **EmotionDirector** fill pose/expression templates from beat mappings.
- **CameraDirector** assigns camera angles and layout size classes (small/medium/large/full_page).
Agents run in two stages (sequential core, then concurrent) to reduce wall‑clock time.

**Phase 2: Reference‑Free Identity Anchoring.**
Panel 1 is generated without MDCP. From the output, we extract a classical identity signature: (1) HSV colour histogram (8×8 on H/S, omitting V for illumination invariance), (2) Canny edge density $\rho_{edge}$ (thresholds 50, 150), (3) style Gram matrix (5×5 over RGB + Sobel gradients), and (4) aesthetic baseline (sharpness, contrast, colourfulness). These descriptors form a compact, device‑agnostic signature for consistency checks. Optional deep embeddings (CLIP/DINOv2) are disabled by default to save VRAM.

**Phases 3–4: Unified Generation with MDCP.**
The backend is SDXL in fp16 with DPMSolverMultistepScheduler (SDE‑DPMSolver++, order 2, Karras sigmas). Memory optimisations (`enable_model_cpu_offload`, attention/vae slicing) and FreeU rebalancing (s1=0.6, s2=0.4, b1=1.1, b2=1.2) are applied.
Prompt construction follows a 10‑layer hierarchy: [style] → [narrative position] → [lighting] → [palette] → [atmosphere] → [camera] → [environment] → [pose/expression] → [cinematic action] → [quality boosters]. Prompts often exceed 77 tokens; Compel handles chunking.
CharCom Compositor derives per‑panel parameters (g guidance, S steps, λ LoRA scale) from base values (7.5, 25, 0.8) with additive rules: action intensity, emotion intensity, anchor consistency (+0.25 to g for n>1), and bookend positions (+3 steps for first/last). Final clamps: g ∈ [5, 12], S ∈ [15, 50], λ ∈ [0.3, 1.0].
Optional mitigations (M1–M5)—detail injection, regional masking, saliency, Fourier scaling, AdaIN—are disabled by default; enabling all adds about 5–8% overhead.

**Phase 5: LLM‑Planned Dialogue Placement.**
An LLM plans bubble position/style (fallback chain: JSON cache → Ollama → LangChain → deterministic heuristic). Emotion‑to‑bubble mapping (47 beats to five styles: calm, intense, thought, whisper, shout) and Comic Neue typography are used. Bubbles are rendered after layout fitting (post‑crop) to prevent geometric distortion.

**Phase 6: Automated Quality Gating.**
Composite score $Q = 0.30S_{cons} + 0.25S_{aes} + 0.20S_{narr} + 0.15S_{emo} + 0.10S_{read}$ with thresholds: excellent ($Q \ge 0.70$), pass ($0.55 \le Q < 0.70$), fail ($Q < 0.55$). Fail triggers up to 2 retries with targeted adjustments (+1.0 guidance, +5 steps, prompt/negative mutations). An optional user‑preference critic (CLIP embedding + logistic regression) can be trained from $\ge3$ ratings.

**Phase 7: Cadence Layout Engine.**
Dynamic page layout (1000×1500 px, margins 40 px, gutters 12 px). Layout weight $w_i = 0.7 + I_i \cdot 1.0$ where $I_i$ is action intensity. Deterministic partitioning for $N=1, 2, 3, 4, \ge5$ (full, vertical stack, mixed, dominant row/grid, three‑tier). Centre‑focal cropping with Lanczos preserves aspect ratio.

**Phase 8: Export and Feedback Tuning.**
Outputs CBZ (with metadata.xml), CBR (fallback to CBZ), PDF (ReportLab/PIL), and HTML (scrollable). Telemetry logs ratings; heuristic tuning adjusts quality thresholds, CFG scale, LoRA weight, critic weights, and style templates based on complaint categories, with safe file locking.

In summary, MDCP provides a mathematically grounded, training‑free mechanism for cross‑panel identity consistency with $O(1)$ memory and bounded stability. The full eight‑phase pipeline orchestrates narrative parsing, agentic enrichment, MDCP generation, quality control, layout, and export—all under local‑first, hardware‑democratic constraints. The complete implementation is reproducible from the provided code and configuration.

## Appendix: Pipeline Algorithms

```text
Algorithm 1: MDCP Denoising Step
Input: timestep t, latent z_t, anchor cache (K_a, V_a), stats (μ_a, σ_a)
Output: z'_t
if 0.20 ≤ t/T ≤ 0.80 then
    α_eff ← 0.03 * (t/T - 0.20) / 0.60
    z_t ← z_t + α_eff (GaussianBlur(z_t) - z_t)
end if
(K_t, V_t) ← UNet cross-attn projections
K_d, V_d ← AsyncPrefetch(K_a, V_a)
attn_cur ← Softmax(Q_t K_t^T / √d) V_t
attn_anc ← Softmax(Q_t K_d^T / √d) V_d
β_eff ← β = 0.15 (or adaptive with mask)
z_attn ← (1 - β_eff) attn_cur + β_eff attn_anc
if 0.30 ≤ t/T ≤ 0.60 then
    (μ_c, σ_c) ← stats of z_attn
    r ← clamp(σ_a / σ_c, 0.80, 1.20)
    z_corr ← z_attn * r + 0.08 * (μ_a - μ_c)
    ω ← 0.08 * (t/T - 0.30) / 0.30
    z'_t ← (1 - ω) z_attn + ω z_corr
else
    z'_t ← z_attn
end if
return z'_t
```

```text
Algorithm 2: Master Eight-Phase Pipeline Orchestration
Input:  prompt P, character name C, panel count N, story_mode MODE
Output: assembled pages (CBZ/PDF/HTML), feedback telemetry log

1:  /* Phase 0 — Story Intake */
2:  story_config ← StoryIntakeEngine.process_prompt(P, N, C, MODE)
3:
4:  /* Phase 1 — Multi-Agent Enrichment (blackboard) */
5:  storyboard ← AgentController.run_planning(story_config)
6:
7:  /* Phase 2 — Anchor Panel (must complete first, sequential) */
8:  anchor_result ← generate_panel_with_retry(panel_id = 1)
9:  identity_signature ← IdentityEmbeddingExtractor.extract(anchor_result.image)
10:
11: /* Phases 3–6 — remaining panels, MDCP-consistent, parallelizable */
12: for panel_id in 2..N do parallel
13:     panel_result ← generate_panel_with_retry(panel_id)   # Algorithm 1 runs inside this call
14: end for
15:
16: /* Phase 7 — Cadence Layout */
17: pages ← LayoutEngine.assemble(sorted_panels_by_page)
18:
19: /* Phase 8 — Export and Feedback Logging */
20: Exporter.export(pages, formats = [CBZ, PDF, HTML])
21: FeedbackTuner.log_telemetry()
22: return pages
```

```text
Algorithm 6: Evaluation Suite and Performance Benchmarking
────────────────────────────────────────────────────────────────────────────────
Input:  Generated Image I_gen, Reference Image I_ref, Target Prompt Prompt_t, 
        Bounding Boxes [B_pred, B_gt], Ground-truth Mask M_gt
Output: JSON report containing 14 metrics and performance profiling results
────────────────────────────────────────────────────────────────────────────────
Step 1: Compute Low-Level Image Metrics
  Aesthetic = ComputeAestheticScore(I_gen)
  Density   = ComputeCannyEdgeDensity(I_gen, 50, 150)
  PSNR_val  = CalculatePSNR(I_gen, I_ref)
  SSIM_val  = CalculateSSIM(I_gen, I_ref)

Step 2: Load Embeddings and Compute Perceptual Similarity
  CLIP_Img  = CalculateCLIPImageSimilarity(I_gen, I_ref)
  DINOv2    = CalculateDINOv2Similarity(I_gen, I_ref)
  DINOv3    = CalculateDINOv3Similarity(I_gen, I_ref)
  SigLIP    = CalculateSigLIPSimilarity(I_gen, I_ref)
  LPIPS_val = CalculateLPIPSDistance(I_gen, I_ref)
  FID_val   = CalculateFID(I_gen, I_ref)

Step 3: Compute Text-Image and Spatial Metrics
  CLIP_Text = CalculateCLIPTextAlignment(I_gen, Prompt_t)
  IoU_val   = CalculateIoU(B_pred, B_gt)
  DetectMetrics = CalculateDetectionMetrics(B_pred, B_gt, threshold=0.5)
  SegmentMetrics = CalculateSegmentationMetrics(SAM2_Predict(I_gen), M_gt)
  BLEU_val  = CalculateBLEU(Generated_Dialogue, Reference_Script)

Step 4: Serialize Report and Free VRAM
  MetricsJSON = PackageMetrics(Aesthetic, Density, PSNR_val, SSIM_val, 
                               CLIP_Img, DINOv2, DINOv3, SigLIP, LPIPS_val, 
                               FID_val, CLIP_Text, IoU_val, DetectMetrics, 
                               SegmentMetrics, BLEU_val)
  SaveJSON("metrics_report.json", MetricsJSON)
  FreeEvaluatorVRAM()
  
  Return MetricsJSON
────────────────────────────────────────────────────────────────────────────────
```
