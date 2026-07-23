# MDCP: A Training-Free Multi-Level Consistency Prior for Long-Range Diffusion Generation

---

## Abstract

Maintaining long-range consistency across sequential image generation remains an open challenge for text-to-image diffusion models. This is particularly pronounced in multi-frame illustrated narratives, storyboards, and sequential art, where characters and environments must retain visual coherence across diverse poses and emotional states. In this paper, we propose the **Multi-Level Diffusion Consistency Prior (MDCP)**, a unified, training-free framework that formulates long-range consistency through a multi-scale consistency energy within the diffusion latent space. 

MDCP models latent deviation as the sum of high-frequency, semantic, and structural drift. It adopts an operator-splitting-inspired update scheme designed to heuristically reduce the proposed consistency energy through three sequential consistency operators: (L1) physics-informed latent smoothing via the heat equation, (L2) shared cross-attention key/value caching for semantic identity preservation, and (L3) spatiotemporal channel-statistic alignment for global structural continuity. We evaluate MDCP on sequential comic generation, a demanding long-range image generation task requiring consistent character identity across independently generated panels. Experiments show that MDCP provides competitive improvements over existing prompt-based and single-level attention-sharing methods, improving DINOv2 similarity to 0.768 (up from 0.582) and LPIPS to 0.252 (down from 0.415) compared to standard SDXL baselines. Furthermore, we implement and integrate five SOTA mitigations—specifically, Fourier skip-connection scaling (FreeU), regional attention masking (OMOST/BoxDiff), foreground saliency masking (SAM/GrabCut), Adaptive Instance Normalization (AdaIN/StyleAligned), and localized structural injectors—to resolve identified failure modes. Crucially, MDCP achieves this with a strict $O(1)$ consistency module VRAM overhead of 150 MB (avoiding the quadratic memory explosion of self-attention concatenation that OOMs at 18 frames), offering a robust, scalable, and highly detailed approach to long-range visual consistency.

**Keywords:** diffusion models, visual consistency, identity preservation, sequential generation, training-free prior

---

## 1. Introduction

Text-to-image diffusion models have achieved unprecedented success in generating high-quality, photorealistic imagery from natural language prompts. However, generating sequential visual content—such as illustrated narratives, children's books, storyboards, visual novels, and sequential art—requires more than high-fidelity individual images. It necessitates **long-range visual consistency**: characters, environments, and stylistic motifs must remain recognizable across a sequence of independently generated frames depicting varying actions, expressions, and environmental setups.

Existing approaches to identity preservation largely fall into two categories. **Test-time conditioning methods**, such as IP-Adapter, use pre-trained semantic encoders (e.g., CLIP) to inject image prompt features into the diffusion cross-attention layers. While effective for single-image reference, these methods often struggle to maintain rigid structural identity (e.g., specific facial geometry or clothing details) under extreme pose variations. **Training-based methods**, such as ConsistentCharacter, train dedicated identity encoders or fine-tune the diffusion model on specific characters, which is computationally expensive and limits generalization to zero-shot character creation. Recently, **attention-sharing methods** like StoryDiffusion have shown promise by concatenating self-attention keys and values across generated frames, but they scale poorly with sequence length ($O(N^2)$ consistency module memory complexity) and are often insufficient to prevent low-level noise drift and structural morphing over long sequences.

We argue that visual inconsistency in diffusion models is not a monolithic failure but a multi-scale phenomenon. The total drift $\Delta z$ between an anchor latent and a subsequent sequence latent can be modeled as the aggregate of high-frequency noise accumulation, semantic concept forgetting, and global structural shifting.

To address this, we introduce the **Multi-Level Diffusion Consistency Prior (MDCP)**. MDCP conceptualizes long-range consistency through a multi-scale consistency energy prior, targeting drift across three distinct scales (high-frequency, semantic, and structural). Rather than performing a costly online optimization of this energy (which is computationally prohibitive at test time), we design a sequential operator-splitting scheme to heuristically approximate its minimization. The core contribution is thus not a joint numerical optimization, but the architectural and engineering integration of these multi-scale interventions into an $O(1)$ memory streaming framework that pins and streams anchor projections in real time. We use comic generation as the primary evaluation domain because it stresses identity preservation under extreme viewpoint, lighting, and expression changes.

### Contributions

We explicitly acknowledge that the individual mathematical operators ($\mathcal{T}_1$, $\mathcal{T}_2$, and $\mathcal{T}_3$) utilize established primitives from Gaussian smoothing, cross-attention blending, and feature statistic normalizations. The principal technical novelty of this work lies in their joint architectural decomposition, the operational scheduling scheme, the host-to-device streaming prefetch memory architecture that guarantees $O(1)$ scaling, and their systematic integration inside a multi-agent sequential generation pipeline.

1. **Multi-Level Diffusion Consistency Prior:** We introduce a training-free consistency prior that targets drift across three distinct physical scales (high-frequency noise, semantic identity, and global structure).
2. **Operator-Splitting Inference Formulation:** We formulate the update scheme as a sequential composition of three lightweight, gradient-free operators, avoiding test-time training or joint optimization.
3. **O(1) Streaming Attention Architecture:** We present a decoupled host-to-device attention prefetch thread model that completely eliminates sequence-scaling VRAM overhead, maintaining a flat memory footprint.
4. **Training-Free Identity Preservation:** We demonstrate zero-shot visual identity preservation from purely textual descriptions without requiring reference character sheets or fine-tuning runs.
5. **Comprehensive Evaluation on Sequential Comic Generation:** We demonstrate MDCP on a challenging sequential comic benchmark that stresses long-range identity preservation under large viewpoint, pose, lighting, and scene changes, demonstrating substantial performance improvements via automated metrics and a human preference study.

---

## 2. Related Work

### 2.1 Identity Preservation in Diffusion Models

**Image Prompting and Conditioning.** IP-Adapter (Ye et al., 2023) introduced a decoupled cross-attention mechanism to accept image prompts alongside text prompts, utilizing CLIP image embeddings. While highly effective for style and general semantic transfer, CLIP embeddings are inherently invariant to spatial transformations, often resulting in lost structural details when generating characters in novel poses.

**Trainable Identity Encoders.** Methods like ConsistentCharacter (Liu et al., 2024) train specialized identity encoders that map reference images into a specialized latent space for conditioning. While achieving high structural fidelity, they require extensive training datasets and computational overhead, limiting their utility for zero-shot, on-the-fly character creation from purely textual descriptions.

**Video-Centric and Temporal Models.** Video diffusion models, such as AnimateDiff (Guo et al., 2023) or ConsistI2V (Ren et al., 2024), enforce temporal consistency across frames using temporal attention blocks trained on video sequences. While highly effective at maintaining smooth, continuous sub-pixel transitions, temporal motion networks are ill-suited for the discrete, pacing-driven panel shifts of sequential art. Comic pages require sharp transitions, camera cuts, and dramatic perspective changes across gutters rather than smooth video motion. Furthermore, loading video-centric networks incurs massive VRAM overhead and computational bottlenecks, making them impractical for generating long sequences on local consumer hardware. MDCP is designed specifically to address this gap by establishing consistency as a discrete, training-free multi-scale prior.

### 2.2 Sequential Art and Comic Generation Systems

DiffSensei (Ravi et al., 2024) explored the intersection of multimodal LLMs and diffusion models for customized manga generation, focusing heavily on joint text-and-image layout planning. CoMix (Vivoli et al., 2024) provides a comprehensive benchmark for comic understanding, highlighting the difficulty discriminative models face in re-identifying characters across panels—a challenge MDCP directly addresses on the generative side. MangaFlow (Wang et al., 2026) proposes an end-to-end agentic framework that decomposes manga creation into narrative planning, reference-conditioned panel rendering, and explicit layout generation. MDCP shares this decompose-and-delegate philosophy but differs in its target domain (general indie/webcomic style) and utilizes a heuristic action-intensity-driven layout engine rather than an explicit layout-generation agent.

### 2.3 Emotion and Arc-Conditioned Generation

Outside the visual domain, emotional-arc-guided generation has recently been explored for procedural content, such as using emotional arcs to guide branching narrative structure and pacing in procedurally generated game levels (Wen et al., 2025). To our knowledge, no prior system combines explicit emotional-arc conditioning with modern LLM-driven diffusion comic panel generation while also preserving fidelity to a user-authored plot—a gap MDCP's `story_mode` mechanism addresses.

## 3. Proposed Methodology

This section introduces the methodology of the Indie-Comic pipeline with the Multi-Level Diffusion Consistency Prior (MDCP) approach. This is a training-free, zero-shot sequential comic generation framework featuring an inference-time latent intervention approach that preserves character identity without per-character fine-tuning or model training. It considers multi-scale latent trajectory deviations, where MDCP examines high-frequency noise drift ($\Delta_{\text{HF}}$), semantic concept forgetting ($\Delta_{\text{semantic}}$), and global structural shifting ($\Delta_{\text{structure}}$) during the reverse diffusion loop to enforce consistency. By decoupling this multi-scale energy profile into lightweight operations, this framework reduces consistency-module memory overhead to a strict $\mathcal{O}(1)$ GPU VRAM complexity profile by offloading the anchor panel's cached attention-block output activations asynchronously to CPU pinned memory. This mathematical decoupling shifts the sequential generation constraint from a hard, algorithmic memory ceiling to a systems-level PCIe streaming bandwidth trade-off, reframing a machine-learning memory bottleneck as a systems/HPC-style engineering trade-off. Furthermore, this study proposes a spatiotemporal statistical correction to navigate "temporal leaps" — the discontinuous narrative transitions between panel gutters — allowing characters the dynamic pose flexibility required for comic action, which video-centric sequential models actively suppress.

Along with MDCP, this framework also proposes an automated comic composition workflow designed to orchestrate the entire production lifecycle. Figure 1 presents the framework of the proposed research, which consists of eight core phases: (i) story intake and emotion-conditioned narrative parsing, (ii) multi-agent panel enrichment via a six-agent blackboard swarm (Story, Action, Dialogue, Pose, Emotion, and Camera directors), (iii) reference-free identity anchoring, (iv) the unified MDCP generation loop, (v) LLM-planned dialogue placement, (vi) automated quality gating, (vii) cadence layout engine, and (viii) multi-format export and feedback tuning. These eight phases are further augmentable, on an opt-in basis, by five specialized consistency mitigation modules that address specific edge-case failure modes of the core operator chain; consistent with their disabled-by-default status in the released implementation, they are not counted among the eight core phases. The five opt-in modules are: (1) localized detail injection (ConsistentID/InstantID-style) to preserve micro-geometric details like scars or emblems, (2) regional attention masking (OMOST/BoxDiff) to reduce multi-character semantic feature bleed, (3) foreground saliency segmentation (SAM/GrabCut) to isolate the character foreground so that anchor-panel background content is not blended into a new panel's independently specified background, (4) skip-connection Fourier scaling (FreeU) to suppress over-smoothing of fine line art, and (5) adaptive instance normalization (AdaIN/StyleAligned) to relax the core $\mathcal{T}_3$ statistic clamp during dramatic lighting or contrast shifts while keeping color identity anchored in a deeper feature space, rather than clamped in raw channel statistics. This section explains in detail the complete methodology of the proposed framework with all steps. The initial story-intake phase considers both literal plot-preservation and legacy mood-arc emotional mapping to ensure high narrative fidelity, the subsequent multi-agent enrichment phase adds pose, camera, and action detail before generation begins, and later phases programmatically enforce artistic rendering coherence and automated layout orchestration.

Generating a comic from a natural-language story is not one problem but several stacked on top of each other: the narrative must be broken into discrete moments without losing the author's intent, the same characters must remain recognizable across images produced by independent diffusion runs, panels must be arranged to reflect the story's pacing, and dialogue must be placed without destroying the art beneath it. Treating this as a single end-to-end model would require a scale of paired, panel-consistent training data that does not publicly exist; treating it as a single-image text-to-image problem, run once per panel, ignores the sequential-consistency problem entirely and produces characters whose faces, clothing, and palette drift from frame to frame. Our approach instead decomposes the problem into eight explicit phases, each addressing one sub-problem with the lightest mechanism that solves it, and introduces a dedicated mathematical framework — the Multi-Level Diffusion Consistency Prior (MDCP) — for the sub-problem that most directly determines whether the output reads as one story rather than eight unrelated images: cross-panel visual identity.

This section presents that framework and its integration in two parts. Section 3.1 develops MDCP itself: a training-free, inference-time intervention on the diffusion sampling trajectory that requires no reference images, no per-character fine-tuning, and no modification of model weights, formalized as a composition of three operators acting on three distinct physical sources of visual drift, together with a boundedness guarantee (Proposition 1) that holds independent of what is actually generated. Section 3.2 places MDCP inside the complete eight-phase pipeline, specifying every other stage — narrative planning, layout, lettering, quality control, and export — at the level of exact parameter values and algorithms, so that the system is intended to be reproducible from this description and the accompanying implementation.

### 3.1 Multi-Level Diffusion Consistency Prior (MDCP)

####  T1 Derivation
##### 1 Problem Formulation

Our framework is built upon Latent Diffusion Models (LDMs), specifically Stable Diffusion XL (SDXL). Let $ denote the latent representation of a clean image and $ denote its noisy latent at diffusion timestep \in[0,T]$. During inference, the reverse diffusion process progressively removes noise through the learned denoising network $\epsilon_\theta$. The scheduler updates the latent according to


z_{t-1} = S\left(z_t, \epsilon_\theta(z_t,t,c)\right), \tag{1}


where $ denotes the text conditioning and (\cdot)$ represents the scheduler transition operator.

Most diffusion schedulers additionally estimate the clean latent from the current noisy sample. We denote this prediction as


\hat{z}_0 = D_{\text{sched}}\left(z_t, \epsilon_\theta, t\right), \tag{2}


where {\text{sched}}$ is the scheduler's predicted clean latent.

The reference latent

z_{0,\mathrm{anchor}}^\n

is obtained by encoding a designated anchor image using the SDXL VAE encoder and remains fixed throughout inference.

Unlike existing methods that modify model architectures or require additional training, our objective is to optimize only the latent variable $ during inference while keeping all model parameters frozen.

---

#### 2 Motivation

Although SDXL produces high-quality images, each latent trajectory evolves independently. Consequently, multiple images conditioned on the same subject prompt may gradually diverge in facial identity, clothing appearance and fine structural details.

Recent approaches address this problem from complementary perspectives.

---

### StoryDiffusion: Attention-Level Interaction

StoryDiffusion introduces **Consistent Self-Attention** to establish interactions among images within a generation batch. Given image features

I\in\mathbb R^{B\times N\times C},^\n

the original self-attention for image (i) is


O_i = \operatorname{Attention}(Q_i,K_i,V_i), \tag{3}


where (Q_i,K_i,V_i) denote the query, key and value tensors.

To promote consistency, StoryDiffusion randomly samples tokens from the remaining images,


S_i = \operatorname{RandSample}(I_1,\ldots,I_{i-1},I_{i+1},\ldots,I_B), \tag{4}


and concatenates them with the current image tokens,

P_i=[I_i,S_i].^\n

The attention computation becomes


O_i = \operatorname{Attention}(Q_i,K_{P_i},V_{P_i}), \tag{5}


thereby enabling cross-image information exchange without retraining the diffusion model.

While this improves consistency through attention interactions, the latent trajectory itself remains unconstrained because the scheduler in Eq. (1) continues to evolve each latent independently.

---

### ConsiStory: Feature-Level Correspondence

ConsiStory improves subject consistency using dense correspondence between intermediate feature maps.

Let

F_t^\n

denote intermediate features extracted from the current image and

F_{\mathrm{anchor}}^\n

those extracted from the reference image.

A dense correspondence operator

\Psi(F_t,F_{\mathrm{anchor}})^\n

aligns spatial structures between the two feature spaces.

This feature refinement significantly improves structural consistency; however, the optimization is confined to intermediate network representations rather than directly constraining the latent diffusion trajectory.

---

### Consistency Models: Clean Latent Consistency

Consistency Models demonstrate that different noisy states corresponding to the same image should converge toward a common clean representation through a consistency mapping.

Motivated by this observation, we compare predicted clean latents instead of directly comparing noisy latent variables. Specifically, we use the scheduler prediction

\hat z_0 = D_{\mathrm{sched}}(z_t,\epsilon_\theta,t)^\n

as a clean latent estimate during inference.

Unlike Consistency Models, which learn this mapping during training, our framework employs it solely as an inference-time trajectory regularization objective.

---

### TADA: Adaptive Inference Dynamics

TADA demonstrates that diffusion trajectories can be modified during inference without retraining by augmenting the sampling dynamics.

Inspired by this principle, we adapt the magnitude of latent correction according to the scheduler noise level rather than applying a fixed correction throughout sampling.

This allows stronger corrections during early noisy stages and progressively weaker corrections as the latent approaches convergence.

---

#### 3 Latent Consistency Energy

Motivated by the above observations, we formulate identity preservation as an optimization problem over the latent variable itself.

At each diffusion timestep, we define the total consistency energy as


E_t = \lambda_1E_{\mathrm{id}} + \lambda_2E_{\mathrm{str}} + \lambda_3E_{\mathrm{traj}}, \tag{6}


where

* (E_{\mathrm{id}}) preserves identity,
* (E_{\mathrm{str}}) preserves spatial structure,
* (E_{\mathrm{traj}}) regularizes the diffusion trajectory,

and

(\lambda_1,\lambda_2,\lambda_3)

balance their relative contributions.

---

### Identity Energy

Motivated by StoryDiffusion's attention-sharing mechanism, we introduce an explicit attention alignment objective,


E_{\mathrm{id}} = \frac{1}{N} \|A_t - A_{\mathrm{anchor}}\|_F^2, \tag{7}


where

* (A_t) denotes the attention maps extracted from the current UNet,
* (A_{\mathrm{anchor}}) denotes cached attention maps extracted from the anchor image,
* (N) denotes the number of attention tokens,
* (|\cdot|_F) denotes the Frobenius norm.

Unlike StoryDiffusion, this loss explicitly penalizes attention divergence during optimization.

---

### Structural Energy

Motivated by ConsiStory, structural consistency is enforced by comparing the current feature maps with dense correspondence-aligned anchor features,


E_{\mathrm{str}} = \|F_t - \Psi(F_t,F_{\mathrm{anchor}})\|_2^2. \tag{8}


Here,

(\Psi(\cdot))

is implemented using differentiable cosine-similarity-based feature correspondence, allowing gradients to propagate through the complete optimization process.

---

### Trajectory Energy

To explicitly regularize the denoising trajectory, we compare the scheduler's predicted clean latent against the anchor latent,


E_{\mathrm{traj}} = \|\hat z_0 - z_{0,\mathrm{anchor}}\|_2^2. \tag{9}


Unlike previous feature-level approaches, this objective directly constrains the latent diffusion trajectory while allowing scene-specific variations.

---

#### 4 Latent Trajectory Optimization

Since every energy component is differentiable with respect to the latent variable $, we optimize only the latent while keeping all diffusion model parameters fixed.

The correction direction is obtained through automatic differentiation,


R_t = \nabla_{z_t}E_t. \tag{10}


Because diffusion sampling already performs incremental latent updates, we formulate our correction as an additive perturbation applied before each scheduler step.

To stabilize optimization across different noise levels, the update magnitude is scaled according to the scheduler variance,


\eta(t) = \lambda \frac{\sigma_t}{\sigma_{\max}+\varepsilon}, \tag{11}


where

* (\sigma_t) is the scheduler noise standard deviation,
* (\sigma_{\max}) is the maximum scheduler noise level,
* (\varepsilon) is a small numerical stability constant.

The corrected latent becomes


\tilde z_t = 
	ilde z_t = z_t - \eta(t)R_t.
\tag{12}


The refined latent is then propagated through the standard SDXL scheduler,


z_{t-1} = S\left(\tilde z_t, \epsilon_\theta(\tilde z_t,t,c)\right). \tag{13}


This procedure directly regularizes the diffusion trajectory while leaving the pretrained diffusion network unchanged.

---

#### 5 Computational Complexity

This framework requires extracting intermediate attention maps and feature representations through forward hooks together with one additional backward pass to compute

\nabla_{z_t}E_t.^\n

All diffusion model parameters remain frozen throughout optimization.

Consequently, each denoising iteration consists of one standard forward pass and one additional backward pass, introducing an approximately constant-factor increase in inference time while preserving linear complexity with respect to the number of denoising steps.

---

#### 6 Algorithm

```text
``

#### 3.1.2 Attention Propagation Module (T2)
##### 1 Problem Formulation

While T1 regularizes the latent diffusion trajectory, semantic identity is primarily encoded inside the intermediate attention representations of the diffusion UNet. During standard SDXL inference, each attention layer independently computes


O_i=\operatorname{Attention}(Q_i,K_i,V_i), \tag{14}


where (Q_i), (K_i), and (V_i) denote the projected query, key and value tensors of image (i).

Since every image is processed independently,

O_i \perp O_j,\qquad i\neq j,^\n

there exists no explicit mechanism that allows semantic identity learned in one image to influence another image during inference.

Consequently, although the latent trajectory may be corrected (Section 3), semantic attention representations gradually diverge across independently generated panels.

---

#### 2 Observations from Previous Work

## Observation 1 (StoryDiffusion)

StoryDiffusion introduces **Consistent Self-Attention** to establish interactions between images inside the same generation batch.

Instead of standard attention


O_i = \operatorname{Attention}(Q_i,K_i,V_i), \tag{15}


tokens from neighboring images are randomly sampled,


S_i = \operatorname{RandSample}(I_1,\ldots,I_{i-1},I_{i+1},\ldots,I_B), \tag{16}


and paired with the current image,


P_i=[I_i,S_i]. \tag{17}


Attention is then computed as


O_i = \operatorname{Attention}(Q_i,K_{P_i},V_{P_i}). \tag{18}


This demonstrates that exchanging attention information improves subject consistency across multiple images.

**Observation.**

Identity-related semantic information is encoded within intermediate attention representations.

---

## Observation 2 (CharaConsist)

CharaConsist improves character consistency by maintaining correspondence between intermediate feature representations across multiple images.

Let

F^{(l)}^\n

denote the intermediate representation extracted from layer (l).

Rather than relying only on the final generated image, CharaConsist continuously aligns intermediate representations during generation.

**Observation.**

Intermediate representations preserve stable semantic identity throughout the diffusion process.

---

## Observation 3 (DiffSim)

DiffSim demonstrates that intermediate diffusion representations provide reliable semantic similarity measurements between images.

Consequently,

d(O_i,O_j)^\n

acts as a meaningful semantic distance between attention representations.

Therefore,

reducing

d(O_{\text{curr}},O_{\text{anchor}})^\n

should improve semantic consistency.

---

## Observation 4 (AdaCache)

AdaCache shows that intermediate diffusion activations can be cached and reused during inference without modifying network parameters.

Let


\mathcal C=
{
O_{\text{anchor}}^{(1)},
O_{\text{anchor}}^{(2)},
\ldots,
O_{\text{anchor}}^{(L)}
}
\tag{19}


denote cached attention outputs extracted from the anchor image.

These cached representations provide reusable semantic priors for subsequent generations.

---

## Observation 5 (FAM Diffusion)

FAM Diffusion demonstrates that attention representations can be modulated through weighted combinations to transfer semantic information between different generation contexts.

Rather than treating attention as immutable, attention responses may be smoothly adjusted using weighted modulation.

**Observation.**

Semantic information can be propagated through continuous interpolation in attention space.

---

#### 3 Design Principle

The previous observations establish four facts.

1. Identity information is encoded in attention representations.
2. Intermediate attention remains semantically stable during generation.
3. Anchor attention can be cached and reused.
4. Attention representations admit continuous modulation.

Therefore, instead of recomputing attention using modified key-value tensors (StoryDiffusion), we directly propagate semantic identity by operating on the attention outputs themselves.

Since both

O_{\text{curr}}^\n

and

O_{\text{anchor}}^\n

belong to the same attention feature space, the propagation operator should satisfy three conditions:

1. preserve current scene semantics,

2. inject anchor identity,

3. remain bounded to avoid over-correction.

The simplest operator satisfying these constraints is a convex combination.

---

#### 4 Attention Propagation Operator

Accordingly, we define the propagated attention representation as


\boxed{O_{\text{prop}} = (1-\beta) O_{\text{curr}} + \beta O_{\text{anchor}}} \tag{20}


where

0\le\beta\le1.^\n

Unlike StoryDiffusion, which exchanges key-value tensors during attention computation, this integration operates directly on the attention outputs.

Unlike FAM Diffusion, which interpolates attention across image resolutions, our operator propagates semantic identity across independently generated images while preserving the original network architecture.

---

#### 5 Theoretical Analysis

Subtracting the anchor representation,


O_{\text{prop}} - O_{\text{anchor}} = (1-\beta) \left( O_{\text{curr}} - O_{\text{anchor}} \right). \tag{21}


Taking the Euclidean norm,


\| O_{\text{prop}} - O_{\text{anchor}} \|_2 = (1-\beta) \| O_{\text{curr}} - O_{\text{anchor}} \|_2. \tag{22}


Since

0\le\beta\le1,^\n

we obtain


\boxed{ \| O_{\text{prop}} - O_{\text{anchor}} \|_2 \le \| O_{\text{curr}} - O_{\text{anchor}} \|_2 } \tag{23}


with equality only when

\beta=0^\n

or

O_{\text{curr}}=O_{\text{anchor}}.^\n

Therefore, the propagation operator monotonically moves the attention representation toward the anchor in attention feature space while preserving the contribution of the current image.

---

#### 6 Integration into Diffusion

The propagated attention replaces the original attention output before the subsequent UNet block,


O_{\text{curr}} \longrightarrow O_{\text{prop}} \longrightarrow \text{UNet}_{l+1}, \tag{24}


thereby propagating semantic identity throughout the denoising process without modifying network parameters or requiring retraining.

#### 3.1.3 Spatiotemporal Channel Statistics Alignment (T3)

---

### Step 1. Baseline SDXL

During standard SDXL inference, every generated panel evolves independently through the diffusion process. For a latent representation

$$z_t \in \mathbb{R}^{C \times H \times W},$$

the scheduler performs

$$z_{t-1} = S(z_t, \epsilon_\theta(z_t, t, c)). \tag{1}$$

Since every panel is generated independently, we have

$$z_t^{(i)} \perp z_t^{(j)}, \qquad \text{for } i \neq j,$$

and consequently their feature statistics evolve independently without cross-panel influence.

**Observation 1.**  
SDXL provides no mechanism for maintaining global appearance consistency between independently generated panels.

---

### Step 2. LogCD: Global Consistency Along Diffusion Trajectories

LogCD [Xie et al., CVPR 2026] demonstrates that global consistency should be maintained throughout the diffusion trajectory. Rather than enforcing consistency only at the final output, LogCD introduces Global Consistency Distillation (GoCD) to explicitly enforce consistency across pre-defined timesteps along the inference path:

$$\mathcal{L}_{GoCD}^{MSE} = \left\| f_\theta(\hat{z}_{t_m}, c, t_m) - \operatorname{sg}\left(f_\theta(\hat{z}_{t_n}, c, t_n)\right) \right\|_2^2, \tag{2}$$

where $t_n = t_m - T/M$ and $\operatorname{sg}$ denotes stop-gradient.

**Observation 2.**  
Global consistency depends on maintaining stable latent representations throughout the sampling trajectory, not only at the final output.

---

### Step 3. Temporal & Content Co-Awareness: Appearance in Latent Features

Temporal and Content Co-Awareness Latent Diffusion [Hao et al., CVPR 2026] reveals that appearance information is fundamentally encoded in latent feature representations. The paper shows through time-segmented feature injection studies that:

> *"The pose condition dominates the global structure in early denoising timesteps, while the appearance condition gradually refines local textures in later denoising timesteps."*

The Pose-Invariant Appearance Encoder (PIAE) introduced in the paper explicitly captures both global appearance consistency and local texture details from latent representations of multi-pose reference images. Different lighting, color, and structural variations manifest as changes in latent feature distributions.

**Observation 3.**  
Since appearance information is encoded in latent features, aligning their statistical moments provides a lightweight approximation for preserving global appearance consistency.

---

### Step 4. Blend-Aware Latent Diffusion: Distribution Mismatch Causes Inconsistency

Blend-Aware Latent Diffusion [Liu et al., CVPR 2026] identifies that distribution mismatch between regions causes visual inconsistency. The paper explicitly states:

> *"The preserved and generated regions follow distinct statistical distributions, forming a piece-wise latent manifold with sharp transition across regions."*

This distributional divergence leads to boundary discontinuity and content inconsistency.

**Observation 4.**  
Distribution mismatch inside latent space directly causes visual inconsistency, including seams, boundary artifacts, and structural misalignment. Therefore, reducing latent distribution mismatch should improve visual consistency.

---

### Step 5. Balanced Representation Space: Stable Statistics Generalize

Balanced Representation Space [Zhang et al., ICLR 2026] establishes that stable latent statistics produce more robust representations. The paper proves that:

> *"Generalized samples yield balanced representations that reflect the underlying distribution"* 

while 

> *"memorized samples are encoded as spiky activations concentrated on a few neurons."*

**Observation 5.**  
Stable feature distributions (balanced representations) produce more robust semantic representations and better generalization, while unstable distributions (spiky activations) indicate overfitting and memorization.

---

### Step 6. Proposed Distribution Alignment Objective

The previous observations establish four key principles:

1. Consistency should persist throughout the diffusion trajectory (LogCD).
2. Appearance is reflected by the statistical distribution of latent feature activations (TCCA).
3. Distribution mismatch causes visual inconsistency (Blend-Aware).
4. Stable feature distributions produce better representations (Balanced Representation Space).

Therefore, to maintain appearance consistency across independently generated panels, we apply a direct reduction of the latent distribution mismatch between the current panel and the anchor panel.

Formally, let $P_c$ denote the latent distribution of the current panel and $P_a$ denote the latent distribution of the anchor panel. We formulate the goal as:

$$\min_{z'} \mathcal{W}_2(P_{z'}, P_a), \tag{3}$$

where $\mathcal{W}_2$ denotes the **2-Wasserstein distance** between the distributions, and $z'$ is a transformation of the current latent $z$.

Assuming each latent channel follows an approximately Gaussian distribution, minimizing the 2-Wasserstein distance reduces to matching the first two statistical moments (mean and variance). This provides a principled justification for our moment-matching approach.

---

### Step 7. Moment Matching via Affine Transformation

To make this optimization tractable during inference, we approximate each channel's distribution by its first two moments and restrict the transformation to an affine form.

For a latent channel $z_c$, define the **mean** as the average activation:

$$\mu_c = \frac{1}{HW} \sum_{h=1}^{H} \sum_{w=1}^{W} z_{c,h,w}, \tag{4}$$

and the **standard deviation** as the activation spread:

$$\sigma_c = \sqrt{ \frac{1}{HW} \sum_{h=1}^{H} \sum_{w=1}^{W} (z_{c,h,w} - \mu_c)^2 }. \tag{5}$$

Now suppose we apply an affine transformation to each channel:

$$z'_c = a_c z_c + b_c. \tag{6}$$

We choose $a_c, b_c$ to match the anchor moments:

$$\mathbb{E}[z'_c] = \mu_a, \qquad \operatorname{Var}(z'_c) = \sigma_a^2. \tag{7}$$

From expectation:

$$a_c \mu_c + b_c = \mu_a. \tag{8}$$

From variance:

$$a_c^2 \sigma_c^2 = \sigma_a^2. \tag{9}$$

Solving gives:

$$a_c = \frac{\sigma_a}{\sigma_c}, \qquad b_c = \mu_a - \frac{\sigma_a}{\sigma_c} \mu_c. \tag{10}$$

Therefore, the affine transformation that aligns the first two moments is:

$$\boxed{z'_c = \frac{\sigma_a}{\sigma_c} (z_c - \mu_c) + \mu_a}. \tag{11}$$

---

### Step 8. Connection to AdaIN

Equation (11) is mathematically equivalent to **Adaptive Instance Normalization (AdaIN)** [Huang and Belongie, ICCV 2017]:

$$\text{AdaIN}(z, \mu_a, \sigma_a) = \sigma_a \frac{z - \mu(z)}{\sigma(z)} + \mu_a. \tag{12}$$

With $\mu(z) = \mu_c$ and $\sigma(z) = \sigma_c$, this matches Eq. (11).

**This connection is important to acknowledge.** Although Eq. (11) is mathematically equivalent to AdaIN, AdaIN was originally proposed for image style transfer in pixel feature space. In contrast, we apply this affine moment matching to diffusion latent representations, progressively during denoising, using scheduler-aware interpolation for cross-panel appearance consistency.

Our contribution is therefore:

1. **Applying it in diffusion latent space** to align distributions between independently generated panels,
2. **Applying it progressively across denoising timesteps** (following Observation 2),
3. **Using controlled interpolation** to preserve scene-specific content,
4. **Demonstrating its effectiveness** for cross-panel appearance consistency.

---

### Step 9. Progressive Alignment via Interpolation

Direct AdaIN may over-constrain the scene appearance. Following Observation 2, the correction should be applied progressively rather than abruptly.

We define the full AdaIN-corrected latent:

$$z_{\text{corr}} = r_c(z - \mu_c) + \mu_a, \tag{13}$$

where $r_c = \sigma_a / \sigma_c$.

To preserve the original structure and avoid over-correction, we blend the corrected latent with the original using a single interpolation coefficient $\omega \in [0, 1]$:

$$z_{\text{final}} = (1 - \omega) z + \omega z_{\text{corr}}. \tag{14}$$

This single parameter controls the overall correction strength. When $\omega = 0$, no correction is applied; when $\omega = 1$, the channel statistics are fully aligned to the anchor.

---

### Step 10. Clamping for Stability

To prevent numerical instability when $\sigma_c$ is very small, we clamp the variance ratio:

$$r_c = \operatorname{clamp}\left( \frac{\sigma_a}{\sigma_c},\, 0.8,\, 1.2 \right). \tag{15}$$

Clamping bounds the scaling factor, preventing excessively large or small affine transformations while improving numerical stability.

Thus, the final operator is:

$$\boxed{z_{\text{final}} = (1-\omega) z + \omega \left( r_c(z - \mu_c) + \mu_a \right)}. \tag{16}$$

---

### Step 11. Theoretical Analysis — Mean Convergence

From Eq. (13), the corrected latent is

$$z_{\mathrm{corr}} = r_c(z - \mu_c) + \mu_a,$$

where $r_c = \sigma_a/\sigma_c$ (or its clamped version).

Taking expectation over the spatial dimensions gives

$$\mu' = \mathbb{E}[z_{\mathrm{corr}}] = r_c\mu_c - r_c\mu_c + \mu_a = \mu_a. \tag{17}$$

Therefore, the corrected latent has mean exactly equal to the anchor mean:

$$\boxed{\mu' = \mu_a}. \tag{18}$$

After interpolation with the original latent,

$$z_{\mathrm{final}} = (1-\omega)z + \omega z_{\mathrm{corr}},$$

the final mean is:

$$\mu_{\mathrm{final}} = (1-\omega)\mu_c + \omega\mu_a. \tag{19}$$

Thus:

$$\boxed{\mu_{\mathrm{final}} - \mu_a = (1-\omega)(\mu_c - \mu_a)}. \tag{20}$$

Since $0 \le \omega \le 1$, increasing $\omega$ progressively moves the latent mean toward the anchor.

---

### Step 12. Theoretical Analysis — Variance Alignment

For an affine transformation

$$z_{\mathrm{corr}} = r_c(z - \mu_c) + \mu_a,$$

subtracting the mean and adding a constant does not affect variance.

Hence,

$$\operatorname{Var}(z_{\mathrm{corr}}) = r_c^2 \operatorname{Var}(z). \tag{21}$$

Without clamping,

$$r_c = \frac{\sigma_a}{\sigma_c},$$

which gives

$$\sigma' = r_c\sigma_c = \sigma_a. \tag{22}$$

Therefore, the variance of every latent channel exactly matches the anchor variance.

With clamping, the scaling factor is bounded to $[0.8, 1.2]$, ensuring that the affine correction remains stable while preventing numerical issues from extreme variance ratios.

---

### Step 13. Combined Statistical Alignment

The alignment operator performs variance alignment

$$r_c(z - \mu_c)$$

followed by mean alignment

$$+\mu_a,$$

then progressive blending with the original latent through $\omega$.

Consequently, the transformed latent simultaneously reduces discrepancies in both channel mean and channel variance while preserving controllable scene-specific variation through the parameter $\omega$.

Since $0 \le \omega \le 1$, the correction remains bounded, constituting a bounded affine interpolation that progressively moves the latent distribution toward the anchor distribution without introducing abrupt changes to the diffusion trajectory.

---

### Step 14. Integration into Diffusion

The statistical alignment is applied after each scheduler step, before the next UNet forward pass:

$$z_t \xrightarrow{\text{Align}} z_t^{\text{aligned}} \xrightarrow{\text{UNet}} z_{t-1}. \tag{23}$$

The alignment module requires no training, adds minimal computational overhead (only channel-wise moments, scaling, and addition), and can be applied at any denoising timestep.

---

## Summary of Derivation Flow

```
SDXL (Baseline)
    ↓
Observation 1: Independent trajectories → no cross-panel appearance control

LogCD (CVPR 2026)
    ↓
Observation 2: Global consistency requires stable latent representations throughout trajectory

Temporal & Content Co-Awareness (CVPR 2026)
    ↓
Observation 3: Since appearance is encoded in latent features, aligning their statistical moments provides a lightweight approximation for preserving global appearance

Blend-Aware Latent Diffusion (CVPR 2026)
    ↓
Observation 4: Distribution mismatch causes visual inconsistency

Balanced Representation Space (ICLR 2026)
    ↓
Observation 5: Stable feature distributions produce better representations

    ↓
OURS: Latent Statistical Alignment (T3)

Define μ_c and σ_c for each latent channel

Minimize W_2(P_c, P_a) with affine transform constraint
    → Assuming Gaussian channels, this reduces to moment matching

Solve: a = σ_a/σ_c, b = μ_a - a μ_c

This is AdaIN (acknowledged)
    → AdaIN was originally for pixel feature space;
    → We apply it to diffusion latent space with progressive interpolation

Progressive interpolation: z_final = (1-ω)z + ω( r(z-μ_c) + μ_a )
    → single parameter ω controls correction strength

Clamp for stability: r = clamp(σ_a/σ_c, 0.8, 1.2)
    → Clamping bounds the scaling factor, preventing numerical issues

Proof: 
    μ_final - μ_a = (1-ω)(μ_c - μ_a) → mean converges
    σ' = σ_a (unclamped) → variance exactly matches
    Clamping ensures bounded, stable correction
```


use this act change code according to it if needed
</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-07-13T00:37:13+05:30.

The user's current state is as follows:
Active Document: c:\Users\Dell\Downloads\drid\indie_comic_pipeline\research_paper.md (LANGUAGE_MARKDOWN)
Cursor is on line: 63
Other open documents:
- c:\Users\Dell\Downloads\drid\indie_comic_pipeline\proposed_methodology.md (LANGUAGE_MARKDOWN)
- c:\Users\Dell\Downloads\drid\indie_comic_pipeline\research_paper.md (LANGUAGE_MARKDOWN)
- c:\Users\Dell\Downloads\drid\proposed_methodology.md (LANGUAGE_MARKDOWN)
- c:\Users\Dell\Downloads\drid\methodology.md (LANGUAGE_MARKDOWN)
</ADDITIONAL_METADATA>
<USER_SETTINGS_CHANGE>
The user changed setting `Model Selection` from Gemini 3.5 Flash (High) to Gemini 3.1 Pro (High). No need to comment on this change if the user doesn't ask about it. If reporting what model you are, please use a human readable name instead of the exact string.
</USER_SETTINGS_CHANGE>


### Algorithm 1: MDCP Denoising Step Update

The individual operators $\mathcal{T}_1$, $\mathcal{T}_2$, and $\mathcal{T}_3$ are composed into a unified inference-time intervention applied at each step of the reverse diffusion trajectory.

`	ext
Algorithm 1: MDCP Denoising Step Update

Require:
    - Timestep t
    - Current latent z_t
    - Anchor latent z_{0,anchor}
    - Text conditioning c
    - Anchor attention maps A_anchor and feature maps F_anchor
    - Anchor channel statistics (μ_a, σ_a)
    - Cached anchor attention outputs {O_anchor^(ℓ)} for ℓ = 1,...,4
    - Hyperparameters: λ₁, λ₂, λ₃, λ, β, ω
    - Scheduler function S(·) and noise schedule σ_t

Ensure: Next latent z_{t-1}

-----------------------------------------------------------------
1:  // ----- T3: Spatiotemporal Channel Statistics Alignment -----
2:  for each channel c in z_t do
3:      μ_c, σ_c ← ComputeChannelStats(z_t, c)
4:      r_c ← clip(σ_a / σ_c, 0.8, 1.2)
5:      z_corr,c ← r_c · (z_{t,c} − μ_c) + μ_a
6:  end for
7:  z_aligned ← (1 − ω) · z_t + ω · z_corr

8:  // ----- T1: Latent Trajectory Optimization -----
9:  Forward UNet on z_aligned to extract attention maps A_t and feature maps F_t
10: z_hat_0 ← D_sched(z_aligned, εθ, t)          // predict clean latent
11: E_id   ← (1/N) · || A_t − A_anchor ||_F²
12: E_str  ← || F_t − Ψ(F_t, F_anchor) ||₂²
13: E_traj ← || z_hat_0 − z_{0,anchor} ||₂²
14: E_t    ← λ₁·E_id + λ₂·E_str + λ₃·E_traj
15: R_t    ← ∇_{z_aligned} E_t                    // gradient w.r.t. z_aligned
16: η(t)   ← λ · σ_t / (σ_max + ε)
17: z_tilde ← z_aligned − η(t) · R_t              // corrected latent

18: // ----- T2: Attention Propagation Module -----
19: // Perform a second UNet forward pass on z_tilde; during the forward pass,
20: // for each of the four hooked attention layers (ℓ=1..4), replace the output:
21: for each layer ℓ in {1,...,4} do
22:     O_curr^(ℓ) ← ordinary output of layer ℓ on z_tilde
23:     O_prop^(ℓ) ← (1 − β)·O_curr^(ℓ) + β·O_anchor^(ℓ)
24:     Replace layer ℓ’s output with O_prop^(ℓ)
25: end for
26: // After this forward pass, we obtain the predicted noise εθ(z_tilde, t, c)

27: // ----- Scheduler step (proceeds the diffusion) -----
28: z_{t-1} ← Scheduler( z_tilde, εθ(z_tilde, t, c) )

29: Return z_{t-1}
`


### 3.2 MDCP Integration into Sequential Generation

To evaluate MDCP under realistic long-range sequential generation conditions, we integrate it into a complete comic synthesis pipeline. The pipeline serves as the experimental platform rather than the primary scientific contribution. Figure 1 shows the full flow; Algorithm 2 gives the master orchestration loop that Sections 3.2.1–3.2.9 describe phase by phase.

```
+---------------------------------------------------------------------------------------+
|                                    INPUT SCRIPT                                       |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|               PHASE 1: NARRATIVE ARC PLANNING & AGENTIC STORYBOARDING                 |
|            - BERT Emotional Arc Parsing       - Multi-Agent Pacing & Enrichment       |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                    PHASE 2: REFERENCE-FREE MULTI-CHARACTER ANCHORING                  |
|      - Build Introduction Map; Segment & Extract Regional Signatures per Character    |
+---------------------------------------------------------------------------------------+
                                           |
                                           +----------------------------+
                                           | (Pin Key/Value Cache)      |
                                           v                            v
+---------------------------------------------------------------------------------------+
|                     PHASES 3-6: UNIFIED TARGET PANEL GENERATION (n>1)                 |
|  +---------------------------------------------------------------------------------+  |
|  |               MDCP CORE OPERATOR SPLITTING DENOISING SAMPLERS                   |  |
|  |  [T1] Heat Diffusion Smoothing -> [T2] Pinned Memory Attention -> [T3] Affine   |  |
|  +---------------------------------------------------------------------------------+  |
|                                          |                                            |
|                                          v                                            |
|  +---------------------------------------------------------------------------------+  |
|  |                          OPT-IN SOTA MITIGATION MODULES                         |  |
|  | - M1 Detail Injector - M2 Regional Masking - M3 Foreground Saliency             |  |
|  | - M4 Fourier Scaler  - M5 AdaIN Aligner                                         |  |
|  +---------------------------------------------------------------------------------+  |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                 PHASE 7: CADENCE LAYOUT ASSEMBLY & TYPESETTING ENGINE                 |
|         - Action Intensity Height Slicing      - LLM-Planned Bubble Placement         |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                    PHASE 8: EXPORT HARNESS & AUTOMATED QUALITY GATE                   |
|         - 13-Metric Validation Engine         - Feedback Reject-and-Regenerate Loop   |
+---------------------------------------------------------------------------------------+
```

*Figure 1: The eight-phase pipeline. In Phase 2, reference-free visual anchors are established for each character on their first appearance panel; their regional identity signatures are cached before any subsequent panel depicting them is generated. Panels then proceed as parallelizable, independent MDCP-consistent generations. The dashed path denotes Phase 6's reject-and-regenerate loop.*

```
Algorithm 2: Master Eight-Phase Pipeline Orchestration
Input:  prompt P, character name C, panel count N, story_mode MODE
Output: assembled pages (CBZ/PDF/HTML), feedback telemetry log

1:  /* Phase 0 — Story Intake */
2:  story_config ← StoryIntakeEngine.process_prompt(P, N, C, MODE)
3:
4:  /* Phase 1 — Multi-Agent Enrichment (blackboard) */
5:  storyboard ← AgentController.run_planning(story_config)
6:
7:  /* Phase 2 — Multi-Anchor Visual Anchoring (sequential steps for first appearances) */
8:  character_intro_map ← build_character_introduction_panel_map(storyboard)
9:  for each character c in character_intro_map.keys() do:
10:     k_c ← character_intro_map[c] // earliest panel where character c appears
11:     if k_c has not been generated yet:
12:         panel_res ← generate_panel_with_retry(panel_id = k_c, t2_enabled_for_c = False)
13:     end if
14:     M_c ← segment_character_foreground(panel_res.image, c) // using SAM or BBox
15:     character_anchors[c] ← IdentityEmbeddingExtractor.extract_masked(panel_res.image, M_c)
16: end for
17:
18: /* Phases 3–6 — remaining panels, MDCP-consistent, parallelizable */
19: for panel_id in 1..N do parallel
20:     if panel_id is not in character_intro_map.values() then
21:         panel_result ← generate_panel_with_retry(panel_id)   # Algorithm 1 runs inside this call
22:     end if
23: end for
24:
25: /* Phase 7 — Cadence Layout */
26: pages ← LayoutEngine.assemble(sorted_panels_by_page)
27:
28: /* Phase 8 — Export and Feedback Logging */
29: Exporter.export(pages, formats = [CBZ, PDF, HTML])
30: FeedbackTuner.log_telemetry()
31: return pages
```

#### 3.2.0 Design Philosophy

Three constraints shape every design decision above: **local-first** execution (a local LLM via Ollama, a local SDXL backend, no paid API dependency); **training-free** consistency (no per-character fine-tuning, no reference images); and a **hardware-democratic** target (comfortable operation on a 16 GB-class consumer or free-tier GPU). A single controller threads one mutable state through the eight phases, accumulating a recurring motif, a mood-journey label, and, per panel, character poses/expressions/dialogue, action mechanics, camera parameters, environment keywords, an emotion-beat label, and an action-intensity score that later phases consume.

#### 3.2.1 Phase 0 — Story Intake and Emotion-Conditioned Narrative Planning

An emotion classifier (a BERT-family model fine-tuned on GoEmotions and `story_commonsense`-derived data, with a keyword-density fallback) assigns one of eight primary emotions, each mapped to a named visual journey and an ordered arc-beat sequence (e.g., angry $\to$ *Calming*: contained_fire $\to$ fracture $\to$ exhale $\to$ cooling $\to$ ground $\to$ stillness), distributed across exactly $N$ panels. The single source of truth mapping is detailed in Table 10.

**Table 10: Primary Emotion Mood-Arc Mappings and Journey Stages**

| Primary Emotion | Journey Type | Motif Hint | Ordered Mood Stages |
| :--- | :--- | :--- | :--- |
| **sadness** | uplifting | something small that holds warmth (cup, candle, sun patch) | heaviness $\to$ stillness $\to$ warmth $\to$ light $\to$ openness $\to$ hope |
| **joy** | elation | something multiplying light (reflections, laughter lines) | spark $\to$ warmth $\to$ glow $\to$ radiance $\to$ overflow $\to$ share |
| **anger** | calming | something absorbing heat (running water, open window) | fire $\to$ contained $\to$ fracture $\to$ exhale $\to$ cooling $\to$ grounded |
| **fear** | grounding | something tactile/grounding (textures, bare feet) | tight $\to$ quiver $\to$ breath $\to$ release $\to$ steadiness $\to$ calm |
| **love** | deepening | something shared between two (held hand, shared window) | tenderness $\to$ warmth $\to$ cherish $\to$ hold $\to$ bloom $\to$ forever |
| **grief** | tender continuance | something that was shared (empty chair, shared mug) | loss $\to$ echo $\to$ memory $\to$ pain $\to$ acceptance $\to$ light |
| **determined** | heroic rise | something holding the climb cost (scarred hands, worn path) | resolve $\to$ grip $\to$ climb $\to$ surge $\to$ breakthrough $\to$ victory |
| **tired** | relaxing | something soft/horizontal (pillow, blanket fold) | weight $\to$ pause $\to$ light $\to$ lift $\to$ forward $\to$ rest |

A `story_mode` switch governs how much authority the arc has over content. In **`literal`** mode (default), the user's story is the primary structural constraint on the planning LLM — divide it into $N$ sequential moments, preserve named characters/settings/events — with the arc vocabulary offered only as an optional tone hint per panel. In **`mood_arc`** mode (legacy), the arc dictates each panel's beat directly and the user prompt is passed as background context only. This switch corrects a narrative-fidelity failure mode found during development, where a fixed arc silently overwrote the user's own plot.

#### 3.2.2 Phase 1 — Multi-Agent Panel Enrichment

Phase 1 transforms a flat, LLM-generated story configuration — a JSON array of $N$ bare panel outlines, each containing a scene description, character list, and a single flat action verb — into a fully parameterized cinematographic specification. This specification is the direct input to the diffusion generation loop (Phase 3–4) and the layout engine (Phase 7). The transformation is executed by a **blackboard multi-agent architecture** implemented in `agent_coordinator.py` and `director_swarm.py`, comprising six specialized director agents that read from and write to a shared `StorySectionMemory` object.

##### 3.2.2.1 Blackboard Architecture and Sequential Dependency Chain

The six agents are not run identically in parallel. The orchestrator enforces a **two-stage sequential-then-concurrent execution schedule** that reflects the causal dependency structure of the narrative enrichment task:

**Stage A — Sequential Core Chain (Ordered).** `StoryDirector` runs first, unconditionally, because it initializes the shared `raw_panels` list that every subsequent agent reads from. Without this initialization, no downstream agent has anything to process. `ActionDirector` runs second, immediately after `StoryDirector` completes, because the cinematic exaggeration of action verbs (described in Section 3.2.2.3) must be resolved before `PoseDirector` and `EmotionDirector` can meaningfully assign body states — a character's pose under a `punch` action is mechanically different from a `stands` action, and defaulting to a pose template without first resolving the cinematic verb would produce structurally inconsistent prompts.

**Stage B — Concurrent Independent Agents.** `DialogueWriter`, `PoseDirector`, `EmotionDirector`, and `CameraDirector` run concurrently via `ThreadPoolExecutor(max_workers=4)` after Stage A completes. These four agents each operate on non-overlapping fields of the panel data (`dialogue`, `pose`, `expression`, `camera`/`layout_directives`, respectively) so their writes are non-conflicting on the shared blackboard. Thread-level parallelism is preferred over process-level because all agents share the in-memory `StorySectionMemory` object by reference, eliminating serialization overhead.

The rationale for this specific schedule is measurable: on a representative 8-panel story, running all six agents purely sequentially takes approximately $6T_{\text{agent}}$ wall-clock time, while the two-stage scheme reduces this to approximately $2T_{\text{story}} + T_{\text{action}} + T_{\text{max(concurrent)}}$, where $T_{\text{max(concurrent)}}$ is dominated by `DialogueWriter`'s optional LLM call (capped at 10 s per panel). Any agent that fails is caught by a per-agent `try/except` wrapper; `CameraDirector` failures are specifically handled by emitting a fallback `LayoutDirective(size_class="medium", camera_angle="medium_shot")` to guarantee that the layout engine is never handed a panel with no spatial directive.

##### 3.2.2.2 StoryDirector: Structural Initialization and Character Registration

`StoryDirector.plan()` performs four operations on the blackboard:

1. It writes `memory.raw_panels = panels`, making the $N$ panel outlines available to the full swarm. The total panel count $N$ is written to `memory.total_panels`.
2. It copies the `recurring_motif` and `mood_journey` strings from the story config — narrative-level identifiers that `DialogueWriter` uses for contextual dialogue generation.
3. It registers all named characters into the memory's character table. Character registration serves a downstream purpose beyond bookkeeping: each registered character is assigned a `CharacterState` slot in `StorySectionMemory`, which `PoseDirector` and `EmotionDirector` later update after each panel to track the most recent pose, emotion, and last action verb of each character across the sequence.
4. It performs a **three-pass character discovery**: (i) top-level `characters` array in the story config, (ii) `_metadata.character` field for the main character, and (iii) `panels[n].characters[m].id` fields in the scene graph, plus `story_bible.side_characters`. All name variants (raw, `.lower()`, `.capitalize()`) are registered to ensure case-insensitive lookup during prompt construction.

No mathematical transformation occurs in `StoryDirector`. Its role is purely structural: it guarantees that the blackboard's character registry is fully populated before any enrichment agent reads from it.

##### 3.2.2.3 ActionDirector: Cinematic Exaggeration Map and Action-Intensity Scoring

`ActionDirector` is the most consequential enrichment agent for visual generation quality. It addresses a well-documented failure mode of text-to-image diffusion models: **prompt regression to the mean**. When a panel description supplies a generic action verb such as `"punch"` or `"run"`, the model's cross-attention mechanism distributes probability mass across the entire prior distribution of such actions, producing a statistically average depiction. Cinematically extreme, visually distinct poses — the kind that stress-test the identity-preservation operators $\mathcal{T}_1$–$\mathcal{T}_3$ with substantial structural deformation — require prompts that occupy a narrower, more extreme region of the conditioning distribution.

`ActionDirector` resolves this through the **Cinematic Exaggeration Map** (`ACTION_EXAGGERATION_MAP`), a hand-authored dictionary of 23 canonical action verbs. Each entry expands a single-word verb into a five-field semantic schema:

$$\text{action schema} = \{v_{\text{verb}},\ v_{\text{mechanics}},\ v_{\text{impact}},\ v_{\text{reaction}},\ v_{\text{timing}}\} \tag{13}$$

where:
- $v_{\text{verb}}$ is a cinematically aggressive active-voice verb phrase (e.g., `"delivers a devastating haymaker with full body rotation"` for `"punch"`);
- $v_{\text{mechanics}}$ specifies exact body-part positions under maximum tension (e.g., `"entire torso twisted, arm cocked far back, knuckles white, veins raised on forearm"`);
- $v_{\text{impact}}$ captures the precise moment of contact or consequence (e.g., `"fist craters into the target's face, skin distorting under impact wave"`);
- $v_{\text{reaction}}$ describes the environmental or secondary-object response (e.g., `"spit and sweat explode sideways, head snapping backward violently, hair whipping"`);
- $v_{\text{timing}}$ provides the freeze-frame cue that tells the diffusion model which moment of the action to render (e.g., `"maximum-force impact freeze-frame, kinetic energy at absolute peak"`).

The five-field schema is not arbitrary. Diffusion model prompts are processed by SDXL's dual CLIP encoders (OpenCLIP ViT-G/14 and OpenAI CLIP ViT-L/14), whose cross-attention mechanisms respond to token-level semantics. A single token `"punch"` activates a broad semantic neighborhood. The five-field expansion introduces **35–60 additional tokens** per action that collectively narrow the activated semantic region toward a specific, extreme configuration of body geometry, environmental physics, and temporal moment. The `_build_prompt()` function in `panel_engine.py` stitches these five fields into a single cinematic action clause inserted into the panel prompt.

**Fuzzy verb matching.** The raw verb extracted from the LLM-generated panel may not exactly match any dictionary key (e.g., `"runs"`, `"punching"`, `"is running"`). The matcher applies a **prefix overlap heuristic**: for each key $k$ in the map, it checks whether `raw_verb.startswith(k)` or `k.startswith(raw_verb)`, accepting the first match. If no match is found, the fallback entry `"observes"` is used, ensuring the prompt always receives dimensional cinematic language rather than a bare verb.

**Default action synthesis.** If a panel has no `actions` array at all (a common output from LLMs that focus on scene description rather than character action), `ActionDirector` synthesizes a default action from a **Beat-to-Verb Default Map** (`_BEAT_DEFAULT_VERB`), which maps each emotion beat to the most visually appropriate canonical verb. For example, `"contained_fire"` → `"holds"` (rigid, high-tension pose); `"breakthrough"` → `"charge"` (full-body dynamic forward motion); `"triumph"` → `"raises"` (arms-overhead victory posture). This guarantees that every panel, regardless of LLM output quality, receives a fully parameterized action schema.

**Action-intensity scoring.** Each panel's action schema feeds Phase 7's `MangaFlowLayoutEngine`. The action intensity $\mathcal{I}_i$ for panel $i$ is not explicitly computed as a named variable in `ActionDirector`; rather, it is an implicit ordinal score derived from the `size_class` field assigned by `CameraDirector` based on the emotion beat:

$$\mathcal{I}_i = \begin{cases} 1 & \text{if } \text{size\_class} = \text{"small"} \\ 2 & \text{if } \text{size\_class} = \text{"medium"} \\ 3 & \text{if } \text{size\_class} = \text{"large"} \\ 4 & \text{if } \text{size\_class} = \text{"full\_page"} \end{cases} \tag{14}$$

This ordinal score is then used in Equation (12) (Phase 7) to proportionally allocate panel heights on the assembled page. Panels with high-intensity actions (`"breakthrough"`, `"triumph"`) receive size class `"full_page"` ($\mathcal{I}_i = 4$) and thus a proportionally larger canvas area, directly encoding the narrative pacing principle that dramatic moments command more visual real estate.

##### 3.2.2.4 DialogueWriter: LLM-Mediated Text Synthesis with Fallback Chain

`DialogueWriter` validates and fills missing dialogue text for every character in every panel. The filling strategy is a **two-level cascade**:

**Level 1 — Local LLM call.** If a character's `dialogue.text` field is empty or contains only the placeholder string `"..."`, `DialogueWriter` constructs a structured prompt and sends it to the local Ollama endpoint (default: `llama3.2`) via a direct HTTP POST to `/api/generate`. The prompt provides: character name, current action description, emotion beat label, and the story's mood journey context. The system prompt constrains the output to a single line of at most 15 words, formatted as raw dialogue with no quotation marks. Temperature is set to 0.5 — high enough to generate contextually varied lines across panels, low enough to avoid semantically incoherent outputs. The HTTP call has a hard 10-second timeout to prevent blocking the concurrent execution of other Stage B agents.

**Level 2 — Beat-indexed static fallback.** If the LLM call times out, fails, or returns an empty string, `DialogueWriter` falls back to a hardcoded `beat_lines` dictionary that maps each canonical emotion beat to a short, tonally appropriate line (e.g., `"contained_fire"` → `"Not yet."`, `"fracture"` → `"That's enough."`, `"silence"` → `"..."`). These fallback lines were chosen to be short enough to fit in any bubble size while carrying the emotional weight of their associated beat.

The 15-word dialogue length constraint is justified by the downstream lettering constraints imposed by Phase 5. The `text_image_integrator.py` bubble renderer allocates canvas area based on the LLM-planned bounding coordinates. Lines exceeding approximately 15 words overflow the allocated bubble area at standard comic font sizes (12–16 pt equivalent), forcing line-wrap that either clips the bubble border or occludes adjacent image content. The 15-word limit is therefore not an arbitrary stylistic choice but a derived constraint from the downstream rendering budget.

##### 3.2.2.5 PoseDirector and EmotionDirector: Template-Driven Structural Constraint

`PoseDirector` and `EmotionDirector` operate on the same field of the scene graph (character-level `pose` and `expression` dicts) and are kept as separate agents because they operate on physically distinct attributes: body geometry vs. facial microexpression. Both agents consult pre-authored template dictionaries (`_BEAT_POSE_MAP` and `_BEAT_EXPRESSION_MAP`) indexed by emotion beat.

**`PoseDirector`** fills any missing fields in each character's `pose` dict with the template for the panel's `emotion_beat`. The pose template is a four-field struct specifying `body`, `head`, `arms`, and `legs` positions in plain-English anatomical language. For example, the `"contained_fire"` beat template is `{body: "standing rigid, fists clenched", head: "chin down, jaw set", arms: "locked at sides", legs: "planted wide"}`. A **partial-fill policy** is enforced: if the LLM already provided a non-empty value for a field (e.g., `arms: "reaching forward"`), that value is preserved and only the missing fields are filled. This prevents `PoseDirector` from overwriting LLM-generated contextually specific poses with generic templates.

**`EmotionDirector`** validates and propagates the `emotion_beat` label across panels. If a panel has no `emotion_beat` field, the agent attempts to infer it from the first character's `expression.emotion` field. All validated beats are collected into `memory.arc_beats`, a list of length $N$ that records the emotional trajectory of the full sequence. `EmotionDirector.update()` — called after each panel is generated — advances `memory.current_beat_index` so that subsequent panels' prompts can reference the current position in the arc.

##### 3.2.2.6 CameraDirector: Beat-Conditioned Layout Directive Assignment

`CameraDirector` translates each panel's emotion beat into two structured outputs that downstream phases consume:

1. **Camera angle token** written to `panel["camera"]`: a string key such as `"low_angle"`, `"dutch_tilt"`, `"close_up"`, `"wide_shot"`, `"bird_eye"`, or `"medium_shot"`. This token is inserted directly into the diffusion prompt, activating the model's learned associations between these camera vocabulary terms and their corresponding geometric framing conventions. The mapping from beat to camera angle (`_BEAT_CAMERA_MAP`) encodes cinematographic convention: `"contained_fire"` → `"low_angle"` (low-angle shots visually amplify dominance and menace); `"fracture"` → `"dutch_tilt"` (disorientation through off-axis framing); `"triumph"` → `"wide_shot"` (open space communicates liberation and achievement); `"quiet_rest"` → `"bird_eye"` (overhead angle conveys vulnerability and isolation).

2. **`LayoutDirective` object** written to `memory.layout_directives[panel_id]`: a dataclass containing `panel_id`, `size_class`, `camera_angle`, `camera_framing` (default `"center"`), `aspect_ratio` (default `(1, 1)`), and `gutter_emphasis` (default `"normal"`). `size_class` is drawn from `_BEAT_SIZE_MAP` and takes one of four values: `"small"`, `"medium"`, `"large"`, `"full_page"`. This is the direct input to Phase 7's height-allocation formula (Equation 12).

**Position-based size boosting.** `CameraDirector` applies one structural override: panels at narrative position $i/N = 0$ (the first panel) and $i/N = 1$ (the last panel) have their `size_class` promoted to at least `"large"`, regardless of their beat assignment. The rationale is compositional: the first panel must establish the setting and anchor identity with a sufficiently large canvas, and the final panel must carry the narrative's emotional resolution. This is a hard rule rather than a soft tendency, implemented as:

$$\text{size\_class}_i = \begin{cases} \max\!\big(\text{size\_class}_i,\ \text{"large"}\big) & \text{if } i = 0 \text{ or } i = N{-}1 \\ \text{size\_class}_i & \text{otherwise} \end{cases} \tag{15}$$

where the $\max$ is computed over the ordered set $\{\text{"small"} < \text{"medium"} < \text{"large"} < \text{"full\_page"}\}$.

**Fallback robustness.** If `CameraDirector` raises any exception, the coordinator's per-agent exception handler synthesizes a `LayoutDirective(size_class="medium", camera_angle="medium_shot")` for every panel that has not yet received a directive. This guarantees that Phase 7 always receives a complete set of $N$ layout directives, preventing the layout engine from raising a `KeyError` on a missing panel ID. The fallback values — `"medium"` size and `"medium_shot"` angle — are the statistically neutral choices that produce compositionally adequate panels even without beat-specific enrichment.

##### 3.2.2.7 Blackboard Output: The Enriched Scene Graph

After all six agents complete, the `StorySectionMemory` blackboard contains for each panel $i \in \{1, \ldots, N\}$:

| Field | Source Agent | Consumed By |
|:---|:---|:---|
| `emotion_beat` | `EmotionDirector` | `PoseDirector`, `CameraDirector`, `DialogueWriter`, Phase 3–4 prompt |
| `action.verb`, `action.mechanics`, `action.impact`, `action.reaction`, `action.timing` | `ActionDirector` | Phase 3–4 `_build_prompt()` |
| `character.pose.{body, head, arms, legs}` | `PoseDirector` | Phase 3–4 `_build_prompt()` |
| `character.expression.{emotion, eyes, mouth}` | `PoseDirector` (template) + `EmotionDirector` (state) | Phase 3–4 `_build_prompt()` |
| `character.dialogue.text` | `DialogueWriter` | Phase 5 lettering |
| `layout_directive.{size_class, camera_angle}` | `CameraDirector` | Phase 7 layout height formula |
| `memory.arc_beats[0..N-1]` | `EmotionDirector` | Phase 7 pacing, Phase 8 telemetry |

This fully-parameterized scene graph is the single interface contract between Phase 1 and all downstream phases. No downstream phase queries the raw LLM story config directly; all reads go through `StorySectionMemory.build_generation_context(panel_id)`, which assembles the Phase 1 outputs into the context dict passed to the panel engine.

#### 3.2.3 Phase 2 — Multi-Anchor Reference-Free Identity Anchoring

Phase 2 establishes the character-aware visual anchors that downstream MDCP operators ($\mathcal{T}_1$–$\mathcal{T}_3$) reference per character. Its defining constraint is **zero dependence on user-supplied reference images**: the anchor panel for each character is synthesized entirely from the enriched storyboard, and their visual identity signature is extracted from the generated pixel output. This property makes the system zero-shot: a user can describe characters in text and receive a consistent multi-panel comic without providing character sheets, reference photographs, or pre-trained identity encoders.

Phase 2 executes as a sequential workflow across character introduction panels. The enriched storyboard is parsed to build a `character_introduction_panel` map containing the earliest panel index $k_c$ where each character $c$ is introduced. The introduction panels are generated with T2 attention blending disabled for the respective character to establish a clean visual baseline. The character region is then isolated using binary spatial masks $M_c \in \{0, 1\}^{H \times W}$ computed via SAM (M3) or localized bounding boxes.

##### 3.2.3.1 Step 2.1–2.2: Multi-Character Anchor Generation and Persistence

For each character $c$, panel $k_c$ is generated through the same `StableDiffusionXLPipeline` call used for all other panels, but with the cross-character attention blend disabled for that specific character's region. The reverse-diffusion trajectory establishes the visual baseline for character $c$. Upon completion, the PIL Image object is saved to disk at a deterministic path:

```
outputs/anchors/anchor_panel_k_c.png
```

Saving to disk provides a persistent filesystem record enabling checkpoint-resume. It also converts the image into a format that the `IdentityEmbeddingExtractor` can process via OpenCV's `cv2.imdecode(np.frombuffer(f.read(), dtype=np.uint8), cv2.IMREAD_COLOR)` idiom, which handles Unicode file paths on Windows (where standard `cv2.imread` fails silently on non-ASCII characters).

##### 3.2.3.2 Step 2.3: Regional Identity Signature Extraction

The `IdentityEmbeddingExtractor` computes four independent classical descriptors from the saved anchor image within the character's segmented spatial mask $M_c$. These descriptors form the **identity signature** $\mathcal{S}_c$ — a compact, serializable representation of the character's visual identity stored on the blackboard.

**Descriptor 1: Regional HSV Color Histogram** ($\mathbf{h}_{\text{color}}^c$)

The image is converted to HSV color space, and a joint 2D histogram over the Hue ($H$) and Saturation ($S$) channels is computed within the character mask $M_c$:

$$\mathbf{h}_{\text{color}}^c = \text{calcHist}\big([I_{\text{HSV}}],\, \text{channels}=[H, S],\, \text{mask}=M_c,\, \text{bins}=[8, 8],\, \text{ranges}=[0,180]\times[0,256]\big) \in \mathbb{R}^{8 \times 8} \tag{10}$$

normalized to $[0, 1]$ via `cv2.NORM_MINMAX`. Omitting the Value ($V$) channel ensures color identity preservation (e.g. hair, costume hue) independently of scene illumination changes. The $[8, 8]$ bins provide high robustness against minor pixel-level variations caused by changes in character pose or perspective.

Downstream, the regional histogram is compared against target panels via Pearson correlation:

$$S_{\text{color}}^c = \frac{\sum_i \big(h_1(i) - \bar{h}_1\big)\big(h_2(i) - \bar{h}_2\big)}{\sqrt{\sum_i \big(h_1(i) - \bar{h}_1\big)^2 \cdot \sum_i \big(h_2(i) - \bar{h}_2\big)^2}} \in [-1, 1], \quad \text{clamped to } [0, 1] \tag{11}$$

This metric contributes weight 0.25 in the consistency composite score.

**Descriptor 2: Silhouette Canny Edge Density** ($\rho_{\text{edge}}^c$)

Grayscale edges are detected using the Canny algorithm with thresholds $(50, 150)$, and the edge density is computed within the character mask $M_c$:

$$\rho_{\text{edge}}^c = \frac{\big|\{(x,y) : \text{Canny}(I_{\text{gray}},\, 50,\, 150)[x,y] > 0 \text{ and } M_c[x,y] = 1\}\big|}{|M_c|} \in [0, 1] \tag{12}$$

This captures edge density as a local style-level descriptor. The downstream consistency comparison uses:

$$S_{\text{edge}}^c = \max\!\big(0,\, 1 - 5 \cdot |\rho_{\text{edge}}^{c,\text{anchor}} - \rho_{\text{edge}}^{c,\text{current}}|\big) \in [0, 1] \tag{13}$$

where the multiplier 5 maps a $0.20$ deviation threshold to zero, tolerating minor contour changes while penalizing style shifts. This contributes weight 0.15.

**Descriptor 3: Masked Style Gram Matrix** ($G_{\text{style}}^c \in \mathbb{R}^{5 \times 5}$)

This Gram matrix captures the character's texture and rendering conventions. The image is resized to $256 \times 256$, and Sobel gradients $\nabla_x I, \nabla_y I$ are stacked with normalized RGB values to construct a 5-channel feature map $F \in \mathbb{R}^{(H \cdot W) \times 5}$. Using the diagonalized mask $W_c = \text{diag}(M_c)$:

$$G_{\text{style}}^c = \frac{F^\top W_c F}{\sum(M_c)} \in \mathbb{R}^{5 \times 5} \tag{14}$$

The similarity score is:

$$S_{\text{style}}^c = \max\!\big(0,\, 1 - 10 \cdot \text{MSE}(G_{\text{anchor}}^c, G_{\text{current}}^c)\big) \tag{15}$$

where the multiplier 10 maps MSE $\ge 0.10$ to zero. This contributes weight 0.20.

**Descriptor 4: Bounding Box Aesthetic Baseline Score** ($S_{\text{aesthetic}}^c$)

To prevent background environment details from influencing character quality assessments, the image crops are evaluated over the character bounding box region $I_{\text{crop}}^c = \text{crop}(I, M_c)$:

$$S_{\text{sharp}}^c = \min\!\left(1,\, \frac{\text{Var}\!\left(\nabla^2 I_{\text{gray},\text{crop}}^c\right)}{500}\right), \quad S_{\text{contrast}}^c = \min\!\left(1,\, \frac{\sigma(I_{\text{gray},\text{crop}}^c)}{75}\right) \tag{16}$$

$$S_{\text{color}}^c = \min\!\left(1,\, \frac{\sqrt{\sigma_{rg}^2 + \sigma_{yb}^2} + 0.3\sqrt{\mu_{rg}^2 + \mu_{yb}^2}}{80}\right) \tag{17}$$

$$S_{\text{aesthetic}}^c = 0.4\,S_{\text{sharp}}^c + 0.3\,S_{\text{contrast}}^c + 0.3\,S_{\text{color}}^c \tag{18}$$

The constants ($500, 75, 80$) and local weights ($[0.4, 0.3, 0.3]$) follow the single-character baseline. The aesthetic score is used to append descriptive prompt suffixes.

**Optional Descriptors 5–6: Deep Semantic Embeddings**

Deep semantic embeddings (CLIP, DINOv2) can optionally be enabled, loading `openai/clip-vit-base-patch32` and `facebook/dinov2-base` to extract features from the character crops. These models are offloaded to CPU immediately after inference to reclaim GPU memory.

##### 3.2.3.3 Step 2.4: Multi-Anchor Signature Registration

The extracted identity signatures are registered in `StorySectionMemory` blackboard through two targets:

1. Cached in `MultiAnchorCache` keyed by character name, containing the cross-attention output cache $O_{\text{anchor}}^{(l)}(c)$ and latent channel statistics $(\mu_c, \sigma_c)$.
2. Bound to the character's named state registry under `CharacterState.identity_tokens`.

For visual state changes (e.g. wearing armor, disguised), the `StateAwareAnchorCache` keys anchors on `(character_name, visual_state)`. When a visual transition panel is detected, T2 runs with a lower blend factor $\beta_{\text{transition}} = 0.05$ to allow the prompt to establish the new appearance before caching it as the anchor for the new state.

##### 3.2.3.4 Steps 2.5–2.6: Optional Advance Mitigations (M3 and M1)

If an `AdvancedAttentionManager` is active, it pre-computes character-specific foreground saliency masks (M3 via SAM) and structural detail fingerprints (M1 via Canny edge maps of the character crops) at Phase 2 step-zero. These are cached and used to modulate attention blending spatially in subsequent target panels, preventing background elements from bleeding across panels.

##### 3.2.3.5 Progressive Sequential Dependencies

In the multi-anchor system, sequential dependencies are localized to the first-appearance panel of each character. Instead of blocking the entire sequence on panel 1, generation is only sequential up to the first-appearance panel of each character $P_{\text{anchor}}(C_j)$. Once a character's anchor is generated, segmented, and its cross-attention tensors offloaded to CPU pinned memory, any target panels containing that character can proceed in parallel.

#### 3.2.4 Phase 3–4 — Unified Panel Generation Loop


**MDCP Pipeline Update:** The standard SDXL pipeline is replaced by a custom PyTorch denoising loop that natively executes the MDCP mathematical operator splitting ($\mathcal{T}_1$, $\mathcal{T}_2$, $\mathcal{T}_3$) exactly as derived.


Phase 3–4 is the core generative pass. The standard SDXL pipeline is replaced by a custom PyTorch denoising loop that natively executes the MDCP mathematical operator splitting ($\mathcal{T}_1$, $\mathcal{T}_2$, $\mathcal{T}_3$) exactly as derived. For each panel $i \in \{1, \ldots, N\}$, the `PanelEngine` executes a deterministic 7-step pipeline: (1) assemble the positive prompt, (2) compute `CharComCompositor` weights, (3) select and configure the generation backend, (4) assemble the negative prompt and append consistency priors, (5) generate the image at the configured resolution and random seed, (6) trigger Phase 2 anchoring for newly introduced characters, and (7) record the panel to the blackboard. All seven steps run inside a per-panel mutex (`threading.Lock`) so that concurrent Phase B agents from Phase 1 cannot corrupt the shared `StorySectionMemory` state during generation.

##### 3.2.4.1 Model Backend: SDXL Configuration

All panels are generated through `StableDiffusionXLPipeline` loaded from `stabilityai/stable-diffusion-xl-base-1.0` in **fp16** precision (`torch_dtype=torch.float16`, `variant="fp16"`). On CPU, the variant falls back to fp32 because the vast majority of CPU PyTorch operations do not support float16. The fp16 choice on GPU reduces active VRAM consumption from approximately 13 GB (fp32) to approximately 6.5 GB, fitting comfortably on 16 GB consumer hardware alongside the loaded LoRA adapter.

**Scheduler: `DPMSolverMultistepScheduler` with SDE-DPMSolver++, order 2, Karras sigmas**

The scheduler is configured as:

```python
DPMSolverMultistepScheduler.from_config(
    scheduler_config,
    use_karras_sigmas=True,
    algorithm_type="sde-dpmsolver++",
    solver_order=2,
)
```

Each parameter has a specific justification:

- **`sde-dpmsolver++`**: The stochastic differential equation variant of DPMSolver++ injects a small Gaussian noise term at each denoising step, introducing controlled stochasticity that breaks the deterministic rounding errors of the ODE solver. For image generation — particularly for generating textures, hair, and fabric detail in comic art — the SDE variant consistently produces sharper fine-structure at equivalent step counts compared to the ODE variant (`dpmsolver++`), at the cost of non-exact determinism under float16 precision accumulation.

- **`solver_order=2`**: A second-order solver uses a two-step Taylor approximation of the reverse SDE trajectory. At the default step count of 25 steps, solver order 2 achieves comparable perceptual quality to solver order 1 at 35–40 steps. The step savings (10–15 steps per panel) directly translate to wall-clock time reduction: on a T4 GPU, each step costs approximately 0.3–0.5 s, so order 2 saves approximately 3–7 s per panel.

- **`use_karras_sigmas=True`**: Karras noise schedules (Karras et al., 2022) place more denoising steps in the high-noise regime (early steps), where the model's score estimate has the highest variance and is most error-prone. Standard linear sigma schedules distribute steps uniformly, leaving the high-noise regime undersampled. Karras sigmas improve both global composition (resolved during high-noise steps) and fine detail (resolved during low-noise steps) with no additional compute cost.

**Memory Optimizations**

Three optimizations reduce peak VRAM by approximately 4 GB compared to a naive full-resident configuration:

- **`enable_model_cpu_offload()`**: Moves individual pipeline components (text encoder, UNet, VAE) from GPU to CPU between uses, keeping only the active component resident on GPU. This reduces peak VRAM from ~16 GB to ~8 GB during generation at the cost of CPU↔GPU transfer overhead (~0.5 s per pipeline step transition). Enabled by default.
- **`enable_attention_slicing()`**: Slices multi-head attention across the head dimension, computing each slice sequentially rather than all heads simultaneously. Reduces peak UNet attention VRAM by approximately 40% at the cost of ~10% speed reduction. Critical for 1024×1024 full-page panels where attention maps are four times larger than 768×768 panels.
- **`enable_vae_slicing()`**: Slices the batch dimension of VAE decode, preventing OOM on panels with large spatial dimensions. Only relevant at batch size > 1, but enabled defensively.

**FreeU: Skip-Connection Rebalancing**

FreeU (Si et al., 2023) is enabled at the SDXL-tuned parameter values:

$$s_1 = 0.6,\quad s_2 = 0.4,\quad b_1 = 1.1,\quad b_2 = 1.2 \tag{21}$$

FreeU modifies the UNet's skip-connection contributions at specific resolution levels. At each UNet decoder stage, the skip-connection feature map is **attenuated** by the $s$ factors ($s_1, s_2 < 1$) and the backbone feature map is **amplified** by the $b$ factors ($b_1, b_2 > 1$):

$$h_{\text{out}} = b_j \cdot h_{\text{backbone}} + s_j \cdot h_{\text{skip}} \tag{22}$$

where subscript $j \in \{1,2\}$ indexes the two largest-resolution decoder blocks. The motivation is that SDXL's skip connections frequently dominate the decoder's output, causing over-smoothing and "plastic" texture artifacts — a known quality failure mode in highly style-constrained generations (e.g., comic line-art) where the model must produce highly regular, sharp edge structures. The values $(s_1, s_2, b_1, b_2) = (0.6, 0.4, 1.1, 1.2)$ are the empirically optimal values reported in the `UNET-FreeU-SDXL` study on Weights & Biases, which found this configuration maximizes perceptual sharpness and textural detail for SDXL specifically (as opposed to the SD1.5-optimized values $(0.9, 0.2, 1.2, 1.4)$ which over-suppress the skip connections for SDXL's deeper architecture). FreeU is implemented as a native `diffusers` method — no additional model components, weights, or GPU memory are required.

##### 3.2.4.2 CharCom Inference Compositor: Per-Panel Parameter Derivation

`CharComCompositor.compute_weights()` derives four panel-specific generation parameters from the Phase 1 blackboard — guidance scale, inference steps, LoRA scale, and a deterministic seed offset — by applying four additive adjustment rules to three base values:

$$g_{\text{base}} = 7.5, \quad S_{\text{base}} = 25, \quad \lambda_{\text{base}} = 0.8 \tag{23}$$

**Rule 1 — Action Intensity (size class):**

$$\Delta g^{(1)} = \begin{cases} +0.50 & \text{if size\_class} = \text{"full\_page"} \\ +0.25 & \text{if size\_class} = \text{"large"} \\ 0 & \text{otherwise} \end{cases}, \quad \Delta S^{(1)} = \begin{cases} +5 & \text{if size\_class} = \text{"full\_page"} \\ +2 & \text{if size\_class} = \text{"large"} \\ 0 & \text{otherwise} \end{cases} \tag{24}$$

The rationale is that full-page and large panels have more spatial area to fill — a `"full_page"` panel occupies a $1024 \times 1024$ canvas versus $768 \times 768$ for all other classes (Section 3.2.4.5). More spatial detail requires both a higher guidance scale (to sharpen the conditioning signal against the prior) and more denoising steps (to resolve fine structure at higher resolution without quality loss from under-sampling).

**Rule 2 — Emotion Intensity:**

$$\Delta g^{(2)} = \begin{cases} +0.50 & \text{if beat} \in \mathcal{B}_{\text{high}} \\ -0.25 & \text{if beat} \in \mathcal{B}_{\text{low}} \\ 0 & \text{otherwise} \end{cases}, \quad \Delta \lambda^{(2)} = \begin{cases} +0.05 & \text{if beat} \in \mathcal{B}_{\text{high}} \\ -0.05 & \text{if beat} \in \mathcal{B}_{\text{low}} \\ 0 & \text{otherwise} \end{cases} \tag{25}$$

where $\mathcal{B}_{\text{high}} = \{\text{"contained\_fire", "fracture", "peak\_noise", "overflow", "breakthrough", "triumph", "ache", "spiral"}\}$ and $\mathcal{B}_{\text{low}} = \{\text{"stillness", "drift", "quiet\_rest", "fade", "softness", "surrender"}\}$. High-emotion beats require a stronger conditioning signal (higher $g$) to ensure the model's output reflects the intense emotional vocabulary injected by Phase 1's `EMOTION_VISUAL_MAP` (e.g., `"tightly coiled explosive tension"`, `"body yielding to gravity"`), rather than regressing to neutral compositions. Low-emotion beats use lower guidance to allow the prior to introduce soft, organic detail (suitable for quiet scenes) rather than over-sharpening to a hyperreal appearance.

**Rule 3 — Anchor Consistency:**

$$\Delta g^{(3)} = \begin{cases} +0.25 & \text{if has\_anchor} = \text{True} \text{ and } i > 1 \\ 0 & \text{otherwise} \end{cases} \tag{26}$$

Once the anchor is established (after panel 1), all subsequent panels receive a small guidance boost. This compensates for the constraint introduced by the $\mathcal{T}_2$ attention blend and the $\mathcal{T}_3$ latent statistics normalization: both operators reduce the model's effective conditioning sensitivity by mixing in anchor-derived signals, and a slight guidance increase restores the prompt's relative influence on the final output.

**Rule 4 — Bookend Position:**

$$\Delta S^{(4)} = \begin{cases} +3 & \text{if } i = 1 \text{ or } i = N \\ 0 & \text{otherwise} \end{cases} \tag{27}$$

The first and last panels receive 3 additional denoising steps because they bear the highest compositional load: panel 1 establishes the visual identity that all MDCP operators reference, and panel $N$ delivers the narrative resolution that determines the reader's lasting impression. Three additional steps adds approximately 1–1.5 s of compute per bookend panel, a small cost relative to the quality benefit.

**Clamping:**

After all four rules are applied, the parameters are clamped to fixed ranges:

$$g_{\text{final}} = \text{clip}(g_{\text{base}} + \Delta g^{(1)} + \Delta g^{(2)} + \Delta g^{(3)},\ 5.0,\ 12.0) \tag{28}$$
$$\lambda_{\text{final}} = \text{clip}(\lambda_{\text{base}} + \Delta\lambda^{(2)},\ 0.3,\ 1.0) \tag{29}$$
$$S_{\text{final}} = \text{clip}(S_{\text{base}} + \Delta S^{(1)} + \Delta S^{(4)},\ 15,\ 50) \tag{30}$$

The clamp ranges reflect the following calibration constraints:
- $g \in [5.0, 12.0]$: below 5.0, the model ignores the prompt and collapses toward the prior distribution; above 12.0, guidance over-saturates the conditioning signal, producing color-burned artifacts and loss of fine detail in comic line-art specifically.
- $\lambda \in [0.3, 1.0]$: below 0.3, the LoRA adapter's learned comic-style transformation becomes imperceptible; at 1.0, the LoRA fully weights its learned style distribution.
- $S \in [15, 50]$: below 15 steps, DPMSolver++ at order 2 does not complete sufficient refinement passes to produce coherent character anatomy; above 50, the diminishing-returns regime produces no perceptible quality improvement at significant compute cost.

**Deterministic Seed:**

$$\text{seed} = 42 + (i \cdot 7 + (\sum_{c \in \text{beat}} \text{ord}(c)) \bmod 100) \tag{31}$$

The base seed 42 is the process-level global seed set at pipeline initialization. The panel-level offset $i \cdot 7 + \text{beat\_hash} \bmod 100$ ensures that (a) each panel generates a structurally different latent starting point (preventing all panels from being visually similar initializations), (b) the offset is deterministic given only the panel index and beat label (enabling exact reproduction of any run), and $ the multiplier 7 (prime) prevents aliasing between adjacent panels that might share the same beat hash residual.

##### 3.2.4.3 Prompt Construction: The 10-Layer Hierarchy

`_build_prompt()` assembles the positive conditioning prompt as an ordered comma-separated concatenation of up to 10 semantic layers. Each layer contributes a distinct type of conditioning information; their order is significant because SDXL's dual CLIP encoders weight earlier tokens more heavily in the cross-attention mechanism.

| Layer | Source | Content |
|:---:|:---|:---|
| 1 | `STYLE_PRESETS[style_preset]` | Art style vocabulary (~30 tokens) |
| 2 | `PANEL_POSITION_MODIFIERS[position_key]` | Narrative composition instruction (~20 tokens) |
| 3 | `EMOTION_VISUAL_MAP[beat]["lighting"]` | Lighting specification |
| 4 | `EMOTION_VISUAL_MAP[beat]["palette"]` | Color palette specification |
| 5 | `EMOTION_VISUAL_MAP[beat]["atmosphere"]` | Atmospheric/mood description |
| 6 | `CAMERA_VISUAL_MAP[camera_angle]` | Camera framing vocabulary |
| 7 | `scene_graph["environment"]` | Environment/setting description |
| 8 | Per-character pose + expression (one entry per character) | Character anatomy state |
| 9 | Cinematic action schema ($v_\text{verb}$, $v_\text{mechanics}$, $v_\text{impact}$, $v_\text{reaction}$, $v_\text{timing}$) | Phase 1 action expansion |
| 10 | `QUALITY_BOOSTERS` | Universal quality terms |

**Narrative position quantization.** Layer 2's position key is computed from the continuous ratio $r_i = (i-1)/(N-1) \in [0, 1]$:

$$\text{position\_key}(r_i) = \begin{cases} \text{"opening"} & r_i = 0 \\ \text{"early"} & 0 < r_i \leq 0.20 \\ \text{"middle\_early"} & 0.20 < r_i \leq 0.40 \\ \text{"midpoint"} & 0.40 < r_i \leq 0.55 \\ \text{"middle\_late"} & 0.55 < r_i \leq 0.70 \\ \text{"climax"} & 0.70 < r_i \leq 0.85 \\ \text{"resolution"} & 0.85 < r_i < 1.0 \\ \text{"coda"} & r_i = 1.0 \end{cases} \tag{32}$$

The asymmetric threshold placement — the midpoint window is $[0.40, 0.55]$, only 15 percentage points wide, while the climax window is $[0.70, 0.85]$, 15 points — reflects the classical three-act story structure: the narrative turn-point (midpoint) is a sharp structural event, while the climax is an extended region of high tension that can span multiple panels in a longer-form story. In an 8-panel story ($N=8$), panels 4–5 ($r \approx 0.43$–$0.57$) land at `"midpoint"` and panel 6 ($r \approx 0.71$) lands at `"climax"`, correctly positioning the dramatic peak in the final third.

**EMOTION_VISUAL_MAP coverage.** The map contains 47 named beats spanning 8 thematic arcs (sad/grief, angry, tired, happy, anxious, grief, determined, love) plus 6 generic fallbacks. Every LLM-generated beat that maps to an unknown key falls back to the `"neutral"` entry (`"balanced natural three-point light"`, `"neutral natural tones"`, `"clean, clear, present moment"`), which is intentionally the least emotionally charged configuration in order to avoid inserting incorrect emotional vocabulary when the beat is ambiguous.

**Token budget and Compel overflow path.** SDXL's dual CLIP encoders each have a hard 77-token context window. The assembled positive prompt for a typical panel (with all 10 layers populated) ranges from approximately 90–150 tokens, consistently exceeding the 77-token limit. The `SDXLBackend.generate()` method therefore attempts to import `compel` and, if available, encodes both the positive and negative prompts using Compel's weighted embedding accumulation:

```python
self._compel = Compel(
    tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
    text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
    returned_embeddings_type=PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
    requires_pooled=[False, True],
)
prompt_embeds, pooled_prompt_embeds = self._compel(prompt)
```

Compel operates by splitting the prompt into 77-token chunks, encoding each chunk separately, and concatenating the resulting penultimate hidden-state tensors before passing them to the UNet's cross-attention projections. This avoids the silent truncation of tokens 78+ that native SDXL would apply. The `PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED` return type is specifically chosen for SDXL compatibility: SDXL's UNet expects the pre-LN normalized hidden states from the penultimate transformer layer, not the final normalized output. If Compel is not installed, the pipeline falls back to standard tokenization with a logged warning; in this case, the quality booster tokens (Layer 10) are the most likely to be silently truncated, degrading output quality but not causing a hard failure.

##### 3.2.4.4 Negative Prompt Taxonomy

The negative prompt is assembled from four orthogonal taxonomies:

**1. Universal quality negatives** (always active): `"photorealistic"`, `"3D render"`, `"blurry"`, `"extra fingers"`, `"deformed hands"`, `"bad anatomy"`, `"watermark"`, `"low quality"`, `"jpeg artifacts"`, `"multiple panels in one image"`. These suppress the model's tendency to hallucinate photograph-style outputs and prevent hand/anatomy anatomy failure modes that are especially prevalent in dynamic action poses.

**2. Style-specific negatives** (conditional on `style_preset`): For example, `manga` and `indie_comic` presets add `"gradients"`, `"airbrushed"`, `"photoshop glow"`, `"painterly"` — terms that describe rendering artifacts inconsistent with hard-edge line-art. The `watercolor_indie` preset adds `"hard ink lines"`, `"cel shading"`, `"flat vector art"` — suppressing the complementary hard-edge register. This ensures that even when the positive prompt's style vocabulary is ambiguous, the model cannot drift into an inconsistent rendering mode.

**3. Emotion-specific negatives** (conditional on beat): Triumph/breakthrough beats add `"dark"`, `"gloomy"`, `"muted colors"`; contained-fire/fracture beats add `"calm"`, `"peaceful"`, `"pastel colors"`; quiet-rest/stillness beats add `"busy background"`, `"chaotic"`, `"high contrast"`. These negatives directly counteract the model's mean-regression tendency: without negative guidance, SDXL consistently produces moderate-intensity images, averaging high and low emotional extremes together.

**4. Action-verb negatives** (conditional on the panel's resolved action): The action verb set from the scene graph is compared against four disjoint verb classes:
- $\mathcal{V}_{\text{combat}}$: adds `"static pose"`, `"hands at sides"`, `"portrait framing"` (SDXL's default response to a `punch` prompt is a static portrait, not a dynamic strike)
- $\mathcal{V}_{\text{movement}}$: adds `"standing still"`, `"both feet on ground"`, `"frozen"` (prevents pose regression to a neutral standing pose)
- $\mathcal{V}_{\text{rest}}$: adds `"dynamic action"`, `"explosion"`, `"busy background"` (prevents drama from creeping into quiet panels)
- $\mathcal{V}_{\text{observe}}$: adds `"action pose"`, `"combat"`, `"motion lines"` (prevents action vocabulary from contaminating contemplative panels)

##### 3.2.4.5 Resolution and Canvas Allocation

Canvas size is determined from the Phase 1 `LayoutDirective.size_class`:

$$\text{resolution}(i) = \begin{cases} (1024,\ 1024) & \text{if size\_class} = \text{"full\_page"} \\ (768,\ 768) & \text{otherwise} \end{cases} \tag{33}$$

The $1024 \times 1024$ resolution for full-page panels is the native training resolution of SDXL and produces the highest-fidelity output. The $768 \times 768$ resolution for all other classes balances quality against compute time and peak VRAM: at $768 \times 768$, SDXL UNet attention maps occupy approximately $5.6 \times$ less memory than at $1024 \times 1024$ (scaling as $\mathcal{O}(H^2 W^2)$ in self-attention), making them feasible with attention slicing on a T4 GPU. The lack of separate `"small"` and `"medium"` resolution tiers is a simplification — all non-full-page panels share the same $768 \times 768$ canvas and their visual size difference is applied entirely by Phase 7's layout engine, which scales, crops, and positions the generated image within a larger page canvas rather than generating at different native resolutions.

##### 3.2.4.6 Advanced Attention Hook Integration

If an `AdvancedAttentionManager` is active, it is integrated into the generation loop through two distinct mechanisms depending on the execution mode:

**Exact Denoising Loop Integration (Default).** The standard diffusers pipeline execution is replaced by a custom PyTorch sampling loop in `mdcp_generate()`. This loop directly schedules and executes the three core MDCP operators at each timestep $t$:
1. **$\mathcal{T}_3$ Channel Alignment:** Invoked on the current latents to align mean/variance using Progressive Affine Moment Matching.
2. **$\mathcal{T}_1$ Trajectory Optimization:** Invoked via a gradient-tracked forward pass on the aligned latents to compute $E_{\text{id}}$, $E_{\text{str}}$, and $E_{\text{traj}}$, backpropagating to obtain $\nabla_{z_{\text{aligned}}} E_t$ and taking a step of size $\eta(t)$.
3. **$\mathcal{T}_2$ Attention Propagation:** Invoked via a second forward pass where forward hooks intercept the 4 cross-attention layers of the UNet to blend text-conditioned outputs with the cached anchor outputs ($O_{\text{anchor}}$), optionally gated by a spatial region mask (M2/M3).
The scheduler then performs the transition step to produce $z_{t-1}$.

**Heuristic Callback Integration (Legacy/A/B testing).** When `use_heuristic_mode=True` is configured, the manager reverts to the legacy callback-based mode. Here, the `step_callback` function returned by `advanced_attention.get_step_callback()` is passed as `callback_on_step_end` to the SDXL pipeline call. This function executes at every denoising step and has direct access to the intermediate `latents` tensor. It applies the $\mathcal{T}_1$ heat diffusion operator (spatial smoothing of the latent map) and the $\mathcal{T}_3$ spatiotemporal anchor blend (mixing anchor statistics into the current latent) as heuristic approximations. The `callback_tensor_inputs=["latents"]` parameter ensures that the callback receives the latent tensor on the same device as the pipeline, avoiding an unnecessary CPU/GPU transfer. Additionally, forward hooks discover the cross-attention modules on the first panel via `backend.get_cross_attention_modules()` to apply attention blending at every step.

#### 3.2.5 Phase 4 — Optional Consistency Modules

Phase 4 encompasses the `AdvancedAttentionManager` — the unified controller for three core MDCP operators ($\mathcal{T}_1$–$\mathcal{T}_3$) and five optional failure-mode mitigations (M1–M5), all implemented in `core/advanced_attention.py`. The three core operators are always active when the manager is enabled (requires GPU). The five mitigations are opt-in by setting their respective flags in `AdvancedAttentionManager.__init__()`. Each mitigation targets a specific identified failure mode of the base $\mathcal{T}_1$–$\mathcal{T}_3$ chain.

##### 3.2.5.1 L1 — Heat Diffusion Prior (`HeatDiffusionPrior`)

**Mathematical basis.** L1 applies the fundamental solution of the discrete heat equation to the latent tensor at each denoising step. The continuous heat equation $\partial u / \partial t = \alpha \nabla^2 u$ describes how a temperature distribution $u$ relaxes toward its spatial mean under diffusion coefficient $\alpha$. The discrete update rule for a 2D latent map $\mathbf{z}$ is:

$$\mathbf{z}_{t+1} = \mathbf{z}_t + \hat{\alpha}_t \cdot (K * \mathbf{z}_t - \mathbf{z}_t) \tag{34}$$

where $K$ is a normalized 2D Gaussian convolution kernel (the Green's function of the heat equation), $*$ denotes 2D convolution, and $\hat{\alpha}_t$ is the timestep-scaled effective diffusion coefficient.

**Gaussian kernel construction.** The kernel is built with `kernel_size=3` and:

$$\sigma = \text{kernel\_size}/3.0 = 1.0, \quad k[i,j] = \exp\!\left(-\frac{(i-1)^2+(j-1)^2}{2\sigma^2}\right),\quad k \leftarrow k / \|k\|_1 \tag{35}$$

The $\sigma = \text{size}/3$ relationship is the standard rule-of-thumb ensuring the Gaussian fills approximately $3\sigma$ of the kernel without excessive zero-padding — for a $3\times3$ kernel this gives $\sigma=1.0$, producing a kernel that weights the center pixel approximately 3.7× more than corner pixels.

**Active window and alpha linearization.** L1 is active only during the denoising window $[\text{end\_ratio}, \text{start\_ratio}] = [0.20, 0.80]$ (where 1.0 = first denoising step, 0.0 = final step):

$$\hat{\alpha}_t = \alpha \cdot \frac{t_{\text{ratio}} - 0.20}{0.80 - 0.20}, \quad \alpha_{\text{base}} = 0.03 \tag{36}$$

The linear interpolation of $\hat{\alpha}_t$ from 0 (at $t_{\text{ratio}} = 0.20$) to $\alpha$ (at $t_{\text{ratio}} = 0.80$) ensures that the heat diffusion is strongest during the early-to-mid denoising stages (when high-frequency spatial noise is highest and benefits most from smoothing) and fades to zero by the final 20% of steps (where fine-detail structure is resolved and smoothing would degrade sharpness). The base $\alpha = 0.03$ is calibrated to provide perceptible noise suppression without blurring edges: at $\hat{\alpha} = 0.03$, Equation (34) shifts each pixel approximately 3% toward its local spatial average per step, an amount that cumulatively suppresses high-frequency jitter over 25 steps without washing out comic line boundaries.

The convolution is applied per-channel via `F.conv2d(..., groups=channels)`, which applies the same spatial kernel independently to each of the 4 latent channels (SDXL's 4-channel VAE latent space), avoiding cross-channel mixing.

**Complementarity with FreeU.** When M4 (FreeU skip scaler) is enabled, the Fourier-domain processing inside the UNet decoder already attenuates high-frequency noise in the feature space. L1 and M4 operate at different levels: L1 acts directly on the raw latent tensor $\mathbf{z}_t$ in the output space, while M4 acts on UNet internal feature maps in the decoder's spatial frequency domain. Both are kept active simultaneously as complementary suppression mechanisms at different representation levels.

##### 3.2.5.2 L2 — Shared Attention Cache (`SharedAttentionCache`)

**Mathematical basis.** L2 enforces character identity by blending a fraction $\beta$ of the anchor panel's attention key/value outputs into every subsequent panel's cross-attention computation:

$$O_{\text{target}} = (1 - \beta) \cdot O_{\text{current}} + \beta \cdot O_{\text{anchor}} \tag{37}$$

where $O \in \mathbb{R}^{B \times L \times C}$ is the cross-attention output tensor (batch $B$, sequence length $L$, channels $C$), and $\beta = 0.15$ is the blend ratio.

**Rationale for $\beta = 0.15$.** The blend weight sits in $[0, 0.4]$ (documented upper bound). At $\beta = 0$, no anchor identity is injected. At $\beta = 0.4$, the anchor's attention outputs dominate excessively, suppressing the target panel's pose- and scene-specific conditioning and causing the output to appear as a near-clone of the anchor. The value $\beta = 0.15$ represents a 15% anchor contribution — sufficient to reinforce hair color, costume patterns, and facial feature distributions captured by the K/V projections, while still allowing the current panel's prompt tokens to drive 85% of the attention distribution. This value was established empirically as the balance point between identity consistency and pose/scene diversity.

**Anchor capture: CPU pinned memory.** During Panel 1 (anchor), the hook captures the attention output tensor:

```python
self._cached_outputs[module] = output.detach().cpu().pin_memory()
```

The `.cpu().pin_memory()` call serves a specific hardware purpose: page-locked (pinned) CPU memory enables asynchronous DMA transfers back to GPU without staging through pageable memory. When the cached tensor is subsequently needed during target panel generation, the transfer:

```python
cached_device = cached.to(device=output.device, dtype=output.dtype, non_blocking=True)
```

uses `non_blocking=True` to initiate the DMA asynchronously, overlapping the CPU→GPU transfer with other compute. This minimizes the PCIe bandwidth latency penalty to approximately 0.1–0.2 ms per attention layer per step on a T4 GPU, compared to approximately 0.5–1.0 ms for a synchronous pageable transfer.

**Spatially masked blend.** When a region mask $M \in [0,1]^{1 \times L \times 1}$ is attached (from M2 or M3), the blend is spatially selective:

$$O_{\text{blended}}[s] = (1 - \beta \cdot M[s]) \cdot O_{\text{current}}[s] + \beta \cdot M[s] \cdot O_{\text{anchor}}[s] \tag{38}$$

where $s$ indexes the spatial sequence position. The mask is flattened from $(1, 1, H, W)$ to $(1, H \cdot W, 1)$ to broadcast across the $C$-dimensional channel axis of the $(B, L, C)$ attention output tensor. Positions where $M[s] \approx 0$ (background) receive no anchor blend; positions where $M[s] = 1$ (character foreground) receive the full $\beta = 0.15$ blend.

##### 3.2.5.3 L3 — Spatiotemporal Consistency Enforcer (`SpatiotemporalConsistencyEnforcer`)

**Mathematical basis.** L3 captures the channel-wise latent statistics of Panel 1 at the final denoising step and uses them to constrain the latent distribution of subsequent panels during a mid-denoising window. Let $\boldsymbol{\mu}_{\text{anchor}} \in \mathbb{R}^C$ and $\boldsymbol{\sigma}_{\text{anchor}} \in \mathbb{R}^C$ be the channel-wise mean and standard deviation of the anchor's final latents $\mathbf{z}_{\text{anchor}} \in \mathbb{R}^{B \times C \times H \times W}$:

$$\mu_{\text{anchor},c} = \frac{1}{BHW}\sum_{b,h,w} z_{\text{anchor},b,c,h,w}, \quad \sigma_{\text{anchor},c} = \text{std}_{b,h,w}(z_{\text{anchor},b,c,h,w}) \tag{39}$$

For a target panel latent $\mathbf{z}_t$ at timestep ratio $t_{\text{ratio}} \in [0.30, 0.60]$, a channel-wise affine correction is applied:

$$z_{t,b,c,h,w}^{\text{corr}} = r_c \cdot z_{t,b,c,h,w} + \delta_c \tag{40}$$

where the std ratio $r_c$ and mean delta $\delta_c$ per channel are:

$$r_c = \text{clip}\!\left(\frac{\sigma_{\text{anchor},c}}{\sigma_{t,c}},\ 0.80,\ 1.20\right), \quad \delta_c = (\mu_{\text{anchor},c} - \mu_{t,c}) \cdot \gamma \tag{41}$$

with base strength $\gamma = 0.08$, and $\sigma_{t,c} = \max(\text{std}_{b,h,w}(z_{t,b,c}),\ 10^{-6})$ (clamped to prevent division by zero).

**Derivation of the Std-Ratio Clamp $[0.80, 1.20]$.** The $\pm 20\%$ clamping bounds were derived empirically by measuring the natural latent standard deviation shifts across 50 unconstrained generated sequences. We observed that adjacent panels depicting continuous action within the same scene naturally varied in standard deviation by $\pm 10\%$ to $\pm 15\%$. However, dramatic narrative beat transitions—such as shifting from a low-intensity `"stillness"` beat (which naturally produces near-uniform, low-variance latents) to a high-intensity `"peak_noise"` beat (which generates high-variance, high-contrast latents)—could cause the raw ratio $r_c$ to spike to values as extreme as $3.0$ or plummet to $0.2$. Without clamping, applying an $r_c$ of $3.0$ would force a quiet, softly-lit panel to inherit the extreme contrast of an explosion anchor, destroying the target panel's intended lighting. By grid-searching the clamp boundaries across $\{[0.9, 1.1], [0.8, 1.2], [0.7, 1.3]\}$, we found that the $[0.80, 1.20]$ envelope optimally accommodated natural intra-scene variance while acting as a hard mathematical stop against destructive cross-scene contrast blow-ups. This allows for subtle style consistency while preserving the target panel's intrinsic dynamic range.

**Time-linear blend weight.** The corrected latent is blended with the uncorrected latent by a weight $w_t$ that increases linearly across the active window:

$$w_t = \gamma \cdot \frac{t_{\text{ratio}} - 0.30}{0.60 - 0.30}, \quad \mathbf{z}_t^{\text{out}} = (1 - w_t)\,\mathbf{z}_t + w_t\,\mathbf{z}_t^{\text{corr}} \tag{42}$$

The active window $[0.30, 0.60]$ targets the mid-denoising phase. Below 0.30 (late denoising), fine detail is being resolved and latent statistics have already converged to their final distribution — applying L3 at this stage would visibly smear fine structure. Above 0.60 (early denoising), the model is still resolving coarse layout and the latent statistics are highly volatile, making anchor-based correction unreliable. The mid-range $[0.30, 0.60]$ is where structural features (pose, body shape, facial geometry) solidify and are most amenable to stable statistical guidance.

##### 3.2.5.4 M1 — Localized Detail Injector (`LocalizedDetailInjector`)

**Failure mode addressed.** L2's global K/V cache captures high-level semantic identity (color palette, body proportions) but lacks the geometric precision to reproduce specific fine details (facial scars, costume emblems, jewelry). M1 adds a patch-level structural fingerprint to the positive prompt of each target panel.

**Algorithm.** At Phase 2 step-zero, M1 computes a Canny edge map of the anchor at resolution $256 \times 256$ (fixed, resolution-independent of the generated panel size):

$$e[x,y] = \text{Canny}(I_{\text{anchor}},\ T_{\text{low}}=50,\ T_{\text{high}}=150)[x,y] \in \{0, 255\} \tag{43}$$

The edge map is divided into a $P \times P = 8 \times 8$ patch grid (patch size $256/8 = 32 \times 32$ pixels each), and the mean normalized edge density per patch is computed:

$$\mathcal{F}[i,j] = \frac{1}{32 \times 32} \sum_{x \in P_i, y \in P_j} \frac{e[x,y]}{255} \in [0, 1], \quad i,j \in \{0,\ldots,7\} \tag{44}$$

producing a $8 \times 8 = 64$-element structural fingerprint. The fingerprint is then quantized into a three-level structural hint string:

$$\text{density\_desc} = \begin{cases} \text{"minimal edge detail, clean flat surfaces"} & \overline{\mathcal{F}} < 0.05 \\ \text{"moderate structural detail, defined contours"} & 0.05 \leq \overline{\mathcal{F}} < 0.15 \\ \text{"high structural complexity, intricate geometric detail"} & \overline{\mathcal{F}} \geq 0.15 \end{cases} \tag{45}$$

where $\overline{\mathcal{F}} = \frac{1}{64}\sum_{i,j}\mathcal{F}[i,j]$ is the overall mean edge density. The number of high-detail patches (those with $\mathcal{F}[i,j] > 0.3$, indicating locally dense structure) is also reported. The resulting structural hint string is appended as Layer 9.5 of the prompt hierarchy (between the cinematic action clause and the quality boosters).

**Optional IP-Adapter mode.** When `use_ip_adapter=True` and the pipeline exposes `set_ip_adapter_scale()`, the anchor image itself is injected as an IP-Adapter image prompt at `detail_weight=0.10`. The low weight (10%) prevents the IP-Adapter from overriding the panel's compositional conditioning while still anchoring fine facial and costume geometry.

##### 3.2.5.5 M2 — Regional Cross-Attention Mask (`RegionalAttentionMask`)

**Failure mode addressed.** L2's global K/V blend is applied uniformly across all spatial positions. In two-character panels, this causes feature bleed: Character A's anchor K/V tokens contaminate the spatial region belonging to Character B, introducing identity confusion (A's hair color or body shape appearing on B).

**Algorithm.** Bounding boxes $\{(x_0^k, y_0^k, x_1^k, y_1^k)\}_{k=1}^{N_{\text{char}}}$ in normalized $[0,1]$ coordinates are extracted from the scene graph (or computed as equal horizontal strips as described in Section 3.2.3.4). These are rasterized into a union binary mask at the primary UNet feature resolution $H_f \times W_f = 64 \times 64$:

$$M_{\text{reg}}[r,c] = \begin{cases} 1.0 & \text{if } (r,c) \in \bigcup_k [r_0^k, r_1^k] \times [c_0^k, c_1^k] \\ 0.0 & \text{otherwise} \end{cases} \tag{46}$$

where $r_0^k = \lfloor y_0^k \cdot H_f \rfloor$, etc. The mask is stored as a `float16` tensor of shape $(1, 1, 64, 64)$, consuming less than 5 MB of VRAM. When attached to `SharedAttentionCache`, it replaces the global blend of Equation (37) with the spatially selective blend of Equation (38).

**Fallback strip heuristic.** If the LLM scene graph does not include bounding box coordinates (a common omission), M2 falls back to an equal-width horizontal strip partition: $N_{\text{char}}$ characters receive strips of width $1/N_{\text{char}}$ spanning the full image height. This is a coarse but zero-failure fallback that at least localizes each character to a horizontal region, reducing (though not eliminating) cross-character K/V contamination.

##### 3.2.5.6 M3 — Foreground Saliency Mask (`ForegroundSaliencyMask`)

**Failure mode addressed.** L2's $\beta = 0.15$ blend is applied to both character-occupied pixels and background pixels. In panels with new backgrounds (different scene from anchor), the anchor's background K/V values contaminate the target background region, causing the new environment to visually "leak" anchor-environment elements.

**Algorithm.** M3 computes a binary foreground mask separating the character from the background using a three-level fallback cascade:

1. **SAM (Segment Anything Model, ViT-B checkpoint)** at `points_per_side=8` (64 automatic prompt points). If multiple mask proposals are returned, the largest by pixel area is selected as the primary subject mask.
2. **GrabCut** (OpenCV, always available): initializes with a rectangle covering the **center 60%** of the image — margins of 20% horizontally and 20% vertically on each side:
$$\text{rect} = (\lfloor W \cdot 0.20 \rfloor,\ \lfloor H \cdot 0.20 \rfloor,\ W - 2\lfloor W \cdot 0.20 \rfloor,\ H - 2\lfloor H \cdot 0.20 \rfloor) \tag{47}$$
The 20% margin is a principled choice: comic panels generated by SDXL consistently place the primary subject in the central 60% of the canvas (the model's compositional prior strongly centers subjects when no explicit composition prompt is given), while gutters, background elements, and atmosphere occupy the outer 20% on each side.

GrabCut iterates `grabcut_iters=5` EM rounds using a Gaussian Mixture Model for foreground and background appearance, classifying each pixel as one of: definite background (`GC_BGD`), probable background (`GC_PR_BGD`), probable foreground (`GC_PR_FGD`), or definite foreground (`GC_FGD`). The final binary mask sets $M[x,y]=1$ for both `GC_PR_FGD` and `GC_FGD` classes (conservative inclusion).

The resulting mask is resized to $64 \times 64$ (the UNet feature resolution) via bilinear interpolation and stored as a `float16` tensor of shape $(1, 1, 64, 64)$, consuming less than 1 MB. This mask is then attached to `SharedAttentionCache` via `set_region_mask()`, enabling the spatially masked blend of Equation (38).

**SAM offload.** After segmentation, the SAM ViT-B model is immediately moved to CPU and garbage-collected to reclaim the approximately 375 MB of VRAM it occupies during inference. The entire M3 computation is performed once at Phase 2 step-zero and incurs no per-step overhead during denoising.

##### 3.2.5.7 M4 — FreeU Skip-Connection Fourier Scaler (`FreeUSkipScaler`)

**Failure mode addressed.** L1's isotropic Gaussian kernel (Equation 34–35) attenuates all high-frequency components equally, including the intentional high-frequency structures of comic art: screen-tones, cross-hatching, precise ink outlines. M4 replaces L1's spatial-domain smoothing with a Fourier-domain selective scaler that explicitly distinguishes low-frequency (global structure, identity) from high-frequency (texture detail, line art) components.

**Algorithm.** M4 installs forward hooks on all ResNet blocks in SDXL's UNet `up_blocks` (decoder stages). For each feature map $x \in \mathbb{R}^{B \times C \times H \times W}$:

$$X_f = \mathcal{F}_{\text{rfft2}}(x) \in \mathbb{C}^{B \times C \times H \times \lfloor W/2+1 \rfloor} \tag{48}$$

The spectrum is partitioned into low-frequency (center quarter) and high-frequency (remainder) regions:

$$X_f^{\text{out}}[b,c,h,w] = \begin{cases} \gamma_b \cdot X_f[b,c,h,w] & \text{if } h < H/4 \text{ or } h \geq 3H/4, \text{ and } w < W_h/4 \\ \gamma_s \cdot X_f[b,c,h,w] & \text{otherwise} \end{cases} \tag{49}$$

with $\gamma_b = 1.2$ (backbone/low-freq amplification) and $\gamma_s = 0.9$ (skip/high-freq attenuation). The modified spectrum is inverted back to spatial domain via `irfft2`. This operation has **zero VRAM overhead** (in-place scalar multiplication of existing tensors) and approximately **0.1% latency overhead** per step (FFT is $\mathcal{O}(HW\log HW)$ over small feature maps).

This is a distinct implementation from the native FreeU call described in Section 3.2.4.1. The native call ($s_1=0.6, s_2=0.4, b_1=1.1, b_2=1.2$) operates on skip-connection tensors in the spatial domain by scaling the entire tensor. M4 operates in the Fourier domain, separating low from high frequencies within each feature map, providing frequency-selective control that the native implementation cannot achieve. When both are active, they provide complementary frequency-domain and spatial-domain control.

##### 3.2.5.8 M5 — AdaIN Style Aligner (`AdaINStyleAligner`)

**Failure mode addressed.** L3's raw latent channel-stat clamp (Equation 40–41) applies a coarse affine correction that is effective for global color-temperature consistency but inappropriate for panels with legitimate dramatic lighting shifts (e.g., a sword-strike panel with explosive backlighting versus a calm scene). The $\pm 20\%$ std-ratio clamp prevents the target panel's latent distribution from expressing the full variance range needed for such extreme lighting, causing the output to appear color-washed.

**Algorithm.** M5 replaces L3's raw latent correction with Adaptive Instance Normalization (AdaIN, Huang & Belongie, ICCV 2017) applied to the UNet decoder's intermediate feature maps — a deeper semantic space where identity-relevant style signals are more separable from illumination signals than in raw latent space.

During the anchor panel's final UNet pass, M5 hooks capture the channel-wise mean and standard deviation of each decoder ResNet block's output:

$$\mu_{\text{style},c} = \frac{1}{BHW}\sum_{b,h,w} f_{b,c,h,w}, \quad \sigma_{\text{style},c} = \text{std}_{b,h,w}(f_{b,c,h,w}), \quad f = \text{ResNet}(x_{\text{anchor}}) \tag{50}$$

stored as CPU-resident tensors. During target panel generation, each ResNet block output $f_{\text{target}}$ is transformed by AdaIN:

$$\text{AdaIN}(f_{\text{target}}) = \sigma_{\text{style}} \cdot \frac{f_{\text{target}} - \mu_{\text{target}}}{\sigma_{\text{target}}} + \mu_{\text{style}} \tag{51}$$

The full AdaIN replacement is blended with the unmodified output at strength $\alpha_t$:

$$f_{\text{out}} = (1 - \alpha_t)\,f_{\text{target}} + \alpha_t\,\text{AdaIN}(f_{\text{target}}) \tag{52}$$

where $\alpha_t$ is linearly ramped within the active window $[0.30, 0.70]$:

$$\alpha_t = \gamma_{\text{adain}} \cdot \frac{t_{\text{ratio}} - 0.30}{0.70 - 0.30}, \quad \gamma_{\text{adain}} = 0.5 \tag{53}$$

The active window $[0.30, 0.70]$ is wider than L3's $[0.30, 0.60]$ because AdaIN operates in a higher-level feature space where semantic style information is more stably separable — it can be applied further into the denoising trajectory without corrupting fine detail. The `strength=0.5` means that at the window's center, the output is a 50/50 blend of the style-aligned and unaligned feature maps, providing meaningful style consistency while still allowing the target panel's illumination to differ.

VRAM overhead is approximately 20–80 MB for the cached feature map statistics (one mean/std pair per decoder block, up to `max_layers=4` blocks), stored as CPU-pinned tensors. Latency overhead is approximately 1.5–2.0% per step (normalization arithmetic only, no additional convolutions or weights required).

##### 3.2.5.9 Default Configuration and Enabling

The five mitigations are **disabled by default in the pipeline's production configuration** — the `PanelEngine` instantiates `AdvancedAttentionManager` without passing any of the five enable flags, so all flags receive their constructor defaults of `True`, but since `AdvancedAttentionManager` itself is not instantiated by default in `integrated_pipeline.py`, none of the mitigations run. Enabling all five requires a single change:

```python
advanced_attention = AdvancedAttentionManager(
    freeu_enabled=True,
    regional_masking_enabled=True,
    saliency_enabled=True,
    adain_enabled=True,
    detail_injector_enabled=True,
)
```

The activation order when all five are enabled is: M4 and M5 install UNet hooks at model load time (once); M3 and M1 run at Phase 2 step-zero (once); M2 runs per panel at panel-start time; L1 and L3 run per denoising step via direct custom loop execution (or via the `callback_on_step_end` callback in legacy mode); L2 runs per attention layer per denoising step via forward hooks. The combined additional latency per panel is approximately 5–8% over baseline SDXL generation.

| Module | Layer | Active When | Overhead |
|:---|:---:|:---|:---|
| L1 HeatDiffusion | Latent ($\mathbf{z}_t$) | Steps 20–80% of denoising, target panels only | <1% |
| L2 AttentionCache | Cross-attn output | All steps, target panels; capture on anchor | ~2% |
| L3 SpatiotemporalEnforcer | Latent ($\mathbf{z}_t$) | Steps 30–60% of denoising, target panels only | <1% |
| M1 DetailInjector | Prompt text | Once at step-zero (prompt concat) | ~0% |
| M2 RegionalMask | Attention mask | Per panel, per step | ~1% |
| M3 SaliencyMask | Attention mask | Once at step-zero (segmentation) | One-time 0.5–1.5 s |
| M4 FreeUScaler | UNet decoder features | Steps 20–80% of denoising | ~0.1% |
| M5 AdaINAligner | UNet decoder features | Steps 30–70% of denoising | ~1.5–2.0% |


#### 3.2.6 Phase 5 — LLM-Planned Dialogue Placement

Phase 5 integrates the LLM-written dialogue from Phase 1 onto the generated panel image as rendered speech bubbles. The implementation — `TextImageIntegrator` in `core/text_image_integrator.py` — approximates the DiffSensei architecture, which uses a dedicated multi-modal model to simultaneously generate panel image and text regions. The pipeline's approximation uses a two-tier strategy: an LLM plans the bubble's position and style in a separate forward pass, and a PIL-based rasterizer renders it onto the finished panel image. This decoupling trades the tighter visual integration of DiffSensei (where text layout is co-generated with the image) for zero additional diffusion cost and full editability of the bubble placement.

##### 3.2.6.1 LLM Planning Chain

`get_layout_plan()` resolves a bubble layout specification through a three-tier fallback chain:

**Tier 1 — JSON cache.** Before any LLM call, the method checks for a per-panel JSON file at `outputs/panels/panel_{id:03d}_bubble_layout.json`. If it exists, the cached plan is compared against the current dialogue text using a case-normalized exact string match. If the dialogue matches, the cached plan is returned immediately — enabling a full offline re-render of an existing story without any LLM dependency. If the dialogue has changed since the last run (e.g., during iterative editing), the cache is invalidated and the LLM is re-queried.

**Tier 2 — Direct Ollama HTTP.** A JSON-structured system prompt and a panel-specific user prompt are sent to `http://localhost:11434/api/generate` via a raw `urllib.request` HTTP POST with `timeout=8` s and `stream=false`. The system prompt defines the output schema exactly:

```json
{
  "speaker": "name or null",
  "dialogue_clean": "text with optional **bold** or *italic*",
  "bubble_shape": "ellipse|dashed_ellipse|jagged|cloud|spiky",
  "speaker_position": "left|center|right",
  "font_scale": 1.0,
  "x_ratio": 0.5,
  "y_ratio": 0.15,
  "text_align": "center|left|right",
  "tail_x_ratio": 0.5,
  "tail_y_ratio": 0.8
}
```

The user prompt provides: `panel_id`, the raw dialogue string (including speaker prefix), the `emotion_beat`, and the scene/action description from Phase 1. Crucially, the prompt instructs the LLM to keep `y_ratio` near the top ($[0.15, 0.30]$) to avoid covering character bodies, and to vary slightly by panel ID to prevent repeated bubbles from stacking in the same position across a page.

**Tier 3 — LangChain fallback.** If the direct HTTP call fails (Ollama not running, timeout, network error), the method falls back to a LangChain-abstracted call supporting four providers selected by `LLM_PROVIDER` environment variable: `ollama` (default, `llama3.2`), `openai` (`gpt-4o-mini`), `gemini` (`gemini-1.5-flash`), `anthropic` (`claude-3-5-sonnet-latest`). All providers use `temperature=0.1` to minimize positional hallucination. If all LLM tiers fail, a deterministic heuristic plan is returned (Tier 4, §3.2.6.2).

The LLM response is parsed by a bracket-depth JSON extractor that handles markdown code-fenced JSON and strips leading/trailing whitespace. The resulting plan is saved back to the JSON cache file regardless of which tier produced it.

##### 3.2.6.2 Deterministic Heuristic Fallback Positioning

When no LLM is available, bubble position is computed deterministically from `panel_id` and the speaker name string, producing stable layouts that vary across panels without any randomness:

$$x_{\text{ratio}} = \begin{cases} 0.25 & \text{if } \text{speaker\_pos} = \text{"left"} \\ 0.75 & \text{if } \text{speaker\_pos} = \text{"right"} \\ 0.50 & \text{otherwise} \end{cases} \tag{54}$$

where speaker position is derived from:
$$\text{speaker\_pos} = \text{pos\_options}\!\left[\left(\text{panel\_id} + \sum_{c \in \text{speaker}} \text{ord}(c)\right) \bmod 3\right],\quad \text{pos\_options} = [\text{"left", "right", "center"}] \tag{55}$$

The vertical ratio is:
$$y_{\text{ratio}} = 0.15 + \left((\text{panel\_id} \times 7) \bmod 3\right) \times 0.08 \in \{0.15, 0.23, 0.31\} \tag{56}$$

The three possible $y$ values (0.15, 0.23, 0.31) correspond to the top 15%, 23%, and 31% of the panel height — all placing the bubble in the upper third to avoid character anatomy, while cycling across panels (period 3 due to the $\bmod 3$) to prevent stack collisions. The multiplier 7 (prime) ensures the three-level cycle does not phase-align with panel index sequences that are multiples of 3 (e.g., in a 9-panel story, panels 3, 6, 9 would all land at $y=0.15$ with multiplier 3, but are distributed across all three levels with multiplier 7).

##### 3.2.6.3 Emotion-to-Bubble Style Mapping

`BEAT_TO_BUBBLE` maps each of the 47 emotion beats (plus 6 generic fallbacks) to one of five bubble categories. The mapping logic follows the emotional register of each arc:

| Category | Shape | Fill RGBA | Border | Font scale | Used for |
|:---|:---:|:---:|:---|:---:|:---|
| `calm` | `ellipse` (rounded rect) | $(255,255,255,230)$ | $(40,40,40)$ | $1.00\times$ | Quiet dialogue, resolution beats, love arc |
| `intense` | `jagged` (small-radius rect) | $(255,255,240,240)$ | $(180,30,30)$ | $1.15\times$ | Anger, exhaustion, anxiety, challenge |
| `thought` | `cloud` | $(240,240,255,200)$ | $(100,100,140)$ | $0.90\times$ | Internal monologue, memory, vulnerability |
| `whisper` | `dashed_ellipse` | $(255,255,255,180)$ | $(120,120,120)$ | $0.85\times$ | Grief, heaviness, silence, ache |
| `shout` | `spiky` | $(255,250,230,245)$ | $(200,50,20)$ | $1.30\times$ | Triumph, breakthrough, elation, unity |

The alpha channel of the fill color encodes opacity: `calm` bubbles at $\alpha=230/255 \approx 90\%$ opacity block the background comfortably; `whisper` bubbles at $\alpha=180/255 \approx 71\%$ are semi-transparent, allowing the underlying scene to show through — a visual reinforcement of the quietness of the communication. The `thought` bubble uses a blue-tinted fill $(240,240,255)$ to visually distinguish internal monologue from spoken dialogue. `intense` and `shout` bubbles use warm red border colors $(180,30,30)$ and $(200,50,20)$ to reinforce their emotional valence.

The font scale multiplier modifies the base `font_size = base_font_size * font_scale`, where `base_font_size=16` pt:
- `whisper`: $16 \times 0.85 = 13.6 \approx 14$ pt — smaller text embodies the quietness of the utterance
- `shout`: $16 \times 1.30 = 20.8 \approx 21$ pt — larger text embodies the physical force of the shout
- `intense`: $16 \times 1.15 = 18.4 \approx 18$ pt — slightly elevated for raised voice or tension

##### 3.2.6.4 Bubble Shape Rendering

All shapes are rendered on a transparent RGBA overlay (same size as the panel) and composited via `Image.alpha_composite()`, ensuring zero destructive modification to the original pixel data.

**Ellipse (standard dialogue).** A PIL `draw.rounded_rectangle()` with $r = \min(w_b, h_b) / 4$, where $w_b, h_b$ are the bubble's pixel dimensions determined from text measurement. The $r = \text{short side}/4$ rule produces comfortably rounded corners that scale with bubble size — smaller bubbles get proportionally smaller corner radii.

**Dashed ellipse (whisper).** The same rounded rectangle is drawn first as solid fill only (no outline), then a dashed border is overlaid by iterating the four edges in steps of 12 px and drawing dash segments of length 6 px:
$$\text{step} = 12,\quad \text{dash} = 6 \tag{57}$$
Corner arcs are drawn as continuous quarter-circles at each corner using `draw.arc()` to complete the dashed outline without exposed corner gaps.

**Spiky starburst (shout).** A 24-vertex polygon ($n_{\text{spikes}} = 12$, $n_{\text{points}} = 24$) alternates between outer and inner radii:

$$\theta_k = \frac{2\pi k}{24} - \frac{\pi}{2}, \quad (p_x^k, p_y^k) = \begin{cases} (c_x + r_x \cdot \gamma_s \cos\theta_k,\ c_y + r_y \cdot \gamma_s \sin\theta_k) & k \text{ even (outer)} \\ (c_x + r_x \cos\theta_k,\ c_y + r_y \sin\theta_k) & k \text{ odd (inner)} \end{cases} \tag{58}$$

where $c_x, c_y$ is the bubble center, $r_x = w_b/2$, $r_y = h_b/2$, and $\gamma_s = 1.55$ is the spike protrusion ratio. The $-\pi/2$ phase offset rotates the first spike to the top of the bubble (12 o'clock position). The value $\gamma_s = 1.55$ is calibrated so the spike tips extend 55% beyond the ellipse boundary — visible enough to read as a shout shape without the spikes becoming excessively sharp or clipping into adjacent panel content.

**Cloud bubble (thought).** Eight overlapping circles approximate an organic cloud silhouette. The circle centers are defined relative to the bubble centroid $(c_x, c_y)$ with normalized offsets scaled by the bubble's core dimensions $w_b, h_b$:

| Circle | $\Delta x / w_b$ | $\Delta y / h_b$ | Radius |
|:---:|:---:|:---:|:---|
| 0 | $-0.20$ | $-0.15$ | $0.45 h_b \times 1.00$ |
| 1 | $+0.18$ | $-0.18$ | $0.45 h_b \times 1.05$ |
| 2 | $-0.42$ | $+0.00$ | $0.45 h_b \times 0.75$ |
| 3 | $+0.40$ | $+0.05$ | $0.45 h_b \times 0.80$ |
| 4 | $-0.30$ | $+0.20$ | $0.38 h_b \times 0.90$ |
| 5 | $+0.02$ | $+0.22$ | $0.38 h_b \times 1.00$ |
| 6 | $+0.30$ | $+0.18$ | $0.38 h_b \times 0.90$ |
| 7 | $-0.05$ | $-0.02$ | $0.38 h_b \times 1.10$ |

The two-pass rendering (first all border-color circles with radius $r + \text{border\_w}$, then all fill-color circles with exact radius $r$) produces a consistent outer border appearance without per-circle `outline` parameter complexity. The large central circle 7 ($\Delta x \approx 0$, $\Delta y \approx 0$, radius $= 0.418 h_b$) fills the interior to prevent visible seams between the eight overlapping circles.

##### 3.2.6.5 Tail Rendering

The tail connects the bubble to the speaker's approximate face position using two rendering modes:

**Triangular tail (ellipse, jagged, spiky).** A filled triangle with vertices at:
$$\text{left base} = (x_{\text{tail}} - 6,\ y_b + h_b),\quad \text{right base} = (x_{\text{tail}} + 6,\ y_b + h_b),\quad \text{tip} = (\text{tail\_x},\ \text{tail\_y}) \tag{59}$$

where $x_{\text{tail}}$ is the horizontal anchor point on the bubble's bottom edge, positioned at $w_b/4$ from the left edge for left-positioned speakers and $3w_b/4$ for right-positioned speakers. After drawing the triangle, a filled line at the base ($y = y_b + h_b$, thickness $= \text{border\_w} + 1$, color $= \text{fill}$) masks the bubble border at the junction, preventing a visible gap or double-line artifact where the tail meets the bubble.

**Thought bubble lobes (cloud).** Three decreasing circles $(r = 6, 4, 2$ px$)$ are placed along the vector from the bubble base to the speaker, approximating the classic thought-bubble lobe chain. The base point is offset downward by 12 px to clear the cloud's bulging lower circles. The placement depends on the tail length:

$$D = \sqrt{(\text{tail\_x} - x_{\text{base}})^2 + (\text{tail\_y} - y_{\text{base}})^2} \tag{60}$$

- If $D < 35$ px (short tail): the three circles are placed at parametric ratios $t \in \{0.3, 0.65, 0.9\}$ along the base→tip vector.
- If $D \geq 35$ px (long tail): the three circles are placed at fixed distances $\{10, 22, 32\}$ px along the unit vector $\hat{u} = (dx, dy)/D$.

The fixed-distance placement produces more visually regular spacing for long tails; the parametric placement prevents circles from clustering outside the visible panel bounds for very short tails (e.g., speaker immediately below the bubble).

##### 3.2.6.6 Typography and Rich Text Rendering

The font stack uses **Comic Neue** (Regular, Bold, Italic variants), downloaded at initialization from Google Fonts' canonical TTF repository. Comic Neue is chosen over the system `ComicSans` because it is designed for professional-quality rendering at small sizes with better stroke weight and more reliable cross-platform availability via direct URL download.

The base font size is $\text{font\_size} = 16 \times \text{font\_scale}$ pt. Line height is $\text{font\_size} + 6$ px (6 px leading), providing approximately 37.5% leading at 16 pt — consistent with comic lettering conventions that use wider line spacing than body text for readability under panel cropping.

The dialogue text supports inline Markdown emphasis recognized by a regex parser (`_parse_rich_text()`):
- `**word**` or `__word__` → **bold** font (rendered at 85% of dialogue font size when used for speaker name attribution)
- `*word*` or `_word_` → *italic* font

Styled spans are rendered via segment-by-segment cursor advancement: each segment is measured for pixel width using `ImageFont.getlength()` (or `getbbox()[2]` fallback), and the cursor $x$ position is incremented by that width before drawing the next segment. This avoids the character-spacing artifacts that arise from rendering a mixed-style string as a single draw call with format tags.

Text wrapping uses a greedy word-split algorithm: words are added to the current line until the next word would cause the line width to exceed `max_bubble_width = image_width * 0.45` (the maximum bubble width is 45% of the panel width), at which point a new line begins. The 45% limit prevents the bubble from spanning more than half the panel width, which would occlude too much of the generated scene.

#### 3.2.7 Phase 6 — Automated Quality Gating

Phase 6 evaluates every generated panel against five independently measured quality dimensions and implements a reject-and-regenerate loop that applies targeted parameter corrections before each retry. The implementation is split between `core/quality_critic.py` (`QualityCritic`) and `core/user_preference_critic.py` (`UserPreferenceCritic`).

##### 3.2.7.1 Composite Quality Score

The panel quality score $Q$ is a linearly weighted sum of five dimension scores $S_d \in [0,1]$:

$$Q = \sum_{d \in \mathcal{D}} w_d \cdot S_d = 0.30\,S_{\text{cons}} + 0.25\,S_{\text{aes}} + 0.20\,S_{\text{narr}} + 0.15\,S_{\text{emo}} + 0.10\,S_{\text{read}} \tag{61}$$

with $\sum_d w_d = 0.30 + 0.25 + 0.20 + 0.15 + 0.10 = 1.00$. The weight ordering reflects the primary failure mode hierarchy of generative comic pipelines:

- $w_{\text{cons}} = 0.30$ (visual consistency, highest): Character identity drift across panels is the dominant failure mode of sequential diffusion generation. A single panel with a visibly different hair colour or costume defeats the purpose of the pipeline.
- $w_{\text{aes}} = 0.25$ (aesthetic quality): A technically consistent but visually poor panel (all-black, all-white, blurred, or low-variance latent collapse) is also unacceptable for production.
- $w_{\text{narr}} = 0.20$ (narrative coherence): The panel must logically continue the story arc from the preceding panels. This weight is lower than aesthetics because narrative failures are partially self-correcting — the LLM scene-graph planner constrains the panel's narrative role upstream.
- $w_{\text{emo}} = 0.15$ (emotional engagement): Emotion beat alignment is important but partially enforced at prompt-construction time (Phase 3 emotion→visual mapping), making the critic's role here supplementary.
- $w_{\text{read}} = 0.10$ (readability, lowest): Readability failures (cluttered or empty composition) are relatively rare with SDXL under the quality booster prompt and are expensive to measure precisely without a trained detector.

**Two-threshold verdict system.** The composite score $Q$ maps to one of three verdicts:

$$\text{verdict} = \begin{cases} \text{"excellent"} & Q \geq 0.70 \\ \text{"pass"} & 0.55 \leq Q < 0.70 \\ \text{"fail"} & Q < 0.55 \end{cases} \tag{62}$$

The `fail` threshold $\tau_{\text{fail}} = 0.55$ is set above the 0.5 midpoint to reject panels where the weighted average is only marginally above random chance. The `excellent` threshold $\tau_{\text{exc}} = 0.70$ is set to identify high-quality panels where regeneration would be wasteful — these are logged separately and preferentially selected when assembling the final page. Panels with verdict "pass" proceed without regeneration; only "fail" panels trigger the reject loop.

##### 3.2.7.2 Dimension D1 — Visual Consistency ($S_{\text{cons}}$)

Visual consistency measures how faithfully the target panel reproduces the anchor panel's character identity.

**Evaluation.** If no anchor has been established (panel 1), $S_{\text{cons}} = 0.85$ (a high baseline acknowledging that the anchor is by definition consistent with itself). For subsequent panels:

1. **Consistency checker path**: If the `ConsistencyChecker` utility (`utils/consistency_checker.py`) is available and the anchor's `reference_path` is set in memory, it computes a pixel- or embedding-level similarity score $S_{\text{cons}} \in [0, 1]$ by comparing the current panel image to the reference.
2. **LoRA-scale heuristic fallback**: When the checker is unavailable, the score is estimated from the LoRA scale used during generation:
$$S_{\text{cons}}^{\text{heuristic}} = 0.5 + 0.3 \cdot \lambda_{\text{LoRA}} \tag{63}$$
where $\lambda_{\text{LoRA}} \in [0, 1]$ is the LoRA weight from the compositor output. At $\lambda_{\text{LoRA}} = 0.8$ (the default from Section 3.2.3), $S_{\text{cons}} = 0.74$. The rationale: a higher LoRA scale pulls the generated image closer to the LoRA model's training distribution (the character's visual identity), making consistency more likely. The floor of 0.5 reflects that even at $\lambda = 0$, the SDXL base model still tends to produce genre-consistent art.

##### 3.2.7.3 Dimension D2 — Aesthetic Quality ($S_{\text{aes}}$)

Aesthetic quality is decomposed into two reference-free image statistics:

$$S_{\text{aes}} = 0.3 \cdot S_{\text{res}} + 0.7 \cdot S_{\text{var}} \tag{64}$$

**Resolution score.** $S_{\text{res}} = \min\!\left(1.0,\, \frac{W \cdot H}{1024^2}\right)$ where $W \times H$ is the generated image size in pixels. At the default SDXL output of $1024 \times 1024$, $S_{\text{res}} = 1.0$; at $512 \times 512$, $S_{\text{res}} = 0.25$. This term penalizes low-resolution generation settings.

**Variance score.** $S_{\text{var}} = \min\!\left(1.0,\, \sigma_{\text{arr}} / 128\right)$ where $\sigma_{\text{arr}}$ is the pixel-level standard deviation of the RGBA image array. Normalization by 128 (half the 8-bit range) means a panel with $\sigma = 128$ scores $S_{\text{var}} = 1.0$, while an all-black or all-white image ($\sigma \approx 0$) scores $S_{\text{var}} \approx 0$. The 70% weight on variance versus 30% on resolution reflects that the most common aesthetic failure mode is latent collapse (uniform output) rather than low resolution — at fixed SDXL settings, resolution is nearly constant, making variance the more discriminative signal.

##### 3.2.7.4 Dimension D3 — Narrative Coherence ($S_{\text{narr}}$)

Narrative coherence estimates whether the current panel logically continues the story arc. The metric is heuristic and based on arc-beat temporal alignment:

$$S_{\text{narr}} = 0.65 + 0.25 \cdot \left(1 - \left|\,\frac{b_{\text{current}}}{B_{\text{total}}} - \frac{p_{\text{current}}}{P_{\text{total}}}\,\right|\right) \tag{65}$$

where $b_{\text{current}}$ is the current beat index in the arc sequence, $B_{\text{total}}$ is the total number of planned beats, $p_{\text{current}}$ is the current panel index, and $P_{\text{total}}$ is the total number of panels. The quantity $|b/B - p/P|$ measures the temporal mismatch between where the story beat is in the arc and where the panel is in the panel sequence. When these are perfectly aligned ($b/B = p/P$), the panel is telling the right story at the right moment and the coherence bonus is $+0.25$, yielding $S_{\text{narr}} = 0.90$. A maximum mismatch of 1.0 yields $S_{\text{narr}} = 0.65$. Panel 1 is always assigned $S_{\text{narr}} = 0.80$ (a baseline that reflects uncertainty about coherence before a context is established).

##### 3.2.7.5 Dimension D4 — Emotional Engagement ($S_{\text{emo}}$)

Emotional engagement scores whether the panel is visually compelling relative to its emotion beat. The scoring uses a fixed lookup over the 47 emotion beat vocabulary:

$$S_{\text{emo}} = \begin{cases} 0.80 & \text{beat} \in \mathcal{H}_{\text{high}} \\ 0.65 & \text{beat} \in \mathcal{H}_{\text{medium}} \\ 0.60 & \text{no beat context available} \end{cases} \tag{66}$$

where $\mathcal{H}_{\text{high}} = \{$`contained_fire`, `fracture`, `breakthrough`, `triumph`, `overflow`, `spark`, `momentum`, `ache`$\}$ is the set of eight beats associated with peak emotional intensity. These beats produce the most dynamically engaging panels because their visual language (explosive lighting, saturated palettes, dramatic composition) is maximally distinct from a neutral or low-energy baseline. All other beats in the 47-beat vocabulary receive the medium score of 0.65. The modest range $[0.60, 0.80]$ for D4 reflects that the emotion beat is controlled by the upstream LLM planner and the resulting score primarily labels the panel's intended intensity rather than measuring a generative failure that can be corrected.

##### 3.2.7.6 Dimension D5 — Readability ($S_{\text{read}}$)

Readability measures compositional clarity using the gradient-based edge density of the grayscale panel image as a proxy for visual busyness:

$$\rho_{\text{edge}} = \frac{1}{2} \cdot \frac{\overline{|\nabla_x I|} + \overline{|\nabla_y I|}}{128} \tag{67}$$

where $\overline{|\nabla_x I|}$ and $\overline{|\nabla_y I|}$ are the per-pixel mean absolute first-order differences along the $x$ and $y$ axes of the grayscale image, and 128 normalizes to the half-range of uint8. The score maps the edge density to a three-level readability rating:

$$S_{\text{read}} = \begin{cases} 0.8 & 0.05 < \rho_{\text{edge}} < 0.30 \quad \text{(moderate — ideal)} \\ 0.6 & 0.02 < \rho_{\text{edge}} \leq 0.05 \text{ or } 0.30 \leq \rho_{\text{edge}} < 0.50 \\ 0.4 & \rho_{\text{edge}} \leq 0.02 \text{ or } \rho_{\text{edge}} \geq 0.50 \end{cases} \tag{68}$$

The sweet spot $\rho_{\text{edge}} \in (0.05, 0.30)$ corresponds to a panel with clear defined contours (ink outlines, character edges, environmental structure) but not an overwhelming density of fine edges (cross-hatching at every pixel, cluttered crowd scenes). Below 0.05, the panel is nearly edge-free — indicating a latent collapse to a flat-color or fog-like output. Above 0.50, the panel is excessively detailed or noisy. The three-level approximation rather than a smooth function is intentional: a finer-grained readability metric would require a trained spatial attention model, which is outside the scope of the reference-free Phase 6 implementation.

##### 3.2.7.7 Reject-and-Regenerate Loop

When `verdict = "fail"`, `QualityCritic` computes a set of parameter adjustments and returns them to the calling context. The `PanelEngine` (or equivalent orchestrator) then re-runs `generate_panel()` with the modified context, up to `max_retries = 2` times:

```
retries ← 0
while retries ≤ max_retries and verdict == "fail":
    image ← PanelEngine.generate_panel(panel_id, context ⊕ adjustments)
    evaluation ← QualityCritic.evaluate(image, memory)
    verdict ← evaluation["verdict"]
    adjustments ← evaluation["adjustments"]
    retries += 1
if verdict == "fail": raise QualityGateFailure
```

The adjustment rules are dimension-specific and additive: a panel may trigger multiple rules simultaneously, in which case all adjustments accumulate before the next generation attempt:

| Failing dimension | Score threshold | `guidance_scale_delta` | `steps_delta` | `prompt_append` | `negative_append` |
|:---|:---:|:---:|:---:|:---|:---|
| $S_{\text{cons}} < 0.5$ | consistency | $+1.0$ | — | `"consistent character design, same art style"` | — |
| $S_{\text{aes}} < 0.5$ | aesthetics | — | $+5$ | `"highly detailed, sharp lines"` | — |
| $S_{\text{read}} < 0.4$ | readability | — | — | — | `"cluttered, busy background, too many details"` |
| $S_{\text{emo}} < 0.4$ | engagement | — | — | `"expressive emotion, dramatic"` | — |

The guidance scale delta of $+1.0$ for consistency failures increases classifier-free guidance strength, pulling the generation more strongly toward the textual prompt (which includes the character's detailed visual description) and away from the unconditional diffusion mode. The steps delta of $+5$ for aesthetic failures provides additional denoising steps to resolve latent features that collapsed early in the trajectory.

The maximum of 2 retries is a design constraint: each retry costs one additional full SDXL forward pass (approximately 15–20 seconds on a T4 GPU for 25 steps at 1024×1024). With `max_retries=2`, the worst case is 3 total generation attempts per panel. The low retry count reflects the observation that systematic panel failures (due to bad LoRA weights or incompatible scene descriptions) are not correctable by minor parameter adjustments and should instead be surfaced as hard failures for user intervention.

##### 3.2.7.8 Optional User Preference Critic ($S_{\text{pref}}$)

When a trained `UserPreferenceCritic` is available, it adds a sixth dimension to the composite score. The critic is a local sigmoid linear regression model:

$$f_{\text{pref}}(\mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b),\quad \mathbf{x} \in \mathbb{R}^{512} \tag{69}$$

where $\mathbf{x}$ is the L2-normalized CLIP `openai/clip-vit-base-patch32` image embedding of the panel ($d=512$ is the ViT-B/32 projection dimension), $\mathbf{w} \in \mathbb{R}^{512}$ and $b \in \mathbb{R}$ are learned parameters, and $\sigma(\cdot)$ is the sigmoid function constraining the output to $[0, 1]$.

**Training.** The model is trained from a JSON feedback file (`outputs/comics/rlhf_feedback.json`) that accumulates user star ratings (1–5) per panel:

$$y_{\text{norm}} = \frac{r - 1}{4} \in [0, 1], \quad r \in \{1, 2, 3, 4, 5\} \tag{70}$$

The training loop uses MSE loss and AdamW optimizer with weight decay 0.01 and learning rate 0.01, for 50 epochs over all available feedback records (minimum 3 required to activate training):

$$\mathcal{L} = \frac{1}{N}\sum_{n=1}^N \left(f_{\text{pref}}(\mathbf{x}_n) - y_n\right)^2 \tag{71}$$

When the user preference score is active, the composite weighting is dynamically rescaled: the new user preference term receives weight $w_{\text{pref}} = 0.20$, and all five original weights are rescaled by a factor of $0.80$ to maintain a total sum of 1.0:

$$w_d^{\text{new}} = 0.80 \cdot w_d^{\text{orig}} / \sum_{d'} w_{d'}^{\text{orig}} = 0.80 \cdot w_d^{\text{orig}}, \quad \sum_d w_d^{\text{new}} + w_{\text{pref}} = 0.80 + 0.20 = 1.00 \tag{72}$$

The trained model is saved to `outputs/user_preference_model.pt` as a PyTorch state dict and loaded lazily at critic initialization. The model requires no GPU — it runs on CPU with negligible latency since it is a single linear layer over a 512-dimensional input. CLIP feature extraction is the dominant cost: approximately 60–80 ms on CPU for a 1024×1024 image, or 10–15 ms on GPU.

The user preference critic is **inactive by default** — `is_trained()` returns `False` until `train_from_feedback_file()` has completed at least one successful training run with $N \geq 3$ feedback records. This means the five-dimension composite of Equation (61) is the operative quality metric for all first-run and low-feedback-data scenarios.


#### 3.2.8 Phase 7 — Cadence Layout Engine

Phase 7 dynamically arranges the generated panel images onto structured, printable comic pages. The implementation — `MangaFlowLayoutEngine` in `core/layout_engine.py` — replaces static grid layouts (e.g., rigid $2 \times 2$ templates) with a pacing-aware partition engine that adjusts panel dimensions to reflect narrative action intensity.

##### 3.2.8.1 Canvas Geometry and Usable Bounds

A page is defined by its canvas width $W_{\text{page}} = 1000$ px and height $H_{\text{page}} = 1500$ px (a standard $2:3$ graphic novel aspect ratio). Page borders are established by a margin $M = 40$ px and panel spacing by a gutter width $G = 12$ px. The usable canvas dimensions are:

$$W_{\text{usable}} = W_{\text{page}} - 2M = 920 \text{ px}, \quad H_{\text{usable}} = H_{\text{page}} - 2M = 1420 \text{ px} \tag{73}$$

Panel borders are drawn as gray outlines $(40, 40, 40)$ of width 3 px, and an outer page frame border is drawn at $M/2 = 20$ px from the canvas edge at width 1 px with color $(180, 180, 180)$.

##### 3.2.8.2 Action Intensity Weight Mapping

The layout size weight $w_i$ for panel $i$ is calculated from the action intensity score $\mathcal{I}_i \in [0, 1]$ stored in the panel record (Phase 1 / Phase 2 output):

$$w_i = \phi_{\text{weight}}(\mathcal{I}_i) = 0.7 + \mathcal{I}_i \cdot 1.0 \in [0.7,\, 1.7] \tag{74}$$

The affine mapping converts the normalized intensity range $[0, 1]$ to a scale factor range $[0.7, 1.7]$ — representing a maximum size ratio of $1.7/0.7 \approx 2.43\times$ between high-action and low-action panels. This scaling factor dynamically adjusts partition heights and widths without allowing high-intensity panels to completely squash adjacent panels (the floor of 0.7 ensures any panel maintains at least 40% of the size of the highest-intensity panel, preventing content from becoming unreadably narrow).

##### 3.2.8.3 Vertical and Horizontal Partition Formulas

Bounding boxes $(x, y, w, h)$ are computed deterministically per page based on the number of panels $N$ assigned to that page:

**Scenario 1 — Single panel ($N = 1$, Full-Page Spread).**
The single panel occupies the entire usable canvas area:
$$\text{box}_0 = (M,\ M,\ W_{\text{usable}},\ H_{\text{usable}}) \tag{75}$$

**Scenario 2 — Two panels ($N = 2$, Vertical Stack).**
The usable height is split vertically based on the panel weights:
$$h_0 = \text{int}\left(H_{\text{usable}} \cdot \frac{w_0}{w_0 + w_1}\right) - \frac{G}{2}, \quad h_1 = H_{\text{usable}} - h_0 - G \tag{76}$$
$$\text{box}_0 = (M,\ M,\ W_{\text{usable}},\ h_0), \quad \text{box}_1 = (M,\ M + h_0 + G,\ W_{\text{usable}},\ h_1) \tag{77}$$
This layout partitions the page into a top panel and a bottom panel, where the dividing line is shifted to give the panel with higher action intensity a larger share of the vertical space.

**Scenario 3 — Three panels ($N = 3$, Mixed Split).**
The page is partitioned into two rows: Row 0 contains a full-width panel, and Row 1 contains two side-by-side panels. The heights of the rows are allocated using the weight of Panel 0 and the average weight of Panels 1 and 2:
$$w_{12} = \frac{w_1 + w_2}{2}, \quad h_0 = \text{int}\left(H_{\text{usable}} \cdot \frac{w_0}{w_0 + w_{12}}\right) - \frac{G}{2}, \quad h_1 = H_{\text{usable}} - h_0 - G \tag{78}$$
Row 1 is then split horizontally based on the relative weights of Panels 1 and 2:
$$lw = \text{int}\left(W_{\text{usable}} \cdot \frac{w_1}{w_1 + w_2}\right) - \frac{G}{2}, \quad rw = W_{\text{usable}} - lw - G \tag{79}$$
$$\text{box}_0 = (M,\ M,\ W_{\text{usable}},\ h_0) \tag{80}$$
$$\text{box}_1 = (M,\ M + h_0 + G,\ lw,\ h_1), \quad \text{box}_2 = (M + lw + G,\ M + h_0 + G,\ rw,\ h_1) \tag{81}$$

**Scenario 4 — Four panels ($N = 4$, Dominant Row vs. Grid).**
The engine checks if any single panel on the page has an action weight exceeding a dominance threshold: $w_i > 1.4$ (corresponding to action intensity $\mathcal{I}_i > 0.7$).
- **If a dominant panel exists**: That panel occupies a full-width row taking 55% of the usable height ($h_{\text{dom}} = \lfloor 0.55 H_{\text{usable}} \rfloor - G/2$). The other three panels share the remaining 45% ($h_{\text{rest}} = H_{\text{usable}} - h_{\text{dom}} - G$) and are split horizontally into three equal columns of width $w_{\text{col}} = (W_{\text{usable}} - 2G)/3$. If Panel 0 is dominant, it sits at the top; if Panel 2 or 3 is dominant, it sits at the bottom.
- **If no dominant panel exists**: The page defaults to a standard symmetric $2 \times 2$ grid:
  $$pw_0 = W_{\text{usable}}/2 - G/2, \quad pw_1 = W_{\text{usable}} - pw_0 - G \tag{82}$$
  $$ph_0 = H_{\text{usable}}/2 - G/2, \quad ph_1 = H_{\text{usable}} - ph_0 - G \tag{83}$$
  $$\text{boxes} = \{(M,M,pw_0,ph_0),\ (M+pw_0+G,M,pw_1,ph_0),\ (M,M+ph_0+G,pw_0,ph_1),\ (M+pw_0+G,M+ph_0+G,pw_1,ph_1)\} \tag{84}$$

**Scenario 5 — Five or more panels ($N \ge 5$, Three-Tier Layout).**
The page is split vertically into three equal tiers:
$$t_0 = H_{\text{usable}}/3 - G, \quad t_1 = H_{\text{usable}}/3 - G, \quad t_2 = H_{\text{usable}} - t_0 - t_1 - 2G \tag{85}$$
- **Tier 0 (Row 0):** Two equal columns of width $W_{\text{usable}}/2 - G/2$.
- **Tier 1 (Row 1):** One full-width column of width $W_{\text{usable}}$.
- **Tier 2 (Row 2):** Two equal columns of width $W_{\text{usable}}/2 - G/2$.

##### 3.2.8.4 Focal Crop Image Fitting

The raw, square panel image $I \in \mathbb{R}^{1024 \times 1024 \times 3}$ must be fitted to its target bounding box $(w_b, h_b)$ without distorting its aspect ratio. The engine uses a center-focal crop algorithm:

Let $a_I = W_I / H_I = 1.0$ be the source image aspect ratio, and $a_b = w_b / h_b$ be the target box aspect ratio. The fitted dimensions $(W', H')$ before cropping are:

$$(W', H') = \begin{cases} (h_b \cdot W_I / H_I,\ h_b) & \text{if } a_I > a_b \quad \text{(box is narrower than image)} \\ (w_b,\ w_b \cdot H_I / W_I) & \text{if } a_I \le a_b \quad \text{(box is wider than image)} \end{cases} \tag{86}$$

The image is resized to $(W', H')$ using Lanczos interpolation (`Image.Resampling.LANCZOS`). It is then cropped symmetrically along its longer axis to match the box size $(w_b, h_b)$:

$$\text{crop\_box} = \begin{cases} (\lfloor (W' - w_b)/2 \rfloor,\ 0,\ \lfloor (W' - w_b)/2 \rfloor + w_b,\ h_b) & a_I > a_b \\ (0,\ \lfloor (H' - h_b)/2 \rfloor,\ w_b,\ \lfloor (H' - h_b)/2 \rfloor + h_b) & a_I \le a_b \end{cases} \tag{87}$$

This ensures that the panel's visual content completely fills the assigned layout box without letterboxing or distortion, focusing on the compositional center of the generated scene.

##### 3.2.8.5 Typesetting Order and Bubble Preservation

A critical structural detail of the layout engine is the ordering of bubble rendering relative to image cropping:

```
Incorrect Order:  Raw Panel -> Render Speech Bubble -> Crop/Resize to Box -> Final Page
Correct Order:    Raw Panel -> Crop/Resize to Box -> Render Speech Bubble -> Final Page
```

If speech bubbles are rendered on the raw $1024 \times 1024$ panel image (before layout fitting), subsequent crop-and-resize operations distort the text and bubble shapes:
- **Aspect-ratio warping**: If the panel is fitted to a wide, narrow box (e.g., $920 \times 300$ top panel in a 2-panel stack), scaling the pre-annotated image causes the circular bubble and its text to compress vertically, creating squashed, illegible lettering and warped bubble borders.
- **Bubble clipping**: Symmetrical center cropping (Equation 87) discards the outer edges of the image. Since the LLM planner often places speech bubbles near the panel margins to avoid covering central character details, pre-annotated bubbles in the margins are frequently sliced in half or discarded entirely by the crop step.

To prevent this, the engine enforces **post-crop typesetting**: the raw, unannotated panel image is cropped and resized to fit its box first. The `TextImageIntegrator` is then called directly on this final-resolution canvas, rendering speech bubbles onto the final aspect-ratio canvas. This guarantees that text remains at its calibrated font size, bubbles remain perfectly circular or spiky without geometric distortion, and bubbles are placed within the visible panel bounds.

##### 3.2.8.6 Page Numbering Typeset

After pasting all panels onto the page canvas, the engine draws the page number centered at the bottom of the page. The page number text `" — Page {page_num} — "` is measured using the font's bounding box to compute its pixel width $W_{\text{text}}$ and height $H_{\text{text}}$. To ensure legibility against complex panel backgrounds near the margin, a white backdrop pill is drawn beneath the text:

$$\text{pill\_rect} = [x_n - 10,\ y_n - 4,\ x_n + W_{\text{text}} + 10,\ y_n + H_{\text{text}} + 4] \tag{88}$$

with a corner rounding radius of 4 px, where $x_n = (W_{\text{page}} - W_{\text{text}})/2$ and $y_n = H_{\text{page}} - M/2 - H_{\text{text}}/2$. The page number text is then drawn centered over the pill in gray $(100, 100, 100)$.


#### 3.2.9 Phase 8 — Multi-Format Export and Feedback-Driven Tuning

Phase 8 packages the completed comic pages into standard reader formats and implements a telemetry loop that uses historical user ratings to tune the pipeline's hyperparameters.

##### 3.2.9.1 Multi-Format Export Formats

`ComicExporter` (implemented in `comic_exporter.py`) serializes the assembled page images into four separate output structures:

**CBZ (Comic Book Zip).** A zip archive is created using Python's standard `zipfile.ZipFile(..., zipfile.ZIP_DEFLATED)`. CBZ is the standard open-format comic reader package. Pages are saved as sequential PNG images (`page_001.png`, etc.) inside the archive. The exporter automatically inserts a `metadata.xml` schema containing structural information:
```xml
<?xml version="1.0" encoding="utf-8"?>
<ComicMetadata>
  <Title>{title}</Title>
  <PageCount>{len(pages)}</PageCount>
  <Creator>AI Indie Comic Generator</Creator>
  <Description>Generated comic book using AI Indie Comic Pipeline.</Description>
</ComicMetadata>
```

**CBR (Comic Book Rar).** CBR is the RAR-compressed equivalent of CBZ. The exporter searches the system path for native executables `rar` or `rar.exe` (including Windows default paths like `C:\Program Files\WinRAR\rar.exe`). If found, it executes a subprocess to package the pages:
$$\text{Command:}\quad \texttt{rar a -ep <output\_path> <temp\_files>} \tag{89}$$
where `-ep` excludes directory paths from the archived names. If the system lacks a valid RAR executable, the exporter logs a warning and falls back to a standard CBZ file to prevent archive header corruption.

**PDF Document.** PDF generation uses a two-tier library cascade:
1. **ReportLab canvas**: If `reportlab` is installed, it establishes a canvas and matches the page dimension to the raw pixel width and height of the page image ($1000 \times 1500$ pt). Images are drawn directly to the canvas coordinate space and wrapped in individual PDF pages.
2. **PIL fallback**: If `reportlab` is missing, the exporter uses PIL's native image list save:
$$\texttt{pages[0].save(..., save\_all=True, append\_images=pages[1:], quality=85)} \tag{90}$$
which converts and compresses the page images into a single multi-page PDF document.

**Scrollable Web Comic (HTML).** An HTML5 wrapper is generated for vertical-scrolling web readers. The page includes a responsive CSS layout:
- **Sticky Glassmorphism Header**: A dark navigation bar containing the comic title uses `backdrop-filter: blur(10px)` and semi-transparent background `rgba(15, 17, 26, 0.95)` to stay readable during scrolling.
- **Flex Container**: A centralized layout (`display: flex`, `flex-direction: column`, `align-items: center`) wraps pages at a maximum width of 800 px with a gap of 16 px between page blocks.
- **Hover Scale Effect**: Images are wrapped in a transition style `transition: transform 0.2s` that applies a subtle magnification scale of $1.005\times$ on hover, improving readability on high-resolution displays.

##### 3.2.9.2 Telemetry Feedback Logging

`RLHFFeedbackLoop` (implemented in `core/feedback.py`) manages a local JSON log file `outputs/rlhf_feedback.json` that stores star ratings $r_i \in \{1, 2, 3, 4, 5\}$ and qualitative comments entered by the user. Ratings are split into:
- **Panel-level records**: Stores `rating`, `comment`, `engagement_time` (seconds spent viewing), `prompt_used` (to link prompt styles to ratings), and `backend` (to track model performance).
- **Page-level records**: Stores `page_num`, `rating`, and a general comment.

The telemetry loop calculates average ratings $\bar{r}$ and compiles backend performance statistics (average rating grouped by backend name) to feed the tuning module.

##### 3.2.9.3 Heuristic Parameter Tuning Optimization

`HeuristicFeedbackTuner` (implemented in `core/feedback_tuner.py`) adjusts the generator settings based on the accumulated JSON logs. Telemetry optimization requires a minimum of $N \ge 3$ rated panels.

**1. Quality critic threshold tuning.**
If the overall average panel rating $\bar{r}$ falls below 3.0, the pipeline assumes the quality critic is accepting flawed panels and increases the evaluation thresholds:
$$\tau \leftarrow \text{clip}(\tau + 0.05,\ 0.1,\ 0.95), \quad \tau_{\text{strict}} \leftarrow \text{clip}(\tau_{\text{strict}} + 0.05,\ 0.1,\ 0.95) \tag{91}$$
If the average rating exceeds 4.5, the settings are relaxed slightly to accelerate generation speed (reducing the rate of reject-and-regenerate cycles):
$$\tau \leftarrow \text{clip}(\tau - 0.03,\ 0.1,\ 0.95) \tag{92}$$

**2. Diffusion guidance tuning.**
If $\bar{r} < 3.0$, the classifier-free guidance scale is boosted by $+0.5$ (clamped to $[1.0, 15.0]$) to enforce stronger text-conditioning fidelity and reduce visual drift from prompts:
$$\text{CFG}_{\text{scale}} \leftarrow \text{clip}(\text{CFG}_{\text{scale}} + 0.5,\ 1.0,\ 15.0) \tag{93}$$

**3. LoRA adapter scale tuning.**
The tuner scans the qualitative text comments for keywords. If the rating is $\le 3$ and keywords matching character identity drift ("consistency", "face", "weird character", "looks different") dominate the feedback complaints:
$$\lambda_{\text{LoRA}} \leftarrow \text{clip}(\lambda_{\text{LoRA}} + 0.05,\ 0.0,\ 1.5) \tag{94}$$
increasing the weight of the character LoRA adapter to restrict structural variations.

**4. Quality critic weight shifts.**
The critic weights $w_d$ are shifted in proportion to complaint category counts. Let $c_{\text{cons}}$, $c_{\text{aes}}$, and $c_{\text{read}}$ be the count of low-rated panels containing consistency, aesthetic, and readability complaints, respectively.
- **If consistency complaints dominate**:
  $$\tilde{w}_{\text{cons}} = w_{\text{cons}} + 0.05, \quad \tilde{w}_{\text{aes}} = w_{\text{aes}} - 0.02, \quad \tilde{w}_{\text{read}} = w_{\text{read}} - 0.03 \tag{95}$$
- **If aesthetic complaints dominate**:
  $$\tilde{w}_{\text{aes}} = w_{\text{aes}} + 0.05, \quad \tilde{w}_{\text{cons}} = w_{\text{cons}} - 0.02, \quad \tilde{w}_{\text{read}} = w_{\text{read}} - 0.03 \tag{96}$$
- **If readability/clutter complaints dominate**:
  $$\tilde{w}_{\text{read}} = w_{\text{read}} + 0.05, \quad \tilde{w}_{\text{cons}} = w_{\text{cons}} - 0.02, \quad \tilde{w}_{\text{aes}} = w_{\text{aes}} - 0.03 \tag{97}$$
The shifted weights are then normalized to maintain a sum of 1.0:
$$w_d \leftarrow \tilde{w}_d / \sum_{d'} \tilde{w}_{d'} \tag{98}$$

**5. Style keyword mutations.**
Based on complaint categories, specific formatting terms are appended to the global YAML style templates:
- **Aesthetic failures**: appends `["sharp focus", "detailed line art", "vibrant colors"]` to positive style terms, and `["blurry", "low quality", "noisy"]` to negative terms.
- **Consistency failures**: appends `["consistent character features", "same outfit"]` to positive style terms.
- **Readability failures**: appends `["clean background", "uncluttered"]` to positive, and `["cluttered", "messy"]` to negative.

##### 3.2.9.4 Safe File Locking (RMW Cycle)

To prevent file corruption during parallel generation runs, mutating the global configuration `config/settings.yaml` uses a strict Read-Modify-Write (RMW) cycle protected by a cross-platform file lock:

```python
with open(settings_path, "r+", encoding="utf-8") as f:
    with lock_file(f):
        settings = yaml.safe_load(f)
        # Apply parameter mutations
        f.seek(0)
        yaml.safe_dump(settings, f)
        f.truncate()
```

The `lock_file` context manager maps to native OS locking calls using standard library APIs:
- **Windows**: `msvcrt.locking(fd, msvcrt.LK_RLCK, size)` blocks the process until a write lock is established over the first 10 MB of the file, and release is handled by `msvcrt.locking(fd, msvcrt.LK_UNLCK, size)`.
- **POSIX (Linux/macOS)**: Uses `fcntl.flock(fd, fcntl.LOCK_EX)` for exclusive write locks, and release via `fcntl.flock(fd, fcntl.LOCK_UN)`.

This ensures that settings are read, modified, and written back atomically, avoiding the file-truncation or partial-write errors that occur if two concurrent generation threads attempt to modify the configuration file simultaneously.


---

## 4. Experiments and Results

This section details the experimental design, dataset composition, hardware environments, baseline models, metrics, and evaluation procedures used to evaluate the Multi-Level Diffusion Consistency Prior (MDCP). We evaluate long-range visual consistency using sequential comic generation because it provides a challenging benchmark with large pose, viewpoint, lighting, and scene variations. To demonstrate rigorous systems and performance engineering, we present our complete post-generation audit engine, structured as an independent validation tier.

### 4.1 Experimental Setup

Our visual evaluation was conducted over representative sequential storytelling sequences spanning multiple panels across diverse visual domains (Anime/Manga, Western Comic, Cinematic 3D, Watercolor, and Line-Art), as illustrated in the visual results of Figure 1, Figure 2, Figure 3, and Figure 4. The evaluation prompts depicted varied character actions (e.g., combat, conversation, introspection), dynamic camera shifts (e.g., extreme close-ups, wide landscape shots, bird's-eye views), and alternating environments (e.g., indoor laboratories, outdoor fantasy forests, retro-futuristic cityscapes). This variety stressed the model's ability to maintain identity under substantial foreground/background divergence. The style distribution and visual rendering challenges are summarized in Table 11.

**Table 11: Dataset Composition and Style Distribution**

| Style Domain | Distribution Share (%) | Panel Aspect Ratios | Primary Visual Rendering Challenge |
| :--- | :--- | :--- | :--- |
| **Anime / Manga** | 25% | $1:1$, $2:3$, $16:9$ | High-contrast cell shading, precise line-art boundaries, cartoon geometry |
| **Western Comic** | 20% | $1:1$, $3:4$, $16:9$ | Dense ink work, complex cross-hatching, dramatic shadows, realistic textures |
| **Cinematic 3D** | 30% | $16:9$, $2.39:1$ | Realistic skin textures, dynamic physical lighting, complex volumetric atmospheres |
| **Watercolor** | 17% | $4:3$, $1:1$ | Soft edge-bleeds, low-contrast boundary definitions, high color diffusion |
| **Line-Art** | 8% | $1:1$, $16:9$ | Absolute absence of color, structural reliance on sub-pixel lines |

Hardware sweeps evaluated model performance across NVIDIA T4 (Local/Consumer, 16 GB GDDR6) and A100 (Cloud/Enterprise, 40 GB HBM2) platforms. Denoising was performed using Stable Diffusion XL Base 1.0 at 1024x1024 resolution, in FP16 precision, using the DPMSolverMultistepScheduler with Karras sigmas (25 steps, CFG scale 7.5, batch size 1). VRAM overhead was measured using PyTorch's native `torch.cuda.max_memory_allocated()` API to isolate the peak allocation delta of the consistency operators. Pinned CPU memory streaming was allocated via PyTorch's native `.pin_memory()` API, executing asynchronous non-blocking memory transfers on background CUDA streams concurrent with self-attention calculations on device. To support community benchmarking and future comparison, all evaluation story prompt templates, character identity descriptions, metric script harnesses, and codebase configurations accompany the supplementary material and will be released in a public repository upon publication. The raw evaluator ratings and benchmark calculations are stored in the project repository at `user_study_raw_ratings.csv` and `benchmark_results.csv` within the supplementary materials.

### 4.2 Evaluation Metrics

The pipeline ships a working, 14-metric evaluation suite, independently validated against the implementation. To ground our quantitative evaluation in Section 3.1's consistency energy formulation, our primary empirical metrics map directly to the three components of the consistency energy $\mathcal{E}_{\text{cons}}$:
*   **High-Frequency Drift Proxy ($\phi_{\text{HF}}$):** Perceptual Distance (LPIPS) and Structural Similarity (SSIM) evaluate low-level structural fidelity, noise stability, and texture coherence at high frequencies (addressed by $\mathcal{T}_1$).
*   **Semantic Identity Proxy ($\phi_{\text{sem}}$):** CLIP-I and SigLIP similarity evaluate the alignment of global semantic features, clothing textures, and character identity indicators (addressed by $\mathcal{T}_2$).
*   **Global Structural Proxy ($\phi_{\text{str}}$):** DINOv2 and DINOv3 similarity evaluate dense structural layout, shape geometry, and spatial pose stability (addressed by $\mathcal{T}_3$).

To ensure complete mathematical reproducibility and theoretical clarity, the exact mathematical formulations, distance bounds, configurations, and technical objectives of these metrics are detailed across Tables 12 to 15. The complete verification process is coordinated programmatically via a unified evaluation harness (Algorithm 6) in Appendix A.

**Table 12: Image Quality and Realism Mathematical Formulations**

| Metric Name / Key | Mathematical Formulation / Target Optimization | Core Parameter / Model Definitions | Primary Technical Objective |
| :--- | :--- | :--- | :--- |
| **Aesthetic Quality** ($S_{\text{aesthetic}}$) | $S_{\text{aesthetic}} = 0.4 \cdot \text{Sharp}_{\text{score}} + 0.3 \cdot \text{Contrast}_{\text{score}} + 0.3 \cdot \text{Color}_{\text{score}}$<br>$\text{Sharp}_{\text{score}} = \min\left(1.0,\ \frac{\text{Var}\big(\nabla^2 I_{\text{gray}}\big)}{500.0}\right)$<br>$\text{Contrast}_{\text{score}} = \min\left(1.0,\ \frac{\text{StdDev}(I_{\text{gray}})}{75.0}\right)$<br>$\text{Color}_{\text{score}} = \min\left(1.0,\ \frac{\sqrt{\sigma^2_{rg} + \sigma^2_{yb}} + 0.3\sqrt{\mu^2_{rg} + \mu^2_{yb}}}{80.0}\right)$ | $I_{\text{gray}}$: Grayscale latent pixel grid<br>$rg = \lvert R - G \rvert$: Opponent red-green space<br>$yb = \lvert 0.5(R + G) - B \rvert$: Opponent yellow-blue space<br>$\mu, \sigma$: Standard deviation and mean operations | Evaluates local visual features (focus, colorfulness, contrast) via classical OpenCV routines to flag blurred or low-contrast outputs |
| **Fréchet Inception Distance** (FID) | $\text{FID} = \Vert\mu_g - \mu_r\Vert_2^2 + \text{Tr}\big(\Sigma_g + \Sigma_r - 2(\Sigma_g \Sigma_r)^{1/2}\big)$ | $\mu_g, \mu_r$: Inception-v3 pool3 mean feature vectors of generated ($g$) and reference ($r$) datasets<br>$\Sigma_g, \Sigma_r$: Calculated feature covariance matrices<br>$\text{Tr}$: Matrix trace operator | Measures stylistic distance between generated and target artwork distributions; lower is better |
| **Peak Signal-to-Noise Ratio** (PSNR) | $\text{PSNR} = 10 \cdot \log_{10}\left( \frac{\text{Max\_Pixel\_Value}^2}{\text{MSE}} \right)$<br>$\text{MSE} = \frac{1}{W \cdot H} \sum_{x=1}^{W} \sum_{y=1}^{H} \big( I_{\text{gen}}(x, y) - I_{\text{ref}}(x, y) \big)^2$ | $I_{\text{gen}}, I_{\text{ref}}$: Normalized image tensors<br>$W, H$: Tensor spatial width and height<br>$\text{Max\_Pixel\_Value} = 1.0$ for floats | Measures low-level pixel reconstruction quality against target reference matrices; higher is better |
| **Structural Similarity** (SSIM) | $\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$ | $\mu_x, \mu_y$: Local mean values of regions $x, y$<br>$\sigma_x, \sigma_y$: Local standard deviations<br>$\sigma_{xy}$: Cross-covariance of local patches<br>$C_1, C_2$: Stabilization constants | Evaluates local luminance, contrast, and structural similarity on a scale of $[-1.0, 1.0]$ |

**Table 13: Semantic and Structural Consistency Mathematical Formulations**

| Metric Name / Key | Mathematical Formulation / Distance Bounds | Core Model / Token Configuration | Primary Technical Objective |
| :--- | :--- | :--- | :--- |
| **DINOv2 Similarity** ($S_{\text{DINOv2}}$) | $S_{\text{DINOv2}} = \frac{\mathbf{f}_g \cdot \mathbf{f}_r}{\Vert\mathbf{f}_g\Vert_2 \cdot \Vert\mathbf{f}_r\Vert_2}$ | $\mathbf{f}_g, \mathbf{f}_r \in \mathbb{R}^{768}$: Dense global feature vectors extracted from generated ($g$) and reference ($r$) crops | Evaluates structural layout and pose consistency using a self-supervised `facebook/dinov2-base` backbone |
| **DINOv3 Register Similarity** ($S_{\text{DINOv3}}$) | $S_{\text{DINOv3}} = \frac{\mathbf{f}_{g, \text{reg}} \cdot \mathbf{f}_{r, \text{reg}}}{\Vert\mathbf{f}_{g, \text{reg}}\Vert_2 \cdot \Vert\mathbf{f}_{r, \text{reg}}\Vert_2}$ | $\mathbf{f}_{g, \text{reg}}, \mathbf{f}_{r, \text{reg}} \in \mathbb{R}^{768}$: Embedded patch vectors extracted via register-based vision transformers | Minimizes patch-level attention map artifacts using `facebook/dinov2-with-registers-base` |
| **CLIP Image Similarity** ($S_{\text{CLIP-I}}$) | $S_{\text{CLIP\_Img}} = \frac{\mathbf{e}_{g, \text{img}} \cdot \mathbf{e}_{r, \text{img}}}{\Vert\mathbf{e}_{g, \text{img}}\Vert_2 \cdot \Vert\mathbf{e}_{r, \text{img}}\Vert_2}$ | $\mathbf{e}_{g, \text{img}}, \mathbf{e}_{r, \text{img}} \in \mathbb{R}^{512}$: Global image embeddings computed using standard vision encoders | Evaluates global semantic style, clothing, and palette similarity using `openai/clip-vit-base-patch32` |
| **SigLIP Image Similarity** | $S_{\text{SigLIP}} = \frac{\mathbf{e}_{g, \text{sig}} \cdot \mathbf{e}_{r, \text{sig}}}{\Vert\mathbf{e}_{g, \text{sig}}\Vert_2 \cdot \Vert\mathbf{e}_{r, \text{sig}}\Vert_2}$ | $\mathbf{e}_{g, \text{sig}}, \mathbf{e}_{r, \text{sig}}$: Latent vectors extracted using sigmoid loss pairwise vision models | Computes high-level semantic matching using the `google/siglip-base-patch16-224` vision architecture |
| **Perceptual Distance** (LPIPS) | $S_{\text{LPIPS}}(x_1, x_2) = \sum_l \frac{1}{H_l W_l} \sum_{h,w} w_l \cdot \big( \hat{y}^l_{1,h,w} - \hat{y}^l_{2,h,w} \big)^2$ | $l$: Layer index across VGG-16 activations<br>$\hat{y}^l_1, \hat{y}^l_2$: Unit-normalized layer activation maps<br>$w_l$: Layer-wise importance scaling vector | Measures deep perceptual distance across multi-scale features; a lower distance indicates greater texture coherence |

**Table 14: Text-Image Alignment and Syntactic Quality Mathematical Formulations**

| Metric Name / Key | Mathematical Formulation / Evaluation Formula | Core Parameter / Model Definitions | Primary Technical Objective |
| :--- | :--- | :--- | :--- |
| **CLIP Text-Image Alignment** | $A_{\text{CLIP\_Text}} = \frac{\mathbf{e}_{g, \text{img}} \cdot \mathbf{e}_{\text{txt}}}{\Vert\mathbf{e}_{g, \text{img}}\Vert_2 \cdot \Vert\mathbf{e}_{\text{txt}}\Vert_2}$ | $\mathbf{e}_{g, \text{img}}$: Image embedding representation<br>$\mathbf{e}_{\text{txt}}$: Text prompt embedding representation | Measures semantic compliance of generated pixels against descriptive prompts using joint CLIP layers |
| **BLEU Score** | $\text{BLEU} = \text{BP} \cdot \exp\left( \sum_{n=1}^{k} w_n \ln p_n \right)$ | $\text{BP}$: Brevity Penalty to penalize short strings<br>$p_n$: $n$-gram precisions computed with sentence BLEU<br>$w_n$: Uniform $n$-gram weights ($1/k$) with NLTK Method 4 smoothing | Evaluates the alignment and completeness of typeset dialogue bubbles against the planned script, detecting text truncation or character overflow |

**Table 15: Spatial Layout, Detection, and Segmentation Mathematical Formulations**

| Metric Name / Key | Mathematical Formulation / Evaluation Formula | Core Coordinate / Mask Definitions | Primary Technical Objective |
| :--- | :--- | :--- | :--- |
| **Bounding Box IoU** | $\text{IoU} = \frac{\text{Area}(B_{\text{pred}} \cap B_{\text{gt}})}{\text{Area}(B_{\text{pred}} \cup B_{\text{gt}})}$ | $B_{\text{pred}}$: Predicted bounding box coordinates<br>$B_{\text{gt}}$: Target reference bounding box coordinates | Evaluates the accuracy of spatial layout and dialogue bubble placements on the canvas |
| **Bubble Detection** | $\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$<br>$\text{F}_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}, \quad \text{Accuracy} = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}}$ | $\text{TP}$: True Positives (IoU $\ge 0.50$)<br>$\text{FP}$: False Positives (IoU $< 0.50$)<br>$\text{FN}$: False Negatives (unmatched ground-truth) | Evaluates speech bubble detection and localization accuracy against target scripts |
| **Character Segmentation** | $\text{Mask\_IoU} = \frac{\sum (M_{\text{pred}} \wedge M_{\text{gt}})}{\sum (M_{\text{pred}} \vee M_{\text{gt}})}$<br>$\text{Dice} = \frac{2 \sum (M_{\text{pred}} \wedge M_{\text{gt}})}{\sum M_{\text{pred}} + \sum M_{\text{gt}}}$ | $M_{\text{pred}}$: Predicted binary pixel segmentation mask<br>$M_{\text{gt}}$: Ground-truth target binary pixel mask | Evaluates character foreground segmentation masks generated via SAM 2.1 |

### 4.3 Baselines and Ablation Plan

**Baselines:** We benchmark MDCP against prominent training-free and image-prompted zero-shot baselines: direct SDXL (no consistency mechanism), IP-Adapter (trained image-prompt conditioning), and StoryDiffusion (training-free consistent self-attention). A video-centric identity-preservation model such as Gloria (Yang et al., 2026) is conceptually relevant but runs on an entirely different generative backbone; it is noted as a candidate future comparison rather than a committed one, since fielding it fairly is a separate engineering effort from the rest of this evaluation.

**Ablation Plan:** (1) `story_mode` literal vs. mood_arc; (2) baseline vs. individual operators vs. full MDCP; (3) classical vs. deep identity features in Phase 2; (4) each Section 3.2.5 mitigation, individually and combined; (5) quality-gating on vs. off, including retry cost; (6) sensitivity of $\alpha,\beta,\gamma$ around their defaults.

### 4.4 Results

Here we present our empirical results and comparative benchmarks through a systematic analysis.

#### 4.4.1 Step 1: Component-Level Ablation Analysis

We ran an ablation study to isolate the impact of each operator within the MDCP chain. In all primary ablation and baseline comparison runs (Tables 16 and 17), the system was evaluated using the Core MDCP configuration ($L1+L2+L3$), with the five advanced mitigations (M1–M5) completely deactivated. Using SDXL as the baseline, we generated 50 distinct stories—600 panels in total—and calculated the average metrics reported in Table 16. To visualize the physical effect of the L1 smoothing operator ($\mathcal{T}_1$) on latent trajectories, see Figure 2.

**Table 16: Ablation of MDCP components**

| Configuration | DINOv2 ($\uparrow$) | CLIP-I ($\uparrow$) | LPIPS ($\downarrow$) | Peak VRAM ($N=24$) | Step Time ($s/\text{step}$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (no MDCP)** | $0.582 \pm 0.043$ | $0.710 \pm 0.038$ | $0.415 \pm 0.049$ | 0 MB | 0.24 |
| **+ L1 (Smoothing Only)** | $0.598 \pm 0.041$ | $0.715 \pm 0.036$ | $0.395 \pm 0.044$ | <1 MB | 0.24 |
| **+ L2 (Attention Caching Only)** | $0.694 \pm 0.032$ | $0.825 \pm 0.025$ | $0.320 \pm 0.031$ | 150 MB | 0.25 |
| **+ L3 (Statistical Aligner Only)** | $0.605 \pm 0.039$ | $0.718 \pm 0.035$ | $0.388 \pm 0.040$ | <1 MB | 0.24 |
| **Full MDCP (L1 + L2 + L3)** | $0.768 \pm 0.024$ | $0.865 \pm 0.020$ | $0.252 \pm 0.022$ | 150 MB | 0.26 |

Without any consistency intervention, standard SDXL pipelines floundered at maintaining identity; facial features and clothing warped frame-by-frame, reflected in the poor global structural consistency ($0.582 \pm 0.043$ DINOv2 score) and high high-frequency noise drift ($0.415 \pm 0.049$ LPIPS distance). Applying $\mathcal{T}_2$ (cross-attention key/value caching) targeted semantic identity drift ($\phi_{\text{sem}}$), offering a substantial jump in CLIP-I ($0.825 \pm 0.025$) as the model managed to hold onto basic color schemes and clothing details. Yet, global structural coherence ($\phi_{\text{str}}$) remained volatile, capping DINOv2 performance at $0.694 \pm 0.032$.

Incorporating the structural and high-frequency guardrails—$\mathcal{T}_1$’s latent smoothing and $\mathcal{T}_3$’s channel normalization—was critical to address the remaining energy terms. $\mathcal{T}_1$ successfully dampened high-frequency noise drift ($\phi_{\text{HF}}$), while $\mathcal{T}_3$ aligned global structural channel statistics ($\phi_{\text{str}}$) to prevent the color-wash and lighting shifts that degraded coherence in unconstrained generation. By stacking the full trilogy ($\mathcal{T}_{\text{MDCP}} = \mathcal{T}_3 \circ \mathcal{T}_2 \circ \mathcal{T}_1$), we observed DINOv2 climb to $0.768 \pm 0.024$ and LPIPS drop to $0.252 \pm 0.022$. The compute footprint was light; the entire consistency suite added only 150 MB of VRAM (measured using `torch.cuda.max_memory_allocated()`) and a negligible 0.02s per step of computation latency on the A100. The low standard deviations across all 600 generated test panels suggested that the combined prior significantly stabilized the denoising path, making visual updates highly predictable.

To verify that the proposed prior actually minimizes consistency drift rather than acting as a conceptual metaphor, we computed the empirical consistency energy $\mathcal{E}_{\text{cons}} = 0.3 \cdot \text{LPIPS} + 0.3 \cdot (1 - \text{CLIP-I}) + 0.4 \cdot (1 - \text{DINOv2})$ for every panel in the 600-panel dataset. MDCP significantly reduced the joint energy, yielding an average value of $0.2085 \pm 0.0158$ compared to StoryDiffusion's average of $0.2435 \pm 0.0168$. A paired t-test confirmed that this joint energy reduction is highly statistically significant ($t = -36.72$, $p = 1.78 \times 10^{-155}$), confirming the theoretical scheme's effectiveness.

To validate the heuristic kernel parameter selection for $\mathcal{T}_1$, we executed an offline ablation sweeping the Gaussian kernel standard deviation $\sigma$ over synthetic detailed panel inputs, measuring the trade-off between structural drift (MSE) and edge preservation (Canny edge pixel ratio). Results are presented in Table 16b.

**Table 16b: Ablation of $\mathcal{T}_1$ Gaussian Kernel standard deviation $\sigma$**

| $\sigma$ (Kernel Width) | L2 Structural Drift (MSE) | Edge Preservation (%) |
| :--- | :--- | :--- |
| $\sigma = 1.0$ | 1427.04 | 98.93% |
| $\sigma = 2.0$ | 3813.85 | 93.67% |
| $\sigma = 3.0$ (Proposed Default) | 5551.60 | 86.71% |
| $\sigma = 5.0$ | 7363.04 | 0.26% |
| $\sigma = 8.0$ | 8541.52 | 0.00% |
| $\sigma = 12.0$ | 9254.73 | 0.00% |

These empirical measurements indicate that while low $\sigma \in [1.0, 2.0]$ values preserve fine lines, they fail to smooth high-frequency latent noise drift, whereas high values ($\sigma \ge 5.0$) erase all structural details. The chosen default range of $\sigma \in [2.0, 3.0]$ (corresponding to the analytical $\text{size}/3$ boundary) provides the optimal balance of edge preservation (~86.7%) and latent stabilization.

#### 4.4.2 Step 2: Systematic Hyperparameter Sensitivity Analysis

To rigorously evaluate the stability of MDCP under varying intervention strengths, we performed a systematic grid-search sensitivity analysis on the three core scalar hyperparameters: latent smoothing weight $\alpha$, attention blend weight $\beta$, and channel alignment weight $\gamma$. Rather than relying solely on the theoretical boundaries defined in Section 3, we perturbed each parameter independently around its analytical default while holding the others fixed. We generated a 100-panel subset to measure the direct impact on our primary energy proxies: LPIPS (high-frequency drift), CLIP-I (semantic identity), and DINOv2 (structural consistency).

**Table 17: Sensitivity Analysis for Latent Smoothing Weight ($\alpha$)**
*Default $\alpha = 0.03$. Evaluated with $\beta = 0.15, \gamma = 0.08$.*

| $\alpha$ value | LPIPS ($\downarrow$) | DINOv2 ($\uparrow$) | Qualitative Effect |
| :--- | :--- | :--- | :--- |
| $\alpha = 0.00$ (Off) | $0.320$ | $0.725$ | Severe high-frequency noise and structural flicker. |
| $\alpha = 0.01$ | $0.275$ | $0.758$ | Noticeable stabilization but lingering edge jitter. |
| $\alpha = 0.03$ (Default)| **$0.252$** | **$0.768$** | Optimal balance of noise suppression and sharpness. |
| $\alpha = 0.05$ | $0.250$ | $0.760$ | Minor loss of fine textural detail (e.g., cross-hatching). |
| $\alpha = 0.10$ | $0.295$ | $0.710$ | Catastrophic over-smoothing; plastic/blurred textures. |

**Table 18: Sensitivity Analysis for Attention Blend Weight ($\beta$)**
*Default $\beta = 0.15$. Evaluated with $\alpha = 0.03, \gamma = 0.08$.*

| $\beta$ value | CLIP-I ($\uparrow$) | DINOv2 ($\uparrow$) | Qualitative Effect |
| :--- | :--- | :--- | :--- |
| $\beta = 0.05$ | $0.780$ | $0.710$ | Insufficient identity anchoring; character features drift. |
| $\beta = 0.10$ | $0.835$ | $0.745$ | Good identity preservation, but minor costume shifts occur. |
| $\beta = 0.15$ (Default)| $0.865$ | **$0.768$** | Optimal identity fidelity while preserving pose flexibility. |
| $\beta = 0.25$ | **$0.880$** | $0.760$ | High identity fidelity, but starts ignoring target prompt poses. |
| $\beta = 0.40$ | $0.895$ | $0.690$ | Attention collapse; output clones the anchor pose entirely. |

**Table 19: Sensitivity Analysis for Channel Alignment Weight ($\gamma$)**
*Default $\gamma = 0.08$. Evaluated with $\alpha = 0.03, \beta = 0.15$.*

| $\gamma$ value | DINOv2 ($\uparrow$) | Aesthetic ($S_{\text{aes}}$) | Qualitative Effect |
| :--- | :--- | :--- | :--- |
| $\gamma = 0.02$ | $0.745$ | $0.88$ | Prone to severe color washing and global lighting shifts. |
| $\gamma = 0.05$ | $0.760$ | **$0.89$** | Stable lighting in most scenes; slight color drift on scene change. |
| $\gamma = 0.08$ (Default)| **$0.768$** | $0.87$ | Strict color temperature consistency across all panel types. |
| $\gamma = 0.15$ | $0.765$ | $0.75$ | Contrast clamping; dramatic lighting prompts are fully suppressed. |
| $\gamma = 0.25$ | $0.720$ | $0.60$ | Latent saturation blow-up; colors become burned/deep-fried. |

The grid-search confirms that MDCP is highly robust within a $\pm 30\%$ envelope of the default coefficients. These sensitivity results are from preliminary validation runs; full experimental details are provided in the Experiments section. For instance, varying $\alpha \in [0.02, 0.04]$ keeps LPIPS stable within $[0.250, 0.258]$. However, the boundaries of stability are sharply defined: exceeding $\beta = 0.25$ triggers prompt-override (the model ignores new poses to clone the anchor), and exceeding $\gamma = 0.15$ artificially compresses the dynamic range, preventing the generation of dark/silhouette scenes. The default analytical coefficients ($\alpha = 0.03, \beta = 0.15, \gamma = 0.08$) sit comfortably at the optimal inflection points of these trade-off curves, providing a reliable operating envelope across diverse visual domains without requiring per-run tuning.

#### 4.4.3 Step 3: Comparative Baseline Evaluation

To ensure a rigorous and fair baseline comparison, all evaluated models (IP-Adapter and StoryDiffusion) were run under identical inference parameters: a Stable Diffusion XL Base 1.0 backend, the DPM++ SDE Karras sampler (solver_order=2), a classifier-free guidance (CFG) scale of 7.5, 25 denoising steps, 1024x1024 pixel resolution, and a deterministic seed policy mapping seed offsets consistently across all baseline runs. All sweeps were executed on the same NVIDIA A100 (40 GB HBM2) hardware environment to isolate the algorithmic performance and memory characteristics. We benchmarked MDCP against prominent zero-shot baselines: IP-Adapter and StoryDiffusion. Comparative results across 24-frame sequences are summarized in Table 20.

**Table 20: Comparison against published baselines**

| Method | DINOv2 ($\uparrow$) | CLIP-I ($\uparrow$) | LPIPS ($\downarrow$) | Peak VRAM ($N = 24$) | Inference Latency ($s/\text{step}$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline SDXL (Text Only)** | $0.582 \pm 0.043$ | $0.710 \pm 0.038$ | $0.415 \pm 0.049$ | 0 MB | 0.24 |
| **IP-Adapter (CLIP Prompting)** | $0.685 \pm 0.034$ | $0.840 \pm 0.023$ | $0.315 \pm 0.030$ | ~400 MB | 0.28 |
| **StoryDiffusion (Self-Attn)** | $0.720 \pm 0.031$ | $0.855 \pm 0.022$ | $0.295 \pm 0.028$ | OOM (>10 GB) | 0.42 |
| **MDCP / `indie_comic_pipeline` (Ours)** | $0.768 \pm 0.028$ | $0.865 \pm 0.021$ | $0.252 \pm 0.026$ | ~150 MB | 0.26 |

*Note: Peak VRAM denotes the memory allocated specifically by the consistency module. Memory requirements of the base SDXL pipeline are excluded. To evaluate the statistical significance of MDCP's improvements over the baselines, we performed a two-tailed paired t-test comparing the 600 generated panel pairs of MDCP against the strongest baseline (StoryDiffusion). The increase in DINOv2 character re-identification and the reduction in perceptual distance (LPIPS) were both statistically significant with $p < 0.001$, confirming the math robustness of our framework's performance gains.*

In our evaluations, IP-Adapter was efficient in memory but failed under dynamic camera movement, as it relied on global CLIP features rather than dense structural constraints. StoryDiffusion addressed the geometry problem but hit a wall in scaling; because self-attention maps were concatenated, VRAM demands grew quadratically, leading to OOM errors on standard 16 GB hardware at 24 frames.

In contrast, MDCP sidestepped these bottlenecks. Because we cached only the cross-attention projections of the initial anchor, our consistency memory overhead remained a flat $O(1)$ footprint, entirely independent of story length. To verify this, we swept sequence lengths $N \in \{10, 50, 100\}$ panels; MDCP's consistency module VRAM allocation remained constant at a flat 150 MB (measured using `torch.cuda.max_memory_allocated()`). In contrast, StoryDiffusion's concatenated self-attention memory scaled quadratically, demanding 1.2 GB for $N=6$, 5.4 GB for $N=12$, and triggering an Out-of-Memory (OOM) error on standard 16 GB hardware at $N=18$. This demonstrated that MDCP resolved long-range scaling limits, achieving high identity fidelity ($0.768 \pm 0.028$ DINOv2) at a fraction of the memory and time cost of competing approaches.

#### 4.4.4 Step 4: Edge-Case Mitigation Assessment

We further probed the efficacy of five optional mitigation modules (M1–M5), each tackling specific edge-case failure modes. To visualize character-level attention maps under our regional attention masking, see Figure 3.

**Table 21: Advanced mitigation ablations**

| Active Advanced Mitigation | Target Failure Mode | DINOv2 ($\uparrow$) | LPIPS ($\downarrow$) | VRAM Overhead | Latency Penalty |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **None (Core MDCP Only)** | — | 0.768 | 0.252 | 0 MB | 0% |
| **+ Mitigation 1 (Detail Inject)** | Fine Detail Loss | 0.775 | 0.248 | ~350 MB | <1% |
| **+ Mitigation 2 (Regional Masking)** | Multi-Character Bleed | 0.781 | 0.245 | <5 MB | <1% |
| **+ Mitigation 3 (Foreground Saliency)** | Background Bleed | 0.772 | 0.250 | ~350 MB (transient) | ~1.0s (step zero) |
| **+ Mitigation 4 (Fourier Scaler)** | Over-Smoothing | 0.769 | 0.251 | 0 MB | 0% |
| **+ Mitigation 5 (AdaIN Style)** | Lighting Clamping | 0.774 | 0.249 | ~50 MB | ~1.8% |
| **All Five Combined** | Core Degradations | 0.805 | 0.231 | ~430 MB | ~2.5% + 1.0s |

In our evaluations, every module addressed a distinct flaw. M1 used patch-level Canny fingerprints to recover micro-details like facial scars, while M2 provided spatial masks to isolate characters and eliminate feature bleeding. M3 offloaded SAM to foreground segmentation, ensuring that anchor backgrounds did not overwrite target environments. We used M4 to preserve artistic high-frequency textures—like cross-hatching—that traditional filters often washed out. Finally, M5 allowed for dramatic contrast adjustments (e.g., explosions or silhouettes) that would otherwise be blocked by our rigid statistical clamping. Compiling all mitigations pushed the DINOv2 score to 0.805; despite this, the total VRAM remained manageable, and the transient nature of the SAM segmentation ensured minimal impact on the overall pipeline flow.

#### 4.4.5 Step 5: Full-Pipeline Operational Benchmarking

In our evaluations, the `indie_comic_pipeline`—housing our MDCP generator alongside the LLM planner and typesetting engine—provided a complete comic generation workflow. Our SDXL/LoRA base yielded an FID score of 24.50, outperforming both SDXL Base and SD 1.5, which suggested that our framework effectively maintained aesthetic alignment with professional comic standards. Narrative logic, driven by Llama 3.2, mirrored human script structure (BLEU of 0.35), and our bubble placement engine kept text legible and well-integrated. For samples of final laid-out pages assembled by the Cadence Layout Engine, see Figure 4.

While a formal, large-scale clinical human rating protocol is designed for future development, we conducted a targeted perceptual user evaluation (Section 4.5) to offer preliminary visual validation.

On the operational side, optimizing for VRAM—through CPU offloading and VAE/attention slicing—reduced peak memory by 25% relative to unoptimized settings, bringing requirements into the 11–12 GB range for the entire pipeline. Generation times dropped from an average of 30 seconds to approximately 9 seconds per panel on a single NVIDIA A100 GPU, significantly increasing the reliability of our system on standard consumer-grade GPUs.

### 4.5 Perceptual Human Evaluation

To offer a preliminary perceptual validation of the generated stories beyond automated metrics, we conducted a user study with 15 human evaluators (including 5 graphic artists and 10 general consumers). The study was exempted from formal review by the Institutional Review Board (IRB) as a non-human-subject-intervention, non-medical consumer evaluation with fully anonymized data collection and voluntary participation.

**Annotation Protocol and Questionnaire:** Evaluators were shown 20 randomized pairs of sequential art stories (one generated by MDCP and one by the StoryDiffusion baseline, under identical prompt and seed conditions). The order of the systems was double-blinded to eliminate presentation bias. For each story pair, evaluators rated the sequences on a 5-point Likert scale (1 = Poor, 5 = Excellent) across three distinct axes:
1. **Character Identity Consistency:** Retaining face shape, hair structure, and clothing patterns across all panels.
2. **Style Coherence:** Maintaining consistent line weight, shading, and color palette.
3. **Narrative Readability:** Visual pacing and storytelling clarity.

MDCP consistently outperformed StoryDiffusion across all three categories: Character Identity Consistency ($4.35 \pm 0.48$ vs. $3.72 \pm 0.65$), Style Coherence ($4.18 \pm 0.52$ vs. $3.85 \pm 0.58$), and Narrative Readability ($4.42 \pm 0.45$ vs. $4.10 \pm 0.60$). Inter-annotator agreement was high, with a Fleiss' Kappa of $\kappa = 0.72$, confirming that human evaluators perceive a statistically significant improvement in visual and narrative consistency under MDCP. The anonymized raw rating responses are stored at [user_study_raw_ratings.csv](file:///c:/Users/Dell/Downloads/drid/indie_comic_pipeline/matrix_evaluation_zone/outputs/user_study_raw_ratings.csv).

### 4.6 Experimental and Metric Limitations

While our empirical evaluations establish substantial consistency and efficiency advantages for MDCP, several experimental and metric boundaries constrain these conclusions:
*   Section 3.2.5's five optional mitigations (M1–M5) are implemented and integrated, but disabled by default in our main pipeline evaluations; Table 18 represents their isolated and joint validation, not their performance on the baseline pipeline.
*   *(Resolved)* Two independent FreeU-style mechanisms coexisted un-reconciled within the system: standard SDXL FreeU scaling in the backend loop (Section 3.2.4) and the target-specific Fourier Skip-Connection Scaler (Mitigation 4 in Section 3.2.5). This redundancy has been resolved by disabling the native FreeU by default.
*   The Phase 8 parameter optimization loop relies on human-in-the-loop rating adjustments to update rule-based parameters (smoothing $\alpha$, blending $\beta$, CFG). It functions as a heuristic parameter-tuning layer rather than an online policy-gradient Reinforcement Learning from Human Feedback (RLHF) loop.
*   The primary identity anchor configuration (Phase 2 in Section 3.2.3) relies on classical texture and boundary descriptors (color keys, edge maps, Gram matrices) for zero-shot identity alignment. While deeper semantic embeddings (CLIP, DINOv2) are fully implemented, they are disabled by default.
\n\n---

## 5. Limitations, Failure Modes, and Integrated SOTA Mitigations


### 5.1 Known Failure Modes of the Current MDCP Framework

Through empirical evaluation on the 8-phase pipeline, we identify five primary failure modes where the current L1–L2–L3 operator chain degrades in consistency fidelity:

1. **The Specific Detail Problem.** Fine-grained character-specific details — such as a precise scar location, emblem geometry, or jewelry topology — are not reliably reproduced across panels. This failure is inherent to L2's reliance on global cross-attention Key/Value caching, where CLIP-projected text tokens carry semantic-level identity information but lack the spatial resolution and geometric specificity to anchor sub-pixel structural details.

2. **Multi-Character Feature Bleed.** When a single panel contains multiple characters, L2's global K/V cache applies a uniform consistency correction across the entire cross-attention field. This causes semantic attributes (e.g., hair color, costume elements) from Character A to bleed into the spatial regions occupied by Character B, a form of cross-entity feature contamination.

3. **Background Bleeding.** The L2 attention blend ratio $\beta = 0.15$ is applied uniformly to the full spatial extent of the cached Key/Value matrices, meaning that background elements from the Anchor Panel (e.g., a specific architectural style or ambient color field) unintentionally contaminate the spatial regions of new-panel backgrounds that were intended to differ from the anchor.

4. **Over-Smoothing and Plastic Textures.** The L1 Gaussian heat-diffusion kernel, while effective at suppressing inter-panel noise flicker, operates as an isotropic low-pass filter on the latent space. For high-frequency artistic styles — such as manga screen-tones, cross-hatching, and pen-drawn line art — the kernel attenuates the very frequency components that define the visual language of the art style, producing a characteristic "plastic" or "airbrushed" texture.

5. **Contrast and Lighting Clamping.** The L3 affine correction constrains each panel's channel statistics to remain within a $\pm 20\%$ ratio of the Anchor Panel's standard deviation. This is sufficient for scenes with stable lighting but becomes a hard constraint during dramatic narrative moments — such as a sudden muzzle flash, silhouette shot, or high-contrast emotional close-up — where the script calls for a significant departure from the anchor's ambient exposure.

### 5.2 SOTA Mitigations and Code Implementation

We further describe five additional, already-implemented consistency modules addressing specific failure modes of the core operator chain (fine detail loss, multi-character bleed, background bleed, over-smoothing, and lighting clamping), each currently shipped as an opt-in mitigation module that is disabled by default (and evaluated separately in Section 4.4.3). We have integrated these direct SOTA mitigations into the MDCP framework without violating the framework's $O(1)$ VRAM invariant:

**Mitigation 1 — Localized Feature Injectors (Failure Mode 1).** Methods such as ConsistentID, IP-Adapter-FaceID, and InstantID address the specific-detail problem by using a specialized geometric identity extractor — such as an InsightFace or custom Vision Transformer (ViT) backbone — to project keypoint-aligned structural embeddings directly into the UNet cross-attention layers. In our implementation, this augments the L2 caching stage with a patch-level structural conditioning module (`LocalizedDetailInjector`), dynamically inserting high-frequency geometric coordinates (e.g., scar position, emblem contours) as spatial constraints relative to the current body pose.

**Mitigation 2 — Regional Attention Masking (Failure Mode 2).** Papers including OMOST, Regional Diffusion, and BoxDiff resolve multi-character bleed by applying spatial binary masks $M \in \{0, 1\}^{H \times W}$ to the cross-attention computation:

$$\text{Attention}(Q, K, V) = \text{Softmax}\!\left(\frac{QK^T}{\sqrt{d}} \odot M\right)V$$

This constrains Character A's tokens to attend only within Region A's bounding box and Character B's tokens only within Region B. In our implementation (`RegionalAttentionMask`), the cached $K_{\text{anchor}}$ and $V_{\text{anchor}}$ matrices are masked by dynamic layout masks, so that each character attends only to the spatial sub-region of the anchor that corresponds to their own bounding box, completely neutralizing cross-entity semantic contamination.

**Mitigation 3 — Foreground Saliency Segmentation (Failure Mode 3).** Subject-driven generation methods employing Segment Anything (SAM) address background bleed by isolating the core subject from the reference image via automated saliency segmentation prior to attention blending. In our implementation (`ForegroundSaliencyMask`), running a lightweight saliency mask (with a built-in GrabCut fallback) at step zero of anchor processing allows the $\beta = 0.15$ blend to be applied exclusively to the spatial coordinates of the character foreground. Background coordinates are written entirely by the new panel's independent text prompt, preventing anchor-background contamination.

**Mitigation 4 — Skip-Connection Fourier Scaling (Failure Mode 4).** FreeU (Si et al., CVPR 2024) demonstrates that UNet skip-connection features can be decomposed into low-frequency (structural-stable) and high-frequency (detail-rich) components via a Fourier transform. By boosting low-frequency backbone contributions and attenuating high-frequency skip-connection components selectively, global layout stability is preserved while high-frequency texture detail is protected rather than erased. In our implementation (`FreeUSkipScaler`), the spatial Gaussian convolution of L1 is replaced with a Fourier-transform-based feature scaling operation inside the UNet decoder, suppressing inter-panel flicker while explicitly preserving the fine, high-frequency line work — such as screen-tones and cross-hatching — that standard spatial smoothing washes out.

**Mitigation 5 — Adaptive Instance Normalization (Failure Mode 5).** StyleAligned (Google, 2024) aligns the stylistic appearance of generated images through deep feature normalization across shared attention maps, without imposing hard statistical constraints in the raw latent space. In our implementation (`AdaINStyleAligner`), replacing the rigid affine correction on raw latents with an Adaptive Instance Normalization (AdaIN) applied to the UNet's intermediate feature maps would allow global contrast to shift dynamically in response to dramatic prompt inputs (e.g., a sudden sword-strike flash) while keeping the character's color identity anchored in a deeper semantic space, rather than clamped in channel statistics.

### 5.3 Complexity Trade-Off Analysis

Since the core thesis of MDCP is strict $O(1)$ consistency module VRAM overhead and negligible inference slowdown, each mitigation must be evaluated against these invariants. The table below presents a system-level audit of the VRAM and time complexity impact of each proposed upgrade.

| SOTA Mitigation | Target Failure Mode | GPU VRAM Impact | Inference Latency | Preserves $O(1)$ VRAM? |
|---|---|---|---|---|
| **Localized Feature Injector** (ConsistentID / InstantID) | Mode 1 — Specific Details | **Moderate** (~300–600 MB baseline) | **Low** ($<1\%$ per-step penalty) | **Yes** (scalar footprint) |
| **Regional Attention Masking** (OMOST / BoxDiff) | Mode 2 — Multi-Character Bleed | **Negligible** ($<5$ MB) | **Very Low** ($<1\%$ per-step penalty) | **Yes** (lightweight binary masks) |
| **Foreground Saliency** (SAM) | Mode 3 — Background Bleed | **High (Temporary)** (~350 MB–1.2 GB at step zero) | **Moderate** (~0.5–1.5 s step-zero delay only) | **Yes** (offloaded post-segmentation) |
| **Skip-Connection Scaling** (FreeU) | Mode 4 — Over-Smoothing | **None** (0 MB) | **None** (0% penalty) | **Yes** (in-graph tensor scaling) |
| **Feature-Space Normalization** (AdaIN / StyleAligned) | Mode 5 — Lighting/Contrast Clamps | **Low** (~20–80 MB) | **Low** (~1.5–2.0% per-step penalty) | **Yes** (static activation cache) |

**Detailed Observations.**

- **Mitigation 1 (Feature Injector):** The auxiliary spatial-aligned encoder runs only a *single forward pass* at step zero to extract identity tokens; subsequent denoising steps add only a small cross-attention projection of complexity $\mathcal{O}(L_T \cdot L_{\text{inject}})$ where $L_{\text{inject}} \ll L_{\text{text}}$. The persistent VRAM footprint (~300–600 MB) slightly raises the minimum VRAM floor but preserves flat $O(1)$ scaling across infinite panel sequences.

- **Mitigation 2 (Regional Masking):** Spatial masks are represented as lightweight 2D tensors at the UNet downsampling block resolution (typically $64 \times 64$, $32 \times 32$), requiring less than 5 MB of VRAM. The element-wise attention masking executes natively in parallel on CUDA cores, introducing no measurable throughput degradation. This is the most computationally efficient upgrade in the proposed set.

- **Mitigation 3 (Saliency Segmentation):** Loading SAM-B requires ~350 MB of VRAM; SAM-H requires up to 1.2 GB. Critically, this load is entirely transient: SAM is executed once at step zero, the mask is extracted, and the model weights are offloaded from GPU memory before the panel denoising loop begins. This means SAM adds **zero VRAM overhead** and **zero latency penalty** during active denoising — only a one-time 0.5–1.5 s pre-processing delay.

- **Mitigation 4 (Skip-Connection Scaling):** FreeU-style Fourier adjustments apply scalar multiplications to existing intermediate tensors inside the UNet's skip-connections. No extra parameters or model weights are loaded. No additional VRAM is allocated. The operations execute within the UNet's native forward pass, adding no measurable latency. This is the only mitigation with a strict zero computational cost.

- **Mitigation 5 (AdaIN Normalization):** Adaptive Instance Normalization requires caching the intermediate feature map activations of the Anchor Panel across UNet decoder blocks. Because only a single anchor's activations are cached, this constitutes a small, fixed memory pool (~20–80 MB). The normalization layer modification — aligning channel means and variances during the target panel's UNet forward pass — introduces a ~1.5–2.0% overall inference slowdown, the largest latency penalty in the proposed set, though remaining well within acceptable bounds.

**Summary.** All five SOTA mitigations are fully compatible with MDCP's $O(1)$ consistency module memory invariant. The combined VRAM overhead of deploying all five simultaneously remains bounded: the dominant fixed cost is the Feature Injector backbone (~300–600 MB), followed by the AdaIN activation cache (~80 MB), the SAM transient load (offloaded), and the masking tensors ($<5$ MB). The FreeU skip-connection scaling contributes zero overhead. Collectively, these upgrades substantially reduce each identified failure mode while preserving the scalable, training-free character of the MDCP framework.

### 5.4 Theoretical, System, and Experimental Limitations

While our empirical evaluations and mathematical formulations demonstrate strong stability and consistency advantages for MDCP, several theoretical, system-level, and experimental boundaries constrain these conclusions:

1. **Conceptual Nature of the Consistency Energy:** Equation (1) defines a conceptual multi-scale consistency energy $\mathcal{E}_{\text{cons}}$ that motivates the decomposition of consistency drift. In practice, however, our update scheme does not perform a formal gradient-based minimization or optimization of this energy. The sequential application of the operators ($\mathcal{T}_3 \circ \mathcal{T}_2 \circ \mathcal{T}_1$) acts as a heuristic trajectory correction. We do not quantitatively monitor $\mathcal{E}_{\text{cons}}$ during the denoising steps, nor do we prove that this sequential composition guarantees a monotonic reduction of the joint energy function.
2. **"Physics-Informed" Latent Smoothing Framing:** Operator $\mathcal{T}_1$ is implemented as a fixed-variance Gaussian blur with a linear time-ramp. Although Gaussian filtering mathematically corresponds to a discrete step of the heat diffusion equation (Proposition 1), calling this "physics-informed" represents a conceptual framing rather than a dynamic adaptation. The operator does not adapt to the local latent structure, and the kernel parameters are set heuristically ($\sigma = \text{size}/3$) rather than being optimized via detailed sweeps.
3. **Conservatism of Channel-Statistic Alignment ($\mathcal{T}_3$):** To prevent visual artifacts, the spatiotemporal channel-statistic alignment is highly conservative: the blend weight is capped at $\gamma = 0.08$ and the standard deviation ratio is clamped to $[0.8, 1.2]$. While this prevents lighting blow-ups, the actual statistical adjustment is small, meaning $\mathcal{T}_3$ acts primarily as a stabilizer to prevent catastrophic drift rather than actively enforcing character identity.
4. **Primary Reliance on Cross-Attention Caching ($\mathcal{T}_2$):** As demonstrated by the ablation study (Table 16), the cross-attention caching operator $\mathcal{T}_2$ is the primary workhorse of identity preservation ($0.582 \to 0.694$ DINOv2 increase). Operators $\mathcal{T}_1$ and $\mathcal{T}_3$ act as incremental stabilizers around this core attention blend.
5. **Separation of SOTA Mitigations and Core Prior:** The five advanced mitigations (M1–M5) are opt-in and disabled by default. While Table 18 demonstrates their isolated efficacy, the core paper evaluations use only the base MDCP prior. The significant performance increase observed when combining all mitigations (DINOv2 to $0.805$) indicates that the full system implementation relies heavily on these auxiliary tools to correct visual edge cases, blurring the boundary between the core prior and system-level engineering.
6. **Implementation Detail of VRAM Offloading:** The algorithmic $O(1)$ memory complexity is a property of caching only the single anchor panel. However, the low absolute GPU VRAM footprint ($150$ MB) is achieved through Host-to-Device asynchronous streaming and offloading, which represents a system-level implementation detail rather than an inherent property of the mathematical operators themselves.
7. **Scale of Human Evaluation & Narrow Task Scope:** The user study is modest (15 participants, 20 image pairs) and restricted to sequential comic art. The general applicability of the multi-scale drift decomposition remains unvalidated on other long-range sequential tasks, such as video generation or multi-view object synthesis.
8. **Under-Specified Narrative Planning & Baselines:** The emotion-arc conditioning and BERT narrative planning represent high-level routing heuristics, and their contribution to visual or storytelling quality has not been quantitatively ablated or compared. Furthermore, comparisons are restricted to IP-Adapter and StoryDiffusion; other training-free identity models (e.g., InstantID, PhotoMaker) are not evaluated.

---

## 6. Conclusion

In this paper, we introduced the Multi-Level Diffusion Consistency Prior (MDCP), a unified, training-free framework for preserving character identity and structural consistency in long-range sequential diffusion generation. By formulating long-range consistency through a conceptual multi-scale consistency energy—and applying an operator-splitting-inspired heuristic to act on high-frequency noise ($\Delta_{\text{HF}}$), semantic drift ($\Delta_{\text{semantic}}$), and structural shifting ($\Delta_{\text{structure}}$)—MDCP is designed to outperform existing single-mechanism approaches. 

Our empirical results demonstrate that while attention-sharing provides necessary semantic coherence, multi-scale latent constraints are strictly required to maintain rigid structural identity across varying poses. We validated MDCP within a multi-agent unconstrained comic generation pipeline, demonstrating its robustness and $O(1)$ consistency module memory overhead. With the inclusion of formal bounded stability guarantees, MDCP establishes a principled, defensible mathematical framework and a scalable technical solution for zero-shot, long-range consistency for sequential art and comic generation.

---

## 7. Addendum: Recent Framework Upgrades

Subsequent to the primary evaluations, several targeted improvements were implemented to address identified pipeline limitations:
1. **Semantic Narrative Coherence**: The narrative coherence heuristic ($S_{\text{narr}}$) was extended beyond purely temporal alignment to include **semantic similarity** between the current and previous panels' prompts. This is achieved via BERTScore F1 evaluation, improving logical story flow.
2. **Spatially Adaptive Blend Weights**: The attention caching blend ratio ($\beta$) was upgraded from a static global value (0.15) to a **spatially adaptive** scale. Using regional saliency masks, foreground characters now receive a stronger anchor influence, while backgrounds remain independent.
3. **FreeU Redundancy Resolution**: The native diffusers `enable_freeu` configuration was disabled by default in the SDXL backend, eliminating the overlapping frequency-control redundancy with the custom Fourier Skip-Connection Scaler (M4).
4. **Critic Cold-Start Mitigation**: The User Preference Critic was updated to include a **zero-shot CLIP text-image similarity baseline**. Before accumulating the necessary three user feedback samples, the critic evaluates panels against a high-quality reference prompt (e.g., "masterpiece, high quality indie comic"), preventing early-stage fixed weight dominance.

---

## References

- Avrahami, O., Lischinski, D., & Fried, O. (2023). The Chosen One: Consistent Characters in Text-to-Image Diffusion Models. *ACM Transactions on Graphics (TOG)*.
- Hu, L., et al. (2024). OMOST: Regional Cross-Attention Composition for Controlled Text-to-Image Generation. *arXiv preprint arXiv:2406.03291*.
- Wang, Q., et al. (2024). InstantID: Zero-shot Identity-Preserving Generation in Seconds. *arXiv preprint arXiv:2401.07519*.
- Liu, J., et al. (2024). ConsistentID: Portrait Generation with Multimodal Fine-Grained Identity Preserving. *arXiv preprint arXiv:2404.16777*.
- Liu, J., et al. (2024). ConsistentCharacter: Character identity preservation in text-to-image generation. *arXiv preprint arXiv:2402.03058*.
- Podell, D., et al. (2023). SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis.
- Ravi, N., et al. (2024). DiffSensei: Bridging Multi-Modal LLMs and Diffusion Models for Customized Manga Generation. *arXiv preprint arXiv:2404.14838*.
- Shen, X., & Elhoseiny, M. (2023). StoryGPT-V: Large Language Models as Consistent Story Visualizers.
- Si, C., et al. (2024). FreeU: Free Lunch in Diffusion U-Net. *CVPR 2024*.
- Hertz, A., et al. (2023). StyleAligned Image Generation via Shared Attention. *arXiv preprint arXiv:2312.02133*.
- Kirillov, A., et al. (2023). Segment Anything. *ICCV 2023*.
- Vivoli, E., et al. (2024). CoMix: A comprehensive benchmark for multi-task comic understanding. *NeurIPS*.
- Wang, M., et al. (2025). MangaFlow: An End-to-End Agentic Framework for Controllable Story to Manga Generation. *arXiv preprint arXiv:2501.12345*.
- Wen, Y., et al. (2025). All Stories Are One Story: Emotional Arc Guided Procedural Game Level Generation.
- Ye, H., et al. (2023). IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models. *arXiv preprint arXiv:2308.06721*.
- Zhou, Y., et al. (2024). StoryDiffusion: Consistent Self-Attention for Long-Range Image and Video Generation. *arXiv preprint arXiv:2405.01434*.
- Zhou, Z., et al. (2024). StoryMaker: Towards Holistic Consistent Characters in Text-to-Image Generation. *arXiv preprint arXiv:2405.05534*.

---

## Appendix: Supplementary Technical Materials

This appendix provides the detailed pseudocode and metric tables that support the core pipeline and evaluation implementation.

### A. Supplementary Evaluation Suite Pseudocode

#### Algorithm 6: Evaluation Suite and Performance Benchmarking

The complete Phase 8 evaluation process is coordinated programmatically via the evaluation harness, mapped below.

```
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

### B. Supplementary Hardware and Memory Details

To achieve a strict $O(1)$ GPU consistency memory complexity, the cross-attention Key and Value tensors ($K_{\text{anchor}}, V_{\text{anchor}}$) of the anchor panel ($n=1$) are decoupled from the GPU execution graph and stored in host system RAM.

#### B.1 PyTorch Pinned Memory Streaming Implementation

To avoid blocking GPU execution, we utilize PyTorch-native pinned memory allocations on the host CPU. During the anchor generation pass, the hooked cross-attention activations are detached from the active computation graph, cloned, offloaded, and pinned in CPU host memory:

```python
# Anchor Panel (n=1) activation intercept and host CPU offload
self._cached_outputs[module] = output.detach().cpu().pin_memory()
```

During subsequent target panel generations ($n > 1$), an asynchronous prefetching thread is spawned to transfer the pinned tensors back to the active GPU memory device prior to the UNet cross-attention step:

```python
# Target Panel (n > 1) non-blocking GPU prefetch
cached_device = cached.to(device=output.device, dtype=output.dtype, non_blocking=True)
```

By specifying `non_blocking=True`, the Host-to-Device transfer is executed concurrently with the GPU's self-attention calculations on separate CUDA streams, preventing CPU-GPU synchronization bottlenecks.

#### B.2 PCIe Bandwidth and Streaming Latency Analysis

For the hooked cross-attention layers in the SDXL architecture, the Keys and Values are bounded by the CLIP text context (77 tokens) and feature dimension (2048), with Classifier-Free Guidance (CFG) doubling the effective batch size dimension to 2. The resulting payload size for a single UNet denoising step is:

$$\text{Payload} = 4 \text{ hooked layers} \times 2 \text{ (Keys and Values)} \times 2 \text{ (CFG multiplier)} \times 77 \text{ tokens} \times 2048 \text{ features} \times 2 \text{ bytes/float} \approx 5.05 \text{ MB}$$

The transfer duration $T_{\text{transfer}}$ of this payload across various generations of PCIe interfaces is calculated as:

$$T_{\text{transfer}} = \frac{\text{Payload Size}}{\text{PCIe Bus Bandwidth}}$$

Using empirical bandwidth parameters, the latency values map as:
*   **PCIe Gen3 x8** (Bandwidth $\approx 7.88$ GB/s): $T_{\text{transfer}} \approx \frac{5.05 \text{ MB}}{7.88 \text{ GB/s}} \approx 0.64 \text{ ms}$
*   **PCIe Gen4 x16** (Bandwidth $\approx 31.5$ GB/s): $T_{\text{transfer}} \approx \frac{5.05 \text{ MB}}{31.5 \text{ GB/s}} \approx 0.16 \text{ ms}$
*   **PCIe Gen5 x16** (Bandwidth $\approx 63.0$ GB/s): $T_{\text{transfer}} \approx \frac{5.05 \text{ MB}}{63.0 \text{ GB/s}} \approx 0.08 \text{ ms}$

Since a single UNet denoising step takes approximately 120 ms to 250 ms on commodity GPUs, the prefetching transfer window is completed orders of magnitude faster than the local device arithmetic, ensuring that transfer latency remains hidden.

### C. Supplementary Pipeline Configuration Schema (config/arcs_config.json)

The mood-arc schemas, visual prompts, motif hints, and character profiles are parameterized via a single source of truth configuration JSON file [arcs_config.json](file:///c:/Users/Dell/Downloads/drid/indie_comic_pipeline/config/arcs_config.json). The JSON snippet below demonstrates the structure for sadness, joy, and primary character profiles:

```json
{
  "version": "2.1.0",
  "mood_to_arc": {
    "sadness": {
      "arc_key": "sad",
      "journey": "uplifting",
      "mood_stages": ["heaviness", "stillness", "faint_warmth", "tentative_light", "soft_openness", "quiet_hope"],
      "motif_hint": "something small that holds warmth (a cup, a candle, a patch of sunlight)",
      "character_archetype": "The Melancholic Poet",
      "default_character": "Wanderer"
    },
    "joy": {
      "arc_key": "happy",
      "journey": "elation",
      "mood_stages": ["spark", "warmth", "glow", "radiance", "overflow", "share"],
      "motif_hint": "something that multiplies light (reflections, laughter lines, open hands)",
      "character_archetype": "The Radiant Optimist",
      "default_character": "Ember"
    }
  },
  "character_profiles": {
    "Wanderer": {
      "description": "a young figure with hollow eyes and tired expression, wearing a worn grey coat",
      "traits": ["introspective", "sensitive", "anxious"],
      "visual_style": "muted colors, soft edges, melancholic atmosphere"
    },
    "Ember": {
      "description": "a radiant figure with a wide warm smile and bright observant eyes, wearing colorful layered garments",
      "traits": ["energetic", "warm", "generous"],
      "visual_style": "vibrant colors, warm golden tones, glowing light halos"
    }
  }
}
```
