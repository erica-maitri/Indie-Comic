# 4. Attention Propagation Module (T2)
## 4.1 Problem Formulation

While T1 regularizes the latent diffusion trajectory, semantic identity is primarily encoded inside the intermediate attention representations of the diffusion UNet. During standard SDXL inference, each attention layer independently computes


O_i=\operatorname{Attention}(Q_i,K_i,V_i), \tag{14}


where (Q_i), (K_i), and (V_i) denote the projected query, key and value tensors of image (i).

Since every image is processed independently,

O_i \perp O_j,\qquad i\neq j,^\n

there exists no explicit mechanism that allows semantic identity learned in one image to influence another image during inference.

Consequently, although the latent trajectory may be corrected (Section 3), semantic attention representations gradually diverge across independently generated panels.

---

# 4.2 Observations from Previous Work

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

# 4.3 Design Principle

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

# 4.4 Attention Propagation Operator

Accordingly, we define the propagated attention representation as


\boxed{O_{\text{prop}} = (1-\beta) O_{\text{curr}} + \beta O_{\text{anchor}}} \tag{20}


where

0\le\beta\le1.^\n

Unlike StoryDiffusion, which exchanges key-value tensors during attention computation, our formulation operates directly on the attention outputs.

Unlike FAM Diffusion, which interpolates attention across image resolutions, our operator propagates semantic identity across independently generated images while preserving the original network architecture.

---

# 4.5 Theoretical Analysis

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

# 4.6 Integration into Diffusion

The propagated attention replaces the original attention output before the subsequent UNet block,


O_{\text{curr}} \longrightarrow O_{\text{prop}} \longrightarrow \text{UNet}_{l+1}, \tag{24}


thereby propagating semantic identity throughout the denoising process without modifying network parameters or requiring retraining.