# 3. T1 Derivation
## 3.1 Problem Formulation

Our framework is built upon Latent Diffusion Models (LDMs), specifically Stable Diffusion XL (SDXL). Let (z_0) denote the latent representation of a clean image and (z_t) denote its noisy latent at diffusion timestep (t\in[0,T]). During inference, the reverse diffusion process progressively removes noise through the learned denoising network (\epsilon_\theta). The scheduler updates the latent according to


z_{t-1} = S\left(z_t, \epsilon_\theta(z_t,t,c)\right), \tag{1}


where (c) denotes the text conditioning and (S(\cdot)) represents the scheduler transition operator.

Most diffusion schedulers additionally estimate the clean latent from the current noisy sample. We denote this prediction as


\hat{z}_0 = D_{\text{sched}}\left(z_t, \epsilon_\theta, t\right), \tag{2}


where (D_{\text{sched}}) is the scheduler's predicted clean latent.

The reference latent

z_{0,\mathrm{anchor}}^\n

is obtained by encoding a designated anchor image using the SDXL VAE encoder and remains fixed throughout inference.

Unlike existing methods that modify model architectures or require additional training, our objective is to optimize only the latent variable (z_t) during inference while keeping all model parameters frozen.

---

# 3.2 Motivation

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

# 3.3 Latent Consistency Energy

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

Unlike StoryDiffusion, this loss is our own formulation and explicitly penalizes attention divergence during optimization.

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

# 3.4 Latent Trajectory Optimization

Since every energy component is differentiable with respect to the latent variable (z_t), we optimize only the latent while keeping all diffusion model parameters fixed.

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
## z_t

\eta(t)R_t.
\tag{12}


The refined latent is then propagated through the standard SDXL scheduler,


z_{t-1} = S\left(\tilde z_t, \epsilon_\theta(\tilde z_t,t,c)\right). \tag{13}


This procedure directly regularizes the diffusion trajectory while leaving the pretrained diffusion network unchanged.

---

# 3.5 Computational Complexity

The proposed framework requires extracting intermediate attention maps and feature representations through forward hooks together with one additional backward pass to compute

\nabla_{z_t}E_t.^\n

All diffusion model parameters remain frozen throughout optimization.

Consequently, each denoising iteration consists of one standard forward pass and one additional backward pass, introducing an approximately constant-factor increase in inference time while preserving linear complexity with respect to the number of denoising steps.

---

# 3.6 Algorithm

```text
Algorithm 1: Latent Trajectory Optimization

Input:
    Initial latent zT
    Text conditioning c
    Anchor latent z0,anchor
    λ1, λ2, λ3

Initialize scheduler statistics

for t = T ... 1 do

    Forward UNet

    Extract attention maps At
    Extract feature maps Ft

    Predict clean latent
        ẑ0 = Dsched(zt)

    Compute
        Eid   = (1/N)||At − Aanchor||²F
        Estr  = ||Ft − Ψ(Ft,Fanchor)||²
        Etraj = ||ẑ0 − z0,anchor||²

    Et = λ1Eid + λ2Estr + λ3Etraj

    Rt = ∇ztEt

    η(t)= λσt/(σmax+ε)

    z̃t = zt − η(t)Rt

    zt−1 = S(z̃t, εθ(z̃t,t,c))

end

Return z0
```