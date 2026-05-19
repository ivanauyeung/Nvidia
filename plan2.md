# Project Plan: Information Geometry and Deep Manifold Theory of LLM Hallucination

**Duration:** 12 Weeks (10 weeks project + 2 weeks technical report & presentation)  
**Goal:** Develop a rigorous information-theoretic framework for understanding LLM hallucinations through the lens of representation manifold geometry. The core contribution is proving an **Information Bottleneck Theorem for Hallucination** — a formal lower bound on hallucination rate as a function of the representation bottleneck at each Transformer layer.  
**Output:** Technical report and presentation for the NVAITC Student Workshop by end of Week 12.

---

## Motivation & Key Insight

Large Language Models hallucinate — they generate fluent, confident, but factually incorrect outputs. Despite extensive empirical work on hallucination detection and mitigation (RLHF, DPO, retrieval augmentation), there is no mathematical theory explaining **why hallucinations are inevitable** given the architecture. Existing work is overwhelmingly empirical: geometric taxonomies, embedding cluster analysis, and behavioral mapping. The few formal results are narrow (e.g., fixed-point non-halting at temperature 0, semantic collapse into spectral basins).

**Our thesis:** Hallucination is not a bug that can be fully patched — it is an **information-theoretic necessity** arising from the finite-dimensional representation bottleneck in Transformers. At every layer $\ell$, the full input context $(x_1, \ldots, x_T)$ is compressed into a hidden state $H_\ell \in \mathbb{R}^{T \times d}$ with effective rank $r_\ell \ll T \cdot d$. By the data processing inequality, any information about the correct output $Y$ that is not preserved in $H_\ell$ is **irrecoverably lost**. Hallucination occurs when the model must generate output that depends on information that was compressed away.

**The key equation:** The hallucination rate is bounded below by:

$$P(\text{hallucination}) \geq 1 - \frac{I(H_\ell; Y)}{H(Y)}$$

where $I(H_\ell; Y)$ is the mutual information between the layer-$\ell$ representation and the correct output, and $H(Y)$ is the entropy of the output distribution. This bound is tight when the model is calibrated (outputs match its internal confidence). The bound reveals that hallucination is fundamentally a **capacity problem**: the representation manifold at layer $\ell$ cannot carry enough information to specify $Y$ when $Y$ is complex, rare, or requires long-range context.

**Why the manifold perspective matters:** The mutual information $I(H_\ell; Y)$ is controlled by the **geometry of the representation manifold** — its intrinsic dimension, curvature, and volume. A manifold with low intrinsic dimension (collapsed representations, as observed by Rigollet's token clustering) has low capacity. A manifold with high curvature concentrates probability mass on a small subset of representable outputs, leaving others in "coverage gaps" where hallucination is inevitable. Our theorems connect these geometric quantities to the information-theoretic hallucination bound.

### Distinction from Prior Work

| Paper | What They Show | What They Don't Show (Our Contribution) |
|---|---|---|
| Wyss (arXiv:2512.05162) "Semantic Collapse" | Activations collapse into finitely many spectral basins | Does not connect spectral basins to hallucination rate or prove a lower bound |
| "Neutral Dynamics" (OpenReview 2025) | Residual Transformers have neutral dynamics (no contraction/expansion) | Shows persistence of hallucination but not inevitability; no quantitative bound |
| Geometric Taxonomy (arXiv:2602.13224) | Three hallucination types with geometric signatures (AUROC 0.958) | Detection, not explanation; no formal theorems |
| Manifold of Failure (arXiv:2602.22291) | Maps behavioral attraction basins empirically | No formal characterization of basin volumes or hallucination rates |
| HELIX (arXiv:2602.17691) | Manifold steering reduces hallucination in quantized models | Engineering framework; no lower bound on achievable hallucination rate |
| Fixed-point non-halting (arXiv:2410.06287) | Cyclic sequences create non-halting at $\tau = 0$ | Adversarial setting only; does not explain natural hallucination |
| Hairy Ball argument (2026 blog) | Odd-dimensional sphere topology may force confidence zeros | Informal; not rigorous; no proof connecting topology to hallucination rate |
| Geometric analysis (arXiv:2602.14778) | Small LMs hallucinate more due to weaker geometric structure | Empirical; no formal bound |

**Our contribution fills a specific gap:** None of the above papers provide a **quantitative lower bound on hallucination rate** as a function of measurable architectural quantities (hidden dimension $d$, effective rank $r_\ell$, manifold volume). We do.

---

## Weeks 1–3: Literature Survey, Framework Setup & Geometric Foundations

**Objective:** Conduct a thorough survey of the existing manifold/geometry-based hallucination literature. Formalize the mathematical framework. Prove preliminary lemmas connecting representation geometry to information capacity.

### Survey Track — Weeks 1–3

Systematically read, summarize, and extract key technical ideas from each major paper in the field. Organize findings into a structured literature review that will form the Related Work section of the final paper.

- **Week 1: Foundational Frameworks**

  | Paper | Key Ideas to Extract |
  |---|---|
  | **Wyss (arXiv:2512.05162)** "Semantic Collapse in Continuous Systems" | Transfer operator formalism for LLM dynamics. Semantic Characterization Theorem: spectral basins as o-minimal structures. How does the number of basins relate to representation capacity? |
  | **"Neutral Dynamics" (OpenReview, fDfctZ8Fhg)** "Not All Who Wander Are Lost" | Exact operator norm analysis of residual Transformers. Neutral spectrum ($|\lambda| \approx 1$) explains hallucination persistence. Extract: the spectral characterization of the residual map Jacobian. |
  | **Rigollet (arXiv:2512.01868)** Mean-field theory of Transformers | Token clustering on $\mathbb{S}^{d-1}$. Asymptotic collapse reduces effective rank. Extract: the clustering rate and its dependence on depth — directly feeds into our capacity bound. |

- **Week 2: Geometric Hallucination Detection & Taxonomy**

  | Paper | Key Ideas to Extract |
  |---|---|
  | **Geometric Taxonomy (arXiv:2602.13224)** | Three hallucination types: unfaithfulness (Type I), confabulation (Type II), factual error (Type III). Geometric signatures: Semantic Grounding Index (SGI), Directional Grounding Index (Γ). Extract: the embedding-space metrics and their connection to information loss. |
  | **Manifold of Failure (arXiv:2602.22291)** | MAP-Elites illumination of failure regions. Behavioral attraction basins. Model-specific topological signatures. Extract: empirical basin volume estimates — will compare to our theoretical predictions. |
  | **Geometric Uncertainty (arXiv:2509.13813)** | Archetypal analysis and convex hull geometry for hallucination detection. Geometric Volume and Geometric Suspicion metrics. Extract: the convex hull perspective on representation coverage. |
  | **Embedding Cluster Geometry (arXiv:2602.14259)** | Hallucination detection via cluster structure: polarity coupling ($\alpha$), cluster cohesion ($\beta$), radial information gradient ($\lambda_s$). Extract: measurable geometric statistics that correlate with hallucination. |
  | **Information Geometry of Hallucination (arXiv:2511.15005)** | Using geometry of information theory (Fisher information metric, statistical manifolds) to analyze hallucination in LLMs. Extract: the Fisher metric formulation on the output distribution manifold, geodesic distances between truthful and hallucinated distributions, and how Fisher information captures sensitivity of model outputs to input perturbations. |
  | **Information Geometry Foundations (MDPI Mathematics 2023)** | Mathematical foundations of information geometry: Fisher-Rao metric, $\alpha$-connections, dually flat structures. Extract: the Riemannian geometry toolkit for probability distribution spaces — apply to the manifold of Transformer output distributions $\{p_\theta(\cdot | x) : x \in \mathcal{X}\}$. |

- **Week 3: Dynamical Systems & Topological Approaches**

  | Paper | Key Ideas to Extract |
  |---|---|
  | **HELIX (arXiv:2602.17691)** "Tethered Reasoning" | Manifold steering via Mahalanobis distance to "truthfulness manifold." Extract: the definition of the truthfulness manifold and how it relates to our information-theoretic framework. |
  | **Agentic Loop Dynamics (arXiv:2512.10350)** | Contractive vs. exploratory regimes in semantic embedding spaces. Semantic dispersion metrics. Extract: the dynamical systems formalization of LLM generation as a discrete map on the representation manifold. |
  | **Fixed-point non-halting (arXiv:2410.06287)** | Formal proof that cyclic sequences at $\tau = 0$ create fixed points. Extract: the fixed-point construction technique — adapt for our false convergence characterization. |
  | **Hairy Ball / Topological argument (2026)** | Poincaré-Hopf theorem applied to activation manifolds. Extract: the topological obstruction idea — formalize as a supporting proposition. |
  | **Emergent Manifold Separability (arXiv:2602.20338)** | Manifold structure changes during reasoning in LLMs. Extract: empirical evidence that the representation manifold's geometry is non-trivial and reasoning-dependent. |

  **Survey Deliverable (End of Week 3):** A 15–20 page structured literature review covering:
  1. Taxonomy of existing approaches (geometric, dynamical, topological, information-theoretic)
  2. Summary of formal results (with complete theorem statements)
  3. Identification of gaps — what is *not* proved
  4. How our Information Bottleneck Theorem fills the central gap

### Theory Track — Week 3: Framework & Preliminary Lemmas

- **Framework Setup:**
  - Define the Transformer as a sequence of maps $F_\ell: \mathbb{R}^{T \times d} \to \mathbb{R}^{T \times d}$, $\ell = 1, \ldots, L$.
  - Define the representation at layer $\ell$: $H_\ell = F_\ell \circ F_{\ell-1} \circ \cdots \circ F_1(X_{\text{embed}})$.
  - Define the **representation manifold** $\mathcal{M}_\ell = \{H_\ell(x) : x \in \mathcal{X}\} \subset \mathbb{R}^{T \times d}$ — the image of the input space under the first $\ell$ layers.
  - Define the **intrinsic dimension** $\dim(\mathcal{M}_\ell)$ as the manifold dimension (estimated via local PCA, MLE, or correlation dimension in practice).
  - Define the **effective rank** $r_\ell = \exp(H(\sigma_1, \ldots, \sigma_d))$ where $\sigma_i$ are the normalized singular values of $H_\ell$ and $H$ is the Shannon entropy (Roy & Vetterli 2007).

- **Lemma 0 (Representation Capacity Bound): Mutual Information is Bounded by Manifold Dimension**
  - **Statement:** Let $\mathcal{M}_\ell \subset \mathbb{R}^{Td}$ be the representation manifold at layer $\ell$ with intrinsic dimension $\dim(\mathcal{M}_\ell) = m_\ell$. For any output variable $Y$ with finite entropy $H(Y)$:
  $$I(H_\ell; Y) \leq \frac{m_\ell}{2} \log\left(1 + \frac{\text{Vol}(\mathcal{M}_\ell)^{2/m_\ell} \cdot \text{SNR}_\ell}{2\pi e}\right)$$
    where $\text{Vol}(\mathcal{M}_\ell)$ is the Riemannian volume of the manifold and $\text{SNR}_\ell$ is the signal-to-noise ratio of the representation.
  - **Proof Strategy:** Apply the entropy power inequality on the manifold. The representation $H_\ell$ lives on an $m_\ell$-dimensional manifold, so its differential entropy is bounded by that of an $m_\ell$-dimensional Gaussian with the same second moment (maximum entropy principle under the manifold constraint). The mutual information $I(H_\ell; Y) = H(H_\ell) - H(H_\ell | Y) \leq H(H_\ell) \leq \frac{m_\ell}{2} \log(\ldots)$ by the Gaussian bound.
  - **Why it matters:** Connects the geometry of the representation manifold (dimension, volume) to the information it can carry about the output. Lower intrinsic dimension → lower capacity → more hallucination. This is the foundation for the main theorem.

- **Lemma 1 (Data Processing Inequality Chain): Information Loss Across Layers**
  - **Statement:** For any output $Y$ and layers $1 \leq \ell_1 < \ell_2 \leq L$:
  $$I(X; Y) \geq I(H_{\ell_1}; Y) \geq I(H_{\ell_2}; Y) \geq I(\hat{Y}; Y)$$
    where $\hat{Y} = \text{decode}(H_L)$ is the model's output. Each inequality is an equality if and only if $H_{\ell+1}$ is a sufficient statistic of $H_\ell$ for $Y$.
  - **Proof:** Direct application of the data processing inequality to the Markov chain $X \to H_{\ell_1} \to H_{\ell_2} \to \hat{Y}$.
  - **Why it matters:** Establishes that information about $Y$ can only decrease through the network. Any layer that compresses too aggressively becomes an irrecoverable bottleneck. Combined with Lemma 0, this gives a per-layer hallucination bound.

### Weeks 1–3 Deliverables

| Deliverable | Week | Status |
|---|---|---|
| Literature survey: foundational frameworks | 1 | To complete |
| Literature survey: geometric detection & taxonomy | 2 | To complete |
| Literature survey: dynamical & topological approaches | 3 | To complete |
| Structured Related Work document (15–20 pages) | 3 | To complete |
| **Lemma 0: Representation Capacity Bound** | 3 | To prove |
| **Lemma 1: Data Processing Chain** | 3 | To prove |
| Mathematical framework document | 3 | To complete |

---

## Weeks 4–7: Information Bottleneck Theorem & Supporting Results

**Objective:** Prove the main Information Bottleneck Theorem for Hallucination. Derive its corollaries for specific architectural choices. Prove the supporting theorem connecting token clustering (Rigollet) to information loss.

### Theory Track

- **Week 4–5: Theorem 1 (Information Bottleneck Theorem for Hallucination — Main Contribution)**

  - **Theorem 1 (Information Bottleneck Theorem): Fundamental Lower Bound on Hallucination Rate**
    - **Statement:** Let $f_\theta$ be an $L$-layer Transformer with hidden dimension $d$. Let $Y$ be the correct next-token output for input $X$, with $H(Y | X) = 0$ (the output is deterministic given the input — e.g., factual question answering). Let $\hat{Y} = f_\theta(X)$ be the model's prediction. Define hallucination as the event $\{\hat{Y} \neq Y\}$. Then:

    $$P(\hat{Y} \neq Y) \geq 1 - \frac{I(H_\ell; Y)}{H(Y)} \geq 1 - \frac{m_\ell \log(1 + \text{Vol}(\mathcal{M}_\ell)^{2/m_\ell} \cdot \text{SNR}_\ell / (2\pi e))}{2 H(Y)}$$

      for any layer $\ell \in \{1, \ldots, L\}$. The bound holds for the **tightest bottleneck layer** $\ell^* = \arg\min_\ell I(H_\ell; Y)$:

    $$P(\hat{Y} \neq Y) \geq 1 - \frac{\min_\ell I(H_\ell; Y)}{H(Y)}$$

      Moreover, if the model is **calibrated** (i.e., $P(\hat{Y} = y | f_\theta(X) = y) = P(Y = y | X)$), then the bound is tight.

    - **Proof Strategy (Three Steps):**

      **Step 1 — Fano's inequality on the output.**
      By Fano's inequality (the information-theoretic bound on decoding error):
      $$P(\hat{Y} \neq Y) \geq 1 - \frac{I(\hat{Y}; Y) + 1}{\log |\mathcal{Y}|}$$
      where $|\mathcal{Y}|$ is the vocabulary size. For large vocabulary ($|\mathcal{Y}| \gg 1$), the $+1$ term is negligible and $\log |\mathcal{Y}| \geq H(Y)$, giving:
      $$P(\hat{Y} \neq Y) \geq 1 - \frac{I(\hat{Y}; Y)}{H(Y)}$$

      **Step 2 — Data processing inequality.**
      By Lemma 1, $I(\hat{Y}; Y) \leq I(H_\ell; Y)$ for any $\ell$. Substituting:
      $$P(\hat{Y} \neq Y) \geq 1 - \frac{I(H_\ell; Y)}{H(Y)}$$

      **Step 3 — Geometric bound on mutual information.**
      By Lemma 0, $I(H_\ell; Y) \leq \frac{m_\ell}{2} \log(1 + \text{Vol}(\mathcal{M}_\ell)^{2/m_\ell} \cdot \text{SNR}_\ell / (2\pi e))$. Substituting gives the final bound.

    - **Why it matters:** This is the paper's central result. It says:
      1. **Hallucination is inevitable** when the output entropy $H(Y)$ exceeds the bottleneck's information capacity $I(H_\ell; Y)$. For rare facts, long-tail knowledge, and multi-hop reasoning, $H(Y)$ is large (many possible outputs), but $I(H_\ell; Y)$ is limited by the manifold dimension.
      2. **The bottleneck is geometric:** Low intrinsic dimension ($m_\ell$ small), low manifold volume, or low SNR all tighten the bound, making hallucination more likely. This connects the abstract information-theoretic bound to measurable geometric properties of the representation.
      3. **Scaling insight:** Increasing $d$ (hidden dimension) increases $m_\ell$ and $\text{Vol}(\mathcal{M}_\ell)$, relaxing the bound — explaining why larger models hallucinate less (but never zero, since $H(Y)$ also grows with task complexity).
      4. **Layer-wise interpretation:** The tightest bottleneck $\ell^*$ is the "weakest link" — the layer where the most information about $Y$ is lost. This identifies which layers are responsible for hallucination, enabling targeted interventions.

  - **Corollary 1.1 (Rank-Deficient Hallucination Bound):** If the representation at layer $\ell$ has effective rank $r_\ell$ (with $r_\ell \leq d$), then:
  $$P(\hat{Y} \neq Y) \geq 1 - \frac{r_\ell \log(1 + d \cdot \text{SNR}_\ell / (r_\ell \cdot 2\pi e))}{2 H(Y)}$$
    This follows from Lemma 0 with $m_\ell = r_\ell$ (the intrinsic dimension equals the effective rank for linear manifolds) and $\text{Vol}(\mathcal{M}_\ell) \leq (d/r_\ell)^{r_\ell/2}$ (volume of the effective subspace).

    **Practical implication:** In deep Transformers, token representations often collapse to low effective rank in later layers (observed by Rigollet and others). Corollary 1.1 shows this rank collapse *directly increases the hallucination floor.*

  - **Corollary 1.2 (Depth-Dependent Bound):** If each layer reduces the effective rank by a multiplicative factor $\rho_\ell$ (i.e., $r_{\ell+1} = \rho_\ell \cdot r_\ell$ with $\rho_\ell < 1$), then after $L$ layers:
  $$r_L = r_0 \cdot \prod_{\ell=1}^{L} \rho_\ell, \qquad P(\hat{Y} \neq Y) \geq 1 - \frac{r_0 \cdot \prod_\ell \rho_\ell \cdot \log(\ldots)}{2 H(Y)}$$
    Deeper models with aggressive rank compression hallucinate more, *unless* the compression is lossless (sufficient statistic preservation).

- **Week 6: Theorem 2 (Token Clustering → Information Loss)**

  - **Theorem 2 (Clustering-Induced Information Loss): Rigollet's Token Collapse Increases Hallucination**
    - **Statement:** Consider the mean-field dynamics of self-attention on $\mathbb{S}^{d-1}$ (Rigollet 2025). Let the angular dispersion at layer $\ell$ be $D_\ell = 1 - \|\bar{h}_\ell\|^2$ where $\bar{h}_\ell = \frac{1}{T} \sum_t h_{\ell,t}$ is the token centroid. As $D_\ell \to 0$ (complete clustering / collapse), the effective rank satisfies $r_\ell \to 1$ and the mutual information satisfies:
    $$I(H_\ell; Y) \leq \frac{1}{2} \log\left(1 + d \cdot \text{SNR}_\ell / (2\pi e)\right) + T \cdot D_\ell \cdot \log d$$
      In particular, at complete collapse ($D_\ell = 0$):
    $$I(H_\ell; Y) \leq \frac{1}{2} \log(1 + d \cdot \text{SNR}_\ell / (2\pi e))$$
      which is the capacity of a *single token* — all $T$ tokens carry the same information.
    - **Proof Strategy:**
      1. When tokens cluster ($D_\ell \to 0$), the token matrix $H_\ell$ has rank approaching 1 (all rows are approximately equal to the centroid $\bar{h}_\ell$).
      2. The mutual information $I(H_\ell; Y) = I(\bar{h}_\ell, \delta H_\ell; Y) = I(\bar{h}_\ell; Y) + I(\delta H_\ell; Y | \bar{h}_\ell)$ where $\delta H_\ell = H_\ell - \mathbf{1}\bar{h}_\ell^\top$ is the deviation from the centroid.
      3. The first term is bounded by the capacity of a single $d$-dimensional vector (Lemma 0 with $m = d$).
      4. The second term is bounded by $T \cdot D_\ell \cdot \log d$ since the deviations have variance $D_\ell$ per dimension and there are $T$ tokens in $d$ dimensions.
      5. At $D_\ell = 0$, the second term vanishes, leaving only single-token capacity.
    - **Why it matters:** This is the formal bridge between Rigollet's clustering theory and hallucination. Rigollet proved that Transformer attention drives tokens to collapse. Our theorem proves that this collapse directly reduces information capacity, increasing the hallucination floor. This also provides a theoretical justification for anti-collapse mechanisms (e.g., gated diversity channels): preventing token clustering preserves information capacity and reduces hallucination. Specifically, any mechanism that prevents $D_\ell \to 0$ by maintaining $D_\ell > D^* > 0$ would preserve $I(H_\ell; Y)$, directly reducing the hallucination lower bound.

- **Week 7: Theorem 3 (Hallucination Type Classification via Manifold Geometry)**

  - **Theorem 3 (Geometric Hallucination Taxonomy): Three Information Loss Modes**
    - **Statement:** Decompose the output entropy as $H(Y) = H_{\text{freq}}(Y) + H_{\text{tail}}(Y) + H_{\text{comp}}(Y)$ where:
      - $H_{\text{freq}}(Y)$: entropy from frequent/common outputs (well-represented in training data)
      - $H_{\text{tail}}(Y)$: entropy from rare/long-tail outputs
      - $H_{\text{comp}}(Y)$: entropy from compositional/multi-hop outputs (requiring combination of multiple facts)

      The information bottleneck creates three distinct hallucination modes, each corresponding to a specific geometric failure:

      | Hallucination Type | Information Loss Mode | Geometric Signature |
      |---|---|---|
      | **Type I: Confabulation** | $I(H_\ell; Y_{\text{tail}}) \approx 0$: rare facts not encoded | Low manifold volume in regions corresponding to rare outputs (coverage gaps) |
      | **Type II: Unfaithfulness** | $I(H_\ell; X_{\text{context}}) < I(H_\ell; X_{\text{prior}})$: context overridden by prior | High Fisher-Rao curvature attractors pulling representations toward high-prior regions (the "prior basin"); geodesic distance from context-dependent to prior-dependent distributions is small |
      | **Type III: Compositional failure** | $I(H_\ell; Y_{\text{comp}}) < \sum_i I(H_\ell; Y_i)$: individual facts encoded but composition lost | Low-rank interaction structure: the manifold captures marginals but not the joint distribution |

    - **Proof Strategy:**
      1. **Type I:** For rare $y$ (appearing $<k$ times in training), the representation manifold $\mathcal{M}_\ell$ has near-zero probability mass in the region $\{h : \text{decode}(h) = y\}$ (by learning theory — rare events cannot be reliably encoded). Formally: by the minimax lower bound for density estimation, the model's implicit density on $\mathcal{M}_\ell$ has error $\Omega(k^{-2/(m_\ell+2)})$ for $k$-sample events, making the coverage gap inevitable for $k \ll n^{m_\ell/(m_\ell+2)}$.
      2. **Type II:** The pre-trained model encodes a strong prior $P_{\text{prior}}(Y)$ in its weights. Context $X_{\text{context}}$ must override this prior via the attention mechanism. When the "prior basin" (the region of $\mathcal{M}_\ell$ encoding $P_{\text{prior}}$) is a high-curvature attractor, the representation resists displacement toward the context-dependent region. Formalize via the **Fisher-Rao metric** on the output distribution manifold $\{p_\theta(\cdot | x)\}$: the Fisher information matrix $\mathcal{I}(\theta) = \mathbb{E}[\nabla \log p_\theta \nabla \log p_\theta^\top]$ defines the Riemannian metric, and the geodesic distance between $p_{\text{prior}}$ and $p_{\text{context}}$ measures how hard it is to move from prior to context-faithful output. High Fisher curvature in the prior basin means the model's output distribution is highly sensitive to parameter changes near the prior but insensitive near the context — requiring disproportionate gradient signal to override the prior.
      3. **Type III:** For compositional outputs $Y = g(Y_1, Y_2)$ requiring two facts, the mutual information satisfies $I(H_\ell; Y) \leq I(H_\ell; Y_1) + I(H_\ell; Y_2) - I(Y_1; Y_2 | H_\ell)$. If the representation encodes $Y_1$ and $Y_2$ independently (low-rank interaction), $I(Y_1; Y_2 | H_\ell) \approx 0$, and $I(H_\ell; Y)$ is bounded by the sum of marginals minus the composition entropy. This is exactly the information lost by the bottleneck's failure to encode the joint.
    - **Why it matters:** Maps the three empirically observed hallucination types (Geometric Taxonomy, arXiv:2602.13224) to three distinct information-theoretic failure modes. Each type has a different geometric signature, explaining *why* different detection methods work for different types. Provides a principled framework for choosing mitigation strategies: Type I needs more data/capacity, Type II needs stronger attention/context grounding, Type III needs higher-rank representations.

### Weeks 4–7 Deliverables

| Deliverable | Week | Status |
|---|---|---|
| **Theorem 1: Information Bottleneck Theorem** | 4–5 | To prove |
| **Corollary 1.1: Rank-Deficient Bound** | 5 | To prove |
| **Corollary 1.2: Depth-Dependent Bound** | 5 | To prove |
| **Theorem 2: Clustering → Information Loss** | 6 | To prove |
| **Theorem 3: Geometric Hallucination Taxonomy** | 7 | To prove |

---

## Weeks 8–10: Empirical Verification & Mitigation Lower Bounds

**Objective:** Verify all theorems empirically on real LLMs. Prove the mitigation lower bound (Theorem 4). Complete experiments.

### Theory Track

- **Week 8–9: Theorem 4 (Mitigation Lower Bound)**

  - **Theorem 4 (Mitigation Floor): Post-Hoc Methods Cannot Eliminate Hallucination Below the Bottleneck Bound**
    - **Statement:** Let $f_\theta$ be a pre-trained Transformer with bottleneck capacity $C^* = \min_\ell I(H_\ell; Y)$. Let $g$ be *any* post-hoc mitigation method that does not modify the internal representations $H_\ell$ (e.g., output filtering, calibration, retrieval-augmented generation that only modifies the final decoding step). Then:
    $$P(g(f_\theta(X)) \neq Y) \geq 1 - \frac{C^*}{H(Y)}$$
      The bound is the same as Theorem 1 — post-hoc methods cannot reduce hallucination below the bottleneck floor.

      However, if the mitigation method modifies the representation (e.g., retrieval-augmented generation that injects retrieved context at intermediate layers, or fine-tuning that changes $F_\ell$), the bound can be improved to:
    $$P(g(f_\theta(X)) \neq Y) \geq 1 - \frac{\min_\ell I(H_\ell^{\text{modified}}; Y)}{H(Y)}$$
      where $H_\ell^{\text{modified}}$ is the new representation. This shows that effective mitigation must *widen the bottleneck* by increasing $m_\ell$, $\text{Vol}(\mathcal{M}_\ell)$, or $\text{SNR}_\ell$.

    - **Proof Strategy:**
      1. Any post-hoc method $g$ acts on the output $\hat{Y} = f_\theta(X)$, forming a Markov chain $Y \to X \to H_\ell \to \hat{Y} \to g(\hat{Y})$.
      2. By the data processing inequality, $I(g(\hat{Y}); Y) \leq I(\hat{Y}; Y) \leq I(H_\ell; Y) = C^*$.
      3. Applying Fano's inequality to $g(\hat{Y})$ gives the same bound as Theorem 1.
      4. For representation-modifying methods: the Markov chain is broken because $H_\ell^{\text{modified}}$ has access to additional information (e.g., retrieved documents). The new bottleneck capacity $C^*_{\text{new}} = \min_\ell I(H_\ell^{\text{modified}}; Y)$ can be strictly larger.
    - **Why it matters:** Has strong practical implications:
      - **RLHF / DPO cannot eliminate hallucination** if they only adjust the output distribution without changing intermediate representations. They can improve calibration (make the model "know what it doesn't know") but cannot increase $C^*$.
      - **RAG (retrieval-augmented generation) helps** because it injects new information at intermediate layers, increasing $C^*$. But the improvement is bounded by the quality and relevance of the retrieved documents.
      - **Scaling (larger $d$) helps** because it increases $m_\ell$ and $\text{Vol}(\mathcal{M}_\ell)$. But the improvement is logarithmic in $d$ (from the Gaussian capacity formula), explaining the diminishing returns of scaling.

  - **Proposition 4.1 (Scaling Law for Hallucination):** If the effective rank scales as $r_\ell = c \cdot d^{\alpha}$ for some $\alpha \in (0, 1]$ (empirically observed: $\alpha \approx 0.5$–$0.8$ in practice), then the hallucination floor decreases as:
  $$P(\hat{Y} \neq Y) \geq 1 - \frac{c \cdot d^{\alpha} \cdot \log(d \cdot \text{SNR})}{2 H(Y)} = 1 - O(d^{\alpha} \log d / H(Y))$$
    This predicts a power-law improvement with hidden dimension: doubling $d$ reduces the hallucination floor by a factor of $2^{\alpha}$, consistent with empirical scaling observations.

- **Week 10: Proposition 5 (Topological Obstruction — Stretch Goal)**

  - **Proposition 5 (Topological Coverage Gap): LayerNorm Geometry Creates Mandatory Blind Spots**
    - **Statement:** If LayerNorm projects token representations onto $\mathbb{S}^{d-1}$ (the $(d-1)$-dimensional unit sphere), and the model must represent a continuous output function $f: \mathcal{X} \to \mathcal{Y}$ where $\mathcal{Y}$ has non-trivial topology (e.g., $\pi_1(\mathcal{Y}) \neq 0$), then there exist inputs $x \in \mathcal{X}$ for which any continuous representation $H_\ell(x) \in \mathbb{S}^{d-1}$ must pass through a region of low confidence (the "coverage gap"). Specifically, by the Borsuk-Ulam theorem, there exists $x$ such that $H_\ell(x) = H_\ell(-x)$ (antipodal inputs are mapped to the same representation), implying the model cannot distinguish them.
    - **Status:** Stretch goal. Prove if time permits; otherwise present as a conjecture with supporting numerical evidence.
    - **Why it matters:** Connects the topology of the representation manifold to the existence of "blind spots" where hallucination is topologically inevitable. This would complement the information-theoretic bound (Theorem 1) with a topological argument.

### Empirical Verification Track — Weeks 8–10

| ID | Experiment | Tests | Setup |
|---|---|---|---|
| E1.1 | **Intrinsic dimension measurement.** For pre-trained LLMs (Llama-3-8B, Mistral-7B, Gemma-2B), feed 1000 prompts from TruthfulQA. At each layer $\ell$, estimate $m_\ell$ using the MLE intrinsic dimension estimator (Levina & Bickel 2004). | L0 | Plot $m_\ell$ vs. $\ell$. Identify the tightest bottleneck layer $\ell^*$. |
| E1.2 | **Effective rank vs. hallucination rate.** Bin prompts by measured hallucination rate (using TruthfulQA ground truth). For each bin, compute average effective rank $r_\ell$ at the bottleneck layer. | T1 | Hallucination rate should correlate inversely with $r_\ell$. Plot and fit to the Theorem 1 bound. |
| E1.3 | **Clustering dispersion vs. hallucination.** Measure angular dispersion $D_\ell$ at each layer. Correlate with per-prompt hallucination rate. | T2 | Lower $D_\ell$ (more clustering) → higher hallucination rate. |
| E1.4 | **Bottleneck bound tightness.** For each prompt, compute the Theorem 1 lower bound using measured $m_\ell$, $\text{Vol}(\mathcal{M}_\ell)$, $\text{SNR}_\ell$. Compare predicted vs. actual hallucination rate. | T1 | Bound should be a valid lower bound (predicted ≤ actual). Measure gap to assess tightness. |
| E1.5 | **Hallucination type classification.** Label TruthfulQA hallucinations as Type I/II/III. For each type, measure the corresponding geometric signature from Theorem 3. | T3 | Type I: low manifold volume. Type II: high curvature toward prior. Type III: low interaction rank. |
| E1.6 | **RAG mitigation test.** Apply retrieval-augmented generation (inject relevant documents at layer $\ell^*/2$). Measure the new $I(H_\ell^{\text{modified}}; Y)$ and hallucination rate. | T4 | RAG should increase $C^*$ and reduce hallucination rate. Verify the improvement matches the new Theorem 4 bound. |
| E1.7 | **Scaling experiment.** Compare models of different sizes (Gemma-2B, Llama-3-8B, Llama-3-70B). Measure $r_\ell$, $m_\ell$, hallucination rate. | P4.1 | Hallucination floor should decrease as $O(d^{\alpha} \log d)$. Fit $\alpha$ from data. |
| E1.8 | **Anti-collapse mechanism validation.** Apply an anti-collapse mechanism (e.g., gated diversity channel or auxiliary orthogonality loss) to a baseline Transformer. Measure $D_\ell$ and hallucination rate for baseline vs. augmented model. | T2 | Augmented model should have higher $D_\ell$ and lower hallucination rate, validating the clustering-hallucination connection. |

---

## Weeks 11–12: Technical Report & Presentation

**Objective:** Write the technical report, prepare the final presentation for the NVAITC Student Workshop, and complete code cleanup and documentation.

- **Technical Report:** Compile all results into a unified technical report:
  - **Title:** "The Information Bottleneck Theory of LLM Hallucination: Why Language Models Must Hallucinate"
  - **Structure:**
    1. Introduction: the hallucination problem and why existing explanations are incomplete
    2. Related work (from Weeks 1–3 survey)
    3. Framework: representation manifold, information capacity
    4. Theorem 1: Information Bottleneck Theorem (main result)
    5. Theorem 2: Token clustering increases hallucination (connection to Rigollet)
    6. Theorem 3: Three hallucination types as three information loss modes
    7. Theorem 4: Mitigation lower bound (what can and cannot help)
    8. Experiments: verification on real LLMs
    9. Discussion: implications for scaling, architecture design, and mitigation

- **Final Presentation:** Prepare the presentation for the NVAITC Student Workshop.

- **Code Cleanup & Documentation:** Finalize codebase with clear documentation.

---

## Summary: Theorem Dependency Map

| Result | Phase | Depends On | Enables |
|---|---|---|---|
| **L0: Representation Capacity Bound** | Weeks 1–3 | Entropy power inequality, manifold dimension | T1, T2 |
| **L1: Data Processing Chain** | Weeks 1–3 | Data processing inequality | T1, T4 |
| **T1: Information Bottleneck Theorem** | Weeks 4–7 | L0, L1, Fano's inequality | Core claim (hallucination lower bound) |
| **C1.1: Rank-Deficient Bound** | Weeks 4–7 | T1, effective rank definition | Practical bound for observed rank collapse |
| **C1.2: Depth-Dependent Bound** | Weeks 4–7 | T1, per-layer compression rates | Scaling insight |
| **T2: Clustering → Information Loss** | Weeks 4–7 | T1, Rigollet's clustering theory | Connects clustering to hallucination floor |
| **T3: Geometric Hallucination Taxonomy** | Weeks 4–7 | T1, manifold curvature, interaction rank | Type-specific mitigation guidance |
| **T4: Mitigation Floor** | Weeks 8–10 | T1, L1, data processing inequality | Practical implications (what works, what doesn't) |
| **P4.1: Scaling Law** | Weeks 8–10 | T4, empirical rank scaling | Prediction for model scaling |
| **P5: Topological Obstruction** | Weeks 8–10 | Borsuk-Ulam, sphere topology | Stretch goal; topological complement to info-theoretic bound |

---

## Timeline Summary

| Week | Phase | Theory | Empirical | Deliverable |
|---|---|---|---|---|
| 1 | Weeks 1–3 | — | — | Survey: foundational frameworks |
| 2 | Weeks 1–3 | — | — | Survey: geometric detection & taxonomy |
| 3 | Weeks 1–3 | **L0, L1 proved** | — | Survey: dynamical & topological; **Related Work document**; mathematical framework |
| 4 | Weeks 4–7 | Begin T1 | — | Main theorem draft |
| 5 | Weeks 4–7 | **T1, C1.1, C1.2 proved** | — | **Information Bottleneck Theorem** |
| 6 | Weeks 4–7 | **T2 proved** | — | Clustering-hallucination connection |
| 7 | Weeks 4–7 | **T3 proved** | — | Geometric taxonomy formalization |
| 8 | Weeks 8–10 | Begin T4 | E1.1, E1.2 | Intrinsic dimension & rank measurements |
| 9 | Weeks 8–10 | **T4, P4.1 proved** | E1.3, E1.4, E1.5 | Mitigation floor; bound tightness verification |
| 10 | Weeks 8–10 | P5 (if time permits) | E1.6, E1.7, E1.8 | RAG test, scaling experiment, anti-collapse validation |
| 11 | Weeks 11–12 | — | — | Technical report writing; presentation preparation |
| 12 | Weeks 11–12 | — | — | **Technical report**; **NVAITC presentation**; code cleanup & documentation |

**Exit criteria per phase:**
- **Weeks 1–3:** Literature survey complete (15–20 pages). L0, L1 proved. Framework formalized.
- **Weeks 4–7:** T1 (main theorem), T2, T3 proved. Core theoretical contribution complete.
- **Weeks 8–10:** T4 proved. All Priority 1 experiments complete. Project work complete.
- **Weeks 11–12:** Technical report complete. Final presentation prepared for NVAITC Student Workshop. Code cleanup and documentation done.

