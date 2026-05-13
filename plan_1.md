# Project Plan: Mathematical Theory of Mamba (Selective State Space Models)

**Duration:** 12 Weeks (10 weeks project + 2 weeks technical report & presentation)  
**Goal:** Develop rigorous mathematical foundations for the Mamba architecture — proving new results on stability, approximation power, expressivity, and generalization of selective state space models (S6), filling critical gaps between Mamba's empirical success and its theoretical understanding.  
**Output:** Technical report and presentation by end of Week 12.

---

## Motivation & Key Insight

Mamba (Gu & Dao, 2023) introduced **selective state space models (S6)** — a breakthrough architecture achieving linear-time sequence modeling by making the SSM parameters ($\mathbf{A}$, $\mathbf{B}$, $\mathbf{C}$, $\Delta$) input-dependent. Unlike Transformers with $O(n^2)$ attention, Mamba processes sequences in $O(n)$ time with $O(1)$ memory per step, enabling efficient modeling of million-token contexts. Mamba-2 (Dao & Gu, 2024) further connected selective SSMs to structured masked attention, bridging the SSM and attention paradigms.

Despite rapid empirical adoption — hybrid Transformer-Mamba architectures (Jamba, Zamba), vision Mamba (Vim), and deployment in long-context applications — **the mathematical theory of Mamba lags far behind its practice:**

- **Generalization:** The first training dynamics analysis (Ren et al., arXiv:2602.12499, Feb 2026) covers only a *single-layer simplified* Mamba block. Multi-layer and deep Mamba generalization bounds are entirely open.
- **Approximation:** Huang et al. (ICML 2025) proved that S6 can represent Haar wavelet projections, but general approximation rates for function classes (Sobolev, Barron, BV) are unknown.
- **Stability:** Lyapunov stability of the recurrent state dynamics has been studied empirically but lacks formal proofs connecting selectivity to stability guarantees.
- **Expressivity vs. Transformers:** No formal separation results exist characterizing what Mamba can compute that Transformers cannot (and vice versa) under equal parameter budgets.

**This project aims to close these gaps** by developing a unified mathematical framework treating Mamba as a *controlled dynamical system* with data-dependent parameters, and deriving new theorems on stability, approximation, expressivity, and generalization.

### The Selective SSM (S6) Formulation

The core Mamba layer evolves a hidden state $h_t \in \mathbb{R}^N$ via:

$$h_t = \bar{\mathbf{A}}_t \, h_{t-1} + \bar{\mathbf{B}}_t \, x_t, \qquad y_t = \mathbf{C}_t \, h_t$$

where the **discretized** parameters are input-dependent:

$$\Delta_t = \text{softplus}(W_\Delta \, x_t + b_\Delta)$$
$$\bar{\mathbf{A}}_t = \exp(\Delta_t \cdot \mathbf{A}), \qquad \bar{\mathbf{B}}_t = (\Delta_t \cdot \mathbf{A})^{-1}(\bar{\mathbf{A}}_t - \mathbf{I}) \cdot \Delta_t \cdot \mathbf{B}_t$$

with $\mathbf{B}_t = W_B \, x_t$, $\mathbf{C}_t = W_C \, x_t$, and $\mathbf{A} \in \mathbb{R}^{N \times N}$ a structured (diagonal, negative) matrix. The selectivity — $\Delta_t$, $\mathbf{B}_t$, $\mathbf{C}_t$ all depending on $x_t$ — is the key innovation: it allows the model to dynamically control what information enters, persists in, and exits the hidden state.

**Mathematical lens:** We view S6 as a *linear time-varying (LTV) system* with input-dependent dynamics. This places Mamba at the intersection of:
- **Control theory:** controllability, observability, and stability of LTV systems
- **Approximation theory:** approximation by recurrent operators with structured state
- **Dynamical systems:** Lyapunov stability, spectral analysis of products of random matrices
- **Statistical learning theory:** generalization of sequence-to-sequence maps with bounded state

### Distinction from Prior Work

| Prior Work | What They Prove | What Remains Open (Our Contributions) |
|---|---|---|
| Gu et al. (2022), S4 theory | HiPPO initialization approximates continuous signals via Legendre/Laguerre bases | Does not cover *selective* (input-dependent) SSMs; S4 parameters are fixed |
| Huang et al. (ICML 2025) | S6 represents Haar wavelet projections; associative recall capacity of S6 | Only Haar basis; no rates for smooth functions; no comparison with Transformer expressivity |
| Ren et al. (Feb 2026) | Single-layer Mamba generalization bound via feature selection | Single layer only; no depth analysis; no architecture comparison |
| Cirone et al. (2025) | SSM stability via Lyapunov exponents (empirical) | Empirical only; no formal stability theorem connecting selectivity to Lyapunov spectrum |

Our contributions fill each gap: **Theorem 1** (stability), **Theorem 3** (approximation rates), **Theorem 5** (expressivity separation), **Theorem 7** (deep generalization).

---

## Weeks 1–3: Dynamical Systems Framework & Stability Theory

**Objective:** Establish the mathematical framework treating Mamba as a controlled LTV system. Prove foundational stability results and the hidden state boundedness theorem. Implement numerical tools for verifying theoretical predictions.

### Theory Track

- **Week 1: Framework Setup & Notation**
  - Formalize the S6 recurrence as an LTV system: $h_t = \mathbf{F}(x_t) h_{t-1} + \mathbf{G}(x_t) x_t$ where $\mathbf{F}(x) = \bar{\mathbf{A}}(x)$ and $\mathbf{G}(x) = \bar{\mathbf{B}}(x)$.
  - Define the function classes for input sequences: $\mathcal{X}_T = \{(x_1, \ldots, x_T) : x_t \in \mathbb{R}^d, \|x_t\| \leq R\}$.
  - Establish the key structural properties of S6 that will be used throughout:
    1. $\mathbf{A}$ is diagonal with negative entries $a_i < 0$ (by design).
    2. $\bar{\mathbf{A}}_t = \text{diag}(e^{\Delta_t \cdot a_i})$ has spectral radius $\rho(\bar{\mathbf{A}}_t) < 1$ for all $t$ (since $a_i < 0$ and $\Delta_t > 0$).
    3. $\Delta_t \in [\delta_{\min}, \delta_{\max}]$ is bounded away from 0 and $\infty$ due to softplus saturation and bounded inputs.

- **Week 2: Lemma 0 & Theorem 1 (Stability)**

  - **Lemma 0 (State Boundedness): Uniform Bound on Hidden State Norms**
    - **Statement:** Let $(x_1, \ldots, x_T) \in \mathcal{X}_T$ be an arbitrary input sequence. The hidden state of a single S6 layer satisfies:
    $$\|h_t\| \leq \frac{\bar{g} \cdot R}{1 - \bar{\rho}} \quad \text{for all } t \geq 1$$
      where $\bar{\rho} = \max_{\|x\| \leq R} \rho(\bar{\mathbf{A}}(x)) = e^{\delta_{\min} \cdot \max_i a_i} < 1$ is the worst-case spectral radius, and $\bar{g} = \max_{\|x\| \leq R} \|\bar{\mathbf{B}}(x)\|$ is the maximum input gain.
    - **Proof Strategy:** Since $\bar{\mathbf{A}}_t$ is diagonal with $|\bar{A}_{t,ii}| = e^{\Delta_t a_i} \leq e^{\delta_{\min} a_{\max}} = \bar{\rho} < 1$, we have $\|\bar{\mathbf{A}}_t\|_2 \leq \bar{\rho}$. Unrolling the recurrence: $\|h_t\| \leq \bar{\rho}^t \|h_0\| + \bar{g} R \sum_{s=0}^{t-1} \bar{\rho}^s \leq \bar{g} R / (1 - \bar{\rho})$ (geometric series). With $h_0 = 0$, the bound holds from $t = 1$.
    - **Why it matters:** Required for all subsequent theorems. Guarantees the hidden state does not explode regardless of sequence length — the fundamental stability property that Transformers lack (attention outputs can grow with sequence length).

  - **Theorem 1 (Exponential Stability): Selective SSM is Uniformly Exponentially Stable**
    - **Statement:** For any two input sequences $(x_1, \ldots, x_T)$ and $(x_1', \ldots, x_T')$ that agree for $t > t_0$ (i.e., $x_t = x_t'$ for $t > t_0$), the corresponding hidden states satisfy:
    $$\|h_t - h_t'\| \leq C \cdot \bar{\rho}^{t - t_0} \quad \text{for } t > t_0$$
      where $C$ depends on $\|h_{t_0} - h_{t_0}'\|$ and the model parameters, and $\bar{\rho} < 1$ is the worst-case contraction rate from Lemma 0.
    - **Proof Strategy:** For $t > t_0$, the difference $e_t = h_t - h_t'$ satisfies:
    $$e_t = \bar{\mathbf{A}}_t \, e_{t-1} \quad (t > t_0, \text{ since } x_t = x_t')$$
      Since $\|\bar{\mathbf{A}}_t\|_2 \leq \bar{\rho} < 1$, we get $\|e_t\| \leq \bar{\rho}^{t-t_0} \|e_{t_0}\|$.
    - **Why it matters:** Proves that Mamba has *finite effective memory* — perturbations in the distant past are exponentially forgotten. This is the formal version of the intuition that SSMs trade off long-range memory for stability, and it sets up the approximation-memory tradeoff analyzed in Theorem 4.
    - **Key distinction from Transformers:** Transformers with full attention have *infinite* effective memory (every past token is directly accessible). Mamba's exponential forgetting is both a strength (stability, generalization) and a limitation (long-range dependency). The selectivity mechanism ($\Delta_t$ modulation) partially compensates by allowing the model to dynamically control the forgetting rate.

- **Week 3: Theorem 2 (Selectivity & Controllability)**

  - **Theorem 2 (Controllability via Selectivity): Input-Dependent $\Delta$ Enables Full State Control**
    - **Statement:** Consider the S6 system with diagonal $\mathbf{A} = \text{diag}(a_1, \ldots, a_N)$ where $a_i < 0$ are distinct. The system is *controllable* in the sense that for any target state $h^* \in \mathbb{R}^N$ with $\|h^*\| \leq H_{\max}$ (Lemma 0), there exists an input sequence $(x_1, \ldots, x_N) \in \mathcal{X}_N$ of length $N$ that drives the state from $h_0 = 0$ to $h_N = h^*$, provided the selectivity parameters $(W_\Delta, W_B)$ satisfy a rank condition.
    - **Proof Strategy:**
      1. Write the $N$-step reachability as $h_N = \sum_{t=1}^{N} \Phi(N, t+1) \bar{\mathbf{B}}_t x_t$ where $\Phi(N, t+1) = \prod_{s=t+1}^{N} \bar{\mathbf{A}}_s$ is the state transition matrix.
      2. Since $\bar{\mathbf{A}}_s$ is diagonal, $\Phi(N, t+1) = \text{diag}(\prod_{s=t+1}^{N} e^{\Delta_s a_i})$, and the controllability Gramian $\mathcal{W}_N = \sum_{t=1}^{N} \Phi(N,t+1) \bar{\mathbf{B}}_t \bar{\mathbf{B}}_t^\top \Phi(N,t+1)^\top$.
      3. With distinct $a_i$ and input-dependent $\Delta_t$, the Gramian $\mathcal{W}_N$ is positive definite (a Vandermonde-type argument on the exponentials $e^{\Delta_t a_i}$). The distinct eigenvalues ensure the columns of the reachability matrix are linearly independent.
      4. For *fixed* $\Delta$ (non-selective S4), controllability requires $N$ steps with the same Gramian structure. Selectivity ($\Delta_t$ varying with input) allows the Gramian to be full-rank with *fewer effective steps* because different $\Delta_t$ values create a richer set of basis functions.
    - **Why it matters:** Formalizes the intuition that selectivity gives Mamba more expressive control over its internal state than fixed-parameter SSMs (S4). The controllability Gramian provides a quantitative measure of "how much information" the S6 mechanism can inject into the state, directly connecting to approximation power (Theorem 3).

  - **Corollary 2.1 (Observability):** By duality (transposing the system with $\mathbf{C}_t$ playing the role of $\mathbf{B}_t$), the S6 system is *observable*: distinct internal states produce distinct output sequences. Combined with controllability, this gives a *minimal realization* result — the S6 state dimension $N$ is the minimal dimension needed to realize the input-output map, and no states are wasted.

- **Week 3: Numerical Verification & Visualization Tools**
  - Implement a standalone S6 simulator in PyTorch (no training, just forward dynamics) for verifying theoretical predictions.
  - **Verification E1.1:** Numerically compute the spectral radius $\bar{\rho}$ and hidden state bound from Lemma 0 across random initializations and input distributions. Compare to the theoretical bound.
  - **Verification E1.2:** Measure the exponential decay rate from Theorem 1 by computing $\|h_t - h_t'\|$ for perturbed inputs and fitting the decay constant. Compare to the predicted $\bar{\rho}$.
  - **Verification E1.3:** Compute the controllability Gramian $\mathcal{W}_N$ for trained Mamba models (using open-source checkpoints) and verify positive definiteness. Plot the condition number $\kappa(\mathcal{W}_N)$ as a function of state dimension $N$ and selectivity strength.
  - Visualize hidden state trajectories in 2D/3D (via PCA) for different input types (periodic, random, structured text) to build intuition for the dynamics.

### Theorem Proofing Track — Weeks 1–3

| Result | Week | Statement | Status |
|---|---|---|---|
| **L0: State Boundedness** | 2 | $\|h_t\| \leq \bar{g}R/(1-\bar{\rho})$ uniformly | To prove |
| **T1: Exponential Stability** | 2 | Past perturbations decay at rate $\bar{\rho}^t$ | To prove |
| **T2: Controllability** | 3 | S6 with distinct $a_i$ is controllable; selectivity improves Gramian conditioning | To prove |
| **C2.1: Observability** | 3 | Dual of T2; distinct states → distinct outputs | To prove |

---

## Weeks 4–7: Approximation Theory & Expressivity Separation

**Objective:** Prove approximation rates for S6 on standard function classes. Establish formal expressivity separation results between Mamba and Transformers. Begin generalization analysis.

### Theory Track

- **Week 4–5: Theorem 3 (Approximation Rates)**

  - **Theorem 3 (Approximation Power): S6 Approximation Rates for Sobolev Functions**
    - **Statement:** Let $f: \mathbb{R}^T \to \mathbb{R}$ be a causal sequence-to-sequence function in the Sobolev space $W^{s,2}([0,1]^T)$ with smoothness $s > 0$. An $L$-layer S6 network with state dimension $N$ per layer achieves approximation error:
    $$\inf_\theta \|f - f_\theta^{\text{S6}}\|_{L^2} \leq C(f, s) \cdot (L \cdot N)^{-s/d}$$
      where $d$ is the input dimension and $C(f,s)$ depends on the Sobolev norm $\|f\|_{W^{s,2}}$.
    - **Proof Strategy (Three Parts):**

      **Part A — Single-Layer Approximation via Exponential Sums.**
      A single S6 layer with state dimension $N$ computes $y_t = \sum_{i=1}^{N} c_i(x_t) \sum_{\tau \leq t} \left(\prod_{s=\tau+1}^{t} e^{\Delta_s a_i}\right) b_i(x_\tau) x_\tau$. The inner sum is a weighted exponential sum over the input history. By the Müntz-Szász theorem, exponential sums with $N$ distinct exponents $\{a_i\}$ can approximate any $L^2$ function on $[0, \infty)$ with error depending on $N$ and the function's regularity. Specifically, for functions with Sobolev regularity $s$, the $N$-term exponential sum achieves $O(N^{-s})$ approximation.

      **Part B — Depth Amplification via Composition.**
      An $L$-layer S6 network composes $L$ such approximators. By the standard composition-approximation argument (Telgarsky 2016, adapted to recurrent architectures): if each layer contributes $O(N^{-s})$ approximation in a different "frequency band" (controlled by the distinct $a_i$ per layer), then $L$ layers achieve $O((LN)^{-s/d})$ by a tensor product argument across dimensions.

      **Part C — Selectivity Enhancement.**
      The input-dependent $\Delta_t$ allows the effective exponents $\Delta_t a_i$ to vary across time steps, creating a *non-uniform* exponential basis adapted to the input. We show this is equivalent to an adaptive quadrature rule: the S6 mechanism places more "resolution" (smaller effective time constants) where the input has high variation and less where it is smooth. This yields a log-factor improvement over fixed-$\Delta$ SSMs for functions with spatially varying regularity (the Besov space $B_{s}^{p,q}$).

    - **Why it matters:** Provides the first approximation rates for selective SSMs on standard function classes, directly comparable to known results for Transformers (Yun et al. 2020: Transformers are universal approximators but without explicit rates) and RNNs (Schäfer & Zimmermann 2006). The selectivity enhancement (Part C) formally justifies why Mamba outperforms S4 on non-stationary signals.

  - **Proposition 3.1 (Approximation of Discontinuous Functions via Haar Wavelets)**
    - **Statement:** For piecewise constant functions with $K$ discontinuities, an S6 layer with state dimension $N \geq K$ achieves *exact* representation (zero error), while a fixed-parameter S4 layer requires $N = \Omega(K \log K)$.
    - **Proof:** Follows from Huang et al. (ICML 2025) — the selective $\Delta_t$ mechanism can set $\Delta_t \to \infty$ at discontinuities (resetting the state) and $\Delta_t \to 0$ in smooth regions (preserving the state). We extend their Haar wavelet result to give the explicit $N$-vs-$K$ tradeoff. The S4 lower bound follows from the fixed-basis limitation of Legendre/Laguerre polynomials.

- **Week 6: Theorem 4 (Memory-Stability Tradeoff) & Theorem 5 (Expressivity Separation)**

  - **Theorem 4 (Memory-Stability Tradeoff): Fundamental Limit of Finite-State Recurrence**
    - **Statement:** For any S6 model with state dimension $N$ and contraction rate $\bar{\rho}$, the *effective memory length* $M_\text{eff}$ (the number of past tokens that influence the current output above a threshold $\varepsilon$) satisfies:
    $$M_\text{eff} \leq \frac{N \cdot \log(H_{\max}/\varepsilon)}{\log(1/\bar{\rho})}$$
      where $H_{\max}$ is the state bound from Lemma 0. Moreover, for any causal function $f$ with *true* dependency length $M > M_\text{eff}$, the S6 approximation error is at least:
    $$\|f - f_\theta^{\text{S6}}\|_{L^2} \geq c \cdot \bar{\rho}^{M - M_\text{eff}} \cdot \text{Var}(f)$$
    - **Proof Strategy:** The $M_\text{eff}$ upper bound follows from Theorem 1: after $M_\text{eff}$ steps, the contribution of token $x_{t - M_\text{eff}}$ to $h_t$ is at most $\bar{\rho}^{M_\text{eff}} H_{\max} < \varepsilon$ per state dimension, and there are $N$ state dimensions. The lower bound on approximation error uses an information-theoretic argument: the S6 state $h_t \in \mathbb{R}^N$ is an $N$-dimensional bottleneck through which all past information must flow, and by the data processing inequality, $I(x_{t-M}; y_t | h_t) = 0$.
    - **Why it matters:** This is arguably the most important result for understanding *when Mamba fails*. It quantifies the fundamental limitation of linear recurrence: Mamba cannot represent functions with long-range dependencies beyond $M_\text{eff}$ tokens. This explains why hybrid Transformer-Mamba architectures (Jamba) outperform pure Mamba on tasks requiring very long-range retrieval.

  - **Theorem 5 (Expressivity Separation): Mamba vs. Transformer**
    - **Statement:** Under equal parameter budgets:
      1. **(Mamba $\supsetneq$ Transformer for sequential tasks):** There exists a family of causal functions $\{f_L\}$ (specifically, finite automaton simulation with $L$ states) that an $L$-layer Mamba with state dimension $N = O(L)$ computes exactly, but any Transformer of depth $D$ requires $D = \Omega(L / \log L)$ layers.
      2. **(Transformer $\supsetneq$ Mamba for retrieval tasks):** There exists a family of functions $\{g_K\}$ (specifically, $K$-nearest-neighbor retrieval over the input sequence) that a single Transformer layer computes exactly, but any Mamba model requires state dimension $N = \Omega(K \cdot T)$ where $T$ is the sequence length.
      3. **(Implication):** Neither architecture strictly dominates the other. Mamba excels at tasks with bounded-state sequential structure; Transformers excel at tasks requiring arbitrary pairwise comparisons.
    - **Proof Strategy:**
      1. For claim 1: Construct a deterministic finite automaton (DFA) with $L$ states. Mamba simulates it by encoding the DFA transition table in the state matrix: $h_t = \bar{\mathbf{A}}(x_t) h_{t-1}$ where $h_t$ is a one-hot encoding of the current DFA state. This requires $N = L$. A Transformer must simulate the sequential state transitions via positional attention chains, which requires depth $\Omega(L/\log L)$ by the circuit complexity argument of Merrill et al. (2022) adapted to causal attention.
      2. For claim 2: The $K$-nearest-neighbor function requires comparing all $O(T)$ past tokens to select the $K$ closest. Attention performs this in one layer. Mamba's state $h_t \in \mathbb{R}^N$ must encode the full sorted list of distances, which requires $N = \Omega(KT)$ by an information-theoretic counting argument on the number of possible orderings.
    - **Why it matters:** Provides the first formal framework for understanding *when to use Mamba vs. Transformer vs. hybrid*. The separation is constructive: claim 1 gives tasks where Mamba wins; claim 2 gives tasks where Transformers win. This directly informs architecture design for practical applications.

- **Week 7: Begin Generalization Framework**
  - Set up the PAC-learning framework for multi-layer S6 networks.
  - Define the hypothesis class $\mathcal{H}_{L,N,d}$ of $L$-layer S6 networks with state dimension $N$ and input dimension $d$.
  - Compute the VC dimension and Rademacher complexity bounds for single-layer S6 (extending Ren et al. Feb 2026) as a warm-up for the multi-layer result in Weeks 8–10.
  - Begin deriving the covering number bounds needed for Theorem 7.

### Numerical Verification Track — Weeks 4–7

| ID | Experiment | Tests | Setup |
|---|---|---|---|
| E2.1 | **Approximation rate curve.** Fix a target function $f \in W^{s,2}$ (e.g., $s = 2$, a smooth causal convolution). Train S6 models with $N \in \{4, 8, 16, 32, 64, 128\}$ and $L \in \{1, 2, 4, 8\}$. Plot $\log(\text{error})$ vs. $\log(LN)$. | T3 | Slope should match $-s/d$ from the theorem. |
| E2.2 | **Selectivity vs. fixed-$\Delta$ comparison.** Repeat E2.1 with (a) selective $\Delta_t$ (Mamba) and (b) fixed $\Delta$ (S4-style). Compare approximation curves. | T3, Part C | Mamba should show log-factor improvement on non-stationary targets. |
| E2.3 | **Memory length measurement.** Feed sequences with a planted signal at position $t - M$ for $M \in \{10, 50, 100, 500, 1000\}$. Measure detection accuracy vs. $M$ for models with $N \in \{16, 64, 256\}$. | T4 | Accuracy should drop sharply at $M \approx M_\text{eff}(N, \bar{\rho})$. |
| E2.4 | **DFA simulation benchmark.** Construct DFAs with $L \in \{4, 8, 16, 32, 64\}$ states. Train Mamba ($N = L$) and Transformer (varying depth). Compare minimum depth for perfect accuracy. | T5, Claim 1 | Mamba: constant depth. Transformer: depth grows with $L$. |
| E2.5 | **Retrieval benchmark.** $K$-nearest-neighbor retrieval task with $K \in \{1, 5, 10\}$ and $T \in \{64, 256, 1024\}$. Compare Mamba (varying $N$) vs. single-layer Transformer. | T5, Claim 2 | Transformer: perfect accuracy. Mamba: accuracy degrades as $KT/N$ grows. |

---

## Weeks 8–10: Generalization Theory & Deep Analysis

**Objective:** Prove the multi-layer generalization bound (the paper's capstone result). Derive the spectral characterization of the state transition matrix. Complete all proofs and run remaining experiments.

### Theory Track

- **Week 8–9: Theorem 6 (Spectral Characterization) & Theorem 7 (Generalization)**

  - **Theorem 6 (Spectral Characterization): Lyapunov Exponents of the Selective State Transition**
    - **Statement:** Consider the product of state transition matrices $\Phi(T, 1) = \prod_{t=1}^{T} \bar{\mathbf{A}}(x_t)$ along an input sequence drawn from a stationary ergodic process. The Lyapunov exponents $\lambda_1 \geq \cdots \geq \lambda_N$ (defined as $\lambda_i = \lim_{T \to \infty} \frac{1}{T} \log \sigma_i(\Phi(T,1))$ where $\sigma_i$ are singular values) satisfy:
      1. $\lambda_i < 0$ for all $i$ (uniform contraction, consistent with T1).
      2. $\lambda_i = \mathbb{E}[\Delta_t] \cdot a_i + O(\text{Var}(\Delta_t))$ — the Lyapunov exponents are, to first order, the time-averaged eigenvalues of the continuous-time system, with a correction from the selectivity variance.
      3. The *Lyapunov dimension* $d_L = \max\{k : \sum_{i=1}^k \lambda_i > 0\}$ determines the effective dimensionality of the attractor of the state dynamics. Higher selectivity variance increases $d_L$, allowing the model to maintain a higher-dimensional state representation.
    - **Proof Strategy:** Since $\bar{\mathbf{A}}_t = \text{diag}(e^{\Delta_t a_i})$ is diagonal, the Lyapunov exponents decouple:
    $$\lambda_i = \lim_{T \to \infty} \frac{1}{T} \sum_{t=1}^{T} \Delta_t \cdot a_i = \mathbb{E}[\Delta_t] \cdot a_i$$
      by the ergodic theorem. The $O(\text{Var}(\Delta_t))$ correction comes from the non-commutativity when $\mathbf{A}$ is not exactly diagonal (off-diagonal perturbation analysis via Furstenberg's theorem). For exactly diagonal $\mathbf{A}$, the exponents are exact. The Lyapunov dimension follows from the Kaplan-Yorke conjecture applied to the product of diagonal matrices.
    - **Why it matters:** Connects Mamba's empirical observation of "the model learns to modulate $\Delta_t$" to a precise dynamical systems quantity. The Lyapunov dimension characterizes the effective capacity of the hidden state — a model with higher $d_L$ utilizes its state dimensions more efficiently. This provides a principled way to analyze and compare different trained Mamba models.

  - **Theorem 7 (Generalization Bound): Multi-Layer Mamba Generalization via State Compression**
    - **Statement:** Let $f_\theta: \mathbb{R}^{T \times d} \to \mathbb{R}^{T \times d}$ be an $L$-layer S6 network with state dimension $N$, parameter count $P = O(L \cdot (N \cdot d + d^2))$, input bound $R$, and contraction rate $\bar{\rho}$ (from T1). Given $n$ i.i.d. training sequences of length $T$, the generalization gap satisfies:
    $$\mathbb{E}[\mathcal{L}_\text{test}] - \mathcal{L}_\text{train} \leq O\left(\sqrt{\frac{L \cdot N \cdot d \cdot \log(T/\delta)}{n}} \cdot \frac{1}{(1 - \bar{\rho})^L}\right)$$
      with probability $\geq 1 - \delta$.
    - **Proof Strategy (Four Steps):**

      **Step 1 — Single-layer covering number.** For a single S6 layer with parameter space $\Theta_1 \subset \mathbb{R}^{P_1}$ and input bound $R$, the output map $x \mapsto y$ has covering number $\log \mathcal{N}(\varepsilon, \mathcal{H}_1, \|\cdot\|_\infty) \leq O(P_1 \log(R H_{\max}/\varepsilon))$ where $H_{\max}$ is from Lemma 0. This uses the Lipschitz property of the S6 output w.r.t. parameters (each parameter appears in a sigmoid/exponential, both Lipschitz).

      **Step 2 — Multi-layer composition.** For $L$ layers composed, the covering number multiplies: $\log \mathcal{N}(\varepsilon, \mathcal{H}_L) \leq L \cdot \log \mathcal{N}(\varepsilon / (L \cdot \text{Lip}^{L-1}), \mathcal{H}_1)$ where $\text{Lip}$ is the per-layer Lipschitz constant. Crucially, for S6 with contraction rate $\bar{\rho} < 1$, the per-layer Lipschitz constant is $\text{Lip} \leq \bar{\rho} + \bar{g} \cdot \|W_C\| < 1 + \bar{g}\|W_C\|$. Unlike Transformers, where the Lipschitz constant grows with sequence length (due to attention), S6's Lipschitz constant is independent of $T$.

      **Step 3 — Rademacher complexity bound.** Convert the covering number to a Rademacher complexity bound via Dudley's entropy integral: $\mathcal{R}_n(\mathcal{H}_L) \leq O(\sqrt{L \cdot N \cdot d \cdot \log T / n})$.

      **Step 4 — Generalization via uniform convergence.** Apply the standard Rademacher-based generalization bound: $\mathbb{E}[\mathcal{L}_\text{test}] - \mathcal{L}_\text{train} \leq 2\mathcal{R}_n(\mathcal{H}_L) + O(\sqrt{\log(1/\delta)/n})$.

    - **Key insight vs. Transformer generalization:** Transformer generalization bounds (Edelman et al. 2022) scale as $O(\sqrt{T^2 \cdot P / n})$ due to the $O(T^2)$ attention Lipschitz constant. The S6 bound scales as $O(\sqrt{T \cdot P / n})$ (only $O(T)$ from the sequence length, not $O(T^2)$) because the recurrent Lipschitz constant is independent of $T$. This is a **quadratic improvement in $T$**, formalizing the intuition that Mamba generalizes better on long sequences.
    - **Why it matters:** This is the capstone theoretical result. It provides the first generalization bound for deep (multi-layer) Mamba and shows a concrete theoretical advantage over Transformers: better generalization on long sequences due to the state compression bottleneck. The bound also reveals the price of depth: the $(1 - \bar{\rho})^{-L}$ factor grows exponentially with depth, suggesting that very deep Mamba models may overfit unless the contraction rate $\bar{\rho}$ is sufficiently small.

- **Week 10: Theorem 8 (Selectivity as Implicit Regularization)**

  - **Theorem 8 (Implicit Regularization): Selectivity Biases Mamba Toward Low-Complexity Solutions**
    - **Statement:** Consider training a single-layer S6 model with gradient descent on a squared loss. In the early phase of training (before the parameters escape a neighborhood of initialization), the selectivity mechanism ($\Delta_t$ depending on $x_t$) implicitly regularizes the model by:
      1. Biasing $\Delta_t$ toward values that minimize the *effective rank* of the state transition product $\Phi(T, 1) = \prod_t \bar{\mathbf{A}}_t$.
      2. This effective rank minimization is equivalent to maximizing the mutual information $I(h_t; x_t)$ relative to $I(h_t; x_{<t})$ — i.e., the selectivity mechanism favors states that are *informative about the current input* over states that merely memorize the past.
    - **Proof Strategy:** Analyze the gradient flow $\dot{W}_\Delta = -\nabla_{W_\Delta} \mathcal{L}$. In the early training regime (linear approximation around initialization), show that the gradient of $\mathcal{L}$ w.r.t. $W_\Delta$ decomposes into a "task-relevant" component (aligning $\Delta_t$ with discriminative features of $x_t$) and a "compression" component (reducing the effective rank of $\Phi$). The compression component dominates early because the task-relevant gradients are initially small (random initialization). This is analogous to the implicit bias of gradient descent toward minimum-norm solutions in linear models (Gunasekar et al. 2018), adapted to the non-linear S6 setting via a local linearization argument.
    - **Why it matters:** Explains *why* Mamba generalizes well in practice despite having more parameters than necessary for many tasks. The selectivity mechanism acts as a built-in regularizer, automatically compressing the state to retain only task-relevant information. This complements Theorem 7 (which gives an *a priori* bound) with a *training dynamics* explanation.

### Numerical Verification Track — Weeks 8–10

| ID | Experiment | Tests | Setup |
|---|---|---|---|
| E3.1 | **Lyapunov exponent measurement.** For pretrained Mamba models (130M, 370M, 1.4B from open-source), feed 1000 sequences from The Pile. Compute empirical Lyapunov exponents via QR decomposition of $\Phi(T,1)$. | T6 | Compare measured $\lambda_i$ to the predicted $\mathbb{E}[\Delta_t] \cdot a_i$. Plot Lyapunov spectrum. |
| E3.2 | **Generalization gap vs. sequence length.** Train Mamba and Transformer on synthetic tasks with $T \in \{64, 256, 1024, 4096\}$. Measure train/test gap. | T7 | Mamba gap should scale as $\sqrt{T}$; Transformer gap as $\sqrt{T^2} = T$. |
| E3.3 | **Generalization gap vs. depth.** Train $L \in \{2, 4, 8, 16, 32\}$ layer Mamba on a fixed task. Measure train/test gap. | T7 | Gap should grow as $(1-\bar{\rho})^{-L}$, confirming the depth penalty. |
| E3.4 | **Effective rank dynamics during training.** Track the effective rank of $\Phi(T,1)$ throughout training. Correlate with generalization performance. | T8 | Effective rank should decrease during early training (implicit regularization) then stabilize. |
| E3.5 | **Selectivity ablation.** Compare (a) full Mamba (selective $\Delta$), (b) fixed $\Delta$ (S4-style), (c) random $\Delta$ (no learning). Measure generalization gap. | T8 | Selective $\Delta$ should have smallest generalization gap, confirming implicit regularization. |

---

## Weeks 11–12: Technical Report & Presentation

**Objective:** Write the technical report, prepare the final presentation for the NVAITC Student Workshop, and complete code cleanup and documentation.

- **Technical Report:** Compile all theorems, proofs, and numerical verifications into a unified technical report. Write the introduction framing the contribution as "a mathematical theory of selectivity in state space models." Ensure each theorem has both a formal proof and supporting numerical evidence.
- **Presentation:** Prepare the final presentation for the NVAITC Student Workshop.
- **Code Cleanup & Documentation:** Clean up the numerical verification codebase and prepare a companion GitHub repository.

---

## Summary: Theorem Dependency Map

| Result | Phase | Depends On | Enables |
|---|---|---|---|
| **L0: State Boundedness** | Weeks 1–3 | Geometric series, $\bar{\rho} < 1$ | T1, T3, T4, T7 |
| **T1: Exponential Stability** | Weeks 1–3 | L0, diagonal $\bar{\mathbf{A}}$ | T4, T6, T7 |
| **T2: Controllability** | Weeks 1–3 | Distinct $a_i$, Vandermonde argument | T3, T5 |
| **C2.1: Observability** | Weeks 1–3 | Dual of T2 | T5 |
| **T3: Approximation Rates** | Weeks 4–7 | T2 (controllability → reachable states), Müntz-Szász | Core approximation claim |
| **P3.1: Discontinuous Functions** | Weeks 4–7 | Huang et al. ICML 2025, T3 Part C | Selectivity advantage for non-smooth targets |
| **T4: Memory-Stability Tradeoff** | Weeks 4–7 | T1 (decay rate), information theory | Fundamental limitation result |
| **T5: Expressivity Separation** | Weeks 4–7 | T2, T4, circuit complexity | Architecture comparison claim |
| **T6: Spectral Characterization** | Weeks 8–10 | T1, ergodic theorem, Furstenberg | Dynamical systems interpretation |
| **T7: Generalization Bound** | Weeks 8–10 | L0, T1 (Lipschitz bound), covering numbers | Capstone result |
| **T8: Implicit Regularization** | Weeks 8–10 | T7, gradient flow analysis | Training dynamics explanation |

---

## Experiment Plan Summary

### Priority 1 (Required for Paper)

| ID | Experiment | Tests Theorem |
|---|---|---|
| E1.1 | State bound verification | L0 |
| E1.2 | Exponential decay rate | T1 |
| E2.1 | Approximation rate curve | T3 |
| E2.3 | Memory length measurement | T4 |
| E2.4 | DFA simulation (Mamba vs. Transformer) | T5 |
| E2.5 | Retrieval benchmark (Mamba vs. Transformer) | T5 |
| E3.2 | Generalization gap vs. sequence length | T7 |

### Priority 2 (Strengthens the Paper)

| ID | Experiment | Tests Theorem |
|---|---|---|
| E1.3 | Controllability Gramian analysis | T2 |
| E2.2 | Selectivity vs. fixed-$\Delta$ approximation | T3 Part C |
| E3.1 | Lyapunov exponent measurement | T6 |
| E3.3 | Generalization gap vs. depth | T7 |
| E3.4 | Effective rank during training | T8 |
| E3.5 | Selectivity ablation | T8 |

### Priority 3 (Extended / Future Work)

| ID | Experiment | Goal |
|---|---|---|
| E4.1 | Repeat all experiments on Mamba-2 (SSD formulation) | Verify theory extends to structured state space duality |
| E4.2 | Hybrid Mamba-Transformer analysis | Test whether T4's memory limit explains when hybridization helps |
| E4.3 | Scaling law comparison | Mamba vs. Transformer compute-optimal frontier |
| E4.4 | Vision domain transfer | Apply T3 approximation analysis to Vim (Vision Mamba) |

---

## Timeline Summary

| Week | Phase | Theory | Numerical Verification | Deliverable |
|---|---|---|---|---|
| 1 | Weeks 1–3 | Framework setup, notation | — | Mathematical specification |
| 2 | Weeks 1–3 | **L0, T1 proved** | E1.1, E1.2 | State bound & stability proofs |
| 3 | Weeks 1–3 | **T2, C2.1 proved** | E1.3, visualization tools | Verified stability framework |
| 4 | Weeks 4–7 | Begin T3 (Parts A, B) | — | Approximation draft |
| 5 | Weeks 4–7 | **Complete T3, P3.1** | E2.1, E2.2 | Approximation rate proof |
| 6 | Weeks 4–7 | **T4, T5 proved** | E2.3, E2.4, E2.5 | Memory-expressivity proofs |
| 7 | Weeks 4–7 | Begin T7 framework | — | Covering number bounds |
| 8 | Weeks 8–10 | **T6 proved** | E3.1 | Spectral characterization |
| 9 | Weeks 8–10 | **T7 proved** | E3.2, E3.3 | Generalization bound |
| 10 | Weeks 8–10 | **T8 proved** | E3.4, E3.5 | Implicit regularization |
| 11 | Weeks 11–12 | — | — | Technical report, presentation prep |
| 12 | Weeks 11–12 | — | — | **Technical report & presentation** |

**Exit criteria per phase:**
- **Weeks 1–3:** L0, T1, T2, C2.1 proved. Stability verified numerically. S6 simulator built.
- **Weeks 4–7:** T3, P3.1, T4, T5 proved. Approximation rates and expressivity separation verified numerically.
- **Weeks 8–10:** T6, T7, T8 proved. All Priority 1 experiments complete.
- **Weeks 11–12:** Technical report finalized. Presentation prepared for NVAITC Student Workshop. Code cleanup and documentation complete.

