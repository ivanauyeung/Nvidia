# Project Plan: Nemotron-3 + Lean 4 Math Theorem Proving Tool

**Duration:** 12 Weeks (10 weeks project + 2 weeks technical report & presentation)  
**Goal:** Build a practical, end-to-end tool that takes mathematical theorem statements (in natural language or LaTeX), uses NVIDIA Nemotron-3 via API to generate formal Lean 4 proofs, verifies them against the Lean kernel, and iteratively refines failed attempts — targeting researchers who want to formally verify theorems from their own papers.  
**Output:** Tool demo technical report + open-source release on GitHub by end of Week 12.

---

## Motivation & Key Insight

The LLM + Lean theorem proving space is dominated by industry labs (DeepSeek-Prover-V2 at 88.9% on MiniF2F, Seed-Prover solving 5/6 IMO 2025 problems). Competing on benchmark performance with a 671B model trained with massive RL is not viable for an academic team.

**Our angle is different:** We are not building a benchmark competitor. We are building a **usable tool for working mathematicians and ML researchers** — a system that takes a theorem statement from a paper draft, attempts to prove it formally in Lean 4, and returns either a verified proof or a structured explanation of where the proof attempt fails. The target user is someone writing a math-heavy AI paper who wants to formally verify their theorems.

**Why this niche is underserved:**
1. **Existing provers are research artifacts, not tools.** DeepSeek-Prover-V2, Seed-Prover, etc. are trained models with papers, not user-facing applications. There is no `pip install lean-prover` that a researcher can point at their theorem and get a verified proof.
2. **The gap is in the pipeline, not the model.** The hard engineering problems — autoformalization (LaTeX → Lean), Mathlib lemma retrieval, multi-step proof decomposition, error-guided refinement, and human-readable proof explanation — are unsolved at the tool level even though the component models exist.
3. **Nemotron-3 is uniquely suited for this.** Its 1M token context window allows feeding entire Mathlib sections as context. Its strong math reasoning (89.1% AIME25) and code generation capabilities make it a strong base for Lean code generation without requiring fine-tuning.

**Design philosophy:** Build a *pipeline* that orchestrates Nemotron-3's API capabilities with Lean 4's verification engine, rather than training a new model. The intelligence comes from the pipeline design (retrieval, decomposition, refinement loops), not from model weights.

### System Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface (CLI / Web)            │
│  Input: Theorem in LaTeX / natural language              │
│  Output: Verified Lean 4 proof or failure report         │
└──────────────┬──────────────────────────┬───────────────┘
               │                          │
       ┌───────▼───────┐          ┌───────▼───────┐
       │ Autoformalize │          │  Proof Report │
       │ (NL → Lean)   │          │  Generator    │
       └───────┬───────┘          └───────▲───────┘
               │                          │
       ┌───────▼──────────────────────────┴───────┐
       │         Proof Search Controller           │
       │  • Subgoal decomposition                  │
       │  • Strategy selection (tactic vs. term)   │
       │  • Budget management (API calls, time)    │
       └───────┬──────────┬───────────────────────┘
               │          │
       ┌───────▼───┐  ┌──▼──────────────┐
       │ Nemotron-3│  │ Mathlib RAG     │
       │ API       │  │ Retrieval       │
       │ (generate │  │ (find relevant  │
       │  tactics) │  │  lemmas/defs)   │
       └───────┬───┘  └──┬──────────────┘
               │          │
       ┌───────▼──────────▼───────────────────────┐
       │         LeanDojo / Lean 4 REPL            │
       │  • Tactic execution                       │
       │  • Proof state tracking                   │
       │  • Error message extraction               │
       └──────────────────────────────────────────┘
```

### Three Operating Modes

1. **Tactic Mode (Primary):** Given a Lean 4 theorem statement, generate tactics one at a time using Nemotron-3, execute each via LeanDojo, observe the resulting proof state, and iterate. The LLM sees the full proof state at each step, enabling adaptive proof search.

2. **Whole-Proof Mode:** Generate a complete Lean 4 proof in one shot, submit to `lean --run` for verification. If it fails, extract error messages and retry with error context. Faster per attempt but less adaptive.

3. **Decomposition Mode:** For complex theorems, first use Nemotron-3 to decompose into subgoals (lemmas), prove each subgoal independently via Tactic Mode, then assemble the full proof. This mirrors the subgoal decomposition strategy used by DeepSeek-Prover-V2 and Seed-Prover.

---

## Weeks 1–3: Infrastructure, Lean 4 Integration & Basic Proof Pipeline

**Objective:** Set up the development environment. Build the core pipeline connecting Nemotron-3 API → Lean 4 verification. Achieve basic proof generation on simple theorems (propositional logic, basic algebra).

### Implementation Track

- **Week 1: Environment Setup & API Integration**
  - Set up Lean 4 + Mathlib development environment (elan, lake, mathlib4).
  - Integrate Nemotron-3 API: build a Python client wrapper with retry logic, rate limiting, and response parsing.
  - Set up LeanDojo (v4.20.0+) for programmatic Lean interaction: install, test `Dojo.run_tac()` and `check_proof()` on a trivial theorem.
  - **Validation milestone:** Successfully call Nemotron-3 to generate a Lean tactic, execute it via LeanDojo, and observe the resulting proof state — all from a single Python script.
  - Define the project's Python package structure:
    ```
    lean_prover/
    ├── api/              # Nemotron-3 API client
    ├── lean/             # LeanDojo wrapper, Lean process management
    ├── retrieval/        # Mathlib RAG
    ├── pipeline/         # Proof search controller
    ├── formalize/        # LaTeX → Lean autoformalization
    ├── cli/              # Command-line interface
    └── web/              # Web UI (Weeks 8–10)
    ```

- **Week 2: Tactic Mode — Single-Step Proof Loop**
  - Implement the core tactic generation loop:
    1. Start with a theorem statement in Lean 4.
    2. Send the current proof state (goals, hypotheses, context) to Nemotron-3 with a prompt requesting the next tactic.
    3. Execute the tactic via `Dojo.run_tac()`.
    4. If `ProofFinished`: success.
    5. If `TacticState`: update context, loop.
    6. If `LeanError`: extract error message, send back to Nemotron-3 with "this tactic failed because [error], try a different approach."
    7. Repeat up to a maximum of $K$ steps (default $K = 50$).
  - Implement the prompt template for tactic generation:
    - System prompt: role as a Lean 4 proof assistant, Mathlib conventions, common tactic vocabulary.
    - User prompt: current goal state, list of available hypotheses, previous tactics attempted (for diversity).
    - Few-shot examples: 3–5 example proof states → tactic pairs, curated from Mathlib.
  - **Error refinement:** When a tactic fails, append the error message to the conversation history. Nemotron-3's long context (1M tokens) allows keeping the full proof attempt history, so the model can learn from its mistakes within a single session.

- **Week 3: Whole-Proof Mode & Retry Logic**
  - Implement Whole-Proof Mode:
    1. Send the theorem statement to Nemotron-3 with prompt: "Write a complete Lean 4 proof for this theorem."
    2. Extract the Lean code block from the response.
    3. Write to a `.lean` file and run `lake build` or `lean --run`.
    4. If success: return the proof.
    5. If failure: parse compiler errors (line number, error type, message), send back to Nemotron-3 with "Your proof had the following errors: [errors]. Please fix them."
    6. Retry up to $N$ times (default $N = 5$).
  - Implement **best-of-$n$ sampling:** For each proof attempt, generate $n$ candidate proofs (using temperature $\tau = 0.7$) and verify all in parallel. Return the first one that passes.
  - Build a proof caching layer: store successful proofs keyed by theorem statement hash, so repeated queries return instantly.

- **Week 4: Evaluation on Simple Benchmarks & Prompt Tuning**
  - Evaluate on a curated set of 50 "easy" Lean 4 theorems:
    - 10 propositional logic (e.g., `p ∧ q → q ∧ p`)
    - 10 basic natural number arithmetic (e.g., `n + 0 = n`)
    - 10 list operations from Mathlib (e.g., `List.length (l₁ ++ l₂) = List.length l₁ + List.length l₂`)
    - 10 basic algebra (e.g., commutativity, associativity in groups/rings)
    - 10 basic analysis (e.g., continuity of sum, product of continuous functions)
  - Measure: pass rate, average API calls per proof, average wall-clock time.
  - Iterate on prompt templates based on failure analysis: Which types of errors does Nemotron-3 make most frequently? (common issues: wrong tactic name, missing imports, type mismatches, universe level errors).
  - **Target:** ≥ 70% pass rate on this easy set. If below 50%, investigate whether the issue is prompt quality, Nemotron-3's Lean knowledge, or pipeline bugs.

### Design Decision Track — Weeks 1–3

| Decision | Options | Evaluation Criteria | Deadline |
|---|---|---|---|
| **Lean interaction method** | (a) LeanDojo Dojo class, (b) Direct Lean server protocol (LSP), (c) `lean --run` subprocess | Reliability, speed, error message quality | Week 1 |
| **Proof search strategy** | (a) Linear (one tactic at a time), (b) Best-first tree search, (c) Beam search | Pass rate vs. API cost tradeoff | Week 2 |
| **Prompt format** | (a) Plain text, (b) Structured JSON, (c) Chat-style multi-turn | Nemotron-3 response quality | Week 2 |
| **Temperature & sampling** | Sweep $\tau \in \{0.0, 0.3, 0.7, 1.0\}$ and $n \in \{1, 4, 8, 16\}$ | Pass rate at fixed API budget | Week 4 |

---

## Weeks 4–7: Mathlib Retrieval, Subgoal Decomposition & Autoformalization

**Objective:** Build the intelligence layer: Mathlib RAG for relevant lemma retrieval, subgoal decomposition for complex theorems, and LaTeX → Lean autoformalization. Scale evaluation to harder benchmarks.

### Implementation Track

- **Week 4: Mathlib Retrieval-Augmented Generation (RAG)**
  - Build a vector database of Mathlib lemmas and definitions:
    1. Extract all theorem/lemma statements from Mathlib4 using LeanDojo's data extraction.
    2. For each lemma, store: (a) the Lean statement, (b) the docstring (if any), (c) the full proof, (d) the file path, (e) an embedding vector.
    3. Generate embeddings using a code embedding model (e.g., `text-embedding-3-large` or a code-specific model). Embed both the Lean statement and a natural language summary generated by Nemotron-3.
    4. Store in a vector database (FAISS or ChromaDB, local — no cloud dependency).
  - Implement the retrieval pipeline:
    1. Given a theorem statement to prove, retrieve the top-$k$ most similar Mathlib lemmas ($k = 10$ default).
    2. Include them in the Nemotron-3 prompt as "Potentially useful lemmas from Mathlib."
    3. Also retrieve the *imports* needed for those lemmas and include them in the generated Lean file header.
  - **Validation:** On the 50-theorem easy benchmark from Weeks 1–3, measure pass rate with and without RAG. Expect ≥ 10% improvement on the algebra/analysis theorems (which depend heavily on knowing the right Mathlib lemma names).

- **Week 5: Subgoal Decomposition Mode**
  - Implement the decomposition pipeline:
    1. **Decompose:** Send the theorem to Nemotron-3 with prompt: "Break this theorem into a sequence of intermediate lemmas. For each lemma, write the Lean 4 statement."
    2. **Validate structure:** Check that the decomposition is logically sound — the final theorem should follow from the lemmas (ask Nemotron-3 to also generate the "assembly proof" that combines the lemmas).
    3. **Prove subgoals:** For each lemma, attempt a proof via Tactic Mode or Whole-Proof Mode independently.
    4. **Assemble:** If all subgoals are proved, combine into a single Lean file with the assembly proof. Verify the full file compiles.
    5. **Fallback:** If a subgoal fails after $N$ attempts, try an alternative decomposition (ask Nemotron-3 for a different proof strategy).
  - Implement **proof strategy selection:**
    - For each theorem, first ask Nemotron-3: "What proof strategies could work for this theorem? (e.g., induction, contradiction, direct construction, case analysis)"
    - Try the top 3 strategies in parallel (each as a separate decomposition).
    - Return the first one that produces a fully verified proof.
  - **Budget management:** Each proof attempt has an API call budget (default: 200 calls). Track spending and abort gracefully when budget is exhausted, reporting partial progress.

- **Week 6: LaTeX → Lean Autoformalization**
  - Implement the autoformalization pipeline:
    1. **Parse LaTeX:** Extract theorem statements from LaTeX source. Handle common patterns: `\begin{theorem}...\end{theorem}`, `\begin{lemma}...\end{lemma}`, inline `$...$` math.
    2. **Formalize:** Send the LaTeX theorem to Nemotron-3 with prompt: "Translate this mathematical theorem from LaTeX to a Lean 4 theorem statement. Use Mathlib conventions and imports."
    3. **Validate:** Check that the generated Lean statement compiles (even without a proof — just the `theorem` declaration with `sorry`). If it doesn't compile, iterate with error feedback.
    4. **Disambiguate:** For ambiguous notation (e.g., "$f$ is continuous" — continuous where? $\mathbb{R} \to \mathbb{R}$? $X \to Y$?), ask the user for clarification via the CLI/web interface.
  - Handle common challenges:
    - **Type inference:** LaTeX is untyped; Lean is typed. The formalizer must infer types (e.g., "$n \in \mathbb{N}$" → `(n : ℕ)`).
    - **Implicit arguments:** Lean uses implicit arguments extensively; the formalizer must match Mathlib conventions.
    - **Notation translation:** Map LaTeX commands to Lean/Mathlib notation (e.g., `\sum` → `∑`, `\int` → `∫`, `\lim` → `Filter.Tendsto`).
  - Build a notation mapping table covering the 200 most common LaTeX math commands and their Mathlib equivalents.

- **Week 7: Evaluation on Medium-Difficulty Benchmarks**
  - Evaluate on MiniF2F-valid (a subset of MiniF2F with 244 problems, covering AMC/AIME/Olympiad-level math).
  - Evaluate on a custom benchmark of 30 theorems from ML theory papers:
    - 10 from standard learning theory (VC dimension, Rademacher complexity bounds)
    - 10 from optimization (convergence of gradient descent, convexity results)
    - 10 from probability (concentration inequalities, martingale results)
  - Measure: pass rate, API calls per proof, wall-clock time, cost per proof (Nemotron-3 API pricing).
  - **Target:** ≥ 30% on MiniF2F-valid, ≥ 50% on the ML theory benchmark. If below, analyze failure modes and prioritize fixes.

### Pipeline Component Table — Weeks 4–7

| Component | Input | Output | Nemotron-3 Usage |
|---|---|---|---|
| **Autoformalization** | LaTeX theorem | Lean 4 `theorem` statement | 1–3 API calls (formalize + fix errors) |
| **Mathlib RAG** | Lean theorem statement | Top-$k$ relevant lemmas + imports | 0 (embedding model only) |
| **Decomposition** | Lean theorem statement | List of subgoal Lean statements + assembly proof | 1–2 API calls |
| **Tactic Generation** | Proof state (goals + hypotheses) | Next tactic | 1 API call per step |
| **Error Refinement** | Failed tactic + error message | Alternative tactic | 1 API call per retry |
| **Proof Report** | Verification result | Human-readable explanation | 1 API call |

---

## Weeks 8–10: Tree Search, Web UI & Evaluation

**Objective:** Implement advanced proof search (best-first tree search). Build a web UI for interactive use. Run comprehensive evaluations.

### Implementation Track

- **Week 8: Best-First Tree Search**
  - Upgrade from linear tactic generation to best-first tree search:
    1. At each proof state, generate $n$ candidate tactics (using Nemotron-3 with temperature $\tau = 0.7$).
    2. Execute each tactic via LeanDojo, producing $n$ child proof states.
    3. Score each child state using a heuristic:
       - **Goal complexity:** number of remaining goals, total number of hypotheses, estimated "difficulty" (ask Nemotron-3 to rate 1–10 how close the goal looks to being provable).
       - **Progress metric:** did the number of goals decrease? Did the goal term get simpler?
    4. Add child states to a priority queue (lower score = more promising).
    5. Pop the best state, expand it, repeat until `ProofFinished` or budget exhausted.
  - Implement search budget controls:
    - Maximum tree nodes: 500 (default).
    - Maximum depth: 30 tactics per branch.
    - Maximum API calls: 200 per theorem.
    - Time limit: 10 minutes per theorem.
  - **Backtracking:** When a branch fails (all tactics produce errors), backtrack to the parent state and try the next-best tactic.
  - Compare tree search vs. linear search on the MiniF2F-valid benchmark. Expect ≥ 10% improvement.

- **Week 9: Web Interface & Interactive Mode**
  - Build a web UI using Streamlit or Gradio:
    - **Input panel:** Text area for theorem statement (LaTeX or Lean). Option to upload a `.tex` file and select theorems.
    - **Progress panel:** Real-time display of proof search progress — current proof state, tactics being tried, tree search visualization.
    - **Output panel:** Verified proof (syntax-highlighted Lean code), or failure report with partial proof and remaining goals.
    - **Interactive mode:** User can intervene during proof search — provide hints ("try induction on n"), select which subgoal to focus on, approve/reject suggested tactics.
  - Build a CLI interface:
    ```
    # Prove a single theorem from LaTeX
    lean-prover prove "For all n : ℕ, n + 0 = n"
    
    # Prove from a Lean file
    lean-prover prove --file theorem.lean
    
    # Batch prove all theorems in a LaTeX file
    lean-prover batch --input paper.tex --output proofs/
    
    # Interactive mode
    lean-prover interactive
    ```
  - Implement proof export: save verified proofs as standalone `.lean` files with all necessary imports, ready to be included in a Lean project.

- **Week 10: Comprehensive Evaluation & Self-Application**
  - **Benchmark evaluation:**

    | Benchmark | Description | Target |
    |---|---|---|
    | Easy-50 (custom) | Propositional logic, basic algebra, list ops | ≥ 80% |
    | ML-Theory-30 (custom) | Learning theory, optimization, probability | ≥ 50% |
    | MiniF2F-valid | 244 AMC/AIME/Olympiad problems | ≥ 30% |
    | ProofNet (subset) | Undergraduate math | ≥ 20% |

  - **Self-application:** Use the tool to attempt proofs of representative theorems from active ML theory research:
    - State boundedness lemmas for recurrent architectures
    - Kernel characterization results for novel activation functions
    - Gating boundedness results for gated architectures
    - These are "real" theorems from active research — success here would be the strongest demonstration of the tool's value.
  - **Failure analysis:** For each failed proof attempt, categorize the failure mode:
    - Autoformalization error (wrong Lean statement)
    - Missing Mathlib lemma (the needed lemma isn't in Mathlib)
    - Proof strategy failure (wrong approach)
    - Tactic-level failure (right approach, wrong tactic)
    - API/budget exhaustion (right approach, but not enough attempts)
  - **Cost analysis:** Report the average cost per proved theorem (API calls × price per call). Target: ≤ $1 per proved theorem for Easy-50 level.

## Weeks 11–12: Technical Report & Presentation

**Objective:** Write the technical report, prepare the final presentation for the NVAITC Student Workshop, and complete code cleanup and documentation.

- **Technical Report:** Write the technical report (tool/demo format) describing the pipeline architecture and design decisions, reporting benchmark results, including case studies from self-application (attempting proofs from active ML theory research), and discussing failure modes and limitations honestly. Position as a *tool contribution*, not a model contribution.
- **Presentation:** Prepare the final presentation for the NVAITC Student Workshop.
- **Code Cleanup & Documentation:** Write comprehensive documentation (installation guide, tutorial, API reference, architecture guide). Clean up codebase, add unit tests for each pipeline component. Publish to GitHub with MIT license.

---

## Key Technical Challenges & Solutions

### Challenge 1: Nemotron-3's Lean 4 Knowledge

Nemotron-3 is trained on general code and math, not specifically on Lean 4. Its Lean knowledge may be limited compared to fine-tuned models like DeepSeek-Prover-V2.

**Solution — Prompt Engineering + RAG:**
- Use detailed system prompts with Lean 4 syntax reference, common tactic list, and Mathlib conventions.
- RAG retrieves relevant Mathlib lemmas and their proofs, giving Nemotron-3 concrete examples of correct Lean 4 style.
- Include a "Lean 4 cheat sheet" in every prompt: the 50 most common tactics (`simp`, `ring`, `omega`, `norm_num`, `exact`, `apply`, `intro`, `cases`, `induction`, `rw`, `have`, `let`, `constructor`, `ext`, `funext`, `calc`, etc.) with brief descriptions.

### Challenge 2: Autoformalization Accuracy

Converting LaTeX to Lean is ambiguous and error-prone. The same LaTeX can correspond to multiple valid Lean formalizations.

**Solution — Iterative Validation:**
- Generate the Lean statement, check it compiles (with `sorry` as proof), and iterate if it doesn't.
- Present the formalized statement back to the user for confirmation before attempting the proof.
- For ambiguous cases, generate multiple formalizations and let the user choose.

### Challenge 3: API Cost & Latency

Each proof attempt may require many API calls. Complex theorems could cost significant money.

**Solution — Budget Management + Caching:**
- Set per-theorem API call budgets with clear reporting.
- Cache all API responses (keyed by full prompt hash) to avoid redundant calls.
- Use Whole-Proof Mode first (1 call) before falling back to Tactic Mode (many calls).
- Implement local tactic pre-filtering: before calling the API, try common "free" tactics (`simp`, `ring`, `omega`, `norm_num`, `trivial`, `tauto`, `decide`) that don't need LLM guidance.

### Challenge 4: Lean 4 / Mathlib Version Churn

Mathlib4 is actively developed and breaks backward compatibility frequently.

**Solution — Pinned Versions + CI:**
- Pin Lean toolchain and Mathlib versions in `lean-toolchain` and `lakefile.lean`.
- Set up CI to test against the latest Mathlib monthly and update pinned version.
- Document the supported Lean/Mathlib version range.

---

## Evaluation Plan

### E1: Component-Level Evaluation

| ID | Component | Test | Metric | Target |
|---|---|---|---|---|
| E1.1 | **Tactic generation** | Given 100 proof states from Mathlib, generate tactics and check if they advance the proof | Tactic success rate | ≥ 40% |
| E1.2 | **Mathlib RAG** | For 50 theorems, check if the needed lemma appears in top-10 retrieval | Recall@10 | ≥ 60% |
| E1.3 | **Autoformalization** | Formalize 50 LaTeX theorems, check if the Lean statement compiles | Compilation rate | ≥ 80% |
| E1.4 | **Autoformalization semantic** | For the 50 theorems above, have a human judge check if the Lean statement matches the LaTeX meaning | Semantic accuracy | ≥ 70% |
| E1.5 | **Decomposition** | For 20 multi-step theorems, check if the decomposition is logically valid | Validity rate | ≥ 60% |

### E2: End-to-End Evaluation

| ID | Benchmark | Size | Difficulty | Target Pass Rate |
|---|---|---|---|---|
| E2.1 | Easy-50 | 50 | Undergraduate | ≥ 80% |
| E2.2 | ML-Theory-30 | 30 | Graduate ML theory | ≥ 50% |
| E2.3 | MiniF2F-valid | 244 | Competition math | ≥ 30% |
| E2.4 | Self-application | 4–8 | Research-level | Best effort |

### E3: Ablation Studies

| ID | Ablation | Measures |
|---|---|---|
| E3.1 | Remove RAG (no Mathlib retrieval) | Pass rate drop |
| E3.2 | Remove decomposition (Tactic/Whole-Proof only) | Pass rate on multi-step theorems |
| E3.3 | Remove error refinement (no retry on failure) | Pass rate, API calls per proof |
| E3.4 | Linear search vs. tree search | Pass rate, API calls per proof |
| E3.5 | Temperature sweep ($\tau \in \{0.0, 0.3, 0.7, 1.0\}$) | Pass rate vs. diversity |
| E3.6 | Best-of-$n$ sweep ($n \in \{1, 4, 8, 16, 32\}$) | Pass rate vs. cost |

### E4: Cost & Efficiency Analysis

| ID | Metric | Measurement |
|---|---|---|
| E4.1 | API calls per proved theorem (by difficulty tier) | Average, median, P95 |
| E4.2 | Wall-clock time per proved theorem | Average, median, P95 |
| E4.3 | Cost per proved theorem (at API pricing) | Average per tier |
| E4.4 | Proof length (tactics) vs. API calls | Correlation analysis |

---

## Timeline Summary

| Week | Phase | Implementation | Evaluation | Deliverable |
|---|---|---|---|---|
| 1 | Weeks 1–3 | Lean 4 + Mathlib setup, Nemotron-3 API client, LeanDojo integration | — | Working API → Lean pipeline |
| 2 | Weeks 1–3 | Tactic Mode: single-step proof loop, prompt templates, error refinement | — | First proofs generated and verified |
| 3 | Weeks 1–3 | Whole-Proof Mode, best-of-$n$ sampling, proof caching, prompt tuning | E2.1 (Easy-50 baseline) | **≥ 70% on Easy-50** |
| 4 | Weeks 4–7 | Mathlib RAG: embedding, vector DB, retrieval pipeline | E1.2 (RAG recall) | RAG integrated |
| 5 | Weeks 4–7 | Subgoal decomposition, proof strategy selection, budget management | E1.5 (decomposition validity) | Decomposition Mode working |
| 6 | Weeks 4–7 | LaTeX → Lean autoformalization, notation mapping | E1.3, E1.4 (formalization accuracy) | End-to-end LaTeX → proof |
| 7 | Weeks 4–7 | Integration testing, pipeline optimization | E2.2, E2.3 (ML-Theory, MiniF2F) | **≥ 30% on MiniF2F** |
| 8 | Weeks 8–10 | Best-first tree search, backtracking, search budget controls | E3.4 (tree vs. linear) | Tree search integrated |
| 9 | Weeks 8–10 | Web UI (Streamlit/Gradio), CLI interface, proof export | — | Interactive tool ready |
| 10 | Weeks 8–10 | Comprehensive evaluation, self-application, failure analysis | E2.1–E2.4, E3.1–E3.6, E4.1–E4.4 | Full benchmark results |
| 11 | Weeks 11–12 | Technical report, presentation prep | — | Report draft |
| 12 | Weeks 11–12 | Documentation, tests, GitHub release | — | **Technical report & presentation** |

**Exit criteria per phase:**
- **Weeks 1–3:** Tactic Mode and Whole-Proof Mode working. ≥ 70% on Easy-50. Prompt templates tuned.
- **Weeks 4–7:** RAG, decomposition, and autoformalization integrated. ≥ 30% on MiniF2F-valid. End-to-end LaTeX → verified proof demonstrated.
- **Weeks 8–10:** Tree search implemented. Web UI and CLI ready. All evaluations complete.
- **Weeks 11–12:** Technical report finalized. Presentation prepared for NVAITC Student Workshop. GitHub release and documentation complete.

---

## Stretch Goals (If Ahead of Schedule)

| Goal | Description | Value |
|---|---|---|
| **Proof explanation** | After finding a verified proof, use Nemotron-3 to generate a natural language explanation of the proof strategy | Makes the tool useful for learning, not just verification |
| **Proof repair** | Given a *partially correct* proof (from a human), identify the broken step and fix it | Common use case: "my proof compiles except for this one `sorry`" |
| **Multi-model backend** | Support swapping Nemotron-3 for other APIs (GPT-4, Claude, DeepSeek) via a unified interface | Lets users choose cost/quality tradeoff |
| **VS Code extension** | Integrate as a Lean 4 VS Code extension that suggests tactics inline | Most natural UX for Lean users |
| **Theorem dependency graph** | For a full paper, extract all theorems and their dependency order, prove in topological order | Enables "verify my entire paper" workflow |
