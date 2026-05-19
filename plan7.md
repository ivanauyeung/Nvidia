# Project Plan: Geometry and Manifold Analysis of LLM Representations — Survey & Knowledge Graph

**Duration:** 12 Weeks (10 weeks project + 2 weeks technical report & presentation)  
**Goal:** Produce a comprehensive survey paper reviewing geometric and manifold-based methods for analyzing LLM internal representations, including a taxonomy, identification of open problems, and an interactive knowledge graph mapping the theorem–field–author landscape of geometric LLM analysis.  
**Output:** Technical report + presentation for NVAITC Student Workshop + interactive knowledge graph explorer by end of Week 12.  
**Outputs:** (1) Technical report (25–35 pages). (2) Publication-ready knowledge graph figure (PDF/SVG). (3) Interactive web-based knowledge graph explorer (HTML, hosted on GitHub Pages). (4) Structured graph dataset (JSON/CSV) for community reuse.

---

## Motivation & Scope

The internal representations of LLMs (token embeddings, hidden states, attention outputs) live in high-dimensional spaces that carry rich geometric structure. A rapidly growing literature (2024–2026) studies this structure through diverse lenses: intrinsic dimension estimation, manifold capacity theory, Riemannian curvature, topological data analysis, and dimensionality reduction for visualization. However, these approaches are fragmented across communities (NLP, differential geometry, information theory, neuroscience-inspired AI) with little cross-referencing or standardized comparison.

**Why add a knowledge graph?** The geometric LLM analysis landscape is itself a *graph*: theorems depend on earlier theorems, methods belong to mathematical fields, authors bridge subfields. No existing resource maps these connections. A well-designed knowledge graph reveals hidden structure — which clusters of theory are mature, which are isolated, and where the frontier lies. This graph becomes the survey's centerpiece figure, giving readers an instant visual overview of the entire field.

**What this project contributes:**

1. **Taxonomy:** A structured classification of geometric methods applied to LLM representations, organized by the mathematical property studied and the LLM phenomenon analyzed.
2. **Knowledge Graph:** An interactive theorem–field–author map of the geometric LLM analysis landscape, with a publication-ready static figure and an interactive web explorer.
3. **Gap Analysis:** Open problems identified both from the taxonomy (underexplored cells) and from the knowledge graph (disconnected clusters, missing bridges).
4. **Practical Guide:** Recommendations for practitioners on which geometric tools to use for specific questions about LLM behavior.

**What this project does NOT do:** It does not propose new theorems or novel geometric methods. Original contributions are limited to the taxonomy, the knowledge graph, and the gap analysis.

---

## Weeks 1–3: Literature Collection, Taxonomy & Knowledge Graph Data

**Objective:** Systematically collect, read, and annotate the full body of relevant literature. Develop the taxonomy. Build the knowledge graph dataset in parallel.

### Implementation Track

- **Week 1: Seed Paper Collection, Database Setup & Graph Schema**

  - Set up a reference management system (Zotero or BibTeX database) for all papers.
  - Collect the seed papers already identified:

    | ID | Paper | Year | Core Method | Key Contribution |
    |----|-------|------|-------------|-----------------|
    | S1 | "Origins of Representation Manifolds in LLMs" (arXiv:2505.18235) | 2025 | Manifold structure | How representations organize as manifolds across layers |
    | S2 | "Emergent Manifold Separability during Reasoning" (arXiv:2602.20338) | 2026 | Manifold separability | Geometric signatures of chain-of-thought reasoning |
    | S3 | "Visualizing LLM Latent Space Geometry Through Dimensionality Reduction" (arXiv:2511.21594) | 2025 | Visualization methods | Systematic comparison of PCA, t-SNE, UMAP for LLM states |
    | S4 | "Measuring Intrinsic Dimension of Token Embeddings" (arXiv:2503.02142) | 2025 | Intrinsic dimension | MLE and TwoNN estimators for LLM embedding dimension |
    | S5 | "Multi-Scale Manifold Alignment for Interpreting LLMs" (arXiv:2505.20333) | 2025 | Manifold alignment | Cross-layer and cross-model manifold correspondence |
    | S6 | "Shape Happens: Automatic Feature Manifold Discovery" (arXiv:2510.01025) | 2025 | Feature manifolds | Automatic discovery of concept manifolds in LLM representations |
    | S7 | "Shared Global and Local Geometry of Language Model Embeddings" (arXiv:2503.21073) | 2025 | Embedding geometry | Global vs. local geometric structure of embedding spaces |
    | S8 | "The Confidence Manifold" (arXiv:2602.08159) | 2026 | Confidence geometry | Geometric structure of model confidence in representation space |

  - **Define the knowledge graph schema:**

    | Node Type | Attributes |
    |-----------|-----------|
    | **Theorem/Result** | Name, abbreviated statement, year, source paper, arXiv ID, proof technique |
    | **Field/Subfield** | Name, parent field, description |
    | **Author** | Name, affiliation, key contributions |

    | Edge Type | From → To | Meaning |
    |-----------|-----------|---------|
    | `depends_on` | Theorem → Theorem | Proof of A uses result B |
    | `extends` | Theorem → Theorem | A generalizes or strengthens B |
    | `belongs_to` | Theorem → Field | A is a result within field F |
    | `authored_by` | Theorem → Author | A was proved/introduced by author P |
    | `bridges` | Field → Field | Shared theorems/tools connect two fields |
    | `applied_to` | Theorem → LLM Domain | Result A applied to domain D (e.g., attention analysis, embedding geometry, hallucination detection) |

  - Store graph data in structured JSON with a parallel CSV for easy editing.
  - Perform forward and backward citation searches on each seed paper.
  - Target: **40–60 papers** in the initial collection.

- **Week 2: Extended Literature Search & Classical Foundations Graph**

  Expand the search beyond seed papers and simultaneously populate the knowledge graph with foundational results:

  | Category | Search Terms | Expected Papers |
  |----------|-------------|-----------------|
  | **Intrinsic Dimension** | LLM intrinsic dimension, embedding manifold dimension, MLE dimension estimation neural network | 8–12 |
  | **Manifold Capacity & Linear Representation** | manifold capacity theory LLM, linear representation hypothesis, concept manifold neural network | 8–12 |
  | **Curvature & Riemannian Geometry** | Riemannian curvature neural network representations, sectional curvature deep learning, Ricci curvature LLM | 5–8 |
  | **Topological Data Analysis** | persistent homology LLM, topological data analysis transformer, Betti numbers neural network | 5–8 |
  | **Visualization & Dimensionality Reduction** | t-SNE UMAP LLM representations, dimensionality reduction transformer hidden states, diffusion map embeddings | 5–8 |
  | **Positional & Structural Geometry** | positional embedding geometry, attention head geometry, rotary embedding manifold | 5–8 |
  | **Geometric Interpretability** | probing geometry LLM, geometric structure factual knowledge, representation engineering geometry | 5–8 |

  **Knowledge graph: classical foundations layer.** Populate the graph with foundational theorems that the LLM geometry papers build upon:

  | Field | Key Theorems/Results | Key Authors |
  |-------|---------------------|-------------|
  | **Intrinsic Dimension Estimation** | MLE estimator (Levina & Bickel 2004), TwoNN (Facco et al. 2017), correlation dimension (Grassberger & Procaccia 1983) | Levina, Bickel, Facco, Laio |
  | **Manifold Capacity** | Perceptron capacity (Cover 1965), manifold capacity theory (Chung et al. 2018), classification capacity (Cohen et al. 2020) | Cover, Chung, Cohen, Sompolinsky |
  | **Riemannian Geometry** | Ollivier-Ricci curvature (Ollivier 2009), sectional curvature estimation, manifold learning (Belkin & Niyogi 2003) | Ollivier, Belkin, Niyogi |
  | **Information Geometry** | Fisher information metric (Amari 1985), natural gradient, statistical manifolds | Amari, Rao |
  | **Topological Data Analysis** | Persistent homology (Edelsbrunner et al. 2000), stability theorem (Cohen-Steiner et al. 2007) | Edelsbrunner, Harer, Carlsson |
  | **Dimensionality Reduction** | t-SNE (van der Maaten 2008), UMAP (McInnes 2018), diffusion maps (Coifman & Lafon 2006) | van der Maaten, McInnes, Coifman |
  | **Transformer Theory** | Token collapse/clustering (Rigollet 2025), attention kernel analysis, NTK for attention | Rigollet, Jacot |

  - Read abstracts and introductions of all collected papers.
  - Maintain a reading log with: paper ID, title, primary geometric tool, LLM studied, key finding, limitations.

- **Week 3: Deep Reading, Annotation & Graph Population (Batch 1)**

  - Deep read the **20 most important papers** (seed papers + highest-cited extensions).
  - For each paper, create a structured annotation:
    1. **Mathematical framework:** What geometric/topological objects are defined? What space do they live in?
    2. **Methodology:** How are geometric quantities estimated from finite data? What are the statistical assumptions?
    3. **Models studied:** Which LLMs? What scale? Pre-trained or fine-tuned?
    4. **Key results:** What geometric properties were discovered? What do they explain about LLM behavior?
    5. **Limitations:** What assumptions are unrealistic? What models/scales were not tested?
    6. **Connections to other papers:** Does this paper build on, contradict, or complement other work?
  - **While reading, extract graph data:** For each paper, add its key theorem/result nodes, author nodes, and edges (`depends_on`, `extends`, `belongs_to`, `authored_by`) to the knowledge graph dataset.

- **Week 3: Taxonomy Design, Batch 2 Reading & Graph Assembly**

  - **Design the taxonomy.** Organize methods along two axes:

    **Axis 1 — What geometric property is studied:**

    | Property | Definition | Example Methods |
    |----------|-----------|-----------------|
    | Intrinsic Dimension (ID) | Number of degrees of freedom in representation manifold | MLE, TwoNN, correlation dimension |
    | Manifold Capacity | Maximum number of linearly separable object manifolds | Mean-field theory, perceptron capacity |
    | Curvature | Local bending of representation manifold | Sectional curvature, Ricci curvature, scalar curvature |
    | Topology | Global connectivity and holes in representation space | Persistent homology, Betti numbers, Euler characteristic |
    | Distance & Metric | Pairwise structure of representations | Wasserstein distance, geodesic distance, CKA |
    | Visualization | Low-dimensional projections for human interpretation | PCA, t-SNE, UMAP, diffusion maps, LDA |

    **Axis 2 — What LLM phenomenon is analyzed:**

    | Phenomenon | Examples |
    |-----------|---------|
    | Layer-wise evolution | How geometry changes across depth |
    | Training dynamics | How geometry evolves during pre-training/fine-tuning |
    | Factual knowledge | Geometric signatures of stored facts |
    | Reasoning | Geometric changes during chain-of-thought |
    | Hallucination | Geometric markers of unreliable outputs |
    | Scaling | How geometry changes with model size |
    | Cross-model comparison | Geometric similarities/differences across architectures |

  - Deep read the remaining **15–20 papers** (Batch 2), completing annotations and graph data extraction.
  - **Assemble the full knowledge graph:**
    - Target: **80+ theorem/result nodes**, **15+ field nodes**, **60+ author nodes**, **200+ edges**.
    - Validate: no orphan nodes, `depends_on` edges form a DAG, every theorem belongs to at least one field.
    - Compute graph statistics: degree distribution, betweenness centrality, connected components.

### Weeks 1–3 Deliverables

- [ ] Complete reading database with 40–60 annotated papers
- [ ] Finalized two-axis taxonomy (Property × Phenomenon)
- [ ] Literature summary table mapping every paper to its taxonomy cell
- [ ] **Knowledge graph dataset (JSON/CSV):** 80+ theorem nodes, 60+ author nodes, 200+ edges
- [ ] Preliminary outline of survey paper sections

---

## Weeks 4–7: Knowledge Graph Visualization & Survey Writing

**Objective:** Build the knowledge graph visualizations and write the main body of the survey.

### Implementation Track

- **Week 4: Graph Layout Design**

  - **Knowledge graph layout design:**
    - Evaluate layout algorithms:

      | Algorithm | Strengths | Use Case |
      |-----------|----------|----------|
      | Force-directed (ForceAtlas2) | Natural clustering | Interactive explorer |
      | Hierarchical (Sugiyama) | Clear dependency flow | Theorem dependency DAG |
      | Timeline × Field grid | Time + field grouping | Static publication figure |

    - Design static figure layout:
      - **Horizontal axis:** Time (left = classical foundations, right = 2026 frontier).
      - **Vertical grouping:** Mathematical field (ID estimation, manifold capacity, curvature, topology, visualization, information geometry).
      - **Node encoding:** Shape by type (circle = theorem, diamond = field, square = author). Size by citation count or dependent-theorem count. Color by field.
      - **Edge encoding:** Solid = `depends_on`/`extends`. Dashed = `bridges`. Thickness = connection strength.
    - Choose tools: `networkx` + `matplotlib`/`graphviz` for static; D3.js for interactive.

- **Week 5: Static Knowledge Graph Figure**

  - **Build static knowledge graph figure:**
    1. Load graph from JSON.
    2. Apply timeline × field grid layout.
    3. Render nodes with field-based coloring and size scaling.
    4. Render edges with type-based styling (solid/dashed/thickness).
    5. Add labels for high-centrality theorem and author nodes.
    6. Add legend, title, field boundary boxes, and callout boxes for landmark results.
    7. Produce multiple views:
       - **Full landscape:** All theorems, fields, and connections.
       - **Field zoom-ins:** One sub-figure per field showing internal structure.
    8. Export as PDF (vector) and high-resolution PNG (target: A0 poster resolution for supplementary).

- **Week 6: Interactive Explorer & Figure Generation**

  - Generate publication-quality figures:
    - Taxonomy overview figure (the two-axis table as a visual)
    - Knowledge graph figures (full landscape + field zoom-ins)

  - **Build interactive knowledge graph explorer (D3.js):**
    - Force-directed layout with zoom and pan.
    - Click theorem node → show statement, authors, year, dependencies, arXiv link.
    - Click field node → highlight all theorems in that field.
    - Click author node → highlight all their contributions.
    - Search bar: type a theorem name, author, or keyword to filter.
    - Filter toggles: show/hide by field, by year range, by LLM application domain.
    - Guided tour mode: click through a pre-defined path telling the story of how geometric LLM analysis evolved.
    - Clean, modern design with dark theme option.
    - Embed graph data as JSON within the HTML (single self-contained file).

- **Week 7: Graph Analytics & Survey Body Writing**

  - **Compute knowledge graph analytics:**

    | Metric | What it Reveals |
    |--------|----------------|
    | **Betweenness centrality** | "Bridge" theorems connecting distant fields (e.g., covering number arguments bridging ID estimation and manifold capacity) |
    | **PageRank** | Most influential foundational results that many LLM geometry papers depend on |
    | **Community detection** (Louvain) | Natural clusters that may not align with traditional field boundaries |
    | **Temporal density** | Which subfields are accelerating (more results per year recently) vs. plateauing |
    | **White space analysis** | Field pairs with few `bridges` edges = unexplored connections = research opportunities |

  - Write the main sections of the paper:

    | Section | Content | Pages |
    |---------|---------|-------|
    | 1. Introduction | Motivation, scope, contribution summary | 1.5 |
    | 2. Background | Notation, LLM architecture recap, key geometric definitions | 2 |
    | 3. Knowledge Graph & Field Landscape | Present the knowledge graph figure; describe the field's structure, key authors, theorem dependencies, temporal evolution | 3 |
    | 4. Taxonomy of Geometric Methods | Two-axis taxonomy with descriptions of each cell | 3 |
    | 5. Intrinsic Dimension Methods | MLE, TwoNN, correlation dimension applied to LLMs | 3 |
    | 6. Manifold Capacity & Linear Representations | Capacity theory, linear representation hypothesis, concept manifolds | 2.5 |
    | 7. Curvature & Riemannian Analysis | Sectional/Ricci/scalar curvature of representation manifolds | 2.5 |
    | 8. Topological Analysis | Persistent homology, Betti numbers, knowledge structure topology | 2 |
    | 9. Visualization & Dimensionality Reduction | Critical comparison of PCA, t-SNE, UMAP, LDA, diffusion maps | 2.5 |
    | 10. Graph Analytics & White Space Analysis | Centrality results, community detection, unexplored connections | 2 |
    | 11. Open Problems & Future Directions | Gap analysis from taxonomy and graph structure | 2 |
    | 12. Conclusion | Summary and recommendations | 1 |

### Weeks 4–7 Deliverables

- [ ] **Static knowledge graph figure (PDF/SVG):** publication-ready, A0 poster + paper-embedded version
- [ ] **Interactive knowledge graph explorer (HTML):** working prototype with search, filter, guided tour
- [ ] Publication-quality taxonomy and knowledge graph figures
- [ ] First draft of Sections 1–9

---

## Weeks 8–10: Gap Analysis & Graph Analytics

**Objective:** Complete the gap analysis (informed by both taxonomy and graph analytics) and polish all outputs.

### Implementation Track

- **Week 8: Gap Analysis (Taxonomy + Graph-Informed) & Explorer Polish**

  Systematically identify gaps from two complementary sources:

  **Taxonomy-derived gaps (underexplored cells):**

  | Gap | Description | Why It Matters |
  |-----|------------|----------------|
  | **Scale-dependent geometry** | Most papers study ≤7B models. How do geometric properties change at 70B+? | Critical for understanding whether findings transfer to frontier models |
  | **Training dynamics geometry** | Few papers track manifold structure evolution during pre-training checkpoints | Would connect geometric analysis to learning theory |
  | **Fine-tuning geometry** | How does LoRA/full fine-tuning alter manifold structure? ID changes? Curvature? | Practical implications for efficient adaptation |
  | **Cross-architecture comparison** | No unified geometric comparison of Transformer vs. Mamba vs. MoE representations | Could explain architectural advantages geometrically |
  | **Curvature–performance connection** | Curvature is measured but rarely connected to downstream task performance | Missing link between geometric description and utility |
  | **Causal vs. correlational** | Most findings are correlational. Causal interventions on geometry are rare | Needed for geometric analysis to become a design tool |
  | **Computational cost** | Many methods (persistent homology, geodesic estimation) do not scale to large models | Practical barrier to adoption |
  | **Geometric analysis of generation** | Most work analyzes encoder representations; autoregressive generation geometry is understudied | Relevant to understanding hallucination, creativity, diversity |

  **Graph-derived gaps (structural holes):**

  | Gap | Graph Signal | Research Opportunity |
  |-----|-------------|---------------------|
  | **Disconnected clusters** | Two field communities with no `bridges` edges between them | A paper connecting these two approaches would be highly novel |
  | **Missing dependency chains** | Recent results that cite no classical foundations | Either the foundations are implicit or genuinely missing |
  | **Author silos** | Author clusters with no co-authorship or co-citation across fields | Collaboration opportunities |
  | **Temporal dead zones** | Fields with no new results in 2025–2026 despite open problems | Ripe for revisitation with modern LLMs |

  - Write Sections 10 (Graph Analytics & White Space) and 11 (Open Problems).
  - **Polish interactive explorer:**
    - Performance optimization for smooth interaction with 200+ nodes.
    - Add guided tour annotations for the 5 most interesting graph insights.
    - Mobile-responsive layout.

- **Week 9: Complete First Draft & Internal Review**

  - Write remaining sections (12: Conclusion).
  - Assemble full draft with all figures, tables, and the knowledge graph figure.
  - Create the bibliography (expected: 80–120 references).
  - Internal review:
    - Check every claim against source papers
    - Ensure taxonomy is exhaustive
    - Verify knowledge graph data accuracy (spot-check 20 random edges)
    - Check notation consistency across sections

- **Week 10: Project Wrap-Up**

  - Finalize all project work.
  - **Deliverable:** All project work complete; full manuscript draft; gap analysis; explorer polished.

---

## Weeks 11–12: Technical Report & Presentation

**Objective:** Write the technical report, prepare the final presentation for the NVAITC Student Workshop, and complete deployment and release.

- **Technical Report Writing**
  - Revise draft based on internal review.
  - Add a **"Practitioner's Guide" appendix:** a decision tree for "I want to understand X about my LLM → use geometric method Y."
  - **Prepare knowledge graph dataset for community release:**
    - `nodes.csv`: All theorem, field, and author nodes with attributes.
    - `edges.csv`: All edges with types and metadata.
    - `graph.json`: Full graph in standard format (compatible with D3.js, Cytoscape, NetworkX).
    - `schema.md`: Documentation of schema, node types, edge types, attribute definitions.
    - `README.md`: How to load, explore, and contribute to the graph.
  - Ensure figures are colorblind-friendly and readable at single-column width.
  - **Deliverable:** Complete technical report.

- **Final Presentation for NVAITC Student Workshop**
  - Prepare slides summarizing the project: motivation, taxonomy, knowledge graph, key findings, gap analysis, and recommendations.
  - **Deliverable:** Presentation ready for NVAITC Student Workshop.

- **Deployment & Release**
  - Final proofread and formatting (LaTeX).
  - **Deploy interactive explorer to GitHub Pages.**
  - Prepare supplementary materials:
    - Graph generation scripts
    - Extended tables with per-paper taxonomy classification
    - Knowledge graph dataset files
  - **Deliverable:** Technical report + deployed explorer + knowledge graph dataset.

---

## Deliverable Dependency Map

| Deliverable | Month | Depends On | Enables |
|-------------|-------|------------|---------|
| **D1: Reading database (40–60 papers)** | 1 | Seed papers, citation search | D2, D3, D6 |
| **D2: Taxonomy (Property × Phenomenon)** | 1 | D1 | D6, D8 |
| **D3: Knowledge graph dataset (JSON/CSV)** | 1 | D1, schema design | D4, D5, D7, D9 |
| **D4: Static knowledge graph figure (PDF/SVG)** | 2 | D3, layout design | D6, D9 |
| **D5: Interactive knowledge graph explorer (HTML)** | 2 | D3, D3.js implementation | D9, D10 |
| **D6: Survey body (Sections 1–9)** | 2 | D1, D2, D4 | D8 |
| **D7: Graph analytics (centrality, communities, white space)** | 3 | D3 | D8 |
| **D8: Gap analysis & open problems** | 3 | D2, D7 | D6 (Sections 10–11) |
| **D9: Final manuscript** | 3 | D4, D5, D6, D7, D8 | Final deliverable |
| **D10: GitHub Pages deployment** | 3 | D5 | Public release |

---

## Quality Checks

### Q1: Knowledge Graph Quality

| ID | Check | Target | Pass Criterion |
|----|-------|--------|----------------|
| Q1.1 | Total theorem/result nodes | ≥ 80 | Covers all major geometric LLM analysis results |
| Q1.2 | Total field nodes | ≥ 15 | All fields from the taxonomy represented |
| Q1.3 | Total author nodes | ≥ 60 | Key contributors in each field included |
| Q1.4 | Total edges | ≥ 200 | Dense enough to reveal non-obvious connections |
| Q1.5 | Orphan nodes | 0 | Every node connected to at least one other |
| Q1.6 | DAG validation | No cycles in `depends_on` | Theorem dependencies are acyclic |
| Q1.7 | Spot-check accuracy | 20 random edges verified against source papers | ≥ 95% accuracy |

### Q2: Visualization Quality

| ID | Check | Pass Criterion |
|----|-------|----------------|
| Q2.1 | Static figure label readability | No overlapping labels at A0 print resolution |
| Q2.2 | Color distinguishability | Colorblind-safe palette (≥ 8 distinguishable colors, WCAG compliant) |
| Q2.3 | Interactive explorer performance | ≥ 30 FPS with 200+ nodes on standard laptop |
| Q2.4 | Search/filter usability | A colleague can find a specific theorem in < 10 seconds |

### Q3: Graph Analytics Validation

| ID | Analysis | Validation |
|----|----------|------------|
| Q3.1 | Top-10 central theorems by PageRank | Sanity check: foundational results (MLE, Cover's theorem, persistence stability) should rank high |
| Q3.2 | Community detection (Louvain) | Communities should roughly align with field boundaries; surprising merges indicate genuine connections |
| Q3.3 | White space analysis | Identified opportunities verifiable by literature search (no existing work connects the two fields) |
| Q3.4 | Temporal density | Growth of LLM-specific geometry papers should spike in 2024–2026 |

---

## Paper Structure Map

| Section | Content | Key Figures/Tables |
|---------|---------|-------------------|
| 1. Introduction | Motivation, scope | Figure: growth of LLM geometry papers over time |
| 2. Background | Definitions, notation | Table: notation reference |
| 3. Knowledge Graph & Field Landscape | **Knowledge graph figure**, field structure, temporal evolution, key bridges | **Figure: full knowledge graph (centerpiece)**, field zoom-ins |
| 4. Taxonomy | Two-axis classification | Figure: taxonomy grid (Property × Phenomenon) |
| 5. Intrinsic Dimension | MLE, TwoNN, correlation dim | Figure: layer-wise ID curves |
| 6. Manifold Capacity | Capacity theory, linear rep hypothesis | Figure: capacity vs. layer depth |
| 7. Curvature | Sectional, Ricci, scalar | Figure: curvature evolution |
| 8. Topology | Persistent homology | Figure: persistence barcodes |
| 9. Visualization | PCA, t-SNE, UMAP, LDA, diffusion maps | Figure: 6-panel visualization comparison |
| 10. Graph Analytics & White Space | Centrality, communities, unexplored connections | Figure: community detection overlay, white space map |
| 11. Open Problems | Gap analysis from taxonomy + graph | Table: gaps with research directions |
| 12. Conclusion | Summary, recommendations | — |
| Appendix A | Practitioner's decision tree | Figure: flowchart |
| Appendix B | Knowledge graph dataset documentation | Schema reference, contribution guide |

---

## Timeline Summary

| Week | Phase | Activity | Deliverable |
|------|-------|----------|-------------|
| 1 | Weeks 1–3 | Seed paper collection, database setup, graph schema design | Paper database with 40+ entries, **D0: graph schema** |
| 2 | Weeks 1–3 | Extended search by category, classical foundations graph population | Complete literature collection (50–60 papers), classical graph layer |
| 3 | Weeks 1–3 | Deep reading, taxonomy design, Batch 2 reading, graph assembly + validation | **D2: Taxonomy**, **D3: knowledge graph dataset** |
| 4 | Weeks 4–7 | Knowledge graph layout design | Layout prototype |
| 5 | Weeks 4–7 | Build static knowledge graph figure | **D4: static figure** |
| 6 | Weeks 4–7 | Interactive explorer build, figure generation | **D5: interactive explorer**, publication figures |
| 7 | Weeks 4–7 | Graph analytics, write survey body (Sections 1–9) | **D7: graph analytics**, first draft of main sections |
| 8 | Weeks 8–10 | Gap analysis (taxonomy + graph), explorer polish | **D8: open problems** identified |
| 9 | Weeks 8–10 | Complete full draft, internal review | Full manuscript draft |
| 10 | Weeks 8–10 | Project wrap-up | All project work complete |
| 11 | Weeks 11–12 | Technical report writing, presentation prep | Technical report draft, presentation |
| 12 | Weeks 11–12 | Final formatting, GitHub Pages deploy | **D9: Technical report**, **D10: live explorer**, knowledge graph dataset |

**Exit criteria per phase:**
- **Weeks 1–3:** 50+ papers read and annotated. Taxonomy finalized. Knowledge graph dataset assembled with 80+ theorem nodes and 200+ edges. Survey outline approved.
- **Weeks 4–7:** Static knowledge graph figure publication-ready. Interactive explorer functional. First draft of core sections complete.
- **Weeks 8–10:** Gap analysis identifies at least 5 concrete future directions from taxonomy and 3 from graph structure. Full manuscript draft complete.
- **Weeks 11–12:** Technical report complete. Presentation ready for NVAITC Student Workshop. Interactive explorer deployed on GitHub Pages. Knowledge graph dataset public.

---

## Key Tools & Libraries

| Tool | Purpose |
|------|---------|
| `networkx` | Graph construction, analytics (centrality, community detection, PageRank) |
| `graphviz` / `pygraphviz` | Hierarchical layout for dependency DAG |
| `matplotlib` | Static figure rendering |
| D3.js | Interactive force-directed graph visualization |
| Sigma.js | WebGL fallback for large graphs |
| `pandas` | Dataset management (CSV import/export) |
| GitHub Pages | Interactive explorer hosting |

---

## Key References

### Core Seed Papers
| ID | Title | arXiv | Year |
|----|-------|-------|------|
| S1 | Origins of Representation Manifolds in LLMs | 2505.18235 | 2025 |
| S2 | Emergent Manifold Separability during Reasoning in LLMs | 2602.20338 | 2026 |
| S3 | Visualizing LLM Latent Space Geometry Through Dimensionality Reduction | 2511.21594 | 2025 |
| S4 | Measuring Intrinsic Dimension of Token Embeddings | 2503.02142 | 2025 |
| S5 | Multi-Scale Manifold Alignment for Interpreting LLMs | 2505.20333 | 2025 |
| S6 | Shape Happens: Automatic Feature Manifold Discovery in LLMs | 2510.01025 | 2025 |
| S7 | Shared Global and Local Geometry of Language Model Embeddings | 2503.21073 | 2025 |
| S8 | The Confidence Manifold | 2602.08159 | 2026 |

### Background / Theory Papers
- Manifold capacity: Cover (1965), Chung et al. (2018), Cohen et al. (2020)
- Riemannian geometry: Ollivier (2009), Belkin & Niyogi (2003), Amari (1985)
- Linear representation hypothesis: Park et al. (2024), Nanda et al. (2023)
- Rigollet's token clustering: arXiv:2512.01868
- Knowledge graph visualization: Bronstein et al. (2021) "Geometric Deep Learning" survey
