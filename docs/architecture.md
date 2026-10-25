# XAI-RAG System Architecture

## Overview
XAI-RAG is a decision-auditing and explainability platform designed for enterprise credit risk systems. It addresses the enterprise ML "last-mile gap" where tabular models generate risk scores and SHAP provides local attributions, but analysts lack verified policy grounding, feasible remediation paths, and tamper-evident auditability.

```mermaid
flowchart TD
    subgraph Inputs
        A[Applicant Data] --> B[Input Validation & Derivation]
    end

    subgraph PillarA [Pillar A: Deterministic Scoring & SHAP]
        B --> C[XGBoost Predictor]
        C --> D[TreeSHAP Explainer]
        D --> E[Additivity & Output-Space Check]
    end

    subgraph PillarB [Pillar B: Constrained Counterfactuals]
        C --> F[DiCE Optimizer / Genetic Engine]
        F --> G[Primitive Action Levers Perturbation]
        G --> H[Derived Feature Recalculation]
        H --> I[Immutability & Domain Validator]
        I --> J[Direct Production Re-Scorer]
    end

    subgraph PillarC [Pillar C: Ontology Bridge - Part 2]
        D --> K[SHAP-to-Query Ontology Translator]
    end

    subgraph PillarD [Pillar D: Retrieval & Grounding - Part 2]
        K --> L[FAISS Dense + BM25 Lexical Retrieval]
        L --> M[Structured Report Generator]
        M --> N[Citation & Quote Verifier]
        N --> O[Tamper-Evident Audit Ledger]
    end

    subgraph Presentation [Presentation - Part 3]
        E --> P[Analyst Dashboard UI]
        J --> P
        O --> P
        P --> Q[PDF / JSON Dossier Export]
    end
```

## Core Architectural Principles

1. **Separation of Concerns & Single Source of Truth**:
   - The **XGBoost model** is the sole source of truth for predictions.
   - **SHAP** is the sole source of truth for feature attributions.
   - The **Policy Corpus** is the sole source of truth for regulations and guidelines.
   - The **LLM** (introduced in Part 2) acts strictly as a synthesis engine over verified evidence; it is never allowed to fabricate scores, probabilities, laws, citations, or counterfactuals.

2. **Mathematical Fidelity (P0 Gate)**:
   - All explanations and predictions must declare and respect their coordinate space (`log_odds` margin vs `probability`).
   - Additivity checks verify $base\_value + \sum \phi_i = raw\_margin$ in log-odds, and its calibrated or sigmoid probability equivalent.
   - Cross-space comparisons are rejected.

3. **Counterfactual Integrity (P0 Gate)**:
   - Candidates must never assume the optimizer's objective value; every candidate is directly re-evaluated using the exact production inference pipeline.
   - Immutable attributes (`AGE`, `PAST_DELINQUENCIES`, `CREDIT_BUREAU_SCORE`, `DAYS_EMPLOYED`, `AMT_INCOME_TOTAL`) cannot change to manufacture approvals.
   - Derived ratios (`FOIR_RATIO`, `LTV_RATIO`) are never perturbed independently; they are recomputed from updated primitives before scoring.

4. **Strict Typed Contracts**:
   - Every module interacts solely through versioned, strictly validated Pydantic v2 schemas forbidding unexpected fields (`extra='forbid'`).
