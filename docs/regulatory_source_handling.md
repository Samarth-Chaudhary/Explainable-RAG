# Regulatory Source Handling & Legal Grounding

## Non-Negotiable Grounding Mandates
1. **Never Invent Law**: The system must never cite or claim a regulatory requirement unless the authoritative clause is present in the document corpus and verified.
2. **Provenance & Attribution**:
   - Every citation must preserve source ID, issuer, jurisdiction, document version, publication date, effective date, page number, and section reference.
   - Text quotes must match verbatim against ingested chunk content through exact or normalized substring verification.
3. **Illustrative Scenarios vs Statutory Law**:
   - Scenario references (such as Scenario A or Scenario B mentions of RBI guidelines, FOIR limits, or tenor extensions) are synthetic demonstration fixtures.
   - Real compliance reports in configured mode require authentic primary sources ingested into the source registry.
4. **Human Review Requirement**:
   - The platform is a decision support tool; it does not possess legal authority to grant, reject, or restructure loans. All outputs state `human_review_required: true`.
