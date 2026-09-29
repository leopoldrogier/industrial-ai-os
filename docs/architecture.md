# Architecture notes

## V1 (this repository)

```
Streamlit UI (src/ui)            ── reads only ──►  AnalysisResult
AI layer (src/ai)                ── copilot, rule-based recommendations, LLM interface
Engine (src/engine)              ── detection, scenarios, pipeline
Analytics (src/analytics)        ── pure pandas calculations, unit-tested
Data access (src/data_loader.py) ── CSV / SQLite -> DataBundle
Data (data/*.csv)                ── ERP / WMS / MES / TMS / QMS extracts (synthetic)
```

Design rules:

1. **Numbers are computed once**, in `analytics`, from data + `config.py` assumptions. Text layers (diagnosis, copilot) only format them.
2. **The analysis date is the last data date**, never the wall clock → results are reproducible.
3. **Detection ≠ recommendation**: `engine/detection.py` states what is wrong and why; `ai/recommendations.py` states what to do. A future LLM/agent layer can replace the second without touching the first.
4. **Two views of the plan**: ERP view (promised dates) vs cross-source view (carrier ETAs). The gap between the two is the core value proposition.
5. **Human validation**: actions are drafts. No write-back to any system in V1.

## V2 target

```
User
 ↓
Industrial AI OS (UI / API)
 ↓
AI Orchestrator  ── LLM with grounding fact sheet (src/ai/llm.py)
 ↓
Specialised agents
 ├── Purchasing Agent   → analytics.purchasing
 ├── Supplier Agent     → analytics.suppliers
 ├── Inventory Agent    → analytics.inventory + engine.scenarios
 ├── Quality Agent      → analytics.quality
 ├── Production Agent   → production data (MES)
 └── Cost Agent         → analytics.opportunities
 ↓
Data / Analytics Engine (this code base)
 ↓
ERP / MES / WMS / QMS / Excel / SQL connectors → DataBundle
```

Each agent exposes existing analytics functions as tools; the orchestrator composes answers; every figure still comes from Python.
