# Industrial AI OS

**Intelligent Industrial Operations Platform — V1**

An intelligence layer that connects to the tools a plant already uses (ERP, Excel, SQL, MES, WMS, QMS…) and turns their data into **problem detection, impact analysis, scenario simulation and recommended actions** — with a human always making the decision.

```
DATA → DETECTION → DIAGNOSIS → IMPACT → OPTIONS → RECOMMENDATION → HUMAN VALIDATION → ACTION
```

> Demo company **Méditerranée Aluminium** is fictitious and all data is synthetic. No figure refers to a real company.

---

## 1. The industrial problem

Mid-sized manufacturers run on an ERP plus a constellation of Excel files, MES exports, carrier portals and quality spreadsheets. Each system is correct *in isolation*, but the problems that cost money sit **between** them:

- the ERP shows a purchase order "promised on 30 Sep" while the carrier knows the truck has not left the supplier;
- a supplier's price creeps up month after month, invisible in a PO-by-PO view;
- a supplier's on-time delivery and quality degrade slowly until a line stops;
- scrap contamination rises at one supplier and nobody links it to melt quality.

Buyers, planners and quality engineers find these issues late, by hand, in Excel.

## 2. The concept

Industrial AI OS reads the operational data, **crosses sources**, detects issues, quantifies their impact, simulates options and recommends actions. It explains every number (formula shown in the UI) and never acts on its own.

| Layer | Role |
|---|---|
| **ERP / MES / WMS / QMS** | System of Record — source of operational truth (unchanged) |
| **Industrial AI OS** | Intelligence Layer — analysis, reasoning, simulation, recommendations |
| **Human** | Decision Maker — validates every action |

## 3. Why this is not an ERP

It does not replace SAP or any system of record: no transactions, no master data ownership, no workflows to migrate. It sits on top, read-only, and prepares *drafts* of actions (purchase requisition, expedite request, planning change) that a human validates and executes in the ERP. Deployment is therefore light and the value is visible in weeks, not years.

## 4. Architecture

```
industrial-ai-os/
├── app.py                        # Streamlit entry point (navigation)
├── data/                         # CSV extracts (synthetic): ERP, WMS, MES, TMS, QMS
├── src/
│   ├── config.py                 # analysis windows + ALL business assumptions
│   ├── data_loader.py            # CSV / SQLite loaders -> DataBundle (V2: SAP, SQL, Excel, API)
│   ├── data_generation/
│   │   └── generator.py          # deterministic synthetic dataset with hidden problems
│   ├── analytics/                # pure calculations (no UI, no text)
│   │   ├── purchasing.py         # spend, price evolution, anomalies, contract leakage
│   │   ├── suppliers.py          # OTD, OTIF, delays, lead time, risk score
│   │   ├── inventory.py          # coverage, projection, stockout, overstock, production impact
│   │   ├── quality.py            # PPM, trends, spikes, cost of non-quality
│   │   ├── logistics.py          # freight cost, premium freight
│   │   └── opportunities.py      # cost & opportunity engine (formula per opportunity)
│   ├── engine/                   # business logic
│   │   ├── detection.py          # issues: severity, evidence, impact
│   │   ├── scenarios.py          # stockout options: cost, feasibility, recommendation
│   │   └── analysis.py           # pipeline -> AnalysisResult (single computed truth)
│   ├── ai/                       # recommendation layer (separate from calculations)
│   │   ├── recommendations.py    # rule-based playbooks + diagnoses
│   │   ├── copilot.py            # intent routing + grounded answers
│   │   └── llm.py                # LLM integration point (V2), fact sheet, system prompt
│   ├── ui/                       # Streamlit pages, components, Plotly charts, theme
│   └── utils/formatting.py
├── tests/                        # pytest: calculations, detection, data consistency
└── docs/architecture.md
```

Separation of concerns: **data** (`data_loader`) → **calculations** (`analytics`) → **business logic** (`engine`) → **recommendations** (`ai`) → **interface** (`ui`). The UI and the copilot only read `AnalysisResult`; no figure is hard-coded in the interface.

## 5. Features

| Module | What it does |
|---|---|
| **Dashboard** | Inventory value, 12-month spend, active suppliers, supply risks, critical issues, opportunities, stable areas, potential savings; prioritised issue list with evidence and impact |
| **Supplier Performance** | Scorecard (spend, OTD, OTIF, delay, PPM, price, price evolution, risk score 0-100), risk map, per-supplier overview (rolling OTD, monthly PPM, lead time, price vs peers), AI diagnosis and recommended actions |
| **Inventory Intelligence** | Coverage per material, stock vs safety stock, **projection ERP view vs carrier-ETA view**, stockout alert (date, delivery date, shortage, lines and production impact), overstock |
| **Purchasing Intelligence** | Spend by supplier/material, monthly price evolution, peer benchmark, automatic detection of abnormal price increases and invoicing above contract |
| **Cost & Opportunity Engine** | Opportunities by category (supplier price, inventory, logistics, supplier performance, quality) with criticality, value, **formula**, explanation and recommendation |
| **AI Industrial Copilot** | Questions in natural language (rule-based V1), answers built only from computed facts, data sources crossed, reasoning trace, **scenario simulation (options A/B/C)**, human validation and draft action, decision log |

Key formulas (all in Python, tested):

- `Days of Coverage = Current Stock / Average Daily Consumption (30 days)`
- `OTD = deliveries on or before promised date / deliveries due` (overdue open lines count as late)
- `OTIF = on time AND received ≥ 98% of ordered`
- `PPM = defects / quantity inspected × 1,000,000`
- `Potential Savings = (current price − peer benchmark) × annual volume`
- `Stockout date = first day the projected stock (stock − consumption + open POs at carrier ETA) is negative`
- Supplier risk score = weighted delivery (30%), quality (25%), price (20%), dependency × criticality (15%), trend (10%)

Business assumptions (contribution margin, carrying rate, express freight multiplier, line restart cost…) are centralised and commented in `src/config.py`.

## 6. Dataset

Deterministic synthetic data (seed 42) for 12 months (29 Sep 2025 → 28 Sep 2026), generated by simulation: production drives consumption, replenishment lanes create purchase orders, deliveries (with delays and short shipments) feed a day-by-day stock simulation, each delivery produces a shipment and a goods-receipt inspection.

| File | Rows | Source system |
|---|---|---|
| `products.csv` | 9 | ERP material master |
| `suppliers.csv` | 14 | ERP vendor master |
| `purchase_orders.csv` | 371 | ERP purchasing |
| `inventory.csv` | 3,285 | WMS (daily stock per material) |
| `production.csv` | 1,095 | MES (3 casting lines × 365 days) |
| `logistics.csv` | 371 | TMS / carrier tracking |
| `quality.csv` | 359 | QMS inspections |

Company profile: 120 employees, ~€35M revenue, Fos-sur-Mer, 2 furnaces, 3 casting lines (extrusion billets, rolling slabs, recycled billets), ~€25M purchasing spend.

**Hidden problems (not flagged in the data — the software must discover them):**

1. **Stockout risk** — primary aluminium stock is above safety stock and the ERP shows no issue, but the two open MAS Aluminium orders are held at origin (carrier ETA +12 and +9 days); stock runs out before any other inbound delivery.
2. **Price drift** — MAS Aluminium price rises ~9% over the year while comparable suppliers move ~1%.
3. **Supplier deterioration** — MAS OTD drops from ~95% to ~85%, lead time from 8 to 12 days, PPM more than doubles.
4. **Quality spike** — EcoMetal Iberia scrap PPM ×3.6 over the last 60 days (Fe/Cu contamination).
5. Secondary: invoicing above a fixed contract (silicon), overstock (magnesium, refractory), premium freight, emergency spot buys.

Regenerate: `python -m src.data_generation.generator`

## 7. Installation

Requires Python 3.10+.

```bash
git clone <repo-url> industrial-ai-os
cd industrial-ai-os
pip install -r requirements.txt
```

No API key and no external service are required.

## 8. Launch

```bash
streamlit run app.py
```

Open http://localhost:8501. If the CSV files are missing, they are generated automatically at first launch.

Run the tests:

```bash
python -m pytest -q
```

## 9. Five-minute demo

1. **Dashboard** — "Industrial AI OS reads our ERP, WMS, MES, TMS and QMS extracts. It finds **4 critical issues** and **~€605k/year** of savings, all computed from the data."
2. **Supplier Performance** → select **MAS Aluminium** — OTD down from 94.9% to 84.6%, lead time +49%, PPM 202 → 510, price +6.5% above peers. Read the AI diagnosis and actions.
3. **Inventory Intelligence** — 🔴 **STOCKOUT RISK** on primary aluminium: stock is above safety stock, the ERP projection (grey) looks fine, but the carrier-ETA projection (orange) goes negative on 7 Oct. Shortage ~95 t, lines CL1/CL2, ~€83k impact.
4. **AI Copilot** → *"Why is production at risk?"* — the answer crosses supplier + purchase orders + logistics + inventory + production and shows how it was built.
5. **Options** — A: expedite (~€10.8k, medium confidence), B: alternative supplier (~€3.4k, recommended), C: production adjustment (~€55k impact). The human validates → a **draft** requisition is prepared; nothing is executed in the ERP.

## 10. Roadmap V2

- **LLM layer**: plug OpenAI / Anthropic into `src/ai/llm.py` (`Narrator` interface, fact sheet + grounding system prompt already in place) — the LLM rephrases, Python computes.
- **Specialised agents**: Purchasing, Supplier, Inventory, Quality, Production, Cost agents behind an orchestrator, each calling the existing analytics functions as tools.
- **Connectors**: SAP (OData / RFC), SQL databases, Excel upload, REST API — all returning the same `DataBundle`.
- **Scenario engine**: volume reallocation, safety-stock optimisation, price-index simulation.
- **Human-in-the-loop**: persistent decision log, approval workflow, prepared ERP transactions (purchase requisitions, PO changes) pushed after approval.
- **Automatic reports**: weekly supplier review and savings report (PDF / email).
- **Multi-site, authentication and roles**.

---

*V1 recommends only. It never executes an action in an ERP.*
