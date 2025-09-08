# LLM Beer Supply Chain Game

A research-oriented, **LLM‑driven** re‑imagining of the classic [Beer Distribution Game](https://en.wikipedia.org/wiki/Beer_distribution_game#).  
Three autonomous agents (Factory, Wholesale, Retail) run a simplified beer supply chain: they **chat/coordinate**, sign **fixed or flexible long‑term contracts**, place **one‑time orders**, handle **backlogs with penalties**, and try to avoid bankruptcies while meeting market demand.

This project contains:
- `main.py` – the simulation/orchestrator that runs the game with real LLMs (Together/OpenAI/Anthropic) or a mock.
- `ai_supply_chain_analysis.py` – an offline analyzer that turns one game’s JSON export into charts & CSVs.

> Research goal: study whether capable LLMs can **manage supply chains**, **understand & use long‑term contracts**, and **handle market crises** when allowed to talk without strict information bottlenecks. The classic bullwhip effect is a reference point; here, rich communication and contracts may mitigate it.

---

## Contents
- [Key Features](#key-features)
- [How This Differs from the Classic Game](#how-this-differs-from-the-classic-game)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Gameplay Mechanics & Round Order](#gameplay-mechanics--round-order)
- [Outputs](#outputs)
- [Analysis Script](#analysis-script)
- [Data Schema (JSON Export)](#data-schema-json-export)
- [Switching Models / Providers](#switching-models--providers)
- [Tips & Gotchas](#tips--gotchas)
- [Roadmap Ideas](#roadmap-ideas)

---

## Key Features
- **Three roles**: `Factory → Wholesale → Retail → Market`.
- **Free chat between neighbors** with a hard cap per conversation (`max_messages_per_call`). Conversations are logged.
- **Long‑term contracts**: `fixed` (amount per round) and `flexible` (±% band chosen each round).
- **One‑time (spot) orders** in addition to contract obligations.
- **Backlog with per‑unit penalties** charged to the party failing to deliver (supplier pays buyer each round until cleared).
- **Storage limits & storage costs** each round; bankruptcy ends the game.
- **Token usage logging** per provider:model to `token_usage.txt`.
- **One‑round delivery delay; instant payments.**

---

## How This Differs from the Classic Game
- Agents are **LLMs that can negotiate** and reason about contracts in natural language (with strict output formats for actions).
- **Contracts & prices are explicit**: fixed vs flexible with configurable multipliers; fines for breaking; backlog fees per undelivered unit per round.
- **Backlog penalties** discipline supply reliability (supplier pays buyer until backlog is served).
- Market demand is configurable; by default it’s steady (you can re‑enable seasonal/crisis/boom logic).
- Objective is not only inventory smoothing but also **economic performance** (money, value, stability).

---

## Requirements
Python 3.10+ is recommended.

Install packages (no `requirements.txt` yet):
```bash
pip install python-dotenv openai anthropic pandas numpy matplotlib
```
> **Together** uses the OpenAI‑compatible client with a custom base URL, so only the `openai` package is needed for Together.

Environment variables (via `.env` or your shell):
- `TOGETHER_API_KEY` *(default path in code)*
- (optional) `OPENAI_API_KEY`
- (optional) `ANTHROPIC_API_KEY`

Example `.env`:
```ini
TOGETHER_API_KEY=sk_your_together_key_here
OPENAI_API_KEY=sk_your_openai_key_here
ANTHROPIC_API_KEY=sk-ant-your_key_here
```

---

## Quick Start
Run a single game from the terminal:
```bash
python main.py
```
This will, by default, use **Together** with model:
```python
together_model = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
```
and create three company interfaces via:
```python
llm_interfaces = {
  "factory":   create_company_llm("factory",   "together", api_key, model=together_model, temperature=0.7),
  "wholesale": create_company_llm("wholesale", "together", api_key, model=together_model, temperature=0.7),
  "retail":    create_company_llm("retail",    "together", api_key, model=together_model, temperature=0.7),
}
```

When the run ends you’ll see a JSON export like `game_results_YYYYMMDD_HHMMSS.json` and a token summary in `token_usage.txt`.

---

## Configuration
Open `main.py`, class **`GlobalParameters`** for defaults (selected highlights):
- `num_rounds`: total rounds (e.g., `2` default; increase for real runs).
- `max_messages_per_call`: chat cap in each neighbor conversation (default `10`).
- `max_factory_production`: per‑round production limit (default `500`).
- `starting_money`: initial cash per firm (default `10000.0`).
- `starting_beer`: initial inventory (default `300`).
- `default_max_storage`: cap on on‑hand units (default `1000`).
- `storage_cost_per_unit`: per‑unit storage charge per round (default `0.5`).
- `base_prices`: list `[2.0, 4.0, 6.0]` for edges  
  `Factory→Wholesale`, `Wholesale→Retail`, `Retail→Market`.
- `contract_multipliers`: `{"fixed": 1.1, "flexible": 1.2}` used to compute contract price per unit from the **edge base price**.
- `flexibility_percentage`: default `0.2` → ±20% band for flexible contracts.
- `contracts_enabled`: enable/disable contract mechanics (default `True`).
- `production_cost`: per‑unit cost for Factory production (default `0.0`).

**Market demand**  
`market_demand_function(round_num)` currently returns a constant `300` per round. A seasonal/boom/crisis variant is included as commented code for quick switching.

**Prompts & strict formats**  
Each role gets a strong system prompt with strict output formats:
- simple `Yes/No` checks,
- pure **numbers** when asked (no extra text),
- contract proposals in the CSV form: `type,amount,length,fine`  
  with `type ∈ {fixed, flexible}`.
- conversations must end with the token `TERMINATE_CONVERSATION` when done.

---

## Gameplay Mechanics & Round Order
For each **round**:
1. **Carry‑over deliveries arrive** (1‑round delay).
2. **Retail sells to Market** at `base_prices[2]`. Retail decides the sale quantity (≤ demand and inventory); money is received instantly.
3. **Wholesale ↔ Retail** may **chat** (capped by `max_messages_per_call`). Retail optionally **proposes a contract** (parsed as `type,amount,length,fine`; price per unit is computed = edge base price × multiplier).
4. **Wholesale receives from Factory** similarly (chat + optional contract).
5. **Contracts & Orders are compiled** via `handle_contracts_and_orders`:
   - **Flexible**: buyer selects an amount within ±`flexibility_percentage` of the contract’s nominal amount.
   - **Fixed**: fixed amount.
   - **Backlog** (if any) is prioritized alongside one‑time orders.
   - **One‑time (spot) orders** can be requested; supplier may accept, reject, or counter.
   - **Priority for fulfillment (revenue‑oriented)**: flexible → fixed → backlog/one‑time.
6. **Deliveries executed**, payments are **instant**. Any **shortage** triggers a supplier decision:
   - (1) deliver what’s available & **add the shortfall to backlog** (supplier pays **backlog fee per unit per round** until cleared),
   - (2) **break the contract** (pay `fine`, no delivery),
   - (3) **propose mutual cancellation** (no penalties **if** buyer agrees; otherwise choose (1) or (2)).
7. **Factory plans and produces** for the **next** round (`decide_production` + `produce_beer`).
8. **Storage cost** is charged, **over‑capacity inventory is discarded**, and **bankruptcy check** runs.
9. **Round state & dialogues are logged** to the Database.

**Backlog fees** (supplier → buyer):  
- Factory backlog fee per unit = `base_prices[0]`  
- Wholesale backlog fee per unit = `base_prices[1]`  
- Retail pays no backlog to Market.

---

## Outputs
At the end of a run:
- `game_results_YYYYMMDD_HHMMSS.json`
- `token_usage.txt` (aggregate `prompt_tokens` & `completion_tokens` per `provider:model`)

Console stats include counts of **created / mutually canceled / broken** contracts, **transactions**, **dialogues**, and **rounds**.

---

## Analysis Script
Turn one JSON export into charts + normalized CSVs:
```bash
python ai_supply_chain_analysis.py game_results_2025XXXX_XXXXXX.json --outdir out
```
CLI parameters:
- `input_json` (positional): path to a single game’s JSON export
- `--outdir out` (default: `out/`)
- `--start-money 10000.0` (used for cumulative money/value baselines)
- `--retail-price 6.0` (used for valuation of inventory)

**Figures (PNG)** saved in `out/`:
1. `1_money_cumulative.png` – cumulative money vs. rounds (stacked, market demand overlay).
2. `2_profit_per_round.png` – money delta per round (stacked; market demand overlay).
3. `3_value_cumulative.png` – cumulative **value** = money + inventory×retail_price.
4. `4_value_delta.png` – value delta per round.
5. `5_supply_structure_percent.png` – **stacked %** of deliveries by type (`flexible`, `fixed`, `one_time`, `backlog`). Market sales excluded.
   - Also `5B_supply_structure_money.png` – same by **USD value**.
6. `6_contract_timeline.png` – Gantt‑like chart with direction coloring (`Factory→Wholesale` red, `Wholesale→Retail` blue) and linestyle for type (fixed `-`, flexible `--`).

**CSVs**:
- `transactions_normalized.csv`
- `round_states_normalized.csv`
- `market_demand_normalized.csv`
- `contracts_created_normalized.csv`
- `contracts_events.csv`

The analyzer is defensive: it normalizes multiple possible field names and re‑classifies generic deliveries using **active contracts** for that round.

---

## Data Schema (JSON Export)
Top‑level keys (from `Database.export_to_json`):
- `transactions`: list of objects with
  ```json
  {
    "round": 1,
    "from": "Wholesale",
    "to": "Retail",
    "beer": 120,
    "money": 528.0,
    "type": "delivery_flexible"   // or delivery_fixed, delivery_one_time, delivery_backlog, backlog_fee, market_sale
  }
  ```
- `dialogues`: `{ "round", "participants": ["Retail","Wholesale"], "messages": [{"from": "...", "message": "..."}], "timestamp": ... }`
- `round_states`: `{ "round", "states": { "Factory": {...}, "Wholesale": {...}, "Retail": {...} }, "timestamp": ... }`
- `contracts`: objects with a `contract` payload and `action ∈ {CREATED, MUTUALLY_CANCELLED, BROKEN}`
- `market_demand`: `{ "round", "demand", "timestamp" }`
- `models_used`: `{ role → { "api_type": "...", "model": "..." } }`

> Note: the analyzer accepts both `contracts` or `contracts_log`, and `market_demand` or `market_demand_log` for robustness.

---

## Switching Models / Providers
LLM access is abstracted by **`LLMInterface`**. The default is **Together** using the OpenAI client with a custom `base_url`. To switch:

- **OpenAI**
  ```python
  api_key = os.getenv("OPENAI_API_KEY")
  llm_interfaces = {
    "factory":   create_company_llm("factory",   "openai",   api_key, model="gpt-4o-mini"),
    "wholesale": create_company_llm("wholesale", "openai",   api_key, model="gpt-4o-mini"),
    "retail":    create_company_llm("retail",    "openai",   api_key, model="gpt-4o-mini"),
  }
  ```

- **Anthropic**
  ```python
  api_key = os.getenv("ANTHROPIC_API_KEY")
  llm_interfaces = {
    "factory":   create_company_llm("factory",   "anthropic", api_key, model="claude-3-5-haiku-20241022"),
    "wholesale": create_company_llm("wholesale", "anthropic", api_key, model="claude-3-5-haiku-20241022"),
    "retail":    create_company_llm("retail",    "anthropic", api_key, model="claude-3-5-haiku-20241022"),
  }
  ```

- **Mock (no API cost)** – quick local sanity checks:
  ```python
  llm_interfaces = { "factory": MockLLM(), "wholesale": MockLLM(), "retail": MockLLM() }
  ```

You can override **temperature**, **max_tokens**, and **model** per role.

---

## Tips & Gotchas
- **Follow the formats.** The code enforces numeric‑only answers or `type,amount,length,fine`. If an LLM returns extra prose, you’ll see a warning and be prompted to continue or exit.
- **IRRATIONAL BEHAVIOR DETECTED** appears if a supplier tries to accept one‑time orders while **short** on inventory (a red flag for prompt design).
- **Backlog fees** punish unreliable suppliers. They are computed as the **edge base price** per undelivered unit each round and transferred from **supplier → buyer**.
- **Storage cost** hits every round; overflow inventory is **discarded** to `max_storage`.
- **Bankruptcy ends the game** immediately when any company’s money drops below zero.
- **Token costs** can grow quickly (3 agents × multiple messages). Consider smaller models, lower `max_messages_per_call`, shorter prompts, and lower `max_tokens` for control.

---

## Roadmap Ideas
- Re‑enable **seasonal / crisis / boom** demand patterns.
- Heterogeneous objectives (risk‑aversion, service levels).
- Multi‑SKU extensions; perishability; lead times > 1 round.
- Separate **backlog fee** setting per contract and/or per role.
- Multi‑agent markets; additional tiers (distributor, 2PL/3PL).
- Fairness & transparency audits of negotiation behavior.

---

### Acknowledgement
Original game concept: **Beer Distribution Game** — https://en.wikipedia.org/wiki/Beer_distribution_game#

