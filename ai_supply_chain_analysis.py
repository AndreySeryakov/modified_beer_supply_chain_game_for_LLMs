#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI Supply-Chain Game — Analysis Script
--------------------------------------
Produces six figures from a single JSON export:
  1) Money cumulative vs rounds
  2) Profit per round (money delta)
  3) Value cumulative (money + inventory×retail_price)
  4) Value per round (delta)
  5) Supply structure (stacked %, flexible/fixed/on-spot/backlog) — excludes Market sales
  6) Contract timeline (Factory→Wholesale=red, Wholesale→Retail=blue; linestyle: Fixed='-', Flexible='--')
"""

import argparse
import json
import os
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DELIVERY_TYPE_ALIASES = {
    "delivery_fixed": "fixed",
    "fixed_delivery": "fixed",
    "delivery_flexible": "flexible",
    "flexible_delivery": "flexible",
    "delivery_one_time": "on_spot",
    "delivery_on_spot": "on_spot",
    "on_spot_delivery": "on_spot",
    "delivery_backlog": "backlog",
    "backlog_delivery": "backlog",
}

COMPANY_ORDER = ["Factory", "Wholesale", "Retail"]


def safe_get(d: Dict[str, Any], key: str, default=None):
    return d[key] if key in d else default


def normalize_round_states(round_states_raw: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for i, blob in enumerate(round_states_raw):
        r = safe_get(blob, "round", i + 1)
        states = safe_get(blob, "states", None)
        if isinstance(states, dict) and len(states) > 0:
            for company, comp in states.items():
                rows.append({
                    "round": r,
                    "company": company,
                    "money": safe_get(comp, "money", np.nan),
                    "beer_storage": safe_get(comp, "beer_storage", np.nan),
                    "backlog": safe_get(comp, "backlog", None),
                    "active_contracts_reported": safe_get(comp, "active_contracts", np.nan),
                    "type": safe_get(comp, "type", None),
                    "name": safe_get(comp, "name", company),
                })
        else:
            for company in COMPANY_ORDER:
                comp = safe_get(blob, company, {})
                rows.append({
                    "round": r,
                    "company": company,
                    "money": safe_get(comp, "money", np.nan),
                    "beer_storage": safe_get(comp, "beer_storage", np.nan),
                    "backlog": safe_get(comp, "backlog", None),
                    "active_contracts_reported": safe_get(comp, "active_contracts", np.nan),
                    "type": safe_get(comp, "type", None),
                    "name": safe_get(comp, "name", company),
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["round", "company"])
    return df


def normalize_transactions(transactions_raw: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(transactions_raw)
    for col in ["round", "type", "from", "to", "beer", "money"]:
        if col not in df.columns:
            df[col] = np.nan
    for col in ["round", "beer", "money"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Infer beer volume if needed
    if "beer" not in df.columns or df["beer"].isna().all():
        for cand in ["beer_amount", "quantity", "qty", "units", "volume", "amount"]:
            if cand in df.columns and not pd.to_numeric(df[cand], errors="coerce").isna().all():
                df["beer"] = pd.to_numeric(df[cand], errors="coerce")
                break
    else:
        missing = df["beer"].isna()
        if missing.any():
            for cand in ["beer_amount", "quantity", "qty", "units", "volume", "amount"]:
                if cand in df.columns:
                    fill_vals = pd.to_numeric(df.loc[missing, cand], errors="coerce")
                    df.loc[missing, "beer"] = fill_vals
                    missing = df["beer"].isna()
                    if not missing.any():
                        break

    # Delivery type normalization
    def _infer_type_norm(row):
        t = str(row.get("type", "")).lower()
        base = DELIVERY_TYPE_ALIASES.get(t, None)
        if base is not None:
            return base
        if "delivery" in t or t in ("delivery", "deliveries"):
            for subkey in ["order_type", "delivery_type", "contract_type", "kind", "subtype", "reason"]:
                v = row.get(subkey, None)
                if v is None:
                    continue
                v = str(v).lower()
                if v in ("fixed", "flexible", "on_spot", "on-spot", "one_time", "one-time", "backlog"):
                    if v in ("one_time", "one-time"):
                        return "on_spot"
                    if v == "on-spot":
                        return "on_spot"
                    return v
            if str(row.get("is_backlog", "")).lower() in ("true", "1", "yes"):
                return "backlog"
        return DELIVERY_TYPE_ALIASES.get(t, t)

    if "type" in df.columns:
        df["type_norm"] = df.apply(_infer_type_norm, axis=1)
    else:
        df["type_norm"] = np.nan
    return df


def normalize_market_demand(market_raw: Any) -> pd.DataFrame:
    if isinstance(market_raw, list):
        if len(market_raw) == 0:
            return pd.DataFrame(columns=["round", "demand"])
        if isinstance(market_raw[0], dict):
            tmp = pd.DataFrame(market_raw)
            if "round" not in tmp.columns:
                tmp["round"] = np.arange(1, len(tmp) + 1)
            return tmp[["round", "demand"]].copy()
        else:
            return pd.DataFrame({"round": np.arange(1, len(market_raw) + 1), "demand": market_raw})
    return pd.DataFrame(columns=["round", "demand"])


def normalize_contracts(contracts_raw: List[Dict[str, Any]]):
    events = pd.DataFrame(contracts_raw) if isinstance(contracts_raw, list) else pd.DataFrame([])
    if events.empty:
        return pd.DataFrame([]), pd.DataFrame([])

    if 'contract' in events.columns:
        cdf = pd.json_normalize(events['contract'])
        events = pd.concat([events.drop(columns=['contract']), cdf], axis=1)

    if 'contract_type' in events.columns:
        events['type_norm'] = events['contract_type'].astype(str).str.lower()
    elif 'type' in events.columns:
        events['type_norm'] = events['type'].astype(str).str.lower()
    else:
        events['type_norm'] = np.nan

    if 'parties' in events.columns:
        def _seller(x):
            try:
                return x[0]
            except Exception:
                return None
        def _buyer(x):
            try:
                return x[1]
            except Exception:
                return None
        events['seller'] = events['parties'].apply(_seller)
        events['buyer'] = events['parties'].apply(_buyer)

    for col in ['start_round', 'length', 'fine', 'price_per_unit', 'amount']:
        if col in events.columns:
            events[col] = pd.to_numeric(events[col], errors='coerce')

    if 'contract_id' not in events.columns:
        events['contract_id'] = np.arange(1, len(events) + 1).astype(str)
    else:
        events['contract_id'] = events['contract_id'].astype(str)

    if 'action' in events.columns:
        created = events[events['action'].astype(str).str.upper() == 'CREATED'].copy()
        if created.empty:
            created = events.sort_values(['contract_id']).groupby('contract_id', as_index=False).first()
    else:
        created = events.drop_duplicates('contract_id')

    created['planned_end_round'] = created.get('start_round', pd.Series([np.nan]*len(created))) +                                    created.get('length', pd.Series([np.nan]*len(created))) - 1

    def _dir(row):
        s = row.get('seller', None) or row.get('from', None)
        b = row.get('buyer', None) or row.get('to', None)
        if pd.notna(s) and pd.notna(b):
            return f"{s}→{b}"
        return 'Unknown'
    created['direction'] = created.apply(_dir, axis=1)

    return created, events


def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_money_cumulative(df_state: pd.DataFrame,
                          df_market: pd.DataFrame,
                          outdir: str,
                          start_money: float,
                          retail_price: float):
    pivot_money = df_state.pivot(index="round", columns="company", values="money").sort_index()
    rounds = pivot_money.index.values
    total_money = pivot_money.sum(axis=1)
    total_money_minus_start = total_money - (start_money * len(COMPANY_ORDER))
    each_minus_start = pivot_money.subtract(start_money)

    d = df_market.set_index("round")["demand"].reindex(rounds).fillna(0.0)
    max_profit_cum = (d * retail_price).cumsum()

    plt.figure(figsize=(10, 6))
    plt.plot(rounds, max_profit_cum, label="Max profit (cum): demand × retail_price")
    plt.plot(rounds, total_money_minus_start, label="Supply chain money (cum) − start")
    for c in COMPANY_ORDER:
        if c in each_minus_start.columns:
            plt.plot(rounds, each_minus_start[c], label=f"{c} money (cum) − start")
    plt.xlabel("Round"); plt.ylabel("USD"); plt.title("Money vs Round (cumulative)")
    plt.grid(True, which="both", linestyle="--", alpha=0.4); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "1_money_cumulative.png")); plt.close()

    out = pd.DataFrame({"round": rounds,
                        "max_profit_cum": max_profit_cum.values,
                        "supply_chain_money_cum_minus_start": total_money_minus_start.values})
    for c in COMPANY_ORDER:
        if c in each_minus_start.columns:
            out[f"{c}_money_cum_minus_start"] = each_minus_start[c].values
    out.to_csv(os.path.join(outdir, "1_money_cumulative.csv"), index=False)


def plot_money_delta(df_state: pd.DataFrame,
                     df_market: pd.DataFrame,
                     outdir: str,
                     start_money: float,
                     retail_price: float):
    pivot_money = df_state.pivot(index="round", columns="company", values="money").sort_index()
    rounds = pivot_money.index.values

    total_money = pivot_money.sum(axis=1)
    baseline_prev = pd.Series([start_money * len(COMPANY_ORDER)], index=[rounds[0]])
    total_money_prev = pd.concat([baseline_prev, total_money.iloc[:-1]])
    total_delta = total_money - total_money_prev.values

    each_delta = {}
    for c in COMPANY_ORDER:
        if c in pivot_money.columns:
            baseline_c_prev = pd.Series([start_money], index=[rounds[0]])
            prev = pd.concat([baseline_c_prev, pivot_money[c].iloc[:-1]])
            each_delta[c] = pivot_money[c] - prev.values

    d = df_market.set_index("round")["demand"].reindex(rounds).fillna(0.0)
    market_per_round = d * retail_price

    plt.figure(figsize=(10, 6))
    plt.plot(rounds, market_per_round, label="Market (round): demand × retail_price")
    plt.plot(rounds, total_delta, label="Supply chain money Δ (round)")
    for c, series in each_delta.items():
        plt.plot(rounds, series, label=f"{c} money Δ (round)")
    plt.xlabel("Round"); plt.ylabel("USD"); plt.title("Profit per Round (money delta)")
    plt.grid(True, which="both", linestyle="--", alpha=0.4); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "2_profit_per_round.png")); plt.close()

    out = pd.DataFrame({"round": rounds,
                        "market_round": market_per_round.values,
                        "supply_chain_money_delta": total_delta.values})
    for c, series in each_delta.items():
        out[f"{c}_money_delta"] = series.values
    out.to_csv(os.path.join(outdir, "2_profit_per_round.csv"), index=False)


def plot_value_cumulative(df_state: pd.DataFrame,
                          df_market: pd.DataFrame,
                          outdir: str,
                          start_money: float,
                          retail_price: float):
    pivot_money = df_state.pivot(index="round", columns="company", values="money").sort_index()
    pivot_beer = df_state.pivot(index="round", columns="company", values="beer_storage").sort_index()

    rounds = pivot_money.index.values
    value = pivot_money + pivot_beer.fillna(0.0) * retail_price
    total_value = value.sum(axis=1)
    total_value_minus_start = total_value - (start_money * len(COMPANY_ORDER))
    each_value_minus_start = value.subtract(start_money)

    d = df_market.set_index("round")["demand"].reindex(rounds).fillna(0.0)
    max_profit_cum = (d * retail_price).cumsum()

    plt.figure(figsize=(10, 6))
    plt.plot(rounds, max_profit_cum, label="Max profit (cum): demand × retail_price")
    plt.plot(rounds, total_value_minus_start, label="Supply chain value (cum) − start")
    for c in COMPANY_ORDER:
        if c in each_value_minus_start.columns:
            plt.plot(rounds, each_value_minus_start[c], label=f"{c} value (cum) − start")
    plt.xlabel("Round"); plt.ylabel("USD-equivalent"); plt.title("Value vs Round (cumulative)")
    plt.grid(True, which="both", linestyle="--", alpha=0.4); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "3_value_cumulative.png")); plt.close()

    out = pd.DataFrame({"round": rounds,
                        "max_profit_cum": max_profit_cum.values,
                        "supply_chain_value_cum_minus_start": total_value_minus_start.values})
    for c in COMPANY_ORDER:
        if c in each_value_minus_start.columns:
            out[f"{c}_value_cum_minus_start"] = each_value_minus_start[c].values
    out.to_csv(os.path.join(outdir, "3_value_cumulative.csv"), index=False)


def plot_value_delta(df_state: pd.DataFrame,
                     df_market: pd.DataFrame,
                     outdir: str,
                     start_money: float,
                     retail_price: float):
    pivot_money = df_state.pivot(index="round", columns="company", values="money").sort_index()
    pivot_beer = df_state.pivot(index="round", columns="company", values="beer_storage").sort_index()
    rounds = pivot_money.index.values

    value = pivot_money + pivot_beer.fillna(0.0) * retail_price
    total_value = value.sum(axis=1)
    baseline_prev = pd.Series([start_money * len(COMPANY_ORDER)], index=[rounds[0]])
    total_value_prev = pd.concat([baseline_prev, total_value.iloc[:-1]])
    total_delta = total_value - total_value_prev.values

    each_delta = {}
    for c in COMPANY_ORDER:
        if c in value.columns:
            baseline_c_prev = pd.Series([start_money], index=[rounds[0]])
            prev = pd.concat([baseline_c_prev, value[c].iloc[:-1]])
            each_delta[c] = value[c] - prev.values

    d = df_market.set_index("round")["demand"].reindex(rounds).fillna(0.0)
    market_per_round = d * retail_price

    plt.figure(figsize=(10, 6))
    plt.plot(rounds, market_per_round, label="Market (round): demand × retail_price")
    plt.plot(rounds, total_delta, label="Supply chain value Δ (round)")
    for c, series in each_delta.items():
        plt.plot(rounds, series, label=f"{c} value Δ (round)")
    plt.xlabel("Round"); plt.ylabel("USD-equivalent"); plt.title("Value per Round (delta)")
    plt.grid(True, which="both", linestyle="--", alpha=0.4); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "4_value_per_round.png")); plt.close()

    out = pd.DataFrame({"round": rounds,
                        "market_round": market_per_round.values,
                        "supply_chain_value_delta": total_delta.values})
    for c, series in each_delta.items():
        out[f"{c}_value_delta"] = series.values
    out.to_csv(os.path.join(outdir, "4_value_per_round.csv"), index=False)


def plot_supply_structure(df_tx: pd.DataFrame, df_contracts: pd.DataFrame, outdir: str):
    """
    Plot 5: Supply structure — stacked % bars per round (flexible, fixed, on_spot, backlog)
    Excludes market sales; reclassifies deliveries based on active contracts for that round.
    """
    df = df_tx.copy()
    if "type" in df.columns:
        df["type"] = df["type"].astype(str).str.lower()
    if "from" in df.columns:
        df["from"] = df["from"].astype(str)
    if "to" in df.columns:
        df["to"] = df["to"].astype(str)
    df = df[~df.get("type", pd.Series("", index=df.index)).eq("market_sale")]
    df = df[(~df.get("from", pd.Series("", index=df.index)).eq("Market")) & (~df.get("to", pd.Series("", index=df.index)).eq("Market"))]

    deliveries = df.copy()

    deliveries = deliveries.dropna(subset=["round"])
    if "beer" not in deliveries.columns:
        deliveries["beer"] = np.nan
    if deliveries["beer"].isna().all():
        for cand in ["beer_amount", "quantity", "qty", "units", "volume", "amount"]:
            if cand in deliveries.columns and not pd.to_numeric(deliveries[cand], errors="coerce").isna().all():
                deliveries["beer"] = pd.to_numeric(deliveries[cand], errors="coerce")
                break
    deliveries = deliveries[pd.to_numeric(deliveries["beer"], errors="coerce").fillna(0) > 0].copy()

    if deliveries.empty or "from" not in deliveries.columns or "to" not in deliveries.columns:
        plt.figure(figsize=(10, 5)); plt.title("Supply Structure — no internal deliveries"); plt.savefig(os.path.join(outdir, "5_supply_structure.png")); plt.close()
        pd.DataFrame().to_csv(os.path.join(outdir, "5_supply_structure.csv"), index=False); return

    contracts = df_contracts.copy()
    for c in ["seller", "buyer", "type_norm", "start_round", "planned_end_round"]:
        if c not in contracts.columns:
            contracts[c] = np.nan
    contracts = contracts.dropna(subset=["seller", "buyer", "start_round", "planned_end_round"])
    contracts["type_norm"] = contracts["type_norm"].astype(str).str.lower()
    contracts = contracts[contracts["type_norm"].isin(["fixed", "flexible"])]

    if not contracts.empty:
        deliveries = deliveries.reset_index(drop=False).rename(columns={"index":"_row_id"})
        merged = deliveries.merge(contracts[["seller","buyer","type_norm","start_round","planned_end_round"]],
                                  left_on=["from","to"], right_on=["seller","buyer"], how="left")
        merged["_active"] = (pd.to_numeric(merged["round"], errors="coerce") >= pd.to_numeric(merged["start_round"], errors="coerce")) &                             (pd.to_numeric(merged["round"], errors="coerce") <= pd.to_numeric(merged["planned_end_round"], errors="coerce"))
        priority = {"fixed": 2, "flexible": 1}
        merged["prio"] = merged["type_norm_y"].map(priority).fillna(0)
        best = (merged[merged["_active"]].sort_values(["_row_id", "prio"], ascending=[True, False]).drop_duplicates("_row_id"))
        deliveries["type_norm"] = deliveries.get("type_norm", "").astype(str)
        # Only reclassify UNKNOWN/generic deliveries; keep explicit on_spot as on_spot
        mask_unknown = deliveries["type_norm"].isna() | (~deliveries["type_norm"].isin(["fixed","flexible","on_spot","backlog"]))
        deliveries = deliveries.merge(best[["_row_id", "type_norm_y"]], on="_row_id", how="left")
        deliveries.loc[mask_unknown & deliveries["type_norm_y"].notna(), "type_norm"] = deliveries.loc[mask_unknown, "type_norm_y"]
        deliveries.drop(columns=["type_norm_y"], inplace=True, errors="ignore")

    deliveries["type_norm"] = deliveries["type_norm"].replace({"one_time":"on_spot", "one-time":"on_spot", "on-spot":"on_spot"})
    deliveries["type_norm"] = deliveries["type_norm"].where(deliveries["type_norm"].isin(["flexible","fixed","on_spot","backlog"]), "on_spot")

    grouped = deliveries.groupby(["round", "type_norm"], as_index=False)["beer"].sum()
    pivot = grouped.pivot(index="round", columns="type_norm", values="beer").fillna(0.0)
    for t in ["flexible", "fixed", "on_spot", "backlog"]:
        if t not in pivot.columns:
            pivot[t] = 0.0
    pivot = pivot[["flexible", "fixed", "on_spot", "backlog"]]
    # Ensure we include ALL rounds from original transactions (even if zero internal deliveries)
    all_rounds = sorted(pd.unique(pd.to_numeric(df_tx.get("round", pd.Series([], dtype=float)), errors="coerce").dropna()))
    try:
        all_rounds = [int(r) for r in all_rounds]
    except Exception:
        pass
    if len(all_rounds) > 0:
        pivot = pivot.reindex(all_rounds, fill_value=0.0)

    total = pivot.sum(axis=1)
    pct = pivot.div(total.replace(0, np.nan), axis=0).fillna(0.0) * 100.0

    rounds = pct.index.values
    plt.figure(figsize=(11, 6))
    bottom = np.zeros(len(rounds))
    for col in pct.columns:
        plt.bar(rounds, pct[col].values, bottom=bottom, label=col.replace("_", " "))
        bottom += pct[col].values
    plt.xlabel("Round"); plt.ylabel("% of delivered volume"); plt.title("Supply Structure by Delivery Type (per round)")
    plt.grid(True, axis="y", linestyle="--", alpha=0.4); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "5_supply_structure.png")); plt.close()
    pct.reset_index().to_csv(os.path.join(outdir, "5_supply_structure.csv"), index=False)



def plot_supply_structure_money(df_tx: pd.DataFrame, df_contracts: pd.DataFrame, outdir: str):
    """
    Plot 5B: Supply structure by MONEY — stacked bars per round (flexible, fixed, on_spot, backlog) in USD.
    Excludes market sales; reclassifies generic deliveries based on active contracts for that round.
    """
    df = df_tx.copy()
    if "type" in df.columns:
        df["type"] = df["type"].astype(str).str.lower()
    if "from" in df.columns:
        df["from"] = df["from"].astype(str)
    if "to" in df.columns:
        df["to"] = df["to"].astype(str)
    # Exclude Market
    df = df[~df.get("type", pd.Series("", index=df.index)).eq("market_sale")]
    df = df[(~df.get("from", pd.Series("", index=df.index)).eq("Market")) & (~df.get("to", pd.Series("", index=df.index)).eq("Market"))]

    deliveries = df.copy()

    # Ensure round and money exist
    deliveries = deliveries.dropna(subset=["round"])
    if "money" not in deliveries.columns:
        # nothing to plot
        plt.figure(figsize=(10, 5)); plt.title("Supply Structure (Money) — no money column"); plt.savefig(os.path.join(outdir, "5_supply_structure_money.png")); plt.close()
        pd.DataFrame().to_csv(os.path.join(outdir, "5_supply_structure_money.csv"), index=False); return

    # Keep only rows where money moved (positive); if negatives exist, take absolute value for magnitude
    money_vals = pd.to_numeric(deliveries["money"], errors="coerce").fillna(0.0)
    # use absolute to visualize trade volume in dollars
    deliveries["money_abs"] = money_vals.abs()
    deliveries = deliveries[deliveries["money_abs"] > 0].copy()

    if deliveries.empty or "from" not in deliveries.columns or "to" not in deliveries.columns:
        plt.figure(figsize=(10, 5)); plt.title("Supply Structure (Money) — no internal deliveries"); plt.savefig(os.path.join(outdir, "5_supply_structure_money.png")); plt.close()
        pd.DataFrame().to_csv(os.path.join(outdir, "5_supply_structure_money.csv"), index=False); return

    # Reclassify generic deliveries using contracts (keep explicit on_spot/backlog types)
    contracts = df_contracts.copy()
    for c in ["seller", "buyer", "type_norm", "start_round", "planned_end_round"]:
        if c not in contracts.columns:
            contracts[c] = np.nan
    contracts = contracts.dropna(subset=["seller", "buyer", "start_round", "planned_end_round"])
    contracts["type_norm"] = contracts["type_norm"].astype(str).str.lower()
    contracts = contracts[contracts["type_norm"].isin(["fixed", "flexible"])]

    if not contracts.empty:
        deliveries = deliveries.reset_index(drop=False).rename(columns={"index":"_row_id"})
        merged = deliveries.merge(contracts[["seller","buyer","type_norm","start_round","planned_end_round"]],
                                  left_on=["from","to"], right_on=["seller","buyer"], how="left")
        merged["_active"] = (pd.to_numeric(merged["round"], errors="coerce") >= pd.to_numeric(merged["start_round"], errors="coerce")) & \
                            (pd.to_numeric(merged["round"], errors="coerce") <= pd.to_numeric(merged["planned_end_round"], errors="coerce"))
        priority = {"fixed": 2, "flexible": 1}
        merged["prio"] = merged["type_norm_y"].map(priority).fillna(0)
        best = (merged[merged["_active"]].sort_values(["_row_id", "prio"], ascending=[True, False]).drop_duplicates("_row_id"))
        deliveries["type_norm"] = deliveries.get("type_norm", "").astype(str)
        mask_unknown = deliveries["type_norm"].isna() | (~deliveries["type_norm"].isin(["fixed","flexible","on_spot","backlog"]))
        deliveries = deliveries.merge(best[["_row_id", "type_norm_y"]], on="_row_id", how="left")
        deliveries.loc[mask_unknown & deliveries["type_norm_y"].notna(), "type_norm"] = deliveries.loc[mask_unknown, "type_norm_y"]
        deliveries.drop(columns=["type_norm_y"], inplace=True, errors="ignore")

    deliveries["type_norm"] = deliveries["type_norm"].replace({"one_time":"on_spot", "one-time":"on_spot", "on-spot":"on_spot"})
    deliveries["type_norm"] = deliveries["type_norm"].where(deliveries["type_norm"].isin(["flexible","fixed","on_spot","backlog"]), "on_spot")

    # Group and plot in USD
    grouped = deliveries.groupby(["round", "type_norm"], as_index=False)["money_abs"].sum()
    pivot = grouped.pivot(index="round", columns="type_norm", values="money_abs").fillna(0.0)
    for t in ["flexible", "fixed", "on_spot", "backlog"]:
        if t not in pivot.columns: pivot[t] = 0.0
    pivot = pivot[["flexible", "fixed", "on_spot", "backlog"]]
    # Ensure we include ALL rounds from original transactions (even if zero internal deliveries)
    all_rounds = sorted(pd.unique(pd.to_numeric(df_tx.get("round", pd.Series([], dtype=float)), errors="coerce").dropna()))
    try:
        all_rounds = [int(r) for r in all_rounds]
    except Exception:
        pass
    if len(all_rounds) > 0:
        pivot = pivot.reindex(all_rounds, fill_value=0.0)


    rounds = pivot.index.values
    plt.figure(figsize=(11, 6))
    bottom = np.zeros(len(rounds))
    for col in pivot.columns:
        plt.bar(rounds, pivot[col].values, bottom=bottom, label=col.replace("_", " "))
        bottom += pivot[col].values
    plt.xlabel("Round"); plt.ylabel("USD"); plt.title("Supply Structure by Delivery Type (Money per round)")
    plt.grid(True, axis="y", linestyle="--", alpha=0.4); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "5_supply_structure_money.png")); plt.close()

    pivot.reset_index().to_csv(os.path.join(outdir, "5_supply_structure_money.csv"), index=False)


def plot_contract_timeline(df_contracts: pd.DataFrame, outdir: str, df_events: pd.DataFrame = None, df_tx: pd.DataFrame = None):
    if df_contracts.empty:
        plt.figure(figsize=(10, 5)); plt.title("Contract Timeline — no contracts"); plt.savefig(os.path.join(outdir, "6_contract_timeline.png")); plt.close()
        pd.DataFrame().to_csv(os.path.join(outdir, "6_contract_timeline.csv"), index=False); return

    df = df_contracts.copy()
    df["start_round"] = pd.to_numeric(df.get("start_round", np.nan), errors="coerce")
    df["planned_end_round"] = pd.to_numeric(df.get("planned_end_round", np.nan), errors="coerce")
    df = df.dropna(subset=["start_round", "planned_end_round"]).copy()
    if df.empty:
        plt.figure(figsize=(10, 5)); plt.title("Contract Timeline — no start/length info"); plt.savefig(os.path.join(outdir, "6_contract_timeline.png")); plt.close()
        pd.DataFrame().to_csv(os.path.join(outdir, "6_contract_timeline.csv"), index=False); return

    df["contract_id"] = df["contract_id"].astype(str)
    # Sort IDs numerically when possible
    ids = df["contract_id"].astype(str).unique().tolist()
    ids_num = pd.to_numeric(pd.Series(ids), errors="coerce")
    order_df = pd.DataFrame({"cid": ids, "num": ids_num})
    order_df = order_df.sort_values(by=["num","cid"], na_position="last")
    contract_ids_sorted = order_df["cid"].tolist()

    y_map = {cid: i for i, cid in enumerate(contract_ids_sorted, start=1)}
    df["ypos"] = df["contract_id"].map(y_map)

    plt.figure(figsize=(12, max(6, 0.3 * len(contract_ids_sorted))))
    directions = ["Factory→Wholesale", "Wholesale→Retail"]
    dir_colors = {"Factory→Wholesale": "red", "Wholesale→Retail": "blue", "Other": "gray"}
    direction_to_df = {d: df[df["direction"] == d] for d in directions}
    others_df = df[~df["direction"].isin(directions)]

    # Extract break/cancel rounds from events if provided
    broken_round = {}
    cancelled_round = {}
    if df_events is not None and not df_events.empty:
        # Build timestamp→round map from transactions if df_tx provided
        ts_round_map = None
        if df_tx is not None and not df_tx.empty and "timestamp" in df_tx.columns and "round" in df_tx.columns:
            try:
                tx = df_tx[["round","timestamp"]].copy()
                tx["ts"] = pd.to_datetime(tx["timestamp"], errors="coerce")
                tx = tx.dropna(subset=["ts","round"])
                # Use earliest timestamp per round as round start proxy
                starts = tx.groupby("round", as_index=False)["ts"].min().sort_values("round")
                ts_round_map = starts
            except Exception:
                ts_round_map = None

        ev = df_events.copy()
        # Ensure contract_id and round present
        if "contract" in ev.columns:
            cdf = pd.json_normalize(ev["contract"])
            ev = pd.concat([ev.drop(columns=["contract"]), cdf], axis=1)
        if "contract_id" in ev.columns:
            ev["contract_id"] = ev["contract_id"].astype(str)
            # Coerce round if available
            if "round" in ev.columns:
                ev["round"] = pd.to_numeric(ev["round"], errors="coerce")
            else:
                ev["round"] = np.nan
            ev["action"] = ev.get("action", "").astype(str).str.upper()
            # First broken/cancel per id
            for cid, sub in ev.groupby("contract_id"):
                # Try direct round first
                br = sub.loc[sub["action"] == "BROKEN", "round"].dropna()
                cr = sub.loc[sub["action"] == "MUTUALLY_CANCELLED", "round"].dropna()

                # If missing, infer from timestamp via tx round starts
                if (br.empty or cr.empty) and ts_round_map is not None:
                    sub2 = sub.copy()
                    if "timestamp" in sub2.columns:
                        sub2["ts"] = pd.to_datetime(sub2["timestamp"], errors="coerce")
                    else:
                        sub2["ts"] = pd.NaT

                    def infer_round(t):
                        if pd.isna(t) or ts_round_map is None or ts_round_map.empty:
                            return np.nan
                        # pick the largest round whose start <= t
                        mask = ts_round_map["ts"] <= t
                        if not mask.any():
                            return ts_round_map["round"].min()
                        return ts_round_map.loc[mask, "round"].max()

                    # Fill inferred rounds for missing
                    if br.empty:
                        br_ts = sub2.loc[sub2["action"] == "BROKEN", "ts"].dropna()
                        if not br_ts.empty:
                            inf_r = infer_round(br_ts.iloc[0])
                            if pd.notna(inf_r):
                                br = pd.Series([inf_r])
                    if cr.empty:
                        cr_ts = sub2.loc[sub2["action"] == "MUTUALLY_CANCELLED", "ts"].dropna()
                        if not cr_ts.empty:
                            inf_r = infer_round(cr_ts.iloc[0])
                            if pd.notna(inf_r):
                                cr = pd.Series([inf_r])

                if not br.empty:
                    broken_round[cid] = float(br.iloc[0])
                if not cr.empty:
                    cancelled_round[cid] = float(cr.iloc[0])    
    def ls_for_type(t):
        t = (t or "").lower()
        if "fixed" in t:
            return "-"
        if "flexible" in t:
            return "--"
        return "-."

    for d in directions:
        sub = direction_to_df[d]
        if not sub.empty:
            for _, row in sub.iterrows():
                plt.hlines(y=row["ypos"], xmin=row["start_round"], xmax=row["planned_end_round"],
                           linewidth=3.5, linestyles=ls_for_type(row.get("type_norm", "")),
                           colors=dir_colors.get(d, None))
                # amount label at midpoint
                try:
                    amt = row.get("amount", np.nan)
                    if pd.notna(amt):
                        xm = (float(row["start_round"]) + float(row["planned_end_round"])) / 2.0
                        plt.text(xm, row["ypos"] + 0.15, f"{int(amt)}", ha="center", va="bottom", fontsize=8)
                except Exception:
                    pass
                # markers for break/cancel
                cid = str(row.get("contract_id"))
                if cid in broken_round:
                    plt.plot([broken_round[cid]], [row["ypos"]], marker='x', markersize=8, color=dir_colors.get(d, None))
                if cid in cancelled_round:
                    plt.plot([cancelled_round[cid]], [row["ypos"]], marker='o', markersize=5, fillstyle='none', color=dir_colors.get(d, None))

    if not others_df.empty:
        for _, row in others_df.iterrows():
            plt.hlines(y=row["ypos"], xmin=row["start_round"], xmax=row["planned_end_round"],
                       linewidth=3.5, linestyles=ls_for_type(row.get("type_norm", "")),
                       colors=dir_colors.get("Other", None))
            try:
                amt = row.get("amount", np.nan)
                if pd.notna(amt):
                    xm = (float(row["start_round"]) + float(row["planned_end_round"])) / 2.0
                    plt.text(xm, row["ypos"] + 0.15, f"{int(amt)}", ha="center", va="bottom", fontsize=8)
            except Exception:
                pass
            cid = str(row.get("contract_id"))
            if cid in broken_round:
                plt.plot([broken_round[cid]], [row["ypos"]], marker='x', markersize=8, color=dir_colors.get("Other", None))
            if cid in cancelled_round:
                plt.plot([cancelled_round[cid]], [row["ypos"]], marker='o', markersize=5, fillstyle='none', color=dir_colors.get("Other", None))

    direction_handles = [
        plt.Line2D([0], [0], linewidth=3.5, linestyle="-", color="red", label="Factory→Wholesale"),
        plt.Line2D([0], [0], linewidth=3.5, linestyle="-", color="blue", label="Wholesale→Retail"),
    ]
    legend_ls = [
        plt.Line2D([0], [0], linewidth=3.5, linestyle="-", label="Fixed"),
        plt.Line2D([0], [0], linewidth=3.5, linestyle="--", label="Flexible"),
        plt.Line2D([0], [0], linewidth=3.5, linestyle="-.", label="Other/Unknown"),
    ]

    plt.yticks(list(y_map.values()), list(y_map.keys()))
    plt.xlabel("Round"); plt.ylabel("Contract ID"); plt.title("Contracts Timeline")
    plt.grid(True, axis="x", linestyle="--", alpha=0.4)
    first_legend = plt.legend(handles=direction_handles, loc="upper left", title="Direction")
    plt.gca().add_artist(first_legend)
    plt.legend(handles=legend_ls, loc="upper right", title="Type (linestyle)")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "6_contract_timeline.png")); plt.close()

    out = df[["contract_id", "direction", "type_norm", "start_round", "planned_end_round"]].copy()
    out.to_csv(os.path.join(outdir, "6_contract_timeline.csv"), index=False)


def main():
    parser = argparse.ArgumentParser(description="Analyze AI supply-chain game results (single JSON file).")
    parser.add_argument("input_json", help="Path to game results JSON file")
    parser.add_argument("--outdir", default="out", help="Output directory for charts and CSVs")
    parser.add_argument("--start-money", type=float, default=10000.0, help="Starting money per company")
    parser.add_argument("--retail-price", type=float, default=6.0, help="Retail beer price for market & valuation")
    args = parser.parse_args()

    ensure_outdir(args.outdir)

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    df_tx = normalize_transactions(safe_get(data, "transactions", []))
    df_state = normalize_round_states(safe_get(data, "round_states", []))
    df_market = normalize_market_demand(safe_get(data, "market_demand", safe_get(data, "market_demand_log", [])))
    df_contracts, df_contract_events = normalize_contracts(safe_get(data, "contracts", safe_get(data, "contracts_log", [])))

    plot_money_cumulative(df_state, df_market, args.outdir, args.start_money, args.retail_price)
    plot_money_delta(df_state, df_market, args.outdir, args.start_money, args.retail_price)
    plot_value_cumulative(df_state, df_market, args.outdir, args.start_money, args.retail_price)
    plot_value_delta(df_state, df_market, args.outdir, args.start_money, args.retail_price)
    plot_supply_structure(df_tx, df_contracts, args.outdir)
    plot_supply_structure_money(df_tx, df_contracts, args.outdir)
    plot_contract_timeline(df_contracts, args.outdir, df_contract_events, df_tx)

    df_tx.to_csv(os.path.join(args.outdir, "transactions_normalized.csv"), index=False)
    df_state.to_csv(os.path.join(args.outdir, "round_states_normalized.csv"), index=False)
    df_market.to_csv(os.path.join(args.outdir, "market_demand_normalized.csv"), index=False)
    df_contracts.to_csv(os.path.join(args.outdir, "contracts_created_normalized.csv"), index=False)
    df_contract_events.to_csv(os.path.join(args.outdir, "contracts_events.csv"), index=False)

    print(f"Done. Outputs saved to: {args.outdir}")
    print("Figures: 1_money_cumulative.png, 2_profit_per_round.png, 3_value_cumulative.png, 4_value_per_round.png, 5_supply_structure.png, 6_contract_timeline.png")


if __name__ == "__main__":
    main()
