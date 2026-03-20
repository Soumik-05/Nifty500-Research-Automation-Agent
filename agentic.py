import pandas as pd
import yfinance as yf
import json
import time
import os
import requests
import io
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CONFIGURATION & PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(BASE_DIR, "nifty500_fundamental_data.json")
TOP_STOCKS_PATH = os.path.join(BASE_DIR, "top150_recommended.json")
TOP_N = 150

def get_nifty_500_list():
    """Fetches official NSE list from the source."""
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers)
        df = pd.read_csv(io.StringIO(resp.text))
        return df['Symbol'].tolist()
    except Exception as e:
        print(f"❌ NSE List Error: {e}")
        return []

def calculate_score(fundamentals):
    """
    REAL-WORLD WEIGHTED SCORING SYSTEM — Out of 1.0
    Designed to match how institutional analysts evaluate stocks.

    PHILOSOPHY:
    - Profitability is king  (ROE + Margins     = 40% of score)
    - Growth must be quality (Earnings > Revenue weight)
    - Valuation needs context (PE 10-20 is sweet spot)
    - Debt is sector-sensitive
    - Efficiency separates good from great
    - Liquidity is a safety check, not a reward

    ┌──────────────────────┬────────┬─────────────────────────────────────┬──────────┐
    │ Metric               │ Weight │ Condition                           │ Points   │
    ├──────────────────────┼────────┼─────────────────────────────────────┼──────────┤
    │ ROE                  │ 2.5x   │ > 25% (exceptional)                 │ +5.0     │
    │ PROFITABILITY #1     │        │ > 20% (strong)                      │ +4.0     │
    │                      │        │ > 15% (good)                        │ +2.5     │
    │                      │        │ > 10% (average)                     │ +1.0     │
    │                      │        │ < 0%  (penalty)                     │ -3.0     │
    ├──────────────────────┼────────┼─────────────────────────────────────┼──────────┤
    │ Net Profit Margin    │ 2.0x   │ > 25% (exceptional)                 │ +4.0     │
    │ PROFITABILITY #2     │        │ > 15% (strong)                      │ +3.0     │
    │                      │        │ > 8%  (acceptable)                  │ +1.5     │
    │                      │        │ > 0%  (at least profitable)         │ +0.5     │
    │                      │        │ < 0%  (penalty)                     │ -3.0     │
    ├──────────────────────┼────────┼─────────────────────────────────────┼──────────┤
    │ Earnings Growth      │ 2.0x   │ > 25% (high growth)                 │ +4.0     │
    │ GROWTH #1            │        │ > 15% (strong growth)               │ +3.0     │
    │                      │        │ > 8%  (moderate growth)             │ +1.5     │
    │                      │        │ > 0%  (at least growing)            │ +0.5     │
    │                      │        │ < -10% (penalty)                    │ -2.0     │
    ├──────────────────────┼────────┼─────────────────────────────────────┼──────────┤
    │ PE Ratio             │ 1.5x   │ 0 < PE < 10  (deep value)           │ +2.0     │
    │ VALUATION            │        │ 10 <= PE < 20 (sweet spot)          │ +3.0     │
    │                      │        │ 20 <= PE < 35 (growth premium)      │ +1.5     │
    │                      │        │ 35 <= PE < 50 (expensive)           │ +0.5     │
    │                      │        │ PE >= 50 (penalty)                  │ -1.5     │
    │                      │        │ PE < 0  (penalty)                   │ -2.0     │
    ├──────────────────────┼────────┼─────────────────────────────────────┼──────────┤
    │ Debt/Equity          │ 1.5x   │ SECTOR SENSITIVE (see below)        │ varies   │
    │ FINANCIAL HEALTH     │        │                                     │          │
    ├──────────────────────┼────────┼─────────────────────────────────────┼──────────┤
    │ Revenue Growth       │ 1.0x   │ > 20% (rapid expansion)             │ +2.0     │
    │ GROWTH #2            │        │ > 10% (healthy)                     │ +1.5     │
    │                      │        │ > 5%  (slow but growing)            │ +0.75    │
    │                      │        │ < -5% (penalty)                     │ -1.5     │
    ├──────────────────────┼────────┼─────────────────────────────────────┼──────────┤
    │ ROA                  │ 1.0x   │ > 15% (exceptional)                 │ +2.0     │
    │ EFFICIENCY #1        │        │ > 10% (strong)                      │ +1.5     │
    │                      │        │ > 5%  (acceptable)                  │ +0.75    │
    │                      │        │ < 0%  (penalty)                     │ -1.5     │
    ├──────────────────────┼────────┼─────────────────────────────────────┼──────────┤
    │ Operating Margins    │ 1.0x   │ > 25% (pricing power)               │ +2.0     │
    │ EFFICIENCY #2        │        │ > 15% (strong ops)                  │ +1.5     │
    │                      │        │ > 8%  (acceptable)                  │ +0.75    │
    │                      │        │ < 0%  (penalty)                     │ -2.0     │
    ├──────────────────────┼────────┼─────────────────────────────────────┼──────────┤
    │ Current Ratio        │ 0.5x   │ 1.5 to 3.0 (ideal)                  │ +1.0     │
    │ LIQUIDITY            │        │ 1.0 to 1.5 (acceptable)             │ +0.5     │
    │                      │        │ > 3.0 (idle cash)                   │ +0.25    │
    │                      │        │ < 1.0 (penalty)                     │ -1.0     │
    └──────────────────────┴────────┴─────────────────────────────────────┴──────────┘

    DEBT/EQUITY SECTOR RULES:
    - Banks              → Skip (use deposits not equity)
    - NBFC/Insurance     → DE < 3.0 good, > 7.0 penalty
    - IT/Pharma/Consumer → DE < 0.3 excellent, >= 2.0 penalty
    - Infra/Real Estate  → DE < 2.5 acceptable, >= 4.0 penalty
    - Manufacturing      → DE < 1.0 good, >= 2.0 penalty

    MAX RAW SCORE ~28 → Normalized to 1.0
    """
    score = 0
    sector   = fundamentals.get("sector",   "N/A")
    industry = fundamentals.get("industry", "N/A")

    pe  = fundamentals.get("pe_ratio_ttm",    "N/A")
    roe = fundamentals.get("roe",              "N/A")
    de  = fundamentals.get("debt_to_equity",   "N/A")
    pm  = fundamentals.get("profit_margins",   "N/A")
    om  = fundamentals.get("operating_margins","N/A")
    rg  = fundamentals.get("revenue_growth",   "N/A")
    cr  = fundamentals.get("current_ratio",    "N/A")
    eg  = fundamentals.get("earnings_growth",  "N/A")
    roa = fundamentals.get("roa",              "N/A")

    # ── 1. ROE — Weight 2.5x (Most Important) ────────────────────────
    if roe not in ("N/A", None):
        if roe > 0.25:        score += 5.0   # Exceptional
        elif roe > 0.20:      score += 4.0   # Strong
        elif roe > 0.15:      score += 2.5   # Good
        elif roe > 0.10:      score += 1.0   # Average
        elif roe < 0:         score -= 3.0   # Destroying shareholder value

    # ── 2. Net Profit Margin — Weight 2.0x ───────────────────────────
    if pm not in ("N/A", None):
        if pm > 0.25:         score += 4.0   # Exceptional
        elif pm > 0.15:       score += 3.0   # Strong
        elif pm > 0.08:       score += 1.5   # Acceptable
        elif pm > 0:          score += 0.5   # At least profitable
        elif pm < 0:          score -= 3.0   # Loss making

    # ── 3. Earnings Growth — Weight 2.0x ─────────────────────────────
    if eg not in ("N/A", None):
        if eg > 0.25:         score += 4.0   # High growth
        elif eg > 0.15:       score += 3.0   # Strong growth
        elif eg > 0.08:       score += 1.5   # Moderate growth
        elif eg > 0:          score += 0.5   # At least growing
        elif eg < -0.10:      score -= 2.0   # Shrinking fast

    # ── 4. PE Ratio — Weight 1.5x ────────────────────────────────────
    if pe not in ("N/A", None):
        if pe < 0:            score -= 2.0   # Negative PE = loss making
        elif 0 < pe < 10:     score += 2.0   # Deep value
        elif 10 <= pe < 20:   score += 3.0   # Sweet spot
        elif 20 <= pe < 35:   score += 1.5   # Growth premium, acceptable
        elif 35 <= pe < 50:   score += 0.5   # Expensive
        elif pe >= 50:        score -= 1.5   # Overvalued

    # ── 5. Debt/Equity — Weight 1.5x — SECTOR SENSITIVE ─────────────
    if de not in ("N/A", None):

        # Banks — skip D/E (they use deposits, not equity by nature)
        if "Bank" in str(industry):
            pass

        # NBFC / Insurance / Other Financial
        elif sector == "Financial Services":
            if de < 3.0:      score += 3.0
            elif de < 5.0:    score += 1.5
            elif de < 7.0:    score += 0.5
            elif de >= 7.0:   score -= 1.5   # Too leveraged

        # IT / Pharma / Consumer Defensive — should be near debt free
        elif sector in ("Technology", "Healthcare", "Consumer Defensive"):
            if de < 0.3:      score += 3.0   # Nearly debt free
            elif de < 0.5:    score += 2.0
            elif de < 1.0:    score += 1.0
            elif de >= 2.0:   score -= 2.0   # Red flag

        # Infrastructure / Real Estate / Utilities — capital intensive
        elif sector in ("Real Estate", "Utilities"):
            if de < 1.5:      score += 3.0
            elif de < 2.5:    score += 1.5
            elif de < 4.0:    score += 0.5
            elif de >= 4.0:   score -= 1.5

        # Manufacturing / Industrials / Everything else
        else:
            if de < 0.5:      score += 3.0
            elif de < 1.0:    score += 2.0
            elif de < 2.0:    score += 0.5
            elif de >= 2.0:   score -= 2.0

    # ── 6. Revenue Growth — Weight 1.0x ──────────────────────────────
    if rg not in ("N/A", None):
        if rg > 0.20:         score += 2.0   # Rapid expansion
        elif rg > 0.10:       score += 1.5   # Healthy
        elif rg > 0.05:       score += 0.75  # Slow but growing
        elif rg < -0.05:      score -= 1.5   # Shrinking revenue

    # ── 7. ROA — Weight 1.0x ─────────────────────────────────────────
    if roa not in ("N/A", None):
        if roa > 0.15:        score += 2.0   # Exceptional asset use
        elif roa > 0.10:      score += 1.5   # Strong
        elif roa > 0.05:      score += 0.75  # Acceptable
        elif roa < 0:         score -= 1.5   # Assets not generating returns

    # ── 8. Operating Margins — Weight 1.0x ───────────────────────────
    if om not in ("N/A", None):
        if om > 0.25:         score += 2.0   # Strong pricing power
        elif om > 0.15:       score += 1.5   # Good operations
        elif om > 0.08:       score += 0.75  # Acceptable
        elif om < 0:          score -= 2.0   # Operations losing money

    # ── 9. Current Ratio — Weight 0.5x (Safety Check) ────────────────
    if cr not in ("N/A", None):
        if 1.5 <= cr <= 3.0:  score += 1.0   # Ideal range
        elif 1.0 <= cr < 1.5: score += 0.5   # Acceptable
        elif cr > 3.0:        score += 0.25  # Too much idle cash
        elif cr < 1.0:        score -= 1.0   # Liquidity risk

    # ── Normalize to 0.0–1.0 (max raw ~28) ───────────────────────────
    normalized = score / 28
    return round(max(min(normalized, 1.0), 0.0), 2)

def get_score_label(score):
    if score >= 0.8:   return "🟢 Excellent"
    elif score >= 0.6: return "🟡 Good"
    elif score >= 0.4: return "🟠 Average"
    else:              return "🔴 Weak"

def get_recommendation(score):
    if score >= 0.8:   return "STRONG BUY"
    elif score >= 0.6: return "BUY"
    elif score >= 0.4: return "HOLD"
    else:              return "AVOID"

def fetch_single_stock(symbol):
    """Fetches data for one stock — runs in parallel."""
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        info = ticker.info

        if not info or info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
            print(f"⚠️ No data for {symbol}, skipping.")
            return None

        # Market cap filter — skip stocks below 500 Cr
        market_cap_cr = (info.get("marketCap", 0) or 0) / 10**7
        if market_cap_cr < 500:
            print(f"⚠️ Skipping {symbol} — Market Cap ₹{market_cap_cr:.0f} Cr < ₹500 Cr")
            return None

        fundamentals = {
            "market_cap_cr":    market_cap_cr,
            "pe_ratio_ttm":     info.get("trailingPE",       "N/A"),
            "pb_ratio":         info.get("priceToBook",      "N/A"),
            "debt_to_equity":   info.get("debtToEquity",     "N/A"),
            "roe":              info.get("returnOnEquity",   "N/A"),
            "roa":              info.get("returnOnAssets",   "N/A"),
            "eps_ttm":          info.get("trailingEps",      "N/A"),
            "book_value":       info.get("bookValue",        "N/A"),
            "div_yield":        info.get("dividendYield",    "N/A"),
            "revenue_growth":   info.get("revenueGrowth",   "N/A"),
            "earnings_growth":  info.get("earningsGrowth",  "N/A"),
            "gross_margins":    info.get("grossMargins",     "N/A"),
            "operating_margins":info.get("operatingMargins","N/A"),
            "profit_margins":   info.get("profitMargins",   "N/A"),
            "current_ratio":    info.get("currentRatio",    "N/A"),
            "sector":           info.get("sector",          "N/A"),
            "industry":         info.get("industry",        "N/A")
        }

        score = calculate_score(fundamentals)

        return {
            "symbol":       symbol,
            "company_name": info.get("longName", "N/A"),
            "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "price_info": {
                "current_price":      info.get("currentPrice",    "N/A"),
                "currency":           info.get("currency",        "INR"),
                "day_high":           info.get("dayHigh",         "N/A"),
                "day_low":            info.get("dayLow",          "N/A"),
                "fifty_two_week_high":info.get("fiftyTwoWeekHigh","N/A"),
                "fifty_two_week_low": info.get("fiftyTwoWeekLow", "N/A")
            },
            "fundamentals": fundamentals,
            "fundamental_score": {
                "score":  score,   # 0.0 to 1.0
                "out_of": 1,
                "label":  get_score_label(score)
                # recommendation not in JSON — shown in terminal only
            }
        }

    except Exception as e:
        print(f"⚠️ Skipping {symbol} due to error: {e}")
        return None

def prepare_for_orchestrator(top_stocks):
    """Packages top 150 summary for the orchestrator agent — NO stock list in JSON."""
    return {
        "agent":               "fundamental_agent",
        "generated_at":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_recommended":   len(top_stocks),
        "selection_criteria":  "Top 150 by real-world weighted fundamental score (out of 500)",
        "score_breakdown": {
            "excellent_0.8_to_1.0": len([s for s in top_stocks if s["fundamental_score"]["score"] >= 0.8]),
            "good_0.6_to_0.8":      len([s for s in top_stocks if 0.6 <= s["fundamental_score"]["score"] < 0.8]),
            "average_0.4_to_0.6":   len([s for s in top_stocks if 0.4 <= s["fundamental_score"]["score"] < 0.6]),
            "weak_0_to_0.4":        len([s for s in top_stocks if s["fundamental_score"]["score"] < 0.4]),
        },
        "sector_distribution": get_sector_distribution(top_stocks)
    }

def get_sector_distribution(stocks):
    """Shows how many recommended stocks are from each sector."""
    distribution = {}
    for stock in stocks:
        sector = stock["fundamentals"].get("sector", "Unknown")
        distribution[sector] = distribution.get(sector, 0) + 1
    return dict(sorted(distribution.items(), key=lambda x: x[1], reverse=True))

def fundamental_agent_nifty500():
    symbols = get_nifty_500_list()

    if not symbols:
        print("❌ No symbols fetched. Exiting.")
        return []

    all_data = []
    failed_symbols = []
    start_time = datetime.now()

    print(f"--- 🚀 SCANNING {len(symbols)} STOCKS WITH 20 WORKERS ---")
    print(f"⏱️  Started at: {start_time.strftime('%H:%M:%S')}")
    print(f"🎯 Goal: Score all 500 → Recommend top {TOP_N} to Orchestrator\n")

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(fetch_single_stock, sym): sym for sym in symbols}

        for i, future in enumerate(as_completed(futures), 1):
            symbol = futures[future]
            result = future.result()

            if result:
                all_data.append(result)
            else:
                failed_symbols.append(symbol)

            if i % 20 == 0:
                elapsed = int((datetime.now() - start_time).total_seconds())
                remaining = int((elapsed / i) * (len(symbols) - i))
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ {i}/{len(symbols)} | "
                      f"⏱️ Elapsed: {elapsed}s | ETA: {remaining}s")

    # Retry failed symbols once
    if failed_symbols:
        print(f"\n🔄 Retrying {len(failed_symbols)} failed stocks...")
        for symbol in failed_symbols:
            time.sleep(2)
            result = fetch_single_stock(symbol)
            if result:
                all_data.append(result)
                print(f"✅ Recovered: {symbol}")
            else:
                print(f"❌ Permanently failed: {symbol}")

    # ✅ Sort all 500 by score
    all_data.sort(key=lambda x: x["fundamental_score"]["score"], reverse=True)

    # ✅ Pick top 150
    top_150 = all_data[:TOP_N]

    # ✅ Save ONLY top 150 to main JSON
    with open(JSON_PATH, "w") as f:
        json.dump(top_150, f, indent=4)
    print(f"💾 Top {TOP_N} saved to: {JSON_PATH}")

    # ✅ Save orchestrator summary
    orchestrator_payload = prepare_for_orchestrator(top_150)
    with open(TOP_STOCKS_PATH, "w") as f:
        json.dump(orchestrator_payload, f, indent=4)
    print(f"🤖 Orchestrator payload saved to: {TOP_STOCKS_PATH}")

    # --- FINAL REPORT ---
    total_time = int((datetime.now() - start_time).total_seconds())
    print(f"\n{'='*65}")
    print(f"  📊 FUNDAMENTAL AGENT — FINAL REPORT")
    print(f"{'='*65}")
    print(f"  ✅ Total Fetched   : {len(all_data)}/{len(symbols)}")
    print(f"  ❌ Total Failed    : {len(failed_symbols)}")
    print(f"  ⏱️  Total Time      : {total_time//60}m {total_time%60}s")
    print(f"  💾 Main JSON       : {JSON_PATH}  ({TOP_N} stocks)")
    print(f"  🤖 Orchestrator    : {TOP_STOCKS_PATH}  (summary only)")
    print(f"{'='*65}")

    # ✅ Print ALL 150 in terminal
    print(f"\n🏆 TOP {TOP_N} STOCKS RECOMMENDED TO ORCHESTRATOR:")
    print(f"{'Rank':<5} {'Symbol':<12} {'Company':<35} {'Score':<8} {'Label':<18} {'Action'}")
    print("-" * 90)
    for rank, stock in enumerate(top_150, 1):
        fs = stock["fundamental_score"]
        recommendation = get_recommendation(fs["score"])
        print(f"{rank:<5} {stock['symbol']:<12} {stock['company_name']:<35} "
              f"{fs['score']:<8} {fs['label']:<18} {recommendation}")

    print(f"\n📂 Sector Distribution of Top {TOP_N}:")
    for sector, count in orchestrator_payload["sector_distribution"].items():
        bar = "█" * count
        print(f"  {sector:<30} {count:>3} {bar}")

    return orchestrator_payload

if __name__ == "__main__":
    while True:
        try:
            fundamental_agent_nifty500()
        except Exception as e:
            print(f"❌ Fatal error: {e}. Retrying in 1 hour.")
            time.sleep(3600)
            continue
        print("\n💤 Scan complete. Next full update in 24 hours.")
        time.sleep(86400)