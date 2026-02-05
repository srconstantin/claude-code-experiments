#!/usr/bin/env python3
"""
Step 2: Determine company status and get stock price data.

For each company:
1. Check if ticker is still actively trading
2. Get stock price history (IPO to current/final)
3. For non-trading companies, determine status (acquired/bankrupt/delisted)
4. Calculate returns

Output:
- step2_company_status.csv: Company status and key metrics
- step2_price_history/: Individual CSV files with price history per company
"""

import csv
import time
import re
import os
import requests
from datetime import datetime, timedelta
from pathlib import Path

import yfinance as yf
import pandas as pd

HEADERS = {
    "User-Agent": "Academic Research sarah.constantin@gmail.com",
    "Accept-Encoding": "gzip, deflate",
}

RATE_LIMIT_DELAY = 0.15

# Create directory for price history
PRICE_DIR = Path("step2_price_history")
PRICE_DIR.mkdir(exist_ok=True)


def check_ticker_status(ticker: str) -> dict:
    """
    Check if a ticker is actively trading using yfinance.
    Returns status info including current price, market cap, etc.
    """
    if not ticker:
        return {"status": "no_ticker", "details": {}}

    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Check if we got valid data
        if not info or info.get("regularMarketPrice") is None:
            # Try to get historical data to see if it ever traded
            hist = stock.history(period="max")
            if hist.empty:
                return {"status": "never_traded", "details": {}}
            else:
                # Had history but no current price - likely delisted
                last_date = hist.index[-1].strftime("%Y-%m-%d")
                last_price = hist["Close"].iloc[-1]
                return {
                    "status": "delisted",
                    "details": {
                        "last_trade_date": last_date,
                        "last_price": last_price,
                    }
                }

        # Active ticker
        return {
            "status": "trading",
            "details": {
                "current_price": info.get("regularMarketPrice"),
                "market_cap": info.get("marketCap"),
                "exchange": info.get("exchange"),
                "52wk_high": info.get("fiftyTwoWeekHigh"),
                "52wk_low": info.get("fiftyTwoWeekLow"),
            }
        }

    except Exception as e:
        return {"status": "error", "details": {"error": str(e)}}


def get_price_history(ticker: str, start_date: str = None) -> pd.DataFrame | None:
    """
    Get full price history for a ticker.
    """
    if not ticker:
        return None

    try:
        stock = yf.Ticker(ticker)

        if start_date:
            hist = stock.history(start=start_date)
        else:
            hist = stock.history(period="max")

        if hist.empty:
            return None

        return hist

    except Exception as e:
        print(f"    Error getting history: {e}")
        return None


def get_8k_filings(cik: str, limit: int = 50) -> list[dict]:
    """
    Get recent 8-K filings to look for material events (merger, acquisition, bankruptcy).
    """
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"

    time.sleep(RATE_LIMIT_DELAY)
    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        return []

    data = response.json()
    filings = []
    recent = data.get("filings", {}).get("recent", {})

    if not recent:
        return []

    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    descriptions = recent.get("primaryDocDescription", [])

    count = 0
    for i, form in enumerate(forms):
        if form == "8-K" and count < limit:
            filings.append({
                "date": dates[i] if i < len(dates) else None,
                "description": descriptions[i] if i < len(descriptions) else "",
            })
            count += 1

    return filings


def determine_exit_status(cik: str, company_name: str) -> dict:
    """
    Try to determine how a non-trading company exited (acquired, bankrupt, delisted).
    Looks at 8-K filings for keywords.
    """
    filings = get_8k_filings(cik)

    if not filings:
        return {"exit_type": "unknown", "exit_date": None, "details": ""}

    # Keywords to look for
    acquisition_keywords = ["merger", "acquisition", "acquired", "tender offer", "business combination"]
    bankruptcy_keywords = ["bankruptcy", "chapter 11", "chapter 7", "liquidation", "wind down", "cease operations"]
    delisting_keywords = ["delisting", "delisted", "nasdaq", "removal from listing"]

    for filing in filings:
        desc = (filing.get("description") or "").lower()

        for kw in acquisition_keywords:
            if kw in desc:
                return {
                    "exit_type": "acquired",
                    "exit_date": filing["date"],
                    "details": filing["description"]
                }

        for kw in bankruptcy_keywords:
            if kw in desc:
                return {
                    "exit_type": "bankrupt",
                    "exit_date": filing["date"],
                    "details": filing["description"]
                }

        for kw in delisting_keywords:
            if kw in desc:
                return {
                    "exit_type": "delisted",
                    "exit_date": filing["date"],
                    "details": filing["description"]
                }

    return {"exit_type": "unknown", "exit_date": None, "details": ""}


def calculate_returns(hist: pd.DataFrame, ipo_date: str = None) -> dict:
    """
    Calculate various return metrics from price history.
    """
    if hist is None or hist.empty:
        return {}

    try:
        first_price = hist["Close"].iloc[0]
        last_price = hist["Close"].iloc[-1]
        max_price = hist["Close"].max()
        min_price = hist["Close"].min()

        first_date = hist.index[0].strftime("%Y-%m-%d")
        last_date = hist.index[-1].strftime("%Y-%m-%d")
        max_date = hist["Close"].idxmax().strftime("%Y-%m-%d")

        total_return = (last_price - first_price) / first_price * 100
        max_return = (max_price - first_price) / first_price * 100

        # Calculate days trading
        days_trading = (hist.index[-1] - hist.index[0]).days

        return {
            "first_price": round(first_price, 2),
            "last_price": round(last_price, 2),
            "max_price": round(max_price, 2),
            "min_price": round(min_price, 2),
            "first_date": first_date,
            "last_date": last_date,
            "max_date": max_date,
            "total_return_pct": round(total_return, 1),
            "max_return_pct": round(max_return, 1),
            "days_trading": days_trading,
        }

    except Exception as e:
        print(f"    Error calculating returns: {e}")
        return {}


def main():
    print("=" * 60)
    print("Step 2: Getting company status and stock prices")
    print("=" * 60)

    # Load companies
    with open('step1_biotech_ipos_since_2000_updated.csv', 'r') as f:
        reader = csv.DictReader(f)
        companies = list(reader)

    print(f"\nProcessing {len(companies)} companies...")

    results = []
    status_counts = {"trading": 0, "delisted": 0, "no_ticker": 0, "error": 0, "never_traded": 0}

    for i, co in enumerate(companies):
        name = co["name"]
        ticker = co.get("ticker", "")
        cik = co["cik"]

        print(f"\n[{i+1}/{len(companies)}] {name[:50]}")
        print(f"    Ticker: {ticker or 'N/A'}")

        # Check ticker status
        ticker_info = check_ticker_status(ticker)
        status = ticker_info["status"]
        status_counts[status] = status_counts.get(status, 0) + 1

        print(f"    Status: {status}")

        # Get price history
        hist = None
        returns = {}

        if ticker and status in ["trading", "delisted"]:
            print(f"    Getting price history...")
            hist = get_price_history(ticker)

            if hist is not None and not hist.empty:
                returns = calculate_returns(hist)
                print(f"    Got {len(hist)} days of history")
                print(f"    Return: {returns.get('total_return_pct', 'N/A')}%")

                # Save price history to file
                price_file = PRICE_DIR / f"{ticker}.csv"
                hist.to_csv(price_file)

        # For non-trading companies, try to determine exit type
        exit_info = {"exit_type": "", "exit_date": "", "exit_details": ""}

        if status == "delisted":
            print(f"    Checking SEC filings for exit info...")
            exit_data = determine_exit_status(cik, name)
            exit_info = {
                "exit_type": exit_data["exit_type"],
                "exit_date": exit_data["exit_date"] or "",
                "exit_details": exit_data["details"][:100] if exit_data["details"] else ""
            }
            print(f"    Exit type: {exit_info['exit_type']}")

        # Compile result
        result = {
            "name": name,
            "cik": cik,
            "ticker": ticker,
            "founding_date": co.get("founding_date", ""),
            "s1_filing_date": co.get("s1_filing_date", ""),
            "state": co.get("state", ""),
            "city": co.get("city", ""),
            "sic_code": co.get("sic_code", ""),
            "status": status,
            "current_price": ticker_info["details"].get("current_price", ""),
            "market_cap": ticker_info["details"].get("market_cap", ""),
            "exchange": ticker_info["details"].get("exchange", ""),
            **exit_info,
            **{f"price_{k}": v for k, v in returns.items()},
        }

        results.append(result)

        # Save intermediate results every 50 companies
        if (i + 1) % 50 == 0:
            save_results(results, "step2_company_status_intermediate.csv")
            print(f"\n    Saved intermediate results ({len(results)} companies)")
            print(f"    Status counts: {status_counts}")

        # Small delay to avoid rate limiting
        time.sleep(0.5)

    # Save final results
    save_results(results, "step2_company_status.csv")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total companies processed: {len(results)}")
    print(f"\nStatus breakdown:")
    for status, count in sorted(status_counts.items(), key=lambda x: -x[1]):
        print(f"  {status}: {count}")

    # Calculate some aggregate stats
    trading = [r for r in results if r["status"] == "trading"]
    delisted = [r for r in results if r["status"] == "delisted"]

    print(f"\nCompanies still trading: {len(trading)}")
    print(f"Companies delisted: {len(delisted)}")

    # Exit type breakdown for delisted
    exit_types = {}
    for r in delisted:
        et = r.get("exit_type", "unknown") or "unknown"
        exit_types[et] = exit_types.get(et, 0) + 1

    print(f"\nExit type breakdown (delisted companies):")
    for et, count in sorted(exit_types.items(), key=lambda x: -x[1]):
        print(f"  {et}: {count}")

    # Return stats for companies with data
    with_returns = [r for r in results if r.get("price_total_return_pct")]
    if with_returns:
        returns = [float(r["price_total_return_pct"]) for r in with_returns]
        print(f"\nReturn statistics ({len(with_returns)} companies with data):")
        print(f"  Mean return: {sum(returns)/len(returns):.1f}%")
        print(f"  Median return: {sorted(returns)[len(returns)//2]:.1f}%")
        print(f"  Positive returns: {sum(1 for r in returns if r > 0)}")
        print(f"  Negative returns: {sum(1 for r in returns if r < 0)}")


def save_results(results: list[dict], filename: str):
    """Save results to CSV."""
    if not results:
        return

    # Get all unique keys
    all_keys = set()
    for r in results:
        all_keys.update(r.keys())

    # Order keys sensibly
    key_order = [
        "name", "cik", "ticker", "founding_date", "s1_filing_date",
        "state", "city", "sic_code", "status", "current_price", "market_cap",
        "exchange", "exit_type", "exit_date", "exit_details"
    ]
    price_keys = sorted([k for k in all_keys if k.startswith("price_")])
    other_keys = sorted([k for k in all_keys if k not in key_order and k not in price_keys])

    fieldnames = [k for k in key_order if k in all_keys] + other_keys + price_keys

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved {len(results)} companies to {filename}")


if __name__ == "__main__":
    main()
