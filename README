# UPB - Prediction Markets - Descriptive Analysis

What it does  
1. Collect: Download public market, event, and related metadata from the platforms' APIs. Responses are stored as compressed JSON, with per-page metadata and checkpoints for resuming interrupted collections.
2. Prepare: Merge live/historical and open/closed market records, deduplicate by platform and market ID, enrich records with available event/series information, and export a normalized Parquet table. SQLite is used as a disk-backed staging database.
3. Analyze: Print descriptive statistics and display charts for scheduled market durations, platform comparisons, category coverage, duration distributions, and scheduled versus observed closing durations where available.

Scope: This collects metadata and current fields returned by the selected API endpoints, no individual trades, historical order books, or price-tick histories.  


## Quick Start

Requires Python and an internet connection for the collection step.

```python
python -m venv ./.venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```


Run a small test collection, build the Parquet file, and analyze it:

```python
python collect_prediction_markets.py --out ./raw --platform both --exclude-kalshi-combos --max-pages 2
python build_market_parquet.py --input ./raw --output ./analysis/data.parquet
python analyze_parquet_data.py
```

`--max-pages 2` limits new pages per dataset in that run, **not** the total number of pages across all endpoints.


## Useful collection options

- `--platform kalshi` or `--platform polymarket` : Collect only one platform (default: both).
- `--max-pages <N>`: Fetch at most N new pages per paginated dataset in this invocation.
- `--exclude-kalshi-combos` : Omit Kalshi combination/multivariate market collection.
- `--no-events` : Skip event and series downloads; some enrichment fields may be missing.
- `--kalshi-rps <N>`, `--polymarket-rps <N>`: Adjust request pacing for the respective APIs.
- `--timeout <N>`, `--retries <N>`: Configure network timeout and retry behavior.


## Interpretation and limitations
- Scheduled duration is not necessarily actual trading duration. Kalshi's `open_time`/`close_time` describe its reported trading schedule; Polymarket's `startDate`/`endDate` are platform-reported fields, and `startDate` need not be the first tradable instant.
- Polymarket's `closedTime`, when available, is used for observed closure; Kalshi's `settlement_ts` describes settlement, not trading close. Consequently, observed-close comparisons cover only a subset of markets.
- Missing and negative durations are reported for quality checks. Negative or missing scheduled durations are excluded from the duration statistics and plots.
- Event categories may be missing or not directly comparable across platforms. Switching analysis to the event level selects one market-level duration per event, which may not represent the entire event.
- The dataset reflects API responses at collection time. It is not a historical time series of market prices or trades.

