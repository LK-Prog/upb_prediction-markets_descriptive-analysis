#!/usr/bin/env python3
"""Normalize downloaded Kalshi/Polymarket metadata into one Parquet table.

Usage:
    python -m pip install pyarrow
    python build_market_parquet.py --input ./prediction_data --output ./analysis/markets.parquet

The original .json.gz downloads remain unchanged. A disk-backed SQLite database
is used for deduplication and metadata joins; output is written in batches, so
the whole market universe need not fit in memory.

Timestamp interpretation:
    Kalshi open_time / close_time are the exchange-reported trading schedule.
    Polymarket startDate / endDate are the platform-reported start/end fields;
    startDate must NOT be assumed to be the first executable trading instant.
    Polymarket closedTime is used as the observed closure field if present.
    Kalshi settlement_ts measures settlement, not trading close.
    Negative durations are retained and flagged, never silently removed.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


DAY_SECONDS = 86400.0
MARKET_DATASETS = {
    "kalshi": ("markets_live", "markets_historical"),
    "polymarket": ("markets_open", "markets_closed"),
}
EVENT_DATASETS = {
    "kalshi": ("events", "events_multivariate"),
    "polymarket": ("events_open", "events_closed"),
}


def parse_utc(value: Any) -> datetime | None:
    if value in (None, "", "null"):
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc)
        except (ValueError, OverflowError, OSError):
            return None
    if not isinstance(value, str):
        return None
    try:
        date = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        return date.astimezone(timezone.utc)
    except ValueError:
        return None


def days_between(start: datetime | None, end: datetime | None) -> float | None:
    if start is None or end is None:
        return None
    return (end - start).total_seconds() / DAY_SECONDS


def text_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value or None
    return str(value)


def numeric(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
        return parsed if abs(parsed) != float("inf") and parsed == parsed else None
    except (ValueError, TypeError, OverflowError):
        return None


def compact_json(value: Any) -> str | None:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) if value is not None else None


def event_id_for_market(platform: str, market: dict) -> str | None:
    if platform == "kalshi":
        return text_or_none(market.get("event_ticker"))
    nested = market.get("events")
    if isinstance(nested, list):
        for event in nested:
            if isinstance(event, dict) and event.get("id") is not None:
                return str(event["id"])
    return text_or_none(market.get("eventId") or market.get("event_id"))


def record_id(platform: str, kind: str, value: dict) -> str | None:
    if platform == "kalshi":
        key = "ticker" if kind in ("markets", "series") else "event_ticker"
    else:
        key = "id"
    return text_or_none(value.get(key))


def initialize_db(path: Path, source_root: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS markets (
            platform TEXT NOT NULL, id TEXT NOT NULL, event_id TEXT,
            source TEXT NOT NULL, fetched_at TEXT NOT NULL, raw TEXT NOT NULL,
            PRIMARY KEY (platform, id)
        );
        CREATE TABLE IF NOT EXISTS events (
            platform TEXT NOT NULL, id TEXT NOT NULL, fetched_at TEXT NOT NULL,
            raw TEXT NOT NULL, PRIMARY KEY (platform, id)
        );
        CREATE TABLE IF NOT EXISTS series (
            platform TEXT NOT NULL, id TEXT NOT NULL, fetched_at TEXT NOT NULL,
            raw TEXT NOT NULL, PRIMARY KEY (platform, id)
        );
        CREATE TABLE IF NOT EXISTS ingested_pages (
            path TEXT PRIMARY KEY, size_bytes INTEGER NOT NULL,
            mtime_ns INTEGER NOT NULL, count INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS market_event_index ON markets(platform, event_id);
    """)
    prior = conn.execute("SELECT value FROM settings WHERE key='source_root'").fetchone()
    source = str(source_root.resolve())
    if prior and prior[0] != source:
        conn.close()
        raise ValueError("Staging database belongs to another input directory; use a different --staging")
    conn.execute("INSERT OR IGNORE INTO settings(key,value) VALUES ('source_root',?)", (source,))
    conn.commit()
    return conn


def pages_for(root: Path) -> Iterator[tuple[str, str, str, Path]]:
    for platform in ("kalshi", "polymarket"):
        for name in MARKET_DATASETS[platform]:
            for path in sorted((root / platform / name / "pages").glob("*.json.gz")):
                yield platform, name, "markets", path
        for name in EVENT_DATASETS[platform]:
            for path in sorted((root / platform / name / "pages").glob("*.json.gz")):
                yield platform, name, "events", path
    for path in sorted((root / "kalshi" / "series" / "pages").glob("*.json.gz")):
        yield "kalshi", "series", "series", path


def import_pages(conn: sqlite3.Connection, root: Path) -> tuple[int, int, int]:
    loaded, skipped, records = 0, 0, 0
    for platform, dataset, kind, path in pages_for(root):
        stat = path.stat()
        file_id = str(path.resolve())
        old = conn.execute("SELECT size_bytes,mtime_ns FROM ingested_pages WHERE path=?", (file_id,)).fetchone()
        if old == (stat.st_size, stat.st_mtime_ns):
            skipped += 1
            continue

        with gzip.open(path, "rt", encoding="utf-8") as stream:
            page = json.load(stream)
        items = page.get(kind)
        if not isinstance(items, list):
            raise ValueError(f"Expected {kind!r} array in {path}")
        metadata_file = path.with_name(path.name.replace(".json.gz", ".meta.json"))
        metadata = json.loads(metadata_file.read_text(encoding="utf-8")) if metadata_file.exists() else {}
        fetched_at = metadata.get("retrieved_at_utc") or datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc).isoformat()

        # An entire page and its progress marker commit together.
        with conn:
            for value in items:
                if not isinstance(value, dict):
                    continue
                key = record_id(platform, kind, value)
                if not key:
                    continue
                serialized = compact_json(value)
                if kind == "markets":
                    conn.execute("""
                        INSERT INTO markets(platform,id,event_id,source,fetched_at,raw)
                        VALUES (?,?,?,?,?,?) ON CONFLICT(platform,id) DO UPDATE SET
                          event_id=excluded.event_id, source=excluded.source,
                          fetched_at=excluded.fetched_at, raw=excluded.raw
                        WHERE excluded.fetched_at >= markets.fetched_at
                    """, (platform, key, event_id_for_market(platform, value),
                          dataset, fetched_at, serialized))
                else:
                    table = "events" if kind == "events" else "series"
                    conn.execute(f"""
                        INSERT INTO {table}(platform,id,fetched_at,raw) VALUES(?,?,?,?)
                        ON CONFLICT(platform,id) DO UPDATE SET
                          fetched_at=excluded.fetched_at, raw=excluded.raw
                        WHERE excluded.fetched_at >= {table}.fetched_at
                    """, (platform, key, fetched_at, serialized))
            conn.execute("""
                INSERT INTO ingested_pages(path,size_bytes,mtime_ns,count) VALUES(?,?,?,?)
                ON CONFLICT(path) DO UPDATE SET size_bytes=excluded.size_bytes,
                    mtime_ns=excluded.mtime_ns, count=excluded.count
            """, (file_id, stat.st_size, stat.st_mtime_ns, len(items)))
        loaded += 1
        records += len(items)
        if loaded % 100 == 0:
            print(f"Indexed {loaded} new pages ({records:,} records); {skipped} unchanged pages skipped", flush=True)
    return loaded, skipped, records


def small_event(event: dict | None) -> dict | None:
    if event is None:
        return None
    # The original event pages may contain full lists of associated markets.
    # Keep that bulky material in raw JSON, not repeated for every Parquet row.
    return {k: v for k, v in event.items() if k not in ("markets", "chats", "comments")}


def event_tags(event: dict | None) -> list[str]:
    if not event or not isinstance(event.get("tags"), list):
        return []
    result = []
    for tag in event["tags"]:
        if isinstance(tag, dict):
            label = tag.get("label") or tag.get("slug")
        else:
            label = tag
        if label is not None:
            result.append(str(label))
    return result


def normalize_market(platform: str, identifier: str, source: str, fetched_at: str,
                     market: dict, event: dict | None, series: dict | None) -> dict:
    if event is None and platform == "polymarket":
        embedded = market.get("events")
        if isinstance(embedded, list) and embedded and isinstance(embedded[0], dict):
            event = embedded[0]

    if platform == "kalshi":
        created_raw, start_raw, end_raw = (market.get(x) for x in
                                           ("created_time", "open_time", "close_time"))
        closed_raw = None  # close_time is not an independently verified actual-close event.
        expiry_raw, settled_raw, updated_raw = (market.get(x) for x in
                                                ("expiration_time", "settlement_ts", "updated_time"))
        title = market.get("title")
        description = market.get("rules_primary")
        status = market.get("status")
        event_key = market.get("event_ticker")
        series_ticker = event.get("series_ticker") if event else None
        volume = market.get("volume_fp")
        volume_24h = market.get("volume_24h_fp")
        liquidity = market.get("liquidity_dollars")
        open_interest = market.get("open_interest_fp")
        bid, ask, last = (market.get(x) for x in
                          ("yes_bid_dollars", "yes_ask_dollars", "last_price_dollars"))
        is_closed = status in ("closed", "settled") if status else None
        is_active = status == "active" or status == "open" if status else None
        accepting = status == "active" or status == "open" if status else None
        result = market.get("result")
        version = None
        market_url = None
        resolution_source = None
        outcome_labels = ["Yes", "No"]
    else:
        created_raw = market.get("createdAt")
        start_raw, end_raw = market.get("startDate"), market.get("endDate")
        closed_raw = market.get("closedTime")
        expiry_raw, settled_raw, updated_raw = None, None, market.get("updatedAt")
        title = market.get("question")
        description = market.get("description")
        status = "closed" if market.get("closed") is True else "open" if market.get("closed") is False else None
        event_key = event_id_for_market(platform, market)
        series_ticker = event.get("seriesSlug") if event else None
        volume = market.get("volumeNum", market.get("volume"))
        volume_24h = market.get("volume24hr")
        liquidity = market.get("liquidityNum", market.get("liquidity"))
        open_interest = None
        bid, ask, last = (market.get(x) for x in ("bestBid", "bestAsk", "lastTradePrice"))
        is_closed = market.get("closed") if isinstance(market.get("closed"), bool) else None
        is_active = market.get("active") if isinstance(market.get("active"), bool) else None
        accepting = market.get("acceptingOrders") if isinstance(market.get("acceptingOrders"), bool) else None
        result = market.get("umaResolutionStatuses")
        version = market.get("version")
        slug = market.get("slug")
        market_url = f"https://polymarket.com/market/{slug}" if slug else None
        resolution_source = market.get("resolutionSource")
        outcome_labels = market.get("outcomes")

    created = parse_utc(created_raw)
    start = parse_utc(start_raw)
    end = parse_utc(end_raw)
    actual_close = parse_utc(closed_raw)
    expiry = parse_utc(expiry_raw)
    settled = parse_utc(settled_raw)
    updated = parse_utc(updated_raw)
    fetched = parse_utc(fetched_at)

    listed_to_end = days_between(created, end)
    start_to_end = days_between(start, end)
    listed_to_close = days_between(created, actual_close)
    start_to_close = days_between(start, actual_close)
    expiry_to_settlement = days_between(expiry, settled)
    listed_to_expiry = days_between(created, expiry)
    negative = any(value < 0 for value in (listed_to_end, start_to_end, listed_to_close,
                                            start_to_close, expiry_to_settlement, listed_to_expiry)
                   if value is not None)
    date_warnings = []
    for label, raw, parsed in (("created", created_raw, created), ("start", start_raw, start),
                               ("end", end_raw, end), ("closed", closed_raw, actual_close),
                               ("expiration", expiry_raw, expiry), ("settlement", settled_raw, settled)):
        if raw not in (None, "") and parsed is None:
            date_warnings.append(f"invalid_{label}")
    if negative:
        date_warnings.append("negative_duration")

    return {
        "platform": platform,
        "market_id": identifier,
        "source_dataset": source,
        "event_id": text_or_none(event_key),
        "event_title": text_or_none(event.get("title")) if event else None,
        "event_category": text_or_none(event.get("category")) if event else None,
        "event_tags_json": compact_json(event_tags(event)),
        "series_ticker": text_or_none(series_ticker),
        "series_title": text_or_none(series.get("title")) if series else None,
        "series_category": text_or_none(series.get("category")) if series else None,
        "market_title": text_or_none(title),
        "market_description": text_or_none(description),
        "market_type": text_or_none(market.get("market_type") or market.get("marketType")),
        "market_version": text_or_none(version),
        "market_status": text_or_none(status),
        "is_closed": is_closed,
        "is_active": is_active,
        "accepting_orders": accepting,
        "market_url": market_url,
        "resolution_source": text_or_none(resolution_source),
        "result_json": compact_json(result),
        "outcomes_json": compact_json(outcome_labels),
        "volume": numeric(volume),
        "volume_24h": numeric(volume_24h),
        "liquidity": numeric(liquidity),
        "open_interest": numeric(open_interest),
        "yes_or_best_bid": numeric(bid),
        "yes_or_best_ask": numeric(ask),
        "last_price": numeric(last),
        "created_at": created,
        "start_at": start,
        "scheduled_end_at": end,
        "observed_closed_at": actual_close,
        "expiration_at": expiry,
        "settled_at": settled,
        "updated_at": updated,
        "retrieved_at": fetched,
        "listing_to_scheduled_end_days": listed_to_end,
        "start_to_scheduled_end_days": start_to_end,
        "listing_to_observed_close_days": listed_to_close,
        "start_to_observed_close_days": start_to_close,
        "listing_to_expiration_days": listed_to_expiry,
        "expiration_to_settlement_days": expiry_to_settlement,
        "observed_close_available": actual_close is not None,
        "negative_duration_flag": negative,
        "date_warnings_json": compact_json(date_warnings),
        "market_metadata_json": compact_json(market),
        "event_metadata_json": compact_json(small_event(event)),
        "series_metadata_json": compact_json(series),
    }


def arrow_schema(pa: Any) -> Any:
    text_columns = [
        "platform", "market_id", "source_dataset", "event_id", "event_title",
        "event_category", "event_tags_json", "series_ticker", "series_title", "series_category",
        "market_title", "market_description", "market_type", "market_version", "market_status",
        "market_url", "resolution_source", "result_json", "outcomes_json", "date_warnings_json",
        "market_metadata_json", "event_metadata_json", "series_metadata_json",
    ]
    float_columns = [
        "volume", "volume_24h", "liquidity", "open_interest", "yes_or_best_bid",
        "yes_or_best_ask", "last_price", "listing_to_scheduled_end_days",
        "start_to_scheduled_end_days", "listing_to_observed_close_days",
        "start_to_observed_close_days", "listing_to_expiration_days",
        "expiration_to_settlement_days",
    ]
    datetime_columns = [
        "created_at", "start_at", "scheduled_end_at", "observed_closed_at", "expiration_at",
        "settled_at", "updated_at", "retrieved_at",
    ]
    bool_columns = ["is_closed", "is_active", "accepting_orders",
                    "observed_close_available", "negative_duration_flag"]
    names = {name: pa.large_string() for name in text_columns}
    names.update({name: pa.float64() for name in float_columns})
    names.update({name: pa.timestamp("us", tz="UTC") for name in datetime_columns})
    names.update({name: pa.bool_() for name in bool_columns})
    # Stable column order for reproducible comparisons.
    return pa.schema([(name, dtype) for name, dtype in names.items()])


def write_parquet(conn: sqlite3.Connection, destination: Path, batch_size: int) -> int:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("Parquet export requires pyarrow. Install with: python -m pip install pyarrow") from error

    destination.parent.mkdir(parents=True, exist_ok=True)
    schema = arrow_schema(pa)
    fd, tmp_name = tempfile.mkstemp(prefix=".parquet-build-", suffix=".parquet", dir=destination.parent)
    os.close(fd)
    total = 0
    query = conn.execute("""
        SELECT m.platform, m.id, m.event_id, m.source, m.fetched_at, m.raw,
               e.raw, s.raw
        FROM markets AS m
        LEFT JOIN events AS e ON m.platform=e.platform AND m.event_id=e.id
        LEFT JOIN series AS s ON m.platform=s.platform AND
             json_extract(e.raw, '$.series_ticker')=s.id
        ORDER BY m.platform, m.id
    """)
    try:
        with pq.ParquetWriter(tmp_name, schema, compression="zstd") as writer:
            while True:
                rows = query.fetchmany(batch_size)
                if not rows:
                    break
                processed = []
                for platform, identifier, event_id, source, fetched_at, raw, e_raw, s_raw in rows:
                    market = json.loads(raw)
                    event = json.loads(e_raw) if e_raw else None
                    series = json.loads(s_raw) if s_raw else None
                    processed.append(normalize_market(platform, identifier, source,
                                                      fetched_at, market, event, series))
                writer.write_table(pa.Table.from_pylist(processed, schema=schema))
                total += len(processed)
                if total % (batch_size * 10) == 0:
                    print(f"Exported {total:,} unique markets", flush=True)
        os.replace(tmp_name, destination)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return total


def check_collection(root: Path, require_complete: bool) -> None:
    if not root.is_dir():
        raise ValueError(f"Input data folder does not exist: {root}")
    found = 0
    for checkpoint in root.glob("*/*/checkpoint.json"):
        found += 1
        status = json.loads(checkpoint.read_text(encoding="utf-8"))
        if not status.get("complete"):
            message = f"Unfinished collection: {checkpoint.parent}"
            if require_complete:
                raise ValueError(message)
            print("Warning: " + message, file=sys.stderr)
    if not found:
        raise ValueError(f"No download checkpoints found under {root}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("prediction_data"))
    parser.add_argument("--output", type=Path, default=Path("analysis/markets.parquet"))
    parser.add_argument("--staging", type=Path, default=Path("analysis/metadata.sqlite"))
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--require-complete", action="store_true",
                        help="Fail if any present collection checkpoint is unfinished")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.output.resolve() == args.staging.resolve():
        parser.error("--output and --staging cannot be the same file")
    try:
        check_collection(args.input, args.require_complete)
        conn = initialize_db(args.staging, args.input)
        try:
            new, reused, count = import_pages(conn, args.input)
            unique = conn.execute("SELECT COUNT(*) FROM markets").fetchone()[0]
            print(f"Indexed {new} new pages ({count:,} source records), skipped {reused} unchanged pages")
            print(f"Unique markets after deduplication: {unique:,}")
            if unique == 0:
                raise ValueError("No market records were found")
            exported = write_parquet(conn, args.output, args.batch_size)
        finally:
            conn.close()
        print(f"Exported {exported:,} unique markets to {args.output.resolve()}")
        return 0
    except (ValueError, RuntimeError, OSError, sqlite3.Error, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
