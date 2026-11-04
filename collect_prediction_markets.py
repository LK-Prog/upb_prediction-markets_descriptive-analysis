#!/.venv/bin/python#!/usr/bin/env python3
"""Download public Kalshi and Polymarket market/event metadata as raw JSON.

Pilot (at most two pages PER paginated endpoint):
    python collect_prediction_markets.py --out ./prediction_data --max-pages 2

Full collection, resuming automatically if interrupted:
    python collect_prediction_markets.py --out ./prediction_data

The script archives the full JSON response for each page in .json.gz files.
Progress checkpoints and per-page metadata are written beside the responses.
It does NOT download trade-by-trade data, historical order books or price ticks.

WARNING:
A complete crawl can be large, especially Kalshi combination markets.
Use --exclude-kalshi-combos to reduce the amount of data.
"""

# TODO: Imporve performance via concurrency.
# TODO: Improve typing, this contains an ungodly amount of unknows

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


KALSHI = "https://external-api.kalshi.com/trade-api/v2"
POLYMARKET = "https://gamma-api.polymarket.com"
TRANSIENT_STATUSES = {429, 500, 502, 503, 504}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def atomic_json(path: Path, value: dict) -> None:
    atomic_write(path, (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))


class RequestPacer:
    def __init__(self, requests_per_second: float):
        self.interval = 1.0 / requests_per_second
        self.next_request = 0.0

    def wait(self) -> None:
        delay = self.next_request - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self.next_request = time.monotonic() + self.interval


def retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            date = parsedate_to_datetime(value)
            return max(0.0, (date - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def fetch_json(url: str, pacer: RequestPacer, timeout: float, retries: int) -> tuple[bytes, dict]:
    headers = {
        "Accept": "application/json",
        "User-Agent": "ResearchMarketMetadataCollector/1.0",
    }
    for attempt in range(retries + 1):
        pacer.wait()
        try:
            with urlopen(Request(url, headers=headers), timeout=timeout) as response:
                raw = response.read()
                if response.headers.get("Content-Encoding", "").lower() == "gzip":
                    raw = gzip.decompress(raw)
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected a JSON object from {url}")
            return raw, payload
        except HTTPError as error:
            if error.code not in TRANSIENT_STATUSES or attempt == retries:
                raise
            instructed = retry_after_seconds(error.headers.get("Retry-After"))
            wait = instructed if instructed is not None else min(60.0, 2 ** attempt)
            print(f"  HTTP {error.code}: retry in {wait:.1f}s", file=sys.stderr)
        except (URLError, TimeoutError, ConnectionError) as error:
            if attempt == retries:
                raise
            wait = min(60.0, 2 ** attempt)
            print(f"  Network error ({error}): retry in {wait:.1f}s", file=sys.stderr)
        time.sleep(wait + random.uniform(0, min(0.5, wait * 0.1)))
    raise RuntimeError("Unreachable retry state")


@dataclass(frozen=True)
class Job:
    name: str
    provider: str
    endpoint: str
    array_key: str
    params: dict[str, str] = field(default_factory=dict)
    in_cursor: str = "cursor"
    out_cursor: str = "cursor"
    single: bool = False

    def url(self, cursor: str | None = None) -> str:
        params = dict(self.params)
        if cursor:
            params[self.in_cursor] = cursor
        return self.endpoint + ("?" + urlencode(params) if params else "")

    def signature(self) -> dict:
        return {
            "name": self.name,
            "provider": self.provider,
            "endpoint": self.endpoint,
            "array_key": self.array_key,
            "params": self.params,
            "in_cursor": self.in_cursor,
            "out_cursor": self.out_cursor,
            "single": self.single,
        }


def jobs_for(args: argparse.Namespace) -> list[Job]:
    jobs: list[Job] = []
    if args.platform in ("both", "kalshi"):
        market_filter = {"mve_filter": "exclude"} if args.exclude_kalshi_combos else {}
        jobs.extend([
            Job("markets_live", "kalshi", KALSHI + "/markets", "markets",
                {"limit": "1000", **market_filter}),
            Job("markets_historical", "kalshi", KALSHI + "/historical/markets",
                "markets", {"limit": "1000", **market_filter}),
            Job("historical_cutoff", "kalshi", KALSHI + "/historical/cutoff",
                "", single=True),
        ])
        if not args.no_events:
            jobs.extend([
                Job("events", "kalshi", KALSHI + "/events", "events",
                    {"limit": "200", "with_nested_markets": "false"}),
                Job("series", "kalshi", KALSHI + "/series", "series",
                    {"include_product_metadata": "true"}, single=True),
            ])
            if not args.exclude_kalshi_combos:
                jobs.append(Job("events_multivariate", "kalshi", KALSHI + "/events/multivariate",
                                "events", {"limit": "200", "with_nested_markets": "false"}))
    if args.platform in ("both", "polymarket"):
        for closed in ("false", "true"):
            state = "closed" if closed == "true" else "open"
            jobs.append(Job(f"markets_{state}", "polymarket", POLYMARKET + "/markets/keyset",
                            "markets", {"limit": "100", "closed": closed},
                            "after_cursor", "next_cursor"))
            if not args.no_events:
                jobs.append(Job(f"events_{state}", "polymarket", POLYMARKET + "/events/keyset",
                                "events", {"limit": "100", "closed": closed},
                                "after_cursor", "next_cursor"))
    return jobs


def initial_progress(job: Job) -> dict:
    return {
        "job": job.signature(),
        "pages_completed": 0,
        "records": 0,
        "next_cursor": None,
        "complete": False,
        "started_at_utc": now_utc(),
    }


def inspect_page(raw: bytes, job: Job, input_cursor: str | None) -> tuple[int, str | None]:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected payload shape for {job.name}")
    if job.single:
        values = payload.get(job.array_key) if job.array_key else None
        return (len(values) if isinstance(values, list) else 1), None
    values = payload.get(job.array_key)
    if not isinstance(values, list):
        raise ValueError(f"Missing list field {job.array_key!r} in {job.name}")
    next_cursor = payload.get(job.out_cursor)
    if next_cursor is not None and not isinstance(next_cursor, str):
        raise ValueError(f"Invalid cursor type for {job.name}")
    next_cursor = next_cursor or None
    if next_cursor is not None and next_cursor == input_cursor:
        raise ValueError(f"Pagination cursor did not advance for {job.name}")
    return len(values), next_cursor


def collect_job(job: Job, out: Path, pacer: RequestPacer,
                timeout: float, retries: int, max_pages: int | None) -> None:
    directory = out / job.provider / job.name
    pages_dir = directory / "pages"
    checkpoint_file = directory / "checkpoint.json"
    if checkpoint_file.exists():
        progress = json.loads(checkpoint_file.read_text(encoding="utf-8"))
        if progress.get("job") != job.signature():
            raise ValueError(f"Job configuration changed for {directory}; choose a new --out")
    else:
        progress = initial_progress(job)
        # Catch accidental reuse of an existing directory with no checkpoint.
        if pages_dir.exists() and any(pages_dir.iterdir()):
            raise RuntimeError(f"Pages exist without a checkpoint: {pages_dir}")
        atomic_json(checkpoint_file, progress)

    if progress["complete"]:
        print(f"[{job.provider}/{job.name}] already complete ({progress['records']} records)")
        return

    new_pages = 0
    print(f"[{job.provider}/{job.name}] starting at page {progress['pages_completed'] + 1}", flush=True)
    while max_pages is None or new_pages < max_pages:
        page_number = progress["pages_completed"] + 1
        filename = f"{page_number:08d}"
        page_file = pages_dir / f"{filename}.json.gz"
        info_file = pages_dir / f"{filename}.meta.json"
        previous_cursor = progress["next_cursor"]
        url = job.url(previous_cursor)

        if page_file.exists():
            # Crash recovery: a page may be durable while the checkpoint is stale.
            raw = gzip.decompress(page_file.read_bytes())
            if info_file.exists():
                saved_info = json.loads(info_file.read_text(encoding="utf-8"))
                if saved_info.get("url") != url:
                    raise RuntimeError(f"Saved page has a different request URL: {page_file}")
            print(f"  recovering existing page {page_number}", flush=True)
        else:
            raw, _ = fetch_json(url, pacer, timeout, retries)

        count, next_cursor = inspect_page(raw, job, previous_cursor)
        if not page_file.exists():
            atomic_write(page_file, gzip.compress(raw, compresslevel=6, mtime=0))
        if not info_file.exists():
            atomic_json(info_file, {
                "url": url,
                "retrieved_at_utc": now_utc(),
                "count": count,
                "input_cursor": previous_cursor,
                "next_cursor": next_cursor,
                "sha256_uncompressed": hashlib.sha256(raw).hexdigest(),
            })

        progress["pages_completed"] = page_number
        progress["records"] += count
        progress["next_cursor"] = next_cursor
        progress["complete"] = job.single or next_cursor is None
        progress["updated_at_utc"] = now_utc()
        atomic_json(checkpoint_file, progress)
        new_pages += 1
        print(f"  page {page_number}: +{count} records; total {progress['records']}", flush=True)
        if progress["complete"]:
            break

    status = "complete" if progress["complete"] else "paused (resume by rerunning)"
    print(f"[{job.provider}/{job.name}] {status}\n", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("prediction_data"),
                        help="Root folder for raw JSON and checkpoints")
    parser.add_argument("--platform", choices=("both", "kalshi", "polymarket"), default="both")
    parser.add_argument("--max-pages", type=int, default=None,
                        help="New pages per dataset in this invocation; omit for a full crawl")
    parser.add_argument("--exclude-kalshi-combos", action="store_true",
                        help="Exclude Kalshi multivariate/combination markets")
    parser.add_argument("--no-events", action="store_true",
                        help="Download only markets (and the Kalshi archive cutoff), not events/series")
    parser.add_argument("--kalshi-rps", type=float, default=3.0)
    parser.add_argument("--polymarket-rps", type=float, default=7.0)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=6)
    args = parser.parse_args()
    if args.max_pages is not None and args.max_pages <= 0:
        parser.error("--max-pages must be positive")
    if args.kalshi_rps <= 0 or args.polymarket_rps <= 0 or args.timeout <= 0 or args.retries < 0:
        parser.error("Rates and timeout must be positive; retries must be nonnegative")

    pacers = {
        "kalshi": RequestPacer(args.kalshi_rps),
        "polymarket": RequestPacer(args.polymarket_rps),
    }
    for job in jobs_for(args):
        try:
            collect_job(job, args.out, pacers[job.provider], args.timeout, args.retries, args.max_pages)
        except (HTTPError, URLError, TimeoutError, ValueError, OSError, RuntimeError) as exc:
            print(f"FAILED [{job.provider}/{job.name}]: {exc}", file=sys.stderr)
            print("Previous pages are preserved. Fix the issue and rerun to resume.", file=sys.stderr)
            return 1
    print(f"Responses saved under {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
