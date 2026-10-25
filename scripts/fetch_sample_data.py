"""
==============================================================================
scripts/fetch_sample_data.py
Standalone historical Kline downloader for local development and ML testing.

Pulls 1-minute OHLCV candles from Binance public REST API for specified symbols
and lookback window, formats them according to the canonical schema, saves
them to local Parquet, and verifies Spark reading and schema compliance.
==============================================================================
"""

import argparse
from datetime import datetime, timedelta, timezone
import logging
import os
import sys
import time
from typing import Any, Dict, List

import pyarrow as pa
import pyarrow.parquet as pq
import requests

# Add repo root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from spark.apps.ml.spark_session import get_spark_session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("fetch_sample_data")

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


def fetch_klines_for_symbol(
    symbol: str,
    days: int,
    interval: str = "1m",
    batch_size: int = 1000,
) -> List[Dict[str, Any]]:
    """Fetch consecutive historical klines for a single symbol using Binance public REST API."""
    end_dt = datetime.now(tz=timezone.utc)
    start_dt = end_dt - timedelta(days=days)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    records: List[Dict[str, Any]] = []
    current_start = start_ms
    batch_num = 0

    logger.info(
        "Fetching %s for lookback %d days (%s to %s)...",
        symbol,
        days,
        start_dt.strftime("%Y-%m-%d %H:%M"),
        end_dt.strftime("%Y-%m-%d %H:%M"),
    )

    session = requests.Session()

    while current_start < end_ms:
        batch_num += 1
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": batch_size,
        }

        try:
            resp = session.get(BINANCE_KLINES_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("Error fetching batch %d for %s: %s. Retrying in 2s...", batch_num, symbol, exc)
            time.sleep(2.0)
            continue

        if not data:
            break

        for item in data:
            # Binance Kline Array:
            # 0: open_time, 1: open, 2: high, 3: low, 4: close, 5: volume, 6: close_time,
            # 7: quote_volume, 8: trades, 9: taker_buy_vol, 10: taker_buy_quote_vol, 11: ignore
            open_time = int(item[0])
            close_time = int(item[6])
            open_dt = datetime.fromtimestamp(open_time / 1000.0, tz=timezone.utc)
            close_dt = datetime.fromtimestamp(close_time / 1000.0, tz=timezone.utc)

            record = {
                "open_time": open_time,
                "symbol": symbol.upper(),
                "open": float(item[1]),
                "high": float(item[2]),
                "low": float(item[3]),
                "close": float(item[4]),
                "volume": float(item[5]),
                "close_time": close_time,
                "quote_asset_volume": float(item[7]),
                "number_of_trades": int(item[8]),
                "taker_buy_base_volume": float(item[9]),
                "taker_buy_quote_volume": float(item[10]),
                "data_source": "historical",
                "ingested_at": datetime.now(tz=timezone.utc).isoformat(),
                "open_time_dt": open_dt,
                "close_time_dt": close_dt,
                "kline_date": open_dt.date().isoformat(),
            }
            records.append(record)

        last_open_time = int(data[-1][0])
        # Next window begins 1ms after last candle close or 1 minute after last open
        current_start = last_open_time + 60_000

        if batch_num % 10 == 0:
            logger.info("[%s] Progress: %d records fetched (%d batches)", symbol, len(records), batch_num)

        time.sleep(0.05)  # Rate-limit safety margin

    logger.info("[%s] Ingestion finished. Total rows: %d across %d batches.", symbol, len(records), batch_num)
    return records


def save_to_parquet(records: List[Dict[str, Any]], output_path: str) -> None:
    """Save normalized records to Parquet format using pyarrow."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    arrow_schema = pa.schema([
        ("open_time", pa.int64()),
        ("symbol", pa.string()),
        ("open", pa.float64()),
        ("high", pa.float64()),
        ("low", pa.float64()),
        ("close", pa.float64()),
        ("volume", pa.float64()),
        ("close_time", pa.int64()),
        ("quote_asset_volume", pa.float64()),
        ("number_of_trades", pa.int64()),
        ("taker_buy_base_volume", pa.float64()),
        ("taker_buy_quote_volume", pa.float64()),
        ("data_source", pa.string()),
        ("ingested_at", pa.string()),
        ("open_time_dt", pa.timestamp("ms", tz="UTC")),
        ("close_time_dt", pa.timestamp("ms", tz="UTC")),
        ("kline_date", pa.string()),
    ])

    table = pa.Table.from_pylist(records, schema=arrow_schema)
    pq.write_table(table, output_path, compression="snappy")
    logger.info("Successfully wrote %d records to Parquet file: %s", len(records), output_path)


def verify_with_spark(parquet_path: str) -> None:
    """Verify that PySpark can read the generated Parquet file and write derived data locally."""
    logger.info("Verifying Parquet file via local PySpark session...")
    spark = get_spark_session(app_name="VerifySampleKlines", log_level="ERROR")

    try:
        abs_path = os.path.abspath(parquet_path).replace("\\", "/")
        df = spark.read.parquet(abs_path)
        total_count = df.count()
        logger.info("Spark read verification SUCCESS. Total rows loaded: %d", total_count)

        symbols_df = df.groupBy("symbol").count()
        for row in symbols_df.collect():
            logger.info("  - Symbol %s: %d rows", row["symbol"], row["count"])

        # Test local Parquet write through Spark to ensure no Windows permission failures
        test_write_path = os.path.abspath(os.path.join(os.path.dirname(parquet_path), "spark_verify_temp.parquet")).replace("\\", "/")
        (
            df.limit(10)
            .write
            .mode("overwrite")
            .parquet(test_write_path)
        )
        logger.info("Spark write verification SUCCESS (wrote test slice to %s)", test_write_path)

        # Clean up test write artifact
        if os.path.exists(test_write_path):
            import shutil
            shutil.rmtree(test_write_path, ignore_errors=True)

    finally:
        spark.stop()
        logger.info("Spark verification session terminated cleanly.")


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch Binance sample klines for ML pipeline")
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT", help="Comma-separated trading symbols")
    parser.add_argument("--days", type=int, default=30, help="Lookback window in days (default: 30)")
    parser.add_argument("--output", default="data/sample_klines.parquet", help="Target Parquet output path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    symbol_list = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    all_records: List[Dict[str, Any]] = []
    for sym in symbol_list:
        sym_records = fetch_klines_for_symbol(sym, days=args.days)
        all_records.extend(sym_records)

    save_to_parquet(all_records, args.output)
    verify_with_spark(args.output)
