"""
SentinelPay — /score latency benchmark (Phase 4, Step 3)
========================================================

Fires many requests at the running API and reports the latency distribution
(p50 / p95 / p99), so you can quote a real p95 number in the README.

Two numbers are reported per percentile:

  * client_ms — full round-trip measured here (network + server + JSON parse).
    This is what a real caller experiences.
  * server_ms — the API's own ``latency_ms`` (scoring work only), read straight
    from the response body. Useful to separate model time from transport time.

We warm up first (the very first requests are slower: lazy Redis connect, JIT,
page-ins) and EXCLUDE those from the stats, then measure ``--n`` requests.

Run it AFTER the stack is up (docker compose up), from the project root:

    python scripts/benchmark_latency.py                 # 500 requests, same card
    python scripts/benchmark_latency.py --n 1000        # more samples
    python scripts/benchmark_latency.py --url http://127.0.0.1:8000/score
    python scripts/benchmark_latency.py --new-card-each # worst case: all cold start
"""

from __future__ import annotations

import argparse
import random
import statistics
import time

import requests


def make_transaction(cc_num: int, when: int) -> dict:
    """Build one raw transaction body (Sparkov fields). Location jitters slightly
    so warm requests still exercise the geo/velocity feature path."""
    return {
        "trans_date_trans_time": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(when)),
        "cc_num": cc_num,
        "merchant": "fraud_Rippin, Kub and Mann",
        "category": "misc_net",
        "amt": round(random.uniform(5, 500), 2),
        "first": "Jennifer", "last": "Banks", "gender": "F",
        "street": "561 Perry Cove", "city": "Moravian Falls", "state": "NC", "zip": 28654,
        "lat": 36.0788, "long": -81.1781, "city_pop": 3495,
        "job": "Psychologist, counselling", "dob": "1988-03-09",
        "trans_num": f"txn-{when}-{random.randint(0, 1_000_000)}",
        "unix_time": when,
        "merch_lat": 36.0 + random.uniform(-0.5, 0.5),
        "merch_long": -82.0 + random.uniform(-0.5, 0.5),
    }


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (no interpolation) — simple and robust for latency."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round(pct / 100.0 * len(ordered) + 0.5)) - 1))
    return ordered[k]


def run(url: str, n: int, warmup: int, new_card_each: bool) -> int:
    session = requests.Session()  # reuse the TCP connection (keep-alive)
    base_card = random.randint(1_000_000_000_000_000, 9_999_999_999_999_999)
    now = int(time.time())

    def one_request(i: int) -> tuple[float, float]:
        cc_num = random.randint(1_000_000_000_000_000, 9_999_999_999_999_999) \
            if new_card_each else base_card
        body = make_transaction(cc_num, now + i)
        t0 = time.perf_counter()
        r = session.post(url, json=body, timeout=30)
        client_ms = (time.perf_counter() - t0) * 1000.0
        r.raise_for_status()
        server_ms = float(r.json().get("latency_ms", 0.0))
        return client_ms, server_ms

    print(f"Warming up ({warmup} requests, excluded) ...")
    for i in range(warmup):
        one_request(i)

    print(f"Measuring {n} requests against {url} "
          f"({'new card each (all cold-start)' if new_card_each else 'same card (steady state)'}) ...")
    client, server = [], []
    t_bench = time.perf_counter()
    for i in range(n):
        c, s = one_request(warmup + i)
        client.append(c)
        server.append(s)
    wall_s = time.perf_counter() - t_bench

    def row(name: str, vals: list[float]) -> str:
        return (f"  {name:10s}  p50={percentile(vals, 50):7.2f}  "
                f"p95={percentile(vals, 95):7.2f}  p99={percentile(vals, 99):7.2f}  "
                f"max={max(vals):7.2f}  mean={statistics.mean(vals):7.2f}")

    print("\n=== Latency (ms) over", n, "requests ===")
    print(row("client", client))
    print(row("server", server))
    print(f"\nThroughput: {n / wall_s:.1f} req/s over {wall_s:.2f}s (single-threaded client)")

    p95 = percentile(client, 95)
    target = 100.0
    verdict = "PASS" if p95 < target else "OVER TARGET"
    print(f"\np95 client latency = {p95:.2f} ms  (target < {target:.0f} ms)  -> {verdict}")
    return 0 if p95 < target else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark the /score endpoint latency.")
    ap.add_argument("--url", default="http://127.0.0.1:8000/score", help="Scoring endpoint URL.")
    ap.add_argument("--n", type=int, default=500, help="Number of measured requests.")
    ap.add_argument("--warmup", type=int, default=50, help="Warm-up requests (excluded).")
    ap.add_argument("--new-card-each", action="store_true",
                    help="Use a fresh card per request (worst case: always cold start).")
    args = ap.parse_args()
    return run(args.url, args.n, args.warmup, args.new_card_each)


if __name__ == "__main__":
    raise SystemExit(main())
