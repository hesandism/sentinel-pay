"""
SentinelPay — manual test for Redis online features (Phase 4, Step 2)
=====================================================================

Sends TWO /score requests for the SAME card and prints the interesting parts of
each response so you can see the Redis online features working:

  * Request 1 (a brand-new card)  -> cold_start is True, velocity features are 0.
  * Request 2 (same card again)   -> cold_start is False, velocity features grew
                                     and distance/speed are now non-zero.

Run it AFTER Redis, MLflow, and the API are all running:

    python scripts/test_online_features.py

We use a random card number each run so the "first" transaction is always a true
cold start (no leftover history in Redis from a previous run).
"""

import random
import time

import requests

# Where the running API lives. Change if you started it on a different port.
API_URL = "http://127.0.0.1:8000/score"

# A fresh random card number so this run starts cold (no history yet).
CC_NUM = random.randint(1_000_000_000_000_000, 9_999_999_999_999_999)


def make_transaction(amt, when, merch_lat, merch_long):
    """Build one raw transaction request body (Sparkov fields)."""
    return {
        "trans_date_trans_time": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(when)),
        "cc_num": CC_NUM,
        "merchant": "fraud_Rippin, Kub and Mann",
        "category": "misc_net",
        "amt": amt,
        "first": "Jennifer", "last": "Banks", "gender": "F",
        "street": "561 Perry Cove", "city": "Moravian Falls", "state": "NC", "zip": 28654,
        "lat": 36.0788, "long": -81.1781, "city_pop": 3495,
        "job": "Psychologist, counselling", "dob": "1988-03-09",
        "trans_num": f"txn-{when}",
        "unix_time": when,
        "merch_lat": merch_lat, "merch_long": merch_long,
    }


def show(label, response_json):
    """Print just the fields that prove the online features are working."""
    print(f"\n=== {label} ===")
    print("cold_start      :", response_json["cold_start"])
    print("online_features :", response_json["online_features"])
    print("fraud_probability:", response_json["fraud_probability"], "->", response_json["decision"])
    if response_json.get("warnings"):
        print("warnings        :", response_json["warnings"])


def main():
    print(f"Using card number: {CC_NUM}")

    # Pick a "now" and a point 30 minutes later so the 2nd txn is recent.
    t0 = int(time.time())
    t1 = t0 + 30 * 60   # 30 minutes after the first

    # --- Request 1: brand-new card -> expect cold_start = True ----------------
    txn1 = make_transaction(amt=100.00, when=t0, merch_lat=36.01, merch_long=-82.04)
    r1 = requests.post(API_URL, json=txn1, timeout=30)
    r1.raise_for_status()
    show("Request 1 (new card)", r1.json())

    # --- Request 2: same card, 30 min later, different location ---------------
    # Now Redis has 1 previous transaction, so cold_start should be False and the
    # velocity / distance / speed features should be non-zero.
    txn2 = make_transaction(amt=250.00, when=t1, merch_lat=36.40, merch_long=-82.40)
    r2 = requests.post(API_URL, json=txn2, timeout=30)
    r2.raise_for_status()
    show("Request 2 (same card, 30 min later)", r2.json())

    # --- Simple pass/fail summary --------------------------------------------
    j1, j2 = r1.json(), r2.json()
    print("\n--- Summary ---")
    print("Request 1 cold_start == True :", j1["cold_start"] is True)
    print("Request 2 cold_start == False:", j2["cold_start"] is False)
    print("Request 2 txn_count_1h > 0   :", j2["online_features"]["txn_count_1h"] > 0)
    print("Request 2 dist_from_prev_km>0:", j2["online_features"]["dist_from_prev_km"] > 0)


if __name__ == "__main__":
    main()
