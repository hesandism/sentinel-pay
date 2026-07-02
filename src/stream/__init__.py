"""SentinelPay streaming pipeline (Phase 5).

Turns the request/response scorer into a live transaction feed:

    producer  -> [transactions topic] -> consumer -> /score -> Postgres + [alerts topic]

See ``producer.py`` and ``consumer.py`` for the two moving parts and ``config.py``
for the shared, environment-overridable settings.
"""
