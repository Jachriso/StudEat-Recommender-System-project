# Latency of /recommend

- Date: 2026-11-26 21:09
- Requests: 50 over HTTP, 5 test students, warm-up excluded
- Catalogue: food.com (7492 meals), model loaded: True
- Ranking source: model = 30, popularity = 20

| Median | p95 | Max | Target |
|---|---|---|---|
| 65 ms | 326 ms | 379 ms | < 2 s (OK) |
