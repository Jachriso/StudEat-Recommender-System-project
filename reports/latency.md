# Latency of /recommend

- Date: 2026-11-26 16:31
- Requests: 50 over HTTP, 5 test students, warm-up excluded
- Catalogue: food.com (7492 meals), model loaded: True
- Ranking source: model = 30, popularity = 20

| Median | p95 | Max | Target |
|---|---|---|---|
| 41 ms | 185 ms | 221 ms | < 2 s (OK) |
