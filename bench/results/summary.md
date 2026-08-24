# Prune benchmark summary

| model | size_gb | tools_acc | gsm8k_acc | mmlu_acc | loop_rate | ppl_wikitext2_32chunk | completion_pp_tps_mean | completion_tg_tps_mean | completion_tg_tps_sd | tool_call_tg_tps_mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 27B-A2.8B-finishing | 12.01 | None | None | None | None |  | 39.4 | 35.98 | 1.39 | 39.35 |
| 27B-A2.8B-finishing-off | 12.01 | None | None | None | None |  | 48.03 | 33.7 | 4.85 | 33.81 |
| 27B-A2.8B-k4 | 16.49 | None | None | None | None |  | 41.74 | 33.4 | 1.98 | 34.24 |
| 27B-A2.8B-k6 | 16.49 | None | None | None | None |  | 40.21 | 33.84 | 1.47 | 34.27 |
| 27B-A2.8B-k6-mtp | 16.49 | None | None | None | None |  | 38.07 | 36.26 | 2.1 | 39.21 |
| 27B-A2.8B-mtp | 16.49 | None | None | None | None |  | 39.67 | 35.11 | 1.95 | 38.17 |
| 27B-A2.8B-mtp-off | 16.49 | None | None | None | None |  | 51.31 | 33.42 | 0.79 | 33.33 |
| base-q4km | 21.17 | None | None | None | None |  | 33.86 | 26.19 | 0.23 | 26.23 |
| base-q4km-mtp | 21.71 | None | None | None | None |  | 31.09 | 29.2 | 0.9 | 32.87 |
| base-q4km-mtp-off | 21.71 | None | None | None | None |  | 31.73 | 25.95 | 0.55 | 25.88 |
| healed-C | 15.95 | 0.975 | 0.8667 | 0.7667 | 0.0235 | 8.0105 | 32.71 | 20.51 | 7.75 | 19.96 |
| healedC-c2 | 11.39 | 0.05 | 0.0 | 0.0 | 0.3412 | 3741.1257 | 71.36 | 500025.08 | 534495.68 | 111159.6 |
| healedC-top4 | 15.95 | 0.9 | 0.7333 | 0.7 | 0.0471 |  | 46.33 | 26.16 | 0.97 | 26.42 |
| healedC-top6 | 15.95 | 0.925 | 0.8667 | 0.7667 | 0.0353 |  | 44.89 | 24.49 | 0.57 | 24.65 |
| healedC-w384 | 13.85 | 0.725 | 0.2667 | 0.4333 | 0.0353 | 13.1905 | 21.9 | 9.43 | 7.49 | 8.14 |
| prune-A | 11.38 | 0.15 | 0.0667 | 0.7667 | 0.2235 | 20.3274 | 29.21 | 16.27 | 2.44 | 15.32 |
| prune-AC | 8.64 | 0.075 | 0.0 | 0.1667 | 0.5765 | 101.2726 | 55.46 | 36.78 | 0.86 | 36.96 |
| prune-C | 15.95 | None | None | None | None |  | 44.31 | 30.16 | 6.7 | 31.29 |
| unsloth-base-mtp | 22.85 | None | None | None | None |  | 29.04 | 24.15 | 2.06 | 26.76 |
