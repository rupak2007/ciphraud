### Full table (validation; latency = mean ± std of 5 repeated executions of one input)

| Config | T3 | Quantized / float PR-AUC | Agreement | FHE latency | Compile | Peak RSS | Bootstraps | Ciphertext in / out | Bootstrap key | Latency frontier |
|---|---|---|---|---|---|---|---|---|---|---|
| `lr_top20_bits8` | **fail** | 0.3218 / 0.3484 | 0.9850 | 6.48 ms ± 0.59 ms | 92.9 s | 922 MB | 0 | 480 / 10,192 B | 0 MB | all-config only |
| `lr_top20_bits16` | pass | 0.3482 / 0.3484 | 0.9997 | 8.57 ms ± 0.87 ms | 93.7 s | 920 MB | 0 | 480 / 17,656 B | 0 MB | both |
| `lr_top50_bits8` | **fail** | 0.2496 / 0.3889 | 0.9750 | 11.60 ms ± 3.17 ms | 97.1 s | 1,440 MB | 0 | 1,200 / 9,456 B | 0 MB | — |
| `lr_top50_bits16` | pass | 0.3886 / 0.3889 | 0.9997 | 15.40 ms ± 2.74 ms | 94.5 s | 1,442 MB | 0 | 1,200 / 16,928 B | 0 MB | both |
| `lr_top100_bits8` | **fail** | 0.2829 / 0.4224 | 0.9791 | 18.49 ms ± 0.72 ms | 96.0 s | 2,104 MB | 0 | 2,400 / 10,656 B | 0 MB | — |
| `lr_top100_bits16` | pass | 0.4221 / 0.4224 | 0.9995 | 30.92 ms ± 1.26 ms | 96.3 s | 2,105 MB | 0 | 2,400 / 18,128 B | 0 MB | both |
| `xgboost_top20_bits8` | **fail** | 0.3309 / 0.5227 | 0.6490 | 1,096.7 s ± 12.9 s | 36.3 s | 2,896 MB | 248,452 | 480 / 5,868,336 B | 841 MB | — |
| `xgboost_top20_bits14` | pass | 0.5218 / 0.5227 | 0.9993 | 4,098.7 s ± 1,148.8 s | 34.4 s | 3,433 MB | 383,776 | 480 / 5,868,336 B | 1,796 MB | — |
| `xgboost_top50_bits8` | **fail** | 0.3064 / 0.5442 | 0.6753 | 931.1 s ± 212.8 s | 56.1 s | 2,417 MB | 151,986 | 1,200 / 3,589,848 B | 841 MB | — |
| `xgboost_top50_bits14` | pass | 0.5429 / 0.5442 | 0.9996 | 2,650.6 s ± 771.9 s | 73.1 s | 3,356 MB | 234,768 | 1,200 / 3,589,848 B | 1,796 MB | both |
| `xgboost_top100_bits8` | **fail** | 0.2824 / 0.5678 | 0.8937 | 1,021.2 s ± 237.3 s | 54.8 s | 2,645 MB | 188,074 | 2,400 / 4,442,232 B | 841 MB | — |
| `xgboost_top100_bits14` | pass | 0.5710 / 0.5678 | 0.9972 | 3,687.5 s ± 840.3 s | 111.0 s | 3,468 MB | 290,512 | 2,400 / 4,442,232 B | 1,940 MB | both |

### Pareto frontiers (accuracy = quantized PR-AUC, higher is better; cost lower is better)

| Cost axis | Frontier over all 12 configurations | Frontier over T3-passing configurations |
|---|---|---|
| FHE round-trip latency (mean of 5 repeated executions) (seconds) | `lr_top20_bits8` → `lr_top20_bits16` → `lr_top50_bits16` → `lr_top100_bits16` → `xgboost_top50_bits14` → `xgboost_top100_bits14` | `lr_top20_bits16` → `lr_top50_bits16` → `lr_top100_bits16` → `xgboost_top50_bits14` → `xgboost_top100_bits14` |
| whole-process peak RSS (MB) | `lr_top20_bits16` → `lr_top50_bits16` → `lr_top100_bits16` → `xgboost_top50_bits14` → `xgboost_top100_bits14` | `lr_top20_bits16` → `lr_top50_bits16` → `lr_top100_bits16` → `xgboost_top50_bits14` → `xgboost_top100_bits14` |
| input + output ciphertext per request (bytes) | `lr_top50_bits8` → `lr_top20_bits8` → `lr_top50_bits16` → `lr_top100_bits16` → `xgboost_top50_bits14` → `xgboost_top100_bits14` | `lr_top50_bits16` → `lr_top100_bits16` → `xgboost_top50_bits14` → `xgboost_top100_bits14` |

Latency dominances that hold on the means but not within one standard deviation: `lr_top20_bits16` over `lr_top50_bits8`; `lr_top50_bits16` over `lr_top100_bits8`; `xgboost_top50_bits8` over `xgboost_top100_bits8`; `xgboost_top50_bits14` over `xgboost_top20_bits14`; `xgboost_top100_bits14` over `xgboost_top20_bits14`.

### Feature-count effect (ratio vs `top_20` at the same model and bit-width)

XGBoost cost follows the number of trees the tier's model was trained with (early stopping chose 358 / 219 / 271 for `top_20` / `top_50` / `top_100`), not the feature count directly, so XGBoost cost is not monotonic in features.

| Model, bits | Features | Trees | Quantized PR-AUC | T3 | Latency | × vs top_20 | Peak RSS | Ciphertext total | Bootstraps |
|---|---|---|---|---|---|---|---|---|---|
| lr, 8 | 20 | n/a | 0.3218 | **fail** | 6.48 ms | 1.00× | 922 MB | 10,672 B | 0 |
| lr, 8 | 50 | n/a | 0.2496 | **fail** | 11.60 ms | 1.79× | 1,440 MB | 10,656 B | 0 |
| lr, 8 | 100 | n/a | 0.2829 | **fail** | 18.49 ms | 2.85× | 2,104 MB | 13,056 B | 0 |
| lr, 16 | 20 | n/a | 0.3482 | pass | 8.57 ms | 1.00× | 920 MB | 18,136 B | 0 |
| lr, 16 | 50 | n/a | 0.3886 | pass | 15.40 ms | 1.80× | 1,442 MB | 18,128 B | 0 |
| lr, 16 | 100 | n/a | 0.4221 | pass | 30.92 ms | 3.61× | 2,105 MB | 20,528 B | 0 |
| xgboost, 8 | 20 | 358 | 0.3309 | **fail** | 1,096.7 s | 1.00× | 2,896 MB | 5,868,816 B | 248,452 |
| xgboost, 8 | 50 | 219 | 0.3064 | **fail** | 931.1 s | 0.85× | 2,417 MB | 3,591,048 B | 151,986 |
| xgboost, 8 | 100 | 271 | 0.2824 | **fail** | 1,021.2 s | 0.93× | 2,645 MB | 4,444,632 B | 188,074 |
| xgboost, 14 | 20 | 358 | 0.5218 | pass | 4,098.7 s | 1.00× | 3,433 MB | 5,868,816 B | 383,776 |
| xgboost, 14 | 50 | 219 | 0.5429 | pass | 2,650.6 s | 0.65× | 3,356 MB | 3,591,048 B | 234,768 |
| xgboost, 14 | 100 | 271 | 0.5710 | pass | 3,687.5 s | 0.90× | 3,468 MB | 4,444,632 B | 290,512 |

### Bit-width effect (high vs low, same model and tier)

| Model | Features | Bits low → high | Quantized PR-AUC low → high | PR-AUC gain | T3 low → high | Latency × | Bootstraps × | Bootstrap key × | Peak RSS × |
|---|---|---|---|---|---|---|---|---|---|
| lr | 20 | 8 → 16 | 0.3218 → 0.3482 | +0.0264 | fail → pass | 1.32× | n/a (0) | n/a (0) | 1.00× |
| lr | 50 | 8 → 16 | 0.2496 → 0.3886 | +0.1390 | fail → pass | 1.33× | n/a (0) | n/a (0) | 1.00× |
| lr | 100 | 8 → 16 | 0.2829 → 0.4221 | +0.1392 | fail → pass | 1.67× | n/a (0) | n/a (0) | 1.00× |
| xgboost | 20 | 8 → 14 | 0.3309 → 0.5218 | +0.1908 | fail → pass | 3.74× | 1.54× | 2.14× | 1.19× |
| xgboost | 50 | 8 → 14 | 0.3064 → 0.5429 | +0.2365 | fail → pass | 2.85× | 1.54× | 2.14× | 1.39× |
| xgboost | 100 | 8 → 14 | 0.2824 → 0.5710 | +0.2886 | fail → pass | 3.61× | 1.54× | 2.31× | 1.31× |

### Prior-art sanity check: measured XGBoost latency vs the ~6 ms in `docs/prd.md` §11

Measured cost per programmable bootstrap (mean round-trip latency ÷ bootstrap count; this machine, whole 6-core CPU in use). The cited figure is compared only as a number: the cited work's model size, tree count, bit-width, hardware, threading and whether the figure is per request were not verified here.

| Config | Trees | Bootstraps | Latency | Latency per bootstrap | Latency ÷ 6 ms |
|---|---|---|---|---|---|
| `xgboost_top20_bits8` | 358 | 248,452 | 1,096.7 s | 4.41 ms | 182,782× |
| `xgboost_top20_bits14` | 358 | 383,776 | 4,098.7 s | 10.68 ms | 683,112× |
| `xgboost_top50_bits8` | 219 | 151,986 | 931.1 s | 6.13 ms | 155,177× |
| `xgboost_top50_bits14` | 219 | 234,768 | 2,650.6 s | 11.29 ms | 441,761× |
| `xgboost_top100_bits8` | 271 | 188,074 | 1,021.2 s | 5.43 ms | 170,196× |
| `xgboost_top100_bits14` | 271 | 290,512 | 3,687.5 s | 12.69 ms | 614,578× |

### Model effect (each model at its highest feasible bit-width: LR 16, XGBoost 14)

| Features | LR PR-AUC | XGBoost PR-AUC | XGBoost − LR | LR latency | XGBoost latency | XGBoost ÷ LR latency | LR / XGBoost ciphertext | LR / XGBoost peak RSS |
|---|---|---|---|---|---|---|---|---|
| 20 | 0.3482 | 0.5218 | +0.1735 | 8.57 ms | 4,098.7 s | 478,209× | 18,136 / 5,868,816 B | 920 / 3,433 MB |
| 50 | 0.3886 | 0.5429 | +0.1543 | 15.40 ms | 2,650.6 s | 172,102× | 18,128 / 3,591,048 B | 1,442 / 3,356 MB |
| 100 | 0.4221 | 0.5710 | +0.1489 | 30.92 ms | 3,687.5 s | 119,261× | 20,528 / 4,444,632 B | 2,105 / 3,468 MB |
