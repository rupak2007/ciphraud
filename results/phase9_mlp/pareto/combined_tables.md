### Combined table (validation; latency = mean ± std of 5 repeated executions of one input)

| Config | T3 | Quantized / float(-twin) PR-AUC | Agreement | FHE latency | Compile | Peak RSS | Bootstraps | Ciphertext in / out | Key material | Latency frontier |
|---|---|---|---|---|---|---|---|---|---|---|
| `lr_top20_bits8` | **fail** | 0.3218 / 0.3484 | 0.9850 | 6.48 ms ± 0.59 ms | 92.9 s | 922 MB | 0 | 480 / 10,192 B | 0.00 GB | all-config only |
| `lr_top20_bits16` | pass | 0.3482 / 0.3484 | 0.9997 | 8.57 ms ± 0.87 ms | 93.7 s | 920 MB | 0 | 480 / 17,656 B | 0.00 GB | both |
| `lr_top50_bits8` | **fail** | 0.2496 / 0.3889 | 0.9750 | 11.60 ms ± 3.17 ms | 97.1 s | 1,440 MB | 0 | 1,200 / 9,456 B | 0.00 GB | — |
| `lr_top50_bits16` | pass | 0.3886 / 0.3889 | 0.9997 | 15.40 ms ± 2.74 ms | 94.5 s | 1,442 MB | 0 | 1,200 / 16,928 B | 0.00 GB | both |
| `lr_top100_bits8` | **fail** | 0.2829 / 0.4224 | 0.9791 | 18.49 ms ± 0.72 ms | 96.0 s | 2,104 MB | 0 | 2,400 / 10,656 B | 0.00 GB | — |
| `lr_top100_bits16` | pass | 0.4221 / 0.4224 | 0.9995 | 30.92 ms ± 1.26 ms | 96.3 s | 2,105 MB | 0 | 2,400 / 18,128 B | 0.00 GB | both |
| `xgboost_top20_bits8` | **fail** | 0.3309 / 0.5227 | 0.6490 | 1,096.7 s ± 12.9 s | 36.3 s | 2,896 MB | 248,452 | 480 / 5,868,336 B | 0.96 GB | — |
| `xgboost_top20_bits14` | pass | 0.5218 / 0.5227 | 0.9993 | 4,098.7 s ± 1,148.8 s | 34.4 s | 3,433 MB | 383,776 | 480 / 5,868,336 B | 1.93 GB | — |
| `xgboost_top50_bits8` | **fail** | 0.3064 / 0.5442 | 0.6753 | 931.1 s ± 212.8 s | 56.1 s | 2,417 MB | 151,986 | 1,200 / 3,589,848 B | 0.96 GB | — |
| `xgboost_top50_bits14` | pass | 0.5429 / 0.5442 | 0.9996 | 2,650.6 s ± 771.9 s | 73.1 s | 3,356 MB | 234,768 | 1,200 / 3,589,848 B | 1.93 GB | both |
| `xgboost_top100_bits8` | **fail** | 0.2824 / 0.5678 | 0.8937 | 1,021.2 s ± 237.3 s | 54.8 s | 2,645 MB | 188,074 | 2,400 / 4,442,232 B | 0.96 GB | — |
| `xgboost_top100_bits14` | pass | 0.5710 / 0.5678 | 0.9972 | 3,687.5 s ± 840.3 s | 111.0 s | 3,468 MB | 290,512 | 2,400 / 4,442,232 B | 2.08 GB | both |
| `mlp_top20_bits3` | **fail** | 0.3924 / 0.4892 | 0.9788 | 5.2 s ± 593.93 ms | 5.2 s | 1,671 MB | 560 | 480 / 65,552 B | 0.65 GB | — |
| `mlp_top20_bits4` | **fail** | 0.4070 / 0.4892 | 0.9789 | 72.6 s ± 5.7 s | 7.4 s | 3,530 MB | 640 | 480 / 262,160 B | 2.03 GB | — |
| `mlp_top50_bits3` | **fail** | 0.4173 / 0.4696 | 0.9703 | 17.1 s ± 1.1 s | 5.7 s | 2,856 MB | 1,200 | 1,200 / 131,088 B | 1.19 GB | — |

### MLP configurations without an FHE latency (recorded, not dropped)

| Config | Status | Key material | Bootstraps | T3 | Quantized / float-twin PR-AUC | Reason |
|---|---|---|---|---|---|---|
| `mlp_top50_bits4` | infeasible_key_memory | 4.44 GB | 1800 | **fail** | 0.4324 / 0.4696 | key material 4.44 GB exceeds the 2.4 GB limit for the 3.8 GiB WSL memory; T2/T1/latency not run |
| `mlp_top100_bits3` | infeasible_key_memory | 3.24 GB | 2400 | **fail** | 0.4494 / 0.4908 | key material 3.24 GB exceeds the 2.4 GB limit for the 3.8 GiB WSL memory; T2/T1/latency not run |
| `mlp_top100_bits4` | infeasible_key_memory | 4.80 GB | 3200 | **fail** | 0.4392 / 0.4908 | key material 4.80 GB exceeds the 2.4 GB limit for the 3.8 GiB WSL memory; T2/T1/latency not run |

### Pareto frontiers over LR + XGBoost + MLP (accuracy = quantized PR-AUC ↑; cost ↓)

| Cost axis | Frontier over all configurations | Frontier over T3-passing configurations |
|---|---|---|
| FHE round-trip latency (mean of 5 repeated executions) (seconds) | `lr_top20_bits8` → `lr_top20_bits16` → `lr_top50_bits16` → `lr_top100_bits16` → `xgboost_top50_bits14` → `xgboost_top100_bits14` | `lr_top20_bits16` → `lr_top50_bits16` → `lr_top100_bits16` → `xgboost_top50_bits14` → `xgboost_top100_bits14` |
| whole-process peak RSS (MB) | `lr_top20_bits16` → `lr_top50_bits16` → `mlp_top20_bits3` → `lr_top100_bits16` → `xgboost_top50_bits14` → `xgboost_top100_bits14` | `lr_top20_bits16` → `lr_top50_bits16` → `lr_top100_bits16` → `xgboost_top50_bits14` → `xgboost_top100_bits14` |
| input + output ciphertext per request (bytes) | `lr_top50_bits8` → `lr_top20_bits8` → `lr_top50_bits16` → `lr_top100_bits16` → `xgboost_top50_bits14` → `xgboost_top100_bits14` | `lr_top50_bits16` → `lr_top100_bits16` → `xgboost_top50_bits14` → `xgboost_top100_bits14` |

Latency dominances that hold on the means but not within one standard deviation: `lr_top20_bits16` over `lr_top50_bits8`; `lr_top50_bits16` over `lr_top100_bits8`; `xgboost_top50_bits8` over `xgboost_top100_bits8`; `xgboost_top50_bits14` over `xgboost_top20_bits14`; `xgboost_top100_bits14` over `xgboost_top20_bits14`.

### Tree vs neural network under FHE (each MLP bit-width against LR 16-bit and XGBoost 14-bit at the same tier)

| Features | MLP bits | MLP T3 | MLP PR-AUC | LR-16 PR-AUC | XGB-14 PR-AUC | MLP latency | LR-16 latency | XGB-14 latency | MLP ÷ LR-16 | XGB-14 ÷ MLP | MLP ciphertext | MLP keys | MLP peak RSS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 3 | **fail** | 0.3924 | 0.3482 | 0.5218 | 5.2 s | 8.57 ms | 4,098.7 s | 604× | 792× | 66,032 B | 0.65 GB | 1,671 MB |
| 20 | 4 | **fail** | 0.4070 | 0.3482 | 0.5218 | 72.6 s | 8.57 ms | 4,098.7 s | 8,472× | 56× | 262,640 B | 2.03 GB | 3,530 MB |
| 50 | 3 | **fail** | 0.4173 | 0.3886 | 0.5429 | 17.1 s | 15.40 ms | 2,650.6 s | 1,112× | 155× | 132,288 B | 1.19 GB | 2,856 MB |

### MLP bit-width effect (high vs low)

| Features | Bits | PR-AUC low → high | T3 low → high | Latency × | Bootstraps × | Key material × | Peak RSS × |
|---|---|---|---|---|---|---|---|
| 20 | 3 → 4 | 0.3924 → 0.4070 | fail → fail | 14.04× | 1.14× | 3.13× | 2.11× |

### Prior-art sanity check: MLP latency vs the ~296 ms neural network in `docs/prd.md` §11

The cited figure is compared only as a number; the cited work's network, bit-widths, hardware and definition of latency were not verified here.

| Config | Measured latency | ÷ 296 ms |
|---|---|---|
| `mlp_top20_bits3` | 5.2 s | 17× |
| `mlp_top20_bits4` | 72.6 s | 245× |
| `mlp_top50_bits3` | 17.1 s | 58× |
