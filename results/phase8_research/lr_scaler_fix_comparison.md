| Config | Pre-fix agreement | Pre-fix PR-AUC drop | Corrected agreement | Corrected PR-AUC drop | Corrected quantized / float PR-AUC | Corrected T3 |
|---|---|---|---|---|---|---|
| `lr_top20_bits8` | 0.9576 | 0.2930 | 0.98497 | 0.02662 | 0.3218 / 0.3484 | FAIL |
| `lr_top20_bits16` | 0.9614 | 0.2901 | 0.99969 | 0.00020 | 0.3482 / 0.3484 | PASS |
| `lr_top50_bits8` | 0.9672 | 0.3498 | 0.97500 | 0.13928 | 0.2496 / 0.3889 | FAIL |
| `lr_top50_bits16` | 0.9672 | 0.3496 | 0.99968 | 0.00026 | 0.3886 / 0.3889 | PASS |
| `lr_top100_bits8` | 0.9700 | 0.3834 | 0.97907 | 0.13947 | 0.2829 / 0.4224 | FAIL |
| `lr_top100_bits16` | 0.9700 | 0.3831 | 0.99954 | 0.00030 | 0.4221 / 0.4224 | PASS |
