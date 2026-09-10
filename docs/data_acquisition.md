# Data Acquisition — IEEE-CIS Fraud Detection

This project uses the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)
Kaggle competition dataset. As of Phase 1, this dataset is **not present on the development
machine** and there is no Kaggle CLI or API credential configured -- per project decision, it is
acquired by **manual download**, not a scripted API call.

## Why manual download

- No new dependency (`kaggle` package) or credential (`kaggle.json`) needed.
- You must personally accept the competition rules on Kaggle regardless of the acquisition
  method -- an API token doesn't bypass that requirement, so scripting the download buys little
  here.

## Procedure

1. Go to the [IEEE-CIS Fraud Detection competition page](https://www.kaggle.com/c/ieee-fraud-detection)
   on Kaggle (sign in if needed) and accept the competition rules.
2. Download `ieee-fraud-detection.zip` from the competition's Data tab.
3. Extract **only** these two files into `data/raw/` (create the directory if it doesn't exist):
   - `train_transaction.csv` (~650 MB)
   - `train_identity.csv` (~26 MB)

   Do **not** extract `test_transaction.csv`, `test_identity.csv`, or `sample_submission.csv` --
   the competition's test files are unlabeled (no `isFraud` column), and this project derives its
   own train/validation/test split from a time-based split of the labeled training data
   (`docs/prd.md` FR2). Extracting them wastes ~600 MB of disk for files this project never reads.

   Final layout:
   ```
   data/raw/
     train_transaction.csv
     train_identity.csv
   ```

4. `data/` is fully gitignored (`.gitignore` line 18: `/data/`) -- nothing here is ever committed,
   and that's intentional: the Kaggle competition rules do not permit redistributing the data.

## Integrity verification

`configs/phase1/eda.yaml` has an `data.expected_sha256` block with empty placeholders. On the
**first** run of `python -m src.data.eda`, the script computes the actual SHA256 of each file and
logs it (structured log line, `verified: false`) without failing -- there is nothing to check
against yet. Record those two hashes into the config afterward; every subsequent run then verifies
against them and **fails loudly** on any mismatch, which catches a truncated or corrupted
re-download.

## Disk / memory expectations

- ~1.4 GB free disk for the zip plus the two extracted CSVs (delete the zip after extracting if
  space is tight; only the two CSVs are needed).
- See `docs/environment.md` for the host's measured RAM (7.95 GB) and the memory-aware loading
  strategy `src/data/load.py` and `src/data/eda.py` use to stay well within it.

## After downloading

Once the files are in place, run:

```bash
python -m src.data.eda --config configs/phase1/eda.yaml
```

This runs entirely on the Windows `.venv` -- no WSL2/FHE toolchain is needed for Phase 1. See
`docs/eda.md` for the resulting findings (generated after this step; not written until the dataset
is actually available and the script has been run against it).
