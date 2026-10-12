# Remediation plan: `results/phase2_pipeline/split_assignments.csv` (PLAN ONLY — nothing below has been applied)

## Facts (verified 2026-10-12)
- File: 590,540 rows; columns `TransactionID`, `TransactionDT`, `split`, `cv_eval_fold`; no labels, no features; sha256 `4c58eed9b326c15219bb1ba5ff4ba2df1d290868473601792bf48d25cb2c6567`; 14.4 MB.
- Committed in `8732dcb` (Phase 2) and present on `origin/main`; the GitHub repository `rupak2007/ciphraud` is **public**. It is also referenced by `src/data/pipeline.py`, `src/data/split.py`, `src/train/data.py`, several tests and docs (`docs/pipeline.md`, the audit).
- **Reproducible locally:** I regenerated it from `data/raw/train_transaction.csv` (columns `TransactionID`, `TransactionDT` only) with the repository's own `compute_split_boundaries` and `build_split_assignments`, writing to a scratch directory, and the result is **byte-identical** (same sha256). Nothing in the repository was written. The local copy is untouched.
- Whether Kaggle's rules prohibit publishing it is **not established**; the owner is checking the rules.

## Track 1 — Untrack going forward (normal commit; reversible; does NOT remove history)
1. `git rm --cached results/phase2_pipeline/split_assignments.csv` (the local file stays on disk), add the path to `.gitignore`.
2. Commit a small replacement `results/phase2_pipeline/split_manifest.json`: the two boundaries and quantiles (already in `split_boundaries.json`), per-split row counts, `n_rows`, min/max TransactionDT, and the **sha256 of the CSV** so anyone with the raw data can verify a regenerated file. No per-row data.
3. Code/tests/docs that read the CSV keep working from the regenerated local file; update the docs sentence that says the file is a committed deliverable. Tests that need it should regenerate or skip when absent.
4. Effect: new clones of the new `HEAD` no longer contain the file. **Every earlier commit that contains it, and every existing clone, fork, GitHub cache or pull-request ref, still has it.** Untracking removes nothing already published.

## Track 2 — Reproduce locally from the raw dataset (no repository change)
`python -m src.data.pipeline --config configs/phase2/pipeline.yaml` regenerates it, but also rewrites the other tracked `results/phase2_pipeline/*` files, so use a copy of the config with `output.dir` set to a scratch directory and `cache_processed_data: false`, or the direct two-column recipe I ran:
`compute_split_boundaries(df.TransactionDT, config["split"])` then `build_split_assignments(df.TransactionID, df.TransactionDT, boundaries).to_csv(path, index=False, lineterminator="\n")`, then compare its sha256 with the one above. This is deterministic and needs only `train_transaction.csv`.

## Track 3 — Remove it from published history (ONLY if the rules require; destructive; needs explicit approval)
What it involves, and why it is not to be done casually:
- Rewrite every commit from `8732dcb` onward (`git filter-repo --path results/phase2_pipeline/split_assignments.csv --invert-paths`, on a fresh clone) and **force-push** `main`. Every commit hash changes, so the four local unpushed commits must be re-applied onto the rewritten history, and any other clone must be re-cloned.
- **Provenance breaks:** `provenance.json` files and docs cite old hashes (`afb09da`, `f11d750`, `e85d16a0`, `4945dea`, ...). They would need a mapping table old -> new, or an explicit statement that hashes before the rewrite are superseded.
- Does not remove it from forks/clones already made, GitHub's cached views, or dangling objects until GitHub garbage-collects; a support request to GitHub (cached views / sensitive data removal) would be needed for the latter.
- Alternative: create a new public repository from a cleaned history and archive or delete the old one (also changes the URL and breaks existing links).
- Cheaper interim measure the owner can choose: make the repository private until the rules are read (visibility is the owner's setting; I will not change it).

## Order I recommend
1. Read the Kaggle rules (owner). 2. If redistribution of row identifiers is allowed: Track 1 only (or nothing). 3. If not allowed: Track 1 now, then decide Track 3 (or a new repository), and consider making the repository private meanwhile. 4. Do not push anything further until this is decided.

Nothing in Tracks 1 and 3 has been executed; the four unpushed commits remain local.

## Update 2026-10-12: authoritative terms read (still PLAN ONLY; nothing applied)
Source: Kaggle "IEEE-CIS Fraud Detection" Competition Rules, <https://www.kaggle.com/competitions/ieee-fraud-detection/rules>, rendered page read on 2026-10-12 (Section 7 "Competition Data"; the page states the rules were in force for a competition that ended 3 Oct 2019 and is now a late-submission competition). No separate dataset licence page was found; the Rules are the governing text.
- **7.A (use):** the data may be used for non-commercial purposes only, including the competition, forum use, academic research and education.
- **7.B (security / redistribution):** participants must not transmit, duplicate, publish, redistribute or otherwise make the Competition Data available to anyone who is not a participant in the competition, and must take reasonable measures to prevent access by persons who have not agreed to the Rules.
- **Competition Data** is defined as the data or datasets available from the Competition Website (and code provided there).
- Section 8.B/8.C (code sharing, open-source licensing of *submissions*) concerns competition submissions and Kaggle-hosted code sharing; it does not appear to govern an independent research repository, but I am not giving a legal opinion.

**What this means for the repository (assessment, not legal advice):**
- `results/phase2_pipeline/split_assignments.csv` is a verbatim copy of two columns of the Competition Data (`TransactionID`, `TransactionDT`) for all 590,540 training rows, plus derived split labels. Publishing it in a public repository is plausibly "publish/make available" under 7.B. The rules do not say that a column extract is exempt, and they do not say it is covered; the question is **ambiguous, with a real risk**, so compliance cannot be claimed.
- Smaller excerpts also contain per-row data: `results/phase{5,6}_fhe*/**/execute_sample.csv` (a handful of `TransactionID` + true label rows each). Aggregate metadata (`missingness_global.csv`, `column_inventory.csv`) is not row data.
- The non-commercial clause 7.A also limits what anyone may do with results derived from the data; the README should say the project uses the dataset under the Kaggle competition's non-commercial terms and that the dataset is not redistributed.
- The raw CSVs, processed arrays and the sealed test arrays are gitignored (`/data/`); that part is consistent with 7.B.

**Recommendation (one clear one):** take Track 1 now (untrack the CSV and add a hash-bearing manifest) in a separate commit *before any push*, regenerate it locally (verified byte-identical), and do **not** push the four unpushed commits plus any new ones until you decide on Track 3 for the already-public history. The CSV is already on the public `origin/main`, so only Track 3 or a new repository removes it from published history. Both are your call; I did not apply any of it. Untracking in this task was not performed because it changes a file that tests and docs depend on and it does not remove the published copy; it is prepared as a separate decision.
