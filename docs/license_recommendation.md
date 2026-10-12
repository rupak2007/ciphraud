# License review and recommendation (DRAFT for the owner; no license file has been added)

Not legal advice. Statements marked **[official]** come from the licensor's own repository text fetched 2026-10-12; **[unverified]** items were not confirmed from an authoritative source.

## 0. Repository visibility (found 2026-10-12)

`gh repo view rupak2007/ciphraud` reports **visibility PUBLIC** (https://github.com/rupak2007/ciphraud). Commits up to Phase 9 (`afb09da`) are on `origin/main`, so everything in them, including the file in section 2, is already publicly readable. Four later local commits (Phase 10-13) are not pushed. With no license file, public readers can view but have no granted right to copy or reuse (default copyright); that is separate from the dataset question.

## 1. Our original work (code, configs, tests, docs, report)

Written for this project, so the owner chooses its license. Points to settle: whether an employer or institution has any claim on it, and that parts were produced with an AI assistant (authorship/copyright treatment of AI-assisted material varies by jurisdiction **[unverified]**).
Options: MIT (shortest, no patent clause); Apache-2.0 (explicit patent grant from contributors, NOTICE requirement); source-available (restricts some uses, not open source); no license (current state; nobody has a right to reuse).
Neither MIT nor Apache-2.0 changes the third-party terms in section 3.

## 2. Kaggle IEEE-CIS dataset-derived artifacts

The dataset itself is not committed (`/data/` is ignored; `git ls-files` finds no raw file). Dataset-derived content that *is* committed:

| Artifact | What it contains | Row-level? |
|---|---|---|
| `results/phase2_pipeline/split_assignments.csv` (14.4 MB, committed in `8732dcb`, **already on the public `origin/main`**) | 590,540 rows with columns `TransactionID`, `TransactionDT`, `split` (train 380,815 / val 105,088 / test 104,637) and `cv_eval_fold` (blank, 0, 1 or 2). IDs run 2,987,000 to 3,577,539, all unique; `TransactionDT` 86,400 to 15,811,131. **No labels, no features, no `isFraud`.** | Yes: it enumerates every competition training transaction by its raw ID and timestamp (two raw columns) |
| `results/phase1_eda/*.csv`, `results/*/metrics.json`, feature rankings, tiers | Aggregates (counts, rates, importances, metrics), feature *names* | No |
| `results/phase10_client_server/deploy/*/reference.json` | Validation row positions, not IDs; integer model outputs; a hash of each quantized input | Per-row outputs of two validation rows (positions, not data) |
| Handoffs, caches, models | In ignored directories, not committed | n/a |

I could not retrieve the competition rules text; **the owner will read the IEEE-CIS rules in their Kaggle account.** Until that is settled the CSV is treated as potentially restricted.

**Replacement / removal plan (nothing executed):**
1. Keep the only local copy; it stays on disk. It is also reproducible (deterministic) from the raw data with `python -m src.data.pipeline --config configs/phase2/pipeline.yaml`.
2. Stop tracking it going forward (`git rm --cached`, add the path to `.gitignore`), and keep the two split boundaries (`split_boundaries.json`, 2 numbers + quantiles) and the file's sha256 and row counts in a small committed JSON so the split stays verifiable. Update the few tests/docs that read it to use the regenerated local file.
3. Be clear that step 2 does not remove it from the public history. Purging requires rewriting published history (e.g. `git filter-repo`) and force-pushing, which also changes every commit hash and invalidates the provenance records that cite `afb09da` and earlier commits; alternatively delete and recreate the public repository from a cleaned history. Neither will be done without explicit approval, and the decision depends on what the rules say.
4. Until decided: do not push anything further.

## 3. Third-party dependencies

Not copied into the repository; installed from `requirements*.txt`. Installed-metadata licenses (WSL venv): xgboost Apache-2.0, torch BSD-3, scikit-learn BSD, pandas BSD-3, httpx BSD-3, uvicorn BSD-3, fastapi MIT, numpy BSD-style, concrete-ml and concrete-ml-extensions BSD-3-Clause-Clear, concrete-python BSD-3-Clause; Brevitas's installed license field is empty (**[unverified]**, check the repository; the Xilinx/AMD repository is the source).

**Zama (Concrete-ML, Concrete)** — checked on the licensor's own GitHub repositories:
- `zama-ai/concrete-ml` LICENSE: "BSD 3-Clause Clear License" **[official]**. The file contains the sentence "NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE." and no additional patent exemption **[official]**.
- `zama-ai/concrete` LICENSE.txt: the same BSD 3-Clause Clear text with the same patent sentence **[official]**.
- The README/FAQ of both repositories (read via the repository pages) state that the libraries are free for development, research, prototyping and experimentation, that commercial use needs a separate commercial patent license from Zama, and that Zama's technologies are patented, with `hello@zama.ai` named as the contact **[official, repository README/FAQ summary]**. I did not retrieve a separate legal terms page or the precise wording of the research exemption, so the exact scope of "research/prototyping" and anything beyond that is **not established here**.
- Consequence for this project: using the libraries for this research project is the described use. Anyone who wants to use this repository's code in a commercial setting should obtain Zama's terms themselves; the repository's own license cannot grant Zama's patent rights. State this in the README if a license is added.

## 4. MIT vs Apache-2.0 (for section 1 only)

| | MIT | Apache-2.0 |
|---|---|---|
| Simplicity | Very short | Longer |
| Patent grant | None | Explicit, from contributors |
| Notices | Keep copyright and license text | Plus NOTICE and change notices |
| Fit | Simple portfolio repo | Reuse by others, near patent-heavy FHE code |

Recommendation: Apache-2.0 if reuse by others is expected, MIT otherwise; first settle section 2, add a short third-party and dataset notice (Zama terms, Kaggle data not included and to be obtained from Kaggle under its rules) to the README, and confirm Brevitas's license. No license has been added.
