# History rewrite notice (2026-10-12)

**Why.** The Kaggle IEEE-CIS competition rules (section 7, read 2026-10-12) allow the data to be used for non-commercial purposes only and prohibit making the Competition Data available to anyone outside the competition. Earlier commits published row-level extracts of the data: `results/phase2_pipeline/split_assignments.csv` (590,540 rows of `TransactionID`, `TransactionDT`, split labels; present on the public `main` from the Phase 2 commit onward) and four `results/phase*/**/execute_sample.csv` files (about 100 `TransactionID` + fraud label rows each). Whether publishing them breached the rules is ambiguous; the owner chose to remove them from the published history rather than rely on that ambiguity.

**What was done.** The five paths were removed from every commit with `git filter-repo --invert-paths` on a fresh clone, verified (no reachable blob of those paths, no remaining blob starting with their headers, identical final file list and identical frozen Phase 12 artifacts), and published with a lease-protected force push. A full mirror of the pre-rewrite repository is kept locally outside the repository (`Ciphraud_pre_remediation_backup.git`) together with the removed local files; they are not published. Local copies remain on disk, git-ignored. `results/phase2_pipeline/split_manifest.json` records the row counts and sha256 values for local regeneration (`src.data.split`).

**Consequences (not fixable by a rewrite).** Forks, clones, pull-request refs or GitHub caches made before the rewrite may still hold the old objects; removing those needs GitHub support. Every commit hash from the Phase 2 commit onward changed. Old hashes cited in `provenance.json` files, reports and audit documents (for example `afb09da`, `f11d750`, `4945dea`) refer to the pre-rewrite history; use the table to find the rewritten commit. The `git_commit` values inside committed result files were deliberately not edited (they are historical evidence). Code and result **contents** are unchanged, apart from the removed files, so recorded `code_file_sha256` values still match.

| Old commit | New commit | Subject |
|---|---|---|
| `0b6aa50` | `543f810` | Rename project to Ciphraud |
| `216d88d` | `2e15f34` | Phase 9: MLP API probe, additive y_train handoff, MLP export and QAT b |
| `41894df` | `1bfb7c6` | docs(research): add evidence-grounded Ciphraud report |
| `47fce77` | `e5c66e1` | Fix LR FHE path: apply the StandardScaler client-side before the compi |
| `4945dea` | `5caf0e2` | Phase 10: client/server encrypted inference (LR, MLP, XGBoost) with ev |
| `51962e2` | `f7760a4` | Add Phase 6: FHE fraud inference with XGBoost across all feature tiers |
| `5fe2dcf` | `91f71b5` | Phase 3: Logistic Regression and XGBoost baselines with class-imbalanc |
| `6527e35` | `a22a5d0` | docs(license): license original code under Apache-2.0 and add third-pa |
| `7ae3b82` | `114ea2d` | docs(portfolio): improve README and project documentation |
| `7e750d5` | `8f76cb7` | feat(docker): add server container and integration checks |
| `8732dcb` | `9daa9c3` | Phase 2 complete: leakage-safe time-based split and preprocessing pipe |
| `8a256b5` | `952786d` | Phase 9: feasibility probe, grid runner, result validator, grid config |
| `8c3fd3d` | `8dd7d4a` | Fix Phase 3 Logistic Regression: unscaled features and insufficient ma |
| `8d246b3` | `2dadd9b` | feat(docker): validate CPU-only LR and MLP encrypted inference |
| `a1172d6` | `7167cf5` | fix(data): stop distributing row-level Kaggle data extracts |
| `afb09da` | `a384988` | Phase 9 closeout: full six-cell MLP grid, FR5/D5 seed stability, Paret |
| `c09cd1f` | `170f21b` | Phase 9: provenance-repair snapshot of the D7/D8 QAT trainer code |
| `cdb4133` | `e356469` | docs: document validation results, limitations, and finalization statu |
| `dc81dd5` | `77344df` | Phase 8: results for all 12 grid configurations, Pareto analysis and r |
| `e745b56` | `1b5be3c` | Phase 8: checkpointed research-grid runner and grid config |
| `e85d16a` | `4aab8f8` | Phase 7: benchmark infrastructure and smoke test |
| `e9898c0` | `3e56e5f` | feat(eval): add frozen test-evaluation tooling and verified model free |
| `f11d750` | `6d001d7` | Phase 8 documentation: research write-up, errata, plan status and proj |
| `f8e8bfe` | `fd26b33` | Phase 4: XGBoost total_gain feature ranking and top-20/50/100 tiers |
| `fe66aef` | `721377d` | Phase 5: FHE proof-of-concept for Logistic Regression on top_20 |
