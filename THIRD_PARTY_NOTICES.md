# Third-party notices and scope of the Apache-2.0 license

The Apache License 2.0 in [`LICENSE`](LICENSE) applies **only to the original Ciphraud source code, configuration, tests and documentation** written for this project, which the owner has the right to license. It does **not** relicense, and gives no rights in:

- **The IEEE-CIS Fraud Detection data** (Kaggle competition, sponsored by the IEEE Computational Intelligence Society, data from Vesta). The data are not included in this repository. Obtain them from Kaggle and use them under the [competition rules](https://www.kaggle.com/competitions/ieee-fraud-detection/rules) (read 2026-10-12), which allow non-commercial use only and prohibit redistributing the data to anyone outside the competition. No transaction records, row identifiers, labels or per-row extracts of the data are distributed in this repository. Aggregate statistics and model results computed from the data are included as research outputs; whether and how they may be reused is governed by those rules, not by Apache-2.0.
- **Third-party software and models** used as dependencies. They stay under their own licenses and are not distributed in this repository (they are installed from the pinned `requirements*.txt`). Licenses below are the ones declared in the installed package metadata (checked 2026-10-12, WSL venv); verify before redistributing.

| Package (pinned) | Declared license |
|---|---|
| concrete-ml 1.9.0 (Zama) | BSD-3-Clause-Clear |
| concrete-python 2.10.0 (Zama) | BSD-3-Clause |
| brevitas 0.10.2 (AMD) | BSD-3-Clause-style (`LICENSE` in the package) |
| xgboost 1.6.2 | Apache-2.0 |
| scikit-learn 1.5.0 | BSD-3-Clause |
| torch 2.3.1 | BSD-3-Clause |
| numpy 1.26.4, pandas 3.0.5 | BSD-3-Clause |
| onnx 1.17.0 | Apache-2.0 |
| skorch 0.11.0 | BSD-3-Clause |
| hummingbird-ml 0.4.11, fastapi 0.141.1, pydantic 2.13.5 | MIT |
| uvicorn 0.54.0, starlette 1.7.0, httpx 0.28.1 | BSD-3-Clause |

The remaining pinned packages are listed in `requirements*.txt`; their licenses were not individually reviewed here.

## Zama / Concrete-ML note
Concrete-ML and concrete-python are Zama's, under BSD-3-Clause-Clear / BSD-3-Clause. Zama's own documentation states that the BSD-3-Clause-Clear license does not grant patent rights and that commercial use of its FHE technology requires a separate patent license from Zama (see `docs/license_recommendation.md` for what was read and where). This project is a non-commercial research prototype; nothing here grants or implies a patent license, and this is not legal advice. Anyone using Ciphraud commercially should obtain advice and Zama's terms.

Third-party copyright and license notices are preserved in the upstream packages; this file does not replace them.
