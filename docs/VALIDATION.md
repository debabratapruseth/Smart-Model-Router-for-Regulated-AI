# Validation record

Validated locally on 2026-09-28 using Python 3.13, CPU only. Exact package versions
are recorded in `requirements-lock.txt` and each benchmark manifest.

- `pytest -q`: **71 passed**. One upstream Starlette/httpx deprecation warning; no test failures.
- All four YAML configurations load; Python modules compile.
- FastAPI started successfully and `/health` returned HTTP 200.
- Streamlit started successfully and `/_stcore/health` returned HTTP 200.
- Streamlit AppTest rendered the developer view, submitted a request, and rendered the executive view without exceptions.
- Default `python -m evaluation.benchmark`: **1,000 requests × 7 strategies = 7,000 rows**, without API keys.
- Local available strategies and Jev mock: **zero observed policy violations**; 745 routed and 255 no-route decisions per strategy on this workload.
- Default ML and LLM are explicitly unavailable. An additional trained-ML benchmark is preserved under `reports/trained_ml/`.
- Adversarial evaluation: **56 checks passed** across 12 named cases and five local/mock strategies (missing metadata is rejected once by schema validation).
- Resilience evaluation: **1,050 scenarios evaluated**, zero violations of configured known-state policy. Simulated-world failures are reported separately.
- Ablation, sensitivity, statistical, confidence-calibration, frontier and paraphrase-stability reports generated successfully.
- Audit append/replay, concurrency, prompt omission, invalid-selection rejection, optional-adapter payload handling and execution input integrity have passing tests.
- Downstream execution remains disabled; `data/execution_results.csv` has a header and **zero execution rows**. API contracts were tested with mocks.
- A scoped scan found no API-key/private-key patterns in project sources. No GPU, LangChain or LangGraph packages are required. No secrets were created.

These checks validate the educational implementation, not production compliance.
The synthetic results do not establish real savings or maintained downstream
quality. Weighted routing's acceptable-route accuracy is lower than the strongest
eligible baseline on the supplied quality-band labels; that tradeoff is visible
in the reports. Live OpenAI and Jev services were not exercised.
