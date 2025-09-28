# Repository Guidelines

## Project Structure & Module Organization
- `graphrag/`: core ingestion, graph extraction, and language-model orchestration logic. Key entrypoints live under `graphrag/index/operations` and `graphrag/language_model/`.
- `scripts/`: operational helpers (e.g., `test_llm.py`, `exportcsvtotxt.py`) wrapped for local troubleshooting.
- `tests/`: unit, smoke, and integration suites; mirrors package layout in `graphrag/` for easy discovery.
- `sap/` & `sap_new/`: example datasets/configuration used in smoke runs; outputs land in `sap/output/`.
- `docs/`, `examples_notebooks/`, and `unified-search-app/`: supporting assets and demos—keep them in sync when APIs change.

## Build, Test, and Development Commands
- `uv run poe index --root ./sap --method fast`: rebuilds the sample SAP index end-to-end.
- `uv run poe query --root ./sap --method basic --query "..."`: exercises the QA path against the freshly indexed data.
- `uv run poe check`: runs `ruff format`, `ruff check`, and `pyright`; required before every PR.
- `uv run pytest tests/unit`: targeted unit test execution; add `-k <pattern>` for focused runs.

## Coding Style & Naming Conventions
- Python is formatted with `ruff format`; keep imports sorted and prefer type annotations on public APIs.
- Follow existing naming: modules snake_case, classes PascalCase, async coroutine names suffixed with verbs (e.g., `_process_document`).
- Avoid introducing print statements—use `logging.getLogger(__name__)` with contextual `extra` payloads.

## Testing Guidelines
- Unit tests use `pytest` + asyncio fixtures; place provider-specific cases under `tests/unit/<area>/` with descriptive filenames.
- Smoke scenarios live in `tests/smoke/` and should be deterministic against bundled data.
- When adding features, supply regression coverage and ensure `uv run poe check` passes locally before pushing.

## Commit & Pull Request Guidelines
- Write imperative, scoped commit subjects (e.g., `fix: sanitize graph extractor braces`).
- Each PR should describe the change, list validation commands (`uv run poe check`, targeted `pytest`), and call out any data migrations or config updates (`sap_new/settings.yaml`, LanceDB directories).
- Link related issues and attach logs or screenshots when UI-facing artifacts (e.g., `unified-search-app`) are affected.

## Security & Configuration Tips
- Secrets are read from environment variables; never commit credentials or `.env` files.
- Use the sample configs in `sap_new/settings.yaml` as references—copy before editing and document any new keys in `docs/`.
- Review `SECURITY.md` before disclosing vulnerabilities and follow the Microsoft disclosure process.
