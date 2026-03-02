# External Integrations

**Analysis Date:** 2026-03-01

## APIs & External Services

**None detected**
- No external API calls found in codebase
- No HTTP client libraries (requests, httpx, etc.)
- No cloud service SDKs

## Data Storage

**Databases:**
- None - Uses local files only

**File Storage:**
- Local filesystem - All data stored in `data/` directory
  - Input data: CSV files (e.g., `data/circle_0_01_noise.csv`)
  - Training data: NumPy arrays (`.npy` files loaded via `np.load()`)
  - Experiment results: Stored in `results/` directory

**Caching:**
- None - No caching layer detected

## Authentication & Identity

**None detected**
- No authentication providers
- No identity management
- No API keys or credentials required

## Monitoring & Observability

**Error Tracking:**
- None - No external error tracking services (Sentry, etc.)

**Logs:**
- Standard print statements in `src/primel/run.py`
- No structured logging framework
- No log aggregation services

## CI/CD & Deployment

**Hosting:**
- None - Library package, no web hosting

**CI Pipeline:**
- None - No CI configuration detected (no `.github/`, `.gitlab-ci.yml`, etc.)
- Manual execution via `python -m primel.run` or scripts in `experiments/scripts/`

## Environment Configuration

**Required env vars:**
- None - No environment variables required

**Secrets location:**
- No secrets management needed (no external services)

## Webhooks & Callbacks

**Incoming:**
- None - No webhook endpoints

**Outgoing:**
- None - No external HTTP requests

## Git Dependencies

**External Git Packages:**
- `jernerics` - Git dependency from `https://github.com/jerrydzhang/jernerics.git`
  - Purpose: Experiment base classes and utilities
  - Version: 0.1.0
  - Location: Specified in `[tool.uv.sources]` in `pyproject.toml`

## External Data Sources

**Benchmark Datasets:**
- Zhong benchmark datasets stored locally in `data/zhong/`
- These are referenced from research but accessed locally, not fetched dynamically

---

*Integration audit: 2026-03-01*
