# Technology Stack

**Analysis Date:** 2026-03-01

## Languages

**Primary:**
- Python 3.12+ - Core language for all source code in `src/`

**Secondary:**
- Nix - Reproducible development environment configuration (`flake.nix`)

## Runtime

**Environment:**
- Python 3.13.7 (current runtime)

**Package Manager:**
- uv 0.8.14
- Lockfile: `uv.lock` (present)

## Frameworks

**Core:**
- Not applicable (library package, not a web framework)

**Testing:**
- pytest 8.4.2 - Unit testing framework

**Build/Dev:**
- Nix - Reproducible development shells (`flake.nix`)
- JupyterLab 4.4.9 - Interactive notebooks and experiments
- python-lsp-server 1.13.1 - Language server for IDE support
- ruff 0.13.0 - Linting and formatting

## Key Dependencies

**Critical:**
- numpy 2.3.3 - Core numerical computing for all array operations and tree evaluation
- scipy 1.16.2 - Scientific computing (stats.qmc for Latin Hypercube Sampling, spatial.distance, optimize.nnls, stats.gaussian_kde)
- pandas 2.3.2 - Data manipulation (CSV loading, data structures)

**Scientific Computing:**
- scikit-learn 1.7.2 - Machine learning utilities
- matplotlib 3.10.6 - Visualization and plotting
- seaborn 0.13.2 - Statistical visualization

**Genetic Programming:**
- gplearn 0.4.2 (optional) - Symbolic regression via genetic programming, used in `src/primel/adapters/gplearn/`
- jernerics 0.1.0 - Git dependency for experiment base classes (`https://github.com/jerrydzhang/jernerics.git`)

## Configuration

**Environment:**
- No environment configuration detected (`.env` not found)
- Configuration via command-line arguments in `src/primel/run.py`
- YAML experiment configs in `experiments/*.yaml`

**Build:**
- `pyproject.toml` - Project metadata and dependencies
- `flake.nix` - Nix development environment
- `uv.lock` - Dependency lockfile

## Platform Requirements

**Development:**
- Python 3.12 or higher
- uv package manager (0.8.14+)
- Nix (optional, for reproducible dev shell)

**Production:**
- Any system supporting Python 3.12+
- No specific deployment target (library package)

---

*Stack analysis: 2026-03-01*
