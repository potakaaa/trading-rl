# Learning notes

## Phase 0: Setup

**What was built:** project skeleton, a Python environment, a config file, and a config loader.

**Key concepts**
- **uv** is a fast Python package manager. `uv sync` reads `pyproject.toml`, creates a `.venv`
  folder (an isolated set of packages for this project, like `node_modules`) and installs
  everything. `uv run <cmd>` runs a command inside that environment. `pyproject.toml` is the
  Python equivalent of `package.json`; `uv.lock` is the equivalent of `package-lock.json`.
- **Config in one file:** every tunable number lives in `config/settings.yaml`. The loader
  (`src/config.py`) uses **pydantic**, which validates types (like Zod in TypeScript).
- **Fail safe:** values only you can supply (account size, loss limits, news window, lot value,
  spread) are `null`. `Settings.require_complete()` raises until they are filled in, so the system
  can never silently run with a guessed prop-firm rule.
- **Secrets:** API tokens go in `.env`, which is gitignored. `.env.example` shows the shape.
