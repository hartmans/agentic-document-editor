# Repository Guidelines

## Project Structure & Module Organization
- `src/` contains the React + Vite frontend. Components live in `src/components/`, shared audio logic in `src/audio/`, and context providers in `src/context/`.
- `backend.py` is the FastAPI entry point for the backend.
- `agentic_editor/` is the Python package used by the backend.
- `tests/` holds Python tests (pytest).
- `index.html`, `vite.config.ts`, and `tsconfig*.json` configure the frontend build.

## Build, Test, and Development Commands
- `yarn dev` starts the Vite dev server for the frontend.
- Python requires a venv to work; try .venv in the repo falling back to ~/ai/venv/ as your venv
- PYTHONPATH needs to include ~/ai/ai-tools
- `python backend.py --document-path path/to/file.txt --port 8000` runs the FastAPI backend (expects a UTF-8 text document).
- `pytest` runs backend unit tests in `tests/`.
- Tests require network. Do not mock components to avoid network or to avoid dependencies; ask for network access and ask for help if there is a missing dependency.

## Coding Style & Naming Conventions
- TypeScript/TSX uses ES modules and React function components. Component files are PascalCase (e.g., `src/components/DocumentEditor.tsx`); utilities use descriptive camelCase names.
- Follow `.editorconfig` for 2-space indentation in JS/JSON/YML; keep existing formatting in TS/TSX and Python files.
- Python modules and functions use snake_case; keep type hints and docstrings where present.

## Commit & Pull Request Guidelines
- If it looks like unrelated changes are uncommitted at the beginning of your work, ask for confirmation.
- Ask for help when you are ready to commit; do not commit yourself until you are told to do so.
